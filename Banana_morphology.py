# -----------------------------
# Standard Library Imports
# -----------------------------
import sys
import os

# -----------------------------
# Scientific Computing
# -----------------------------
import numpy as np
import pandas as pd

# -----------------------------
# Visualization Libraries
# -----------------------------
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import plotly.graph_objects as go

# -----------------------------
# Statistical Analysis
# -----------------------------
from scipy import stats, interpolate
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA
from tabulate import tabulate

# -----------------------------
# PyQt6 Widgets and Core
# -----------------------------
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QInputDialog, QSplitter, QTextEdit, QListWidget, QAbstractItemView, 
    QLineEdit, QListWidgetItem, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, Qt, QObject, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

# -----------------------------
# Matplotlib Configuration
# -----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# -----------------------------
# Data import & Cleaning
# -----------------------------
class DataImportCleanTab(QWidget):
    dfChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.df = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        btn_import = QPushButton("Import Data")
        btn_import.clicked.connect(self.import_data)
        control_layout.addWidget(btn_import)

        control_layout.addWidget(QLabel("Missing Value Handling:"))
        self.cb_missing = QComboBox()
        self.cb_missing.addItems(["Delete Missing", "Mean Imputation", "Median Imputation"])
        control_layout.addWidget(self.cb_missing)

        btn_clean = QPushButton("Clean Data")
        btn_clean.clicked.connect(self.clean_data)
        control_layout.addWidget(btn_clean)

        main_layout.addLayout(control_layout)


        self.table_preview = QTableWidget()
        self.table_preview.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        main_layout.addWidget(self.table_preview)

        main_layout.addSpacing(10)

        self.table_stats = QTableWidget()
        self.table_stats.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_stats.setMaximumHeight(250)
        main_layout.addWidget(self.table_stats)

        self.setLayout(main_layout)

    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "Data Files (*.csv *.xlsx *.xls);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            else:
                self.df = pd.read_excel(file_path)
            self.df.columns = self.df.columns.str.strip()
            required_columns = ['Genotype', 'Growth_season', 'DAF', 'Replicate']
            missing = [col for col in required_columns if col not in self.df.columns]
            if missing:
                self.df = None
                return
            self.preview_data()
            self.dfChanged.emit()
        except:
            self.df = None

    def preview_data(self):
        if self.df is None:
            return

        self.table_preview.setRowCount(min(50, len(self.df)))
        self.table_preview.setColumnCount(len(self.df.columns))
        self.table_preview.setHorizontalHeaderLabels(self.df.columns)
        for i in range(self.table_preview.rowCount()):
            for j in range(self.table_preview.columnCount()):
                val = str(self.df.iloc[i, j])
                item = QTableWidgetItem(val[:50])
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.table_preview.setItem(i, j, item)

        self.show_statistics()

    def show_statistics(self):
        if self.df is None:
            return
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        non_metric_cols = ['DAF', 'Replicate']
        metric_cols = [col for col in numeric_cols if col not in non_metric_cols]
        if not metric_cols:
            return

        stats_data = []
        grouped = self.df.groupby(['Genotype', 'Growth_season', 'DAF'])[metric_cols].mean().reset_index()
        for col in metric_cols:
            mean_val = grouped[col].mean()
            std_val = grouped[col].std()
            min_val = grouped[col].min()
            max_val = grouped[col].max()
            count_val = grouped[col].count()
            missing_val = self.df[col].isna().sum()
            stats_data.append([
                col,
                f"{mean_val:.4f}",
                f"{std_val:.4f}",
                f"{min_val:.4f}",
                f"{max_val:.4f}",
                count_val,
                missing_val
            ])

        headers = ['Metric','Mean','Std','Min','Max','Count','Missing']
        self.table_stats.setColumnCount(len(headers))
        self.table_stats.setHorizontalHeaderLabels(headers)
        self.table_stats.setRowCount(len(stats_data))

        for i, row in enumerate(stats_data):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.table_stats.setItem(i, j, item)

    def clean_data(self):
        if self.df is None:
            return
        method = self.cb_missing.currentText()
        try:
            if method == "Delete Missing":
                self.df = self.df.dropna()
            elif method == "Mean Imputation":
                self.df = self.df.fillna(self.df.mean(numeric_only=True))
            else:
                self.df = self.df.fillna(self.df.median(numeric_only=True))
            self.preview_data()
            self.dfChanged.emit()
        except:
            pass

# -----------------------------
# Statistical Analysis (All Metrics ANOVA)
# -----------------------------
class AnalysisTab(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

        # Correlation
        btn_corr = QPushButton("Correlation Analysis")
        btn_corr.clicked.connect(self.show_correlation)

        # Multi-factor ANOVA
        btn_anova_all = QPushButton("Run Multi-factor ANOVA for All Metrics")
        btn_anova_all.clicked.connect(self.run_anova_all)

 
        btn_export_heatmap = QPushButton("Export Heatmap Data")
        btn_export_heatmap.clicked.connect(self.export_heatmap_data)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        btn_export_results = QPushButton("Export ANOVA Results to File")
        btn_export_results.clicked.connect(self.export_results_to_file)

        layout.addWidget(QLabel("=== Correlation Analysis ==="))
        layout.addWidget(btn_corr)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("=== Multi-factor ANOVA ==="))
        layout.addWidget(btn_anova_all)
        layout.addWidget(self.result_text)
        layout.addWidget(btn_export_results)
        layout.addWidget(btn_export_heatmap)
        self.setLayout(layout)

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    def show_correlation(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            QMessageBox.warning(self, "Warning", "Not enough numeric data")
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["DAF", "Replicate"]]
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Warning", "Not enough numeric data")
            return
        df_avg = df.groupby(['Genotype','DAF'])[numeric_cols].mean()
        corr_matrix = df_avg.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Correlation Coefficient'}, square=True, mask=mask)
        ax.set_title("Trait Correlation Heatmap (Upper Triangle)")
        self.canvas.draw()

    # -----------------------------
    # Multi-factor ANOVA for All Metrics
    # -----------------------------
    def run_anova_all(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            QMessageBox.warning(self, "Warning", "Please import data first!")
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["DAF", "Replicate"]]
        if not numeric_cols:
            QMessageBox.warning(self, "Warning", "No numeric metrics available")
            return

        results_text = ""
        try:
            df['Genotype'] = df['Genotype'].astype(str)
            df['Growth_season'] = df['Growth_season'].astype(str)
            df['DAF'] = df['DAF'].astype(str)

            for metric in numeric_cols:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
                df_metric = df.dropna(subset=[metric])

                factors = []
                if df_metric['Genotype'].nunique() > 1:
                    factors.append("C(Genotype)")
                if df_metric['Growth_season'].nunique() > 1:
                    factors.append("C(Growth_season)")
                if df_metric['DAF'].nunique() > 1:
                    factors.append("C(DAF)")

                if not factors:
                    results_text += f"Metric: {metric} — skipped (no varying factors)\n\n"
                    continue

                formula_parts = factors[:]
                for i in range(len(factors)):
                    for j in range(i+1, len(factors)):
                        formula_parts.append(f"{factors[i]}:{factors[j]}")
                if len(factors) == 3:
                    formula_parts.append(":".join(factors))

                formula = f"{metric} ~ " + " + ".join(formula_parts)

                model = ols(formula, data=df_metric).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)

                results_text += f"【Multi-factor ANOVA】\nMetric: {metric}\n\n{anova_table}\n\n"

                if df_metric['Genotype'].nunique() > 1:
                    tukey = pairwise_tukeyhsd(endog=df_metric[metric],
                                            groups=df_metric['Genotype'],
                                            alpha=0.05)
                    results_text += f"【Tukey HSD Post-hoc Test for Genotype】\n{tukey.summary()}\n\n"

            self.result_text.setPlainText(results_text)

        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", str(e))

    # -----------------------------
    # Export ANOVA Results
    # -----------------------------
    def export_results_to_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ANOVA Results", "anova_results.txt", "Text Files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.result_text.toPlainText())
                QMessageBox.information(self, "Saved", f"Results saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", str(e))

    # -----------------------------
    # Heatmap Data Export
    # -----------------------------
    def export_heatmap_data(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            QMessageBox.warning(self, "Warning", "Please import data first!")
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["DAF", "Replicate"]]
        if not numeric_cols:
            QMessageBox.warning(self, "Warning", "No numeric metrics available")
            return

        try:
            df_avg = df.groupby(['Genotype','DAF'])[numeric_cols].mean().reset_index()
            df_heatmap = df_avg.copy()
            df_heatmap.insert(0, 'Sample', df_heatmap['Genotype'] + "_DAF" + df_heatmap['DAF'].astype(str))
            df_heatmap = df_heatmap.drop(columns=['Genotype','DAF'])

            df_std = df_heatmap.copy()
            df_std[numeric_cols] = (df_std[numeric_cols] - df_std[numeric_cols].mean()) / df_std[numeric_cols].std()

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Heatmap Data", "heatmap_data.xlsx", "Excel Files (*.xlsx)"
            )
            if file_path:
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                with pd.ExcelWriter(file_path) as writer:
                    df_heatmap.to_excel(writer, sheet_name="RawMatrix", index=False)
                    df_std.to_excel(writer, sheet_name="ZscoreMatrix", index=False)
                QMessageBox.information(self, "Saved", f"Heatmap data saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

# -----------------------------
# Growth Analysis
# -----------------------------
class GrowthCurveTab(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        self.lst_Genotype = QListWidget()
        self.lst_Genotype.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.lst_Genotype.setFixedHeight(80)
        self.lst_Genotype.setFixedWidth(120)


        self.cb_metric_x = QComboBox()
        self.cb_metric_y = QComboBox()

        btn_plot = QPushButton("Plot Curve")
        btn_plot.clicked.connect(self.plot_curve)

        btn_export = QPushButton("Export Mean and Error")
        btn_export.clicked.connect(self.export_all_stats)

        control_layout.addWidget(QLabel("Select Genotype:"))
        control_layout.addWidget(self.lst_Genotype)
        control_layout.addWidget(QLabel("X-axis Metric:"))
        control_layout.addWidget(self.cb_metric_x)
        control_layout.addWidget(QLabel("Y-axis Metric:"))
        control_layout.addWidget(self.cb_metric_y)
        control_layout.addWidget(btn_plot)
        control_layout.addWidget(btn_export)


        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addLayout(control_layout)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_controls(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            return


        self.lst_Genotype.clear()
        for v in df['Genotype'].unique():
            self.lst_Genotype.addItem(str(v))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_defaults = ["DAF"]
        x_choices = x_defaults + [c for c in numeric_cols if c not in x_defaults + ["Replicate"]]
        y_choices = [c for c in numeric_cols if c not in x_defaults + ["Replicate"]]

        self.cb_metric_x.clear()
        self.cb_metric_x.addItems(x_choices)
        self.cb_metric_y.clear()
        self.cb_metric_y.addItems(y_choices)

    def plot_curve(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            QMessageBox.warning(self, "Warning", "Please import data first!")
            return

        selected_items = self.lst_Genotype.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one Genotype!")
            return
        varieties = [item.text() for item in selected_items]

        x_metric = self.cb_metric_x.currentText()
        y_metric = self.cb_metric_y.currentText()
        if x_metric == y_metric:
            QMessageBox.warning(self, "Error", "X and Y metrics cannot be the same!")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        colors = plt.cm.tab10.colors

        for i, Genotype in enumerate(varieties):
            data = df[df['Genotype'] == Genotype].copy()
            if data.empty:
                continue

            grouped = data.groupby('DAF')[[x_metric, y_metric]].agg(['mean','std']).reset_index()
            grouped.columns = ['DAF', f'{x_metric}_mean', f'{x_metric}_std', f'{y_metric}_mean', f'{y_metric}_std']

            grouped = grouped.sort_values(by=f'{x_metric}_mean')

            x_mean = grouped[f'{x_metric}_mean'].values
            y_mean = grouped[f'{y_metric}_mean'].values
            x_std = grouped[f'{x_metric}_std'].values
            y_std = grouped[f'{y_metric}_std'].values

            ax.scatter(x_mean, y_mean, color=colors[i % 10], label=Genotype, s=40, zorder=3)

            ax.plot(x_mean, y_mean, '-', color=colors[i % 10], linewidth=1)

            if x_metric == "DAF":
                ax.errorbar(x_mean, y_mean, yerr=y_std, fmt='none', color=colors[i % 10], capsize=3)
            else:
                ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='none', color=colors[i % 10], capsize=3)

        if ax.lines:
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
            ax.set_title(f'Growth Curve: {y_metric} vs {x_metric} (Replicate Std Dev)')
            ax.legend()
            ax.grid(True)
            self.canvas.draw()
        else:
            QMessageBox.warning(self, "Warning", "No valid data for selected varieties")

    def export_all_stats(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            QMessageBox.warning(self, "Warning", "Please import data first!")
            return


        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_metric_cols = ['DAF', 'Replicate']
        metric_cols = [col for col in numeric_cols if col not in non_metric_cols]

        all_data = []
        for Genotype in df['Genotype'].unique():
            df_var = df[df['Genotype'] == Genotype]
            grouped = df_var.groupby('DAF')[metric_cols].agg(['mean','std']).reset_index()
            # 扁平化列名
            grouped.columns = ['DAF'] + [f"{col}_{stat}" for col, stat in grouped.columns[1:]]
            grouped.insert(0, 'Genotype', Genotype)
            all_data.append(grouped)

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            # Save to Excel
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save All Mean and Error", "Parameter_Mean_Error.xlsx", "Excel Files (*.xlsx)"
            )
            if file_path:
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                result_df.to_excel(file_path, index=False)
                QMessageBox.information(self, "Complete", f"Data saved to {file_path}")


# -----------------------------
# Growth Modeling
# -----------------------------
class ModelTab(QWidget):
    def __init__(self, parent=None, data_handler=None):
        super().__init__(parent)
        self.df = None
        self.metrics = []
        self.fitted_params = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        
        btn_import = QPushButton("Import Mean and Error Data")
        btn_import.clicked.connect(self.import_data)
        control_layout.addWidget(btn_import)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("X-axis:"))
        self.x_combo = QComboBox()
        hlayout.addWidget(self.x_combo)
        hlayout.addWidget(QLabel("Y-axis:"))
        self.y_combo = QComboBox()
        hlayout.addWidget(self.y_combo)
        control_layout.addLayout(hlayout)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(QLabel("Model:"))
        self.fit_combo = QComboBox()
        self.fit_combo.addItems([
            "Linear", "Quadratic", "Cubic", 
            "Logistic", "Gompertz", "Richards", "Weibull"
        ])
        hlayout2.addWidget(self.fit_combo)

        hlayout2.addWidget(QLabel("By Genotype:"))
        self.by_Genotype_combo = QComboBox()
        self.by_Genotype_combo.addItems(["No", "Yes"])
        hlayout2.addWidget(self.by_Genotype_combo)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(self.plot_and_fit)
        hlayout2.addWidget(btn_fit)

        btn_export = QPushButton("Export Fit Parameters")
        btn_export.clicked.connect(self.export_fit_params)
        hlayout2.addWidget(btn_export)

        control_layout.addLayout(hlayout2)
        layout.addLayout(control_layout)

        self.figure = plt.figure(figsize=(7,5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not file_path: return

        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            else:
                self.df = pd.read_excel(file_path)
            self.process_imported_data()
            self.update_controls()
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", f"Error: {str(e)}")

    def process_imported_data(self):
        if self.df is None: return
        self.metrics = [
            col.replace('_mean','') for col in self.df.columns if col.endswith('_mean')
        ]
        if "DAF" in self.df.columns and "DAF" not in self.metrics:
            self.metrics.insert(0, "DAF")

    def update_controls(self):
        if not self.metrics: return
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(self.metrics)
        self.y_combo.addItems(self.metrics)
        if "DAF" in self.metrics:
            self.x_combo.setCurrentText("DAF")
        if len(self.metrics) > 1:
            self.y_combo.setCurrentText(self.metrics[1])

    def plot_and_fit(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please import data first")
            return

        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        fit_type = self.fit_combo.currentText()
        by_Genotype = self.by_Genotype_combo.currentText() == "Yes"

        if x_col == y_col:
            QMessageBox.warning(self, "Error", "X and Y variables cannot be the same")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if by_Genotype and "Genotype" in self.df.columns:
            groups = list(self.df.groupby("Genotype"))
        else:
            groups = [("Overall", self.df)]

        self.fitted_params.clear()
        n = len(groups)
        formula_start = 0.05
        formula_gap = 0.1
        formula_positions = [formula_start + i*formula_gap for i in range(n)]
        formula_positions.reverse()

        for (label, group), pos_y in zip(groups, formula_positions):
            self._plot_single_fit(ax, group, x_col, y_col, fit_type, str(label), pos_y)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(loc='upper left')
        ax.grid(True)
        self.canvas.draw()

    def _plot_single_fit(self, ax, df, x_col, y_col, fit_type, label, pos_y):
        try:
            x_mean_col = f"{x_col}_mean" if x_col != "DAF" else x_col
            y_mean_col = f"{y_col}_mean"
            if x_mean_col not in df.columns or y_mean_col not in df.columns: return

            x = df[x_mean_col].values
            y = df[y_mean_col].values

            x_std_col = f"{x_col}_std"
            y_std_col = f"{y_col}_std"
            xerr = df[x_std_col].values if x_std_col in df.columns else np.zeros_like(x)
            yerr = df[y_std_col].values if y_std_col in df.columns else np.zeros_like(y)

            line = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=3)
            color = line[0].get_color()

            model_config = {
                "Linear": {"func": lambda x,a,b: a*x+b, "p0":[1,0]},
                "Quadratic": {"func": lambda x,a,b,c: a*x**2+b*x+c, "p0":[0.1,0.1,0]},
                "Cubic": {"func": lambda x,a,b,c,d: a*x**3+b*x**2+c*x+d, "p0":[0.01,0.01,0.1,0]},
                "Logistic": {"func": lambda x,K,r,x0: K/(1+np.exp(-r*(x-x0))), "p0":[np.max(y)*1.1,0.5,np.median(x)]},
                "Gompertz": {"func": lambda x,a,b,c: a*np.exp(-b*np.exp(-c*x)), "p0":[np.max(y)*1.1,1,0.1]},
                "Richards": {"func": lambda x,K,a,v,q: K/((1+a*np.exp(-v*x))**(1/q)), "p0":[np.max(y)*1.1,1,0.5,1]},
                "Weibull": {"func": lambda x,a,b,c: a*(1-np.exp(-b*x**c)), "p0":[np.max(y)*1.1,0.1,1]}
            }

            config = model_config.get(fit_type, model_config["Linear"])
            func = config["func"]
            p0 = config["p0"][:func.__code__.co_argcount-1]

            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=100000)
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = func(x_fit, *popt)
            ax.plot(x_fit, y_fit, color=color, linestyle='--', label=f'{label} Fitted Curve')

            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0

            if label not in self.fitted_params:
                self.fitted_params[label] = {}
            self.fitted_params[label][fit_type] = {"popt": popt, "r2": r2}

            formula_latex = f"{label}: {self._formula_to_latex(fit_type,popt)}\n$R^2={r2:.4f}$"
            ax.text(0.98, pos_y, formula_latex, fontsize=9, color=color,
                    transform=ax.transAxes, va='bottom', ha='right')

        except Exception as e:
            print(f"{label} {fit_type} fitting failed: {e}")
            ax.text(0.98, pos_y, f"{label} {fit_type} Fitting Failed", fontsize=8, color='red',
                    transform=ax.transAxes, va='bottom', ha='right')

    def _formula_to_latex(self, fit_type, popt):
        if fit_type == "Linear": return f"$y={popt[0]:.3f}x+{popt[1]:.3f}$"
        if fit_type == "Quadratic": return f"$y={popt[0]:.3f}x^2+{popt[1]:.3f}x+{popt[2]:.3f}$"
        if fit_type == "Cubic": return f"$y={popt[0]:.3f}x^3+{popt[1]:.3f}x^2+{popt[2]:.3f}x+{popt[3]:.3f}$"
        if fit_type == "Logistic": return f"$y={popt[0]:.3f}/(1+e^(-{popt[1]:.3f}(x-{popt[2]:.3f})))$"
        if fit_type == "Gompertz": return f"$y={popt[0]:.3f}e(-{popt[1]:.3f}e(-{popt[2]:.3f}x))$"
        if fit_type == "Richards": return f"$y={popt[0]:.3f}/(1+{popt[1]:.3f}e(-{popt[2]:.3f}x))^(1/{popt[3]:.3f})$"
        if fit_type == "Weibull": return f"$y={popt[0]:.3f}(1-e(-{popt[1]:.3f}x^{popt[2]:.3f}))$"
        return ""

    def export_fit_params(self):
        if not self.fitted_params:
            QMessageBox.warning(self, "Warning", "No fitted parameters to export!")
            return
        import pandas as pd
        from PyQt6.QtWidgets import QFileDialog

        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        rows = []
        for Genotype, fits in self.fitted_params.items():
            for model, fit_info in fits.items():
                popt = fit_info.get("popt", [])
                r2 = fit_info.get("r2", np.nan)
                row = {"Genotype": Genotype, "Model": model, "R2": r2}
                for i, val in enumerate(popt):
                    row[f"{y_col}_vs_{x_col}_Param{i+1}"] = val
                rows.append(row)
        df = pd.DataFrame(rows)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Fit Parameters", f"{y_col}_vs_{x_col}_Fit_Parameters.xlsx", "Excel Files (*.xlsx)"
        )
        if file_path:
            if not file_path.endswith(".xlsx"): file_path += ".xlsx"
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Complete", f"Fit parameters saved to {file_path}")


# -----------------------------
# Morphological Structure Model
# -----------------------------
# -----------------------------
# Camera Handler
# -----------------------------
class CameraHandler(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._callback = None

    def set_callback(self, callback):
        self._callback = callback

    @pyqtSlot(str)
    def saveCamera(self, cam_json):
        import json
        cam = json.loads(cam_json)
        if self._callback:
            self._callback(cam)


# -----------------------------
# Banana Morphology Tab
# -----------------------------
class BananaMorphologyTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.df = None
        self.varieties = []
        self.days_available = []
        self.min_day = 0
        self.max_day = 1

        self.params_outer = {}
        self.params_inner = {}
        self.params_circumference = {}

        self.default_outer = [19.823, 0.960, 0.019]
        self.default_inner = [17.127, 0.962, 0.017]
        self.default_circumference = [16.128, 0.988, 0.016]

        self.cross_sections = {"DJ": 1, "FJ": 2, "GJ": 4}
        self.saved_camera = dict(eye=dict(x=1.5, y=-1.5, z=1.5))

        self.Wd_to_Wf = {
            "DJ": (0.272674784, -7.948278725),
            "FJ": (0.320382632, -6.912749653),
            "GJ": (0.326001898, -3.205075411)
        }

        main_layout = QHBoxLayout(self)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(5, 5, 5, 5)

        self.import_btn = QPushButton("Import Mean and Error Data")
        self.import_btn.clicked.connect(lambda: self.import_excel(auto=True))
        control_layout.addWidget(self.import_btn)

        self.import_outer_btn = QPushButton("Import Outer Arc Parameters")
        self.import_outer_btn.clicked.connect(lambda: self.import_params("outer", auto=True))
        control_layout.addWidget(self.import_outer_btn)

        self.import_inner_btn = QPushButton("Import Inner Arc Parameters")
        self.import_inner_btn.clicked.connect(lambda: self.import_params("inner", auto=True))
        control_layout.addWidget(self.import_inner_btn)

        self.import_circumference_btn = QPushButton("Import Circumference Parameters")
        self.import_circumference_btn.clicked.connect(lambda: self.import_params("circumference", auto=True))
        control_layout.addWidget(self.import_circumference_btn)

        control_layout.addWidget(QLabel("Select Genotype:"))
        self.list_widget_Genotype = QListWidget()
        self.list_widget_Genotype.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list_widget_Genotype.setFixedWidth(150)
        self.list_widget_Genotype.setFixedHeight(150)
        control_layout.addWidget(self.list_widget_Genotype)

        control_layout.addWidget(QLabel("Select DAF:"))
        self.list_widget_days = QListWidget()
        self.list_widget_days.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list_widget_days.setFixedWidth(150)
        self.list_widget_days.setFixedHeight(200)
        control_layout.addWidget(self.list_widget_days)

        self.axis_checkbox = QCheckBox("Show Axes")
        self.axis_checkbox.setChecked(True)
        self.axis_checkbox.stateChanged.connect(self.update_plot)
        control_layout.addWidget(self.axis_checkbox)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_view_btn)

        self.update_btn = QPushButton("Update Display")
        self.update_btn.clicked.connect(self.update_plot)
        control_layout.addWidget(self.update_btn)

        self.screenshot_btn = QPushButton("Save Screenshot")
        self.screenshot_btn.clicked.connect(self.save_screenshot)
        control_layout.addWidget(self.screenshot_btn)

        control_layout.addStretch()
        main_layout.addWidget(control_widget)

        # WebView
        self.webview = QWebEngineView()
        main_layout.addWidget(self.webview, stretch=1)

        # WebChannel
        self.channel = QWebChannel()
        self.camera_handler = CameraHandler()
        self.camera_handler.set_callback(self._update_saved_camera)
        self.channel.registerObject("pyCamera", self.camera_handler)
        self.webview.page().setWebChannel(self.channel)

    # -----------------------------
    # Auto-load files (when tab opens)
    # -----------------------------
    def auto_load_files(self):
        self.import_excel(auto=True, filename="Parameter_Mean_Error.xlsx")
        self.import_params("outer", auto=True, filename="Lo_vs_Wf_Fit_Parameters.xlsx")
        self.import_params("inner", auto=True, filename="Li_vs_Wf_Fit_Parameters.xlsx")
        self.import_params("circumference", auto=True, filename="Lc_vs_Wf_Fit_Parameters.xlsx")

    def _update_saved_camera(self, cam):
        self.saved_camera = cam
        print("已保存视角:", cam)

    def reset_view(self):
        if hasattr(self, 'current_fig'):
            self.current_fig.update_layout(scene=dict(camera=self.saved_camera))
            self._set_webview_html(self.current_fig)

    def gompertz(self, x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))

    def outer_arc(self, x, Genotype):
        a, b, c = self.params_outer.get(Genotype, self.default_outer)
        return self.gompertz(x, a, b, c)

    def inner_arc(self, x, Genotype):
        a, b, c = self.params_inner.get(Genotype, self.default_inner)
        return self.gompertz(x, a, b, c)

    def circumference(self, x, Genotype):
        a, b, c = self.params_circumference.get(Genotype, self.default_circumference)
        return self.gompertz(x, a, b, c)

    def import_excel(self, auto=False, filename=None):
        if auto and filename:
            if not os.path.exists(filename):
                return
            file_path = filename
        else:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
            if not file_path:
                return

        self.df = pd.read_excel(file_path)
        self.df['DAF'] = self.df['DAF'].astype(int)
        self.varieties = sorted(self.df['Genotype'].unique())
        self.days_available = sorted(self.df['DAF'].unique())
        self.min_day, self.max_day = min(self.days_available), max(self.days_available)

        self.list_widget_Genotype.clear()
        for v in self.varieties:
            item = QListWidgetItem(v)
            self.list_widget_Genotype.addItem(item)
        if self.varieties:
            self.list_widget_Genotype.item(0).setSelected(True)

        self.list_widget_days.clear()
        for d in self.days_available:
            item = QListWidgetItem(str(d))
            self.list_widget_days.addItem(item)
        if self.days_available:
            self.list_widget_days.item(0).setSelected(True)

        print(f"Mean and Error data loaded: {file_path}")
        self.update_plot()

    def import_params(self, param_type, auto=False, filename=None):
        if auto and filename:
            if not os.path.exists(filename):
                return
            file_path = filename
        else:
            file_path, _ = QFileDialog.getOpenFileName(self, f"Select {param_type} Parameters File", "", "Excel Files (*.xlsx *.xls)")
            if not file_path:
                print(f"{param_type} parameter file not selected")
                return

        df_params = pd.read_excel(file_path)
        count = 0
        for idx, row in df_params.iterrows():
            Genotype = row['Genotype']
            params = [row.iloc[3], row.iloc[4], row.iloc[5]]
            if param_type == "outer":
                self.params_outer[Genotype] = params
            elif param_type == "inner":
                self.params_inner[Genotype] = params
            elif param_type == "circumference":
                self.params_circumference[Genotype] = params
            count += 1
        print(f"{param_type} parameter file loaded: {file_path}, {count} records")
        self.update_plot()

    def day_to_color(self, day):
        ratio = (day - self.min_day) / (self.max_day - self.min_day) if self.max_day > self.min_day else 0
        r = int(165 + ratio * (255 - 165))
        g = int(205 + ratio * (220 - 205))
        b = int(115 + ratio * (50 - 115))
        return f'rgb({r},{g},{b})'

    def generate_cross_section(self, radius, interp_points=1):
        n_corners = 3
        base_angles = np.linspace(0, 2*np.pi, n_corners, endpoint=False)
        X, Y = [], []
        for i in range(len(base_angles)):
            theta1 = base_angles[i]
            theta2 = base_angles[(i+1) % len(base_angles)]
            if theta2 < theta1:
                theta2 += 2*np.pi
            thetas = np.linspace(theta1, theta2, interp_points+2)
            for t in thetas[:-1]:
                X.append(radius * np.cos(t))
                Y.append(radius * np.sin(t))
        X.append(X[0]); Y.append(Y[0])
        return np.array(X), np.array(Y)

    def create_banana(self, x, Genotype, section_count=200,
                      end_radius=0.3, smoothness=5, offset=(0,0,0),
                      min_radius=0.5, handle_sections=20):
        lp_params = {
            "DJ": (0.015111173, 3.04229117),
            "FJ": (0.009060502, 3.721891405),
            "GJ": (0.041201134, 0.427253611)
        }
        if Genotype in lp_params:
            a, b = lp_params[Genotype]
        else:
            a, b = (0.015, 3.0)
        handle_length = a * x + b
        handle_length = max(handle_length, 1.0)

        outer_len = self.outer_arc(x, Genotype)
        inner_len = self.inner_arc(x, Genotype)
        max_circumference = self.circumference(x, Genotype)
        if outer_len <= inner_len:
            outer_len = inner_len + 0.1
        if max_circumference < 2*np.pi*end_radius:
            max_circumference = 2*np.pi*end_radius + 0.1
        max_radius = max_circumference/(2*np.pi)
        radian_angle = (outer_len-inner_len)/(2*(max_radius+end_radius))
        center_radius = (outer_len+inner_len)/(2*radian_angle)

        t = np.linspace(0,1,section_count)
        x_center = center_radius*np.cos(radian_angle*(t-0.5)) + offset[0]
        y_center = center_radius*np.sin(radian_angle*(t-0.5)) + offset[1]
        z_center = -t*10 + offset[2]
        center_points = np.stack([x_center,y_center,z_center], axis=1)

        tangents = np.gradient(center_points, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:,None]

        interp_points = self.cross_sections.get(Genotype,3)
        X, Y, Z = [], [], []
        for i,pos_ratio in enumerate(t):
            radius = max_radius-(max_radius-end_radius)*abs(2*pos_ratio-1)**smoothness
            if pos_ratio>0.9:
                radius = max(radius*(1-pos_ratio)*10, min_radius)
            cx,cy = self.generate_cross_section(radius, interp_points)
            tangent = tangents[i]
            not_parallel = np.array([0,0,1]) if abs(tangent[2])<0.9 else np.array([0,1,0])
            normal1 = np.cross(tangent, not_parallel); normal1/=np.linalg.norm(normal1)
            normal2 = np.cross(tangent, normal1); normal2/=np.linalg.norm(normal2)
            circle_points = center_points[i][np.newaxis,:] + np.outer(cx, normal1) + np.outer(cy, normal2)
            X.append(circle_points[:,0]); Y.append(circle_points[:,1]); Z.append(circle_points[:,2])
        X = np.array(X); Y=np.array(Y); Z=np.array(Z)

        num_fit = max(2, section_count//10)
        last_pts = center_points[-num_fit:]
        tangent_fit = np.gradient(last_pts, axis=0)
        tangent_fit /= np.linalg.norm(tangent_fit, axis=1)[:,None]
        avg_tangent = np.mean(tangent_fit, axis=0)

        handle_start_point = center_points[-1]
        handle_t = np.linspace(0, handle_length, handle_sections)
        handle_center = handle_start_point + avg_tangent[None,:] * handle_t[:,None]

        final_radius = max_radius-(max_radius-end_radius)*abs(2 * 1.0-1)**smoothness
        if 1.0 > 0.9:
            final_radius = max(final_radius*(1-1.0)*10, min_radius)

        handle_radius_start = final_radius * 1.0
        handle_radius_end = final_radius * 1.05
        Xh,Yh,Zh=[],[],[]
        for i in range(handle_sections):
            r = handle_radius_start + (handle_radius_end-handle_radius_start)*(i/(handle_sections-1))
            cx,cy = self.generate_cross_section(r, interp_points)
            tangent = avg_tangent
            not_parallel = np.array([0,0,1]) if abs(tangent[2])<0.9 else np.array([0,1,0])
            normal1 = np.cross(tangent, not_parallel); normal1/=np.linalg.norm(normal1)
            normal2 = np.cross(tangent, normal1); normal2/=np.linalg.norm(normal2)
            circle_points = handle_center[i][np.newaxis,:] + np.outer(cx, normal1) + np.outer(cy, normal2)
            Xh.append(circle_points[:,0]); Yh.append(circle_points[:,1]); Zh.append(circle_points[:,2])
        Xh=np.array(Xh); Yh=np.array(Yh); Zh=np.array(Zh)

        return (X,Y,Z),(Xh,Yh,Zh)

    def update_plot(self):
        self.webview.setHtml("")
        if self.df is None:
            return

        selected_varieties = [item.text() for item in self.list_widget_Genotype.selectedItems()]
        selected_days = sorted([int(item.text()) for item in self.list_widget_days.selectedItems()])
        if not selected_varieties or not selected_days:
            return

        xs, ys, zs, colors = [], [], [], []

        row_spacing = 25
        col_spacing = 15
        scale_factor = 1.0

        day_to_x = {day: idx * col_spacing for idx, day in enumerate(selected_days)}
        genotype_to_y = {gen: idx * row_spacing for idx, gen in enumerate(selected_varieties)}

        max_x = col_spacing * (len(selected_days) - 1)

        for Genotype in selected_varieties:
            y_center = genotype_to_y[Genotype]
            df_subset = self.df[(self.df['Genotype'] == Genotype) & (self.df['DAF'].isin(selected_days))]
            sorted_days = sorted(df_subset['DAF'].unique())

            for day in sorted_days:
                x_center = day_to_x[day]
                row = df_subset[df_subset['DAF'] == day]
                if row.empty:
                    continue

                Wd = float(row['Wd_mean'].iloc[0])
                slope, intercept = self.Wd_to_Wf.get(Genotype, (0.3, 0))
                Wf = (Wd - intercept) / slope

                (X, Y, Z), (Xh, Yh, Zh) = self.create_banana(Wf, Genotype)


                all_X = np.concatenate([X.flatten(), Xh.flatten()])
                X_center = np.mean(all_X)
                X_centered = X - X_center + x_center
                Xh_centered = Xh - X_center + x_center

                X = max_x - X_centered
                Xh = max_x - Xh_centered

                Y = Y*scale_factor + y_center
                Yh = Yh*scale_factor + y_center

                Z = Z*scale_factor
                Zh = Zh*scale_factor

                xs.append(X); ys.append(Y); zs.append(Z); colors.append(self.day_to_color(day))
                xs.append(Xh); ys.append(Yh); zs.append(Zh); colors.append(self.day_to_color(day))

        fig = go.Figure()
        for X, Y, Z, c in zip(xs, ys, zs, colors):
            fig.add_trace(go.Mesh3d(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                color=c, opacity=1.0, alphahull=0, flatshading=False,
                lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2, roughness=0.9, fresnel=0.1),
                lightposition=dict(x=500, y=500, z=500)
            ))

        camera = self.saved_camera
        if self.axis_checkbox.isChecked():
            xaxis = dict(title='X (cm)', showgrid=True, zeroline=True, visible=True)
            yaxis = dict(title='Y (cm)', showgrid=True, zeroline=True, visible=True)
            zaxis = dict(title='Z (cm)', showgrid=True, zeroline=True, visible=True)
        else:
            xaxis = yaxis = zaxis = dict(visible=False)

        fig.update_layout(
            scene=dict(
                xaxis=xaxis,
                yaxis=yaxis,
                zaxis=zaxis,
                aspectmode='data',
                camera=camera
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        self.current_fig = fig
        self._set_webview_html(fig)

    def _set_webview_html(self, fig):
        html_str = fig.to_html(include_plotlyjs='cdn')
        js_inject = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.pyCamera = channel.objects.pyCamera;
            var myPlot = document.querySelector(".plotly-graph-div");
            if(myPlot){
                myPlot.on('plotly_relayout', function(eventdata){
                    if(eventdata['scene.camera']){
                        pyCamera.saveCamera(JSON.stringify(eventdata['scene.camera']));
                    }
                });
            }
        });
        </script>
        """
        html_str = html_str.replace("</body>", js_inject + "</body>")
        self.webview.setHtml(html_str)

    def save_screenshot(self):
        if not hasattr(self, "current_fig") or self.current_fig is None:
            print("No current figure found.")
            return

        from PyQt6.QtWidgets import QFileDialog
        import plotly.io as pio

        # Open file dialog to save high-resolution screenshot
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save High-Resolution Screenshot",
            "banana_plot.png",
            "PNG Files (*.png);;SVG Files (*.svg)"
        )
        if not file_path:
            return

        fmt = 'png' if file_path.lower().endswith('.png') else 'svg'

        # Create a copy of the figure to avoid modifying the original
        fig_copy = self.current_fig.full_figure_for_development()
        fig_copy.update_layout(
            width=3500,
            height=2500,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(camera=self.saved_camera)  # Use the current camera view
        )

        try:
            pio.write_image(
                fig_copy,
                file_path,
                format=fmt,
                scale=3
            )
            print(f"Screenshot saved: {file_path}")
        except Exception as e:
            print(f"Failed to save screenshot: {e}")

# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Banana Morphology Simulation System")
        self.setGeometry(100, 100, 1200, 800)  

        self.data_handler = DataImportCleanTab()
        self.analysis_tab = AnalysisTab(self.data_handler)
        self.growth_tab = GrowthCurveTab(self.data_handler)
        self.model_tab = ModelTab(self.data_handler)  
        self.banana_3d_tab = BananaMorphologyTab()   

        self.tabs = QTabWidget()
        self.tabs.addTab(self.data_handler, "Data Preprocessing")
        self.tabs.addTab(self.analysis_tab, "Statistical Analysis")
        self.tabs.addTab(self.growth_tab, "Growth Analysis")
        self.tabs.addTab(self.model_tab, "Growth Simulation")
        self.tabs.addTab(self.banana_3d_tab, "Morphological Structure Model")  

        self.setCentralWidget(self.tabs)

        self.data_handler.dfChanged.connect(self.update_all_tabs)

        # 标记香蕉 3D Tab 是否已自动加载
        self.banana_tab_loaded = False
        # 绑定 Tab 切换信号
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def update_all_tabs(self):
        """Refresh each tab's controls when data is updated"""
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if hasattr(tab, 'update_controls'):
                tab.update_controls()

    def on_tab_changed(self, index):
        """当切换到 Morphological Structure Model Tab 时自动加载文件"""
        if self.tabs.widget(index) == self.banana_3d_tab and not self.banana_tab_loaded:
            self.banana_3d_tab.auto_load_files()  # 调用 BananaMorphologyTab 的自动加载方法
            self.banana_tab_loaded = True

# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')


    font = app.font()
    font.setPointSize(10)
    app.setFont(font)


    window = MainWindow()
    window.show()
    sys.exit(app.exec())