import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QInputDialog, QSplitter, QTextEdit, QListWidget, QAbstractItemView, 
    QLineEdit, QListWidgetItem
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
from scipy import stats, interpolate
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA
from tabulate import tabulate


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
            required_columns = ['Variety', 'DAF', 'Replicate']
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
        grouped = self.df.groupby(['Variety','DAF'])[metric_cols].mean().reset_index()
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
# Statistical Analysis
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

        btn_corr = QPushButton("Correlation Analysis")
        btn_corr.clicked.connect(self.show_correlation)

        self.cb_metric_anova = QComboBox()
        btn_anova = QPushButton("ANOVA Test")
        btn_anova.clicked.connect(self.run_anova)

        btn_pca = QPushButton("PCA Analysis")
        btn_pca.clicked.connect(self.run_pca)

        self.result_label = QLabel("Analysis Results")
        self.result_label.setWordWrap(True)

        layout.addWidget(QLabel("=== Correlation Analysis ==="))
        layout.addWidget(btn_corr)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("=== Significance Test ==="))
        layout.addWidget(QLabel("Select the trait:"))
        layout.addWidget(self.cb_metric_anova)
        layout.addWidget(btn_anova)
        layout.addWidget(QLabel("=== Principal Component Analysis (PCA) ==="))
        layout.addWidget(btn_pca)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def update_controls(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["DAF", "Replicate"]]
        self.cb_metric_anova.clear()
        self.cb_metric_anova.addItems(numeric_cols)

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
        # 按 Variety × DAF 取Replicate均值
        df_avg = df.groupby(['Variety','DAF'])[numeric_cols].mean()
        corr_matrix = df_avg.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Correlation Coefficient'}, square=True, mask=mask)
        ax.set_title("Trait Correlation Heatmap (Upper Triangle)")
        self.canvas.draw()

    def run_anova(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            QMessageBox.warning(self, "Warning", "Please import data first!")
            return
        metric = self.cb_metric_anova.currentText()
        if metric not in df.columns:
            QMessageBox.warning(self, "Error", f"Column '{metric}' does not exist")
            return
        if len(df['Variety'].unique()) < 2:
            QMessageBox.warning(self, "Error", "At least 2 varieties required for comparison")
            return
        try:
            df_avg = df.groupby(['Variety','DAF'])[metric].mean().reset_index()
            groups = [df_avg[df_avg['Variety']==var][metric] for var in df_avg['Variety'].unique()]
            f_val, p_val = stats.f_oneway(*groups)
            tukey = pairwise_tukeyhsd(endog=df_avg[metric], groups=df_avg['Variety'], alpha=0.05)
            result_text = f"【ANOVA Results】\nMetric: {metric}\nF-value = {f_val:.4f}, p-value = {p_val:.4f}\n\n"
            result_text += "【Tukey HSD Post-hoc Test】\n" + str(tukey.summary())
            self.result_label.setText(result_text)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", str(e))

    def run_pca(self):
        df = getattr(self.data_handler, "df", None)
        if df is None:
            QMessageBox.warning(self, "Warning", "Please import data first!")
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["DAF", "Replicate"]]
        if not numeric_cols:
            QMessageBox.warning(self, "Warning", "No numeric metrics available")
            return

        df_avg = df.groupby(['Variety','DAF'])[numeric_cols].mean()
        X = df_avg.dropna()
        if len(X)==0:
            QMessageBox.warning(self, "Warning", "Insufficient data or too many missing values")
            return

        try:
            pca = PCA()
            scores = pca.fit_transform(X)
            loadings = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(len(X.columns))])
            explained = pca.explained_variance_ratio_

            self.figure.clear()
            gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
            ax1 = self.figure.add_subplot(gs[0])
            sns.heatmap(loadings, annot=True, cmap="viridis", fmt=".2f", ax=ax1)
            ax1.set_title("PCA Component Matrix (Loading)")

            ax2 = self.figure.add_subplot(gs[1])

            target_metric = numeric_cols[0]
            if target_metric in loadings.columns:
                target_loadings = loadings[target_metric].sort_values(ascending=False)
                target_loadings.plot(kind='bar', ax=ax2, color='skyblue')
                ax2.set_ylabel("Loading")
                ax2.set_title(f"{target_metric} Driving Metric Loading Ranking")
                ax2.grid(True)
            self.figure.tight_layout()
            self.canvas.draw()


            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save PCA Results", "PCA_results.xlsx", "Excel 文件 (*.xlsx)"
            )
            if file_path:
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                with pd.ExcelWriter(file_path) as writer:
                    # Scores
                    scores_df = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])])
                    scores_df.insert(0, 'Variety', df_avg.index.get_level_values(0))
                    scores_df.insert(1, 'DAF', df_avg.index.get_level_values(1))
                    scores_df.to_excel(writer, sheet_name="Scores", index=False)

                    # Loadings
                    loadings.to_excel(writer, sheet_name="Loadings")

                    # Explained variance
                    ev_df = pd.DataFrame({
                        'PC': [f'PC{i+1}' for i in range(len(explained))],
                        'ExplainedVariance': explained
                    })
                    ev_df.to_excel(writer, sheet_name="ExplainedVariance", index=False)

                QMessageBox.information(self, "Complete", f"PCA results saved to {file_path}")

            result_text = "Principal Component Explained Variance:\n"
            for i,var in enumerate(explained):
                result_text += f"PC{i+1}: {var:.2%}\n"
            result_text += f"\nPCA results automatically exported to file"
            self.result_label.setText(result_text)

        except Exception as e:
            QMessageBox.critical(self, "PCA Failed", str(e))


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

        self.lst_variety = QListWidget()
        self.lst_variety.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.lst_variety.setFixedHeight(80)
        self.lst_variety.setFixedWidth(120)


        self.cb_metric_x = QComboBox()
        self.cb_metric_y = QComboBox()

        btn_plot = QPushButton("Plot Curve")
        btn_plot.clicked.connect(self.plot_curve)

        btn_export = QPushButton("Export Mean and Error")
        btn_export.clicked.connect(self.export_all_stats)

        control_layout.addWidget(QLabel("Select Variety:"))
        control_layout.addWidget(self.lst_variety)
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


        self.lst_variety.clear()
        for v in df['Variety'].unique():
            self.lst_variety.addItem(str(v))

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

        selected_items = self.lst_variety.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one variety!")
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

        for i, variety in enumerate(varieties):
            data = df[df['Variety'] == variety].copy()
            if data.empty:
                continue

            grouped = data.groupby('DAF')[[x_metric, y_metric]].agg(['mean','std']).reset_index()
            grouped.columns = ['DAF', f'{x_metric}_mean', f'{x_metric}_std', f'{y_metric}_mean', f'{y_metric}_std']

            x_mean = grouped[f'{x_metric}_mean'].values
            y_mean = grouped[f'{y_metric}_mean'].values
            x_std = grouped[f'{x_metric}_std'].values
            y_std = grouped[f'{y_metric}_std'].values

            ax.plot(x_mean, y_mean, 'o-', color=colors[i % 10], label=variety, markersize=6)

            if x_metric == "DAF":
                ax.errorbar(x_mean, y_mean, yerr=y_std, fmt='o', color=colors[i % 10], capsize=3)
            else:
                ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='o', color=colors[i % 10], capsize=3)

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
        for variety in df['Variety'].unique():
            df_var = df[df['Variety'] == variety]
            grouped = df_var.groupby('DAF')[metric_cols].agg(['mean','std']).reset_index()
            # 扁平化列名
            grouped.columns = ['DAF'] + [f"{col}_{stat}" for col, stat in grouped.columns[1:]]
            grouped.insert(0, 'Variety', variety)
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

        hlayout2.addWidget(QLabel("By Variety:"))
        self.by_variety_combo = QComboBox()
        self.by_variety_combo.addItems(["No", "Yes"])
        hlayout2.addWidget(self.by_variety_combo)

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
        by_variety = self.by_variety_combo.currentText() == "Yes"

        if x_col == y_col:
            QMessageBox.warning(self, "Error", "X and Y variables cannot be the same")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if by_variety and "Variety" in self.df.columns:
            groups = list(self.df.groupby("Variety"))
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

            # 保存参数和R²
            if label not in self.fitted_params:
                self.fitted_params[label] = {}
            self.fitted_params[label][fit_type] = {"popt": popt, "r2": r2}

            # 显示公式和R²
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
        for variety, fits in self.fitted_params.items():
            for model, fit_info in fits.items():
                popt = fit_info.get("popt", [])
                r2 = fit_info.get("r2", np.nan)
                row = {"Variety": variety, "Model": model, "R2": r2}
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

class BananaMorphologyTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.df = None
        self.varieties = []
        self.days_available = []
        self.min_day = 0
        self.max_day = 1

        # 参数字典
        self.params_outer = {}
        self.params_inner = {}
        self.params_circumference = {}

        # 默认参数（用于没有导入的品种）
        self.default_outer = [19.823, 0.960, 0.019]
        self.default_inner = [17.127, 0.962, 0.017]
        self.default_circumference = [16.128, 0.988, 0.016]

        self.initial_camera_position = dict(eye=dict(x=0, y=-3, z=3))

        main_layout = QHBoxLayout(self)  

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(5,5,5,5)

        # 导入数据按钮
        self.import_btn = QPushButton("Import Mean and Error Data")
        self.import_btn.clicked.connect(self.import_excel)
        control_layout.addWidget(self.import_btn)

        # 导入参数按钮
        self.import_outer_btn = QPushButton("Import Outer Arc Parameters")
        self.import_outer_btn.clicked.connect(lambda: self.import_params("outer"))
        control_layout.addWidget(self.import_outer_btn)

        self.import_inner_btn = QPushButton("Import Inner Arc Parameters")
        self.import_inner_btn.clicked.connect(lambda: self.import_params("inner"))
        control_layout.addWidget(self.import_inner_btn)

        self.import_circumference_btn = QPushButton("Import circumference Parameters")
        self.import_circumference_btn.clicked.connect(lambda: self.import_params("circumference"))
        control_layout.addWidget(self.import_circumference_btn)

        # 品种选择
        control_layout.addWidget(QLabel("Select Variety："))
        self.list_widget_variety = QListWidget()
        self.list_widget_variety.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list_widget_variety.setFixedWidth(150)
        self.list_widget_variety.setFixedHeight(150)
        control_layout.addWidget(self.list_widget_variety)

        # DAF选择
        control_layout.addWidget(QLabel("Select DAF："))
        self.list_widget_days = QListWidget()
        self.list_widget_days.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list_widget_days.setFixedWidth(150)
        self.list_widget_days.setFixedHeight(200)
        control_layout.addWidget(self.list_widget_days)

        # 重置视角和更新显示
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_view_btn)
        
        self.update_btn = QPushButton("Update Display")
        self.update_btn.clicked.connect(self.update_plot)
        control_layout.addWidget(self.update_btn)
        control_layout.addStretch()
        main_layout.addWidget(control_widget)

        # Web view
        self.webview = QWebEngineView()
        main_layout.addWidget(self.webview, stretch=1)

    def reset_view(self):
        if hasattr(self, 'current_fig'):
            self.current_fig.update_layout(
                scene=dict(camera=self.initial_camera_position)
            )
            self.webview.setHtml(self.current_fig.to_html(include_plotlyjs='cdn'))

    # Gompertz公式
    def gompertz(self, x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))

    def outer_arc(self, x, variety):
        a,b,c = self.params_outer.get(variety, self.default_outer)
        return self.gompertz(x, a,b,c)

    def inner_arc(self, x, variety):
        a,b,c = self.params_inner.get(variety, self.default_inner)
        return self.gompertz(x, a,b,c)

    def circumference(self, x, variety):
        a,b,c = self.params_circumference.get(variety, self.default_circumference)
        return self.gompertz(x, a,b,c)

    def import_excel(self):
        from PyQt6.QtWidgets import QFileDialog
        excel_path, _ = QFileDialog.getOpenFileName(self, "选择 Excel 文件", "", "Excel Files (*.xlsx *.xls)")
        if not excel_path: return
        self.df = pd.read_excel(excel_path)
        self.df['DAF'] = self.df['DAF'].astype(int)
        self.varieties = sorted(self.df['Variety'].unique())
        self.days_available = sorted(self.df['DAF'].unique())
        self.min_day, self.max_day = min(self.days_available), max(self.days_available)

        self.list_widget_variety.clear()
        for v in self.varieties:
            item = QListWidgetItem(v)
            self.list_widget_variety.addItem(item)
            item.setSelected(True)
        self.list_widget_days.clear()
        for d in self.days_available:
            item = QListWidgetItem(str(d))
            self.list_widget_days.addItem(item)
            item.setSelected(True)

        self.update_plot()

    def import_params(self, param_type):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {param_type} Parameters File", "", "Excel Files (*.xlsx *.xls)")
        if not file_path: 
            print(f"{param_type} 参数文件未选择")
            return
        df_params = pd.read_excel(file_path)
        count = 0
        for idx,row in df_params.iterrows():
            variety = row['Variety']
            # Gompertz参数顺序: a,b,c
            params = [row.iloc[3], row.iloc[4], row.iloc[5]]
            if param_type=="outer":
                self.params_outer[variety] = params
            elif param_type=="inner":
                self.params_inner[variety] = params
            elif param_type=="circumference":
                self.params_circumference[variety] = params
            count += 1
        print(f"已导入 {param_type} 参数文件: {file_path}，共 {count} 条记录")
        self.update_plot()

    # 颜色渐变
    def day_to_color(self, day):
        ratio = (day - self.min_day) / (self.max_day - self.min_day) if self.max_day>self.min_day else 0
        r = int(165 + ratio * (255-165))
        g = int(205 + ratio * (220-205))
        b = int(115 + ratio * (50-115))
        return f'rgb({r},{g},{b})'

    # 核心生成函数（创建香蕉Mesh）
    def create_banana(self, x, variety, section_count=50, points_per_section=30, end_radius=0.3, smoothness=5, offset=(0,0,0), min_radius=0.25):
        outer_len = self.outer_arc(x, variety)
        inner_len = self.inner_arc(x, variety)
        max_circumference = self.circumference(x, variety)
        if outer_len <= inner_len: outer_len = inner_len + 0.1
        if max_circumference < 2*np.pi*end_radius: max_circumference = 2*np.pi*end_radius + 0.1
        max_radius = max_circumference / (2*np.pi)
        radian_angle = (outer_len - inner_len) / (2*(max_radius + end_radius))
        center_radius = (outer_len + inner_len) / (2*radian_angle)

        t = np.linspace(0, 1, section_count)
        x_center = center_radius * np.cos(radian_angle*(t-0.5)) + offset[0]
        y_center = center_radius * np.sin(radian_angle*(t-0.5)) + offset[1]
        z_center = t*10 + offset[2]
        center_points = np.stack([x_center, y_center, z_center], axis=1)

        tangents = np.gradient(center_points, axis=0)
        tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

        X,Y,Z=[],[],[]
        for i,pos_ratio in enumerate(t):
            radius = max_radius - (max_radius - end_radius) * abs(2*pos_ratio-1)**smoothness
            if pos_ratio>0.9:
                radius = max(radius*(1-pos_ratio)*10, min_radius)
            radius *= 1 + 0.02*np.sin(10*np.pi*pos_ratio)

            tangent = tangents[i]
            not_parallel = np.array([0,0,1]) if abs(tangent[2])<0.9 else np.array([0,1,0])
            normal1 = np.cross(tangent, not_parallel)
            normal1 /= np.linalg.norm(normal1)
            normal2 = np.cross(tangent, normal1)
            normal2 /= np.linalg.norm(normal2)

            theta = np.linspace(0, 2*np.pi, points_per_section)
            circle_points = center_points[i] + radius*np.outer(np.cos(theta), normal1) + radius*np.outer(np.sin(theta), normal2)
            X.append(circle_points[:,0])
            Y.append(circle_points[:,1])
            Z.append(circle_points[:,2])
        return np.array(X), np.array(Y), np.array(Z)

    # 更新显示
    def update_plot(self):
        self.webview.setHtml("")
        if self.df is None: return

        selected_varieties = [item.text() for item in self.list_widget_variety.selectedItems()]
        selected_days = [int(item.text()) for item in self.list_widget_days.selectedItems()]
        if not selected_varieties or not selected_days: return

        xs,ys,zs,colors=[],[],[],[]
        row_spacing = 25
        col_spacing = 6

        for row_idx,variety in enumerate(selected_varieties):
            row_offset_y = row_idx*row_spacing
            sorted_days = sorted([day for day in selected_days if not self.df[(self.df['Variety']==variety)&(self.df['DAF']==day)].empty])
            x_offset = 0
            for day in sorted_days:
                row = self.df[(self.df['Variety']==variety)&(self.df['DAF']==day)]
                fresh_weight = float(row['FW_mean'].iloc[0])
                X,Y,Z = self.create_banana(fresh_weight, variety)
                Z = Z - np.min(Z)

                min_x,max_x = np.min(X), np.max(X)
                width = max_x - min_x
                X = X - min_x + x_offset
                Y = Y + row_offset_y

                xs.append(X)
                ys.append(Y)
                zs.append(Z)
                colors.append(self.day_to_color(day))

                x_offset += width + col_spacing

        fig = go.Figure()
        for X,Y,Z,c in zip(xs,ys,zs,colors):
            fig.add_trace(go.Mesh3d(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                color=c,
                opacity=1.0,
                alphahull=0,
                flatshading=False,
                lighting=dict(
                    ambient=0.5,   # 环境光，调低整体亮度
                    diffuse=0.5,   # 漫反射，调暗表面亮度
                    specular=0.1,  # 高光反射，调低
                    roughness=0.7, # 表面粗糙度，值越大表面越柔和
                    fresnel=0.1
                ),
                lightposition=dict(x=0, y=150, z=150)
            ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                camera=self.initial_camera_position
            ),
            margin=dict(l=0,r=0,t=0,b=0)
        )

        self.current_fig = fig
        self.webview.setHtml(fig.to_html(include_plotlyjs='cdn'))


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

    def update_all_tabs(self):
        """Refresh each tab's controls when data is updated"""
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if hasattr(tab, 'update_controls'):
                tab.update_controls()


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