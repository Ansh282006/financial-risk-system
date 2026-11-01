# modules/chart_generator.py
"""
Chart Generator Module - Handles all chart generation from CSV data
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartGenerator:
    """Handles generation of various charts from CSV data"""
    
    def __init__(self):
        self.data = None
    
    def generate_all_charts(self, filepath):
        """Generate all available charts from CSV file"""
        try:
            self.data = pd.read_csv(filepath)
            charts = {}
            
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for chart generation")
                return {}
            
            # 1. Distribution Histograms
            dist_chart = self._create_distribution_charts(numeric_cols)
            if dist_chart:
                charts['distributions'] = dist_chart
            
            # 2. Correlation Heatmap
            if len(numeric_cols) > 1:
                corr_chart = self._create_correlation_heatmap(numeric_cols)
                if corr_chart:
                    charts['correlation'] = corr_chart
            
            # 3. Missing Values Heatmap
            if self.data.isnull().sum().sum() > 0:
                missing_chart = self._create_missing_values_heatmap()
                if missing_chart:
                    charts['missing_values'] = missing_chart
            
            # 4. Box Plots
            box_chart = self._create_box_plots(numeric_cols)
            if box_chart:
                charts['boxplots'] = box_chart
            
            # 5. Line Chart (if date column exists)
            line_chart = self._create_line_chart(numeric_cols)
            if line_chart:
                charts['line_chart'] = line_chart
            
            logger.info(f"Generated {len(charts)} charts successfully")
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {}
    
    def _create_distribution_charts(self, numeric_cols):
        """Create distribution histogram charts"""
        try:
            n_cols = min(len(numeric_cols), 6)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:6]):
                self.data[col].dropna().hist(bins=30, edgecolor='black', alpha=0.7, ax=axes[i], color='#2563eb')
                axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(col, fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                axes[i].grid(True, alpha=0.3)
            
            for i in range(n_cols, 6):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            chart_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            return chart_base64
        except Exception as e:
            logger.error(f"Error creating distribution charts: {e}")
            return None
    
    def _create_correlation_heatmap(self, numeric_cols):
        """Create correlation heatmap"""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = self.data[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, ax=ax, annot_kws={'size': 8})
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            chart_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            return chart_base64
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return None
    
    def _create_missing_values_heatmap(self):
        """Create missing values heatmap"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(self.data.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Pattern', fontsize=14, fontweight='bold')
            plt.tight_layout()
            chart_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            return chart_base64
        except Exception as e:
            logger.error(f"Error creating missing values heatmap: {e}")
            return None
    
    def _create_box_plots(self, numeric_cols):
        """Create box plots for outlier detection"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            self.data[numeric_cols[:8]].boxplot(ax=ax, rot=45)
            ax.set_title('Box Plots - Outlier Detection', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            chart_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            return chart_base64
        except Exception as e:
            logger.error(f"Error creating box plots: {e}")
            return None
    
    def _create_line_chart(self, numeric_cols):
        """Create line chart if date column exists"""
        try:
            date_cols = [col for col in self.data.columns if 'date' in col.lower() or 'Date' in col.lower()]
            if date_cols and len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                date_col = date_cols[0]
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                df_sorted = self.data.sort_values(date_col)
                ax.plot(df_sorted[date_col], df_sorted[numeric_cols[0]], linewidth=2, color='#2563eb')
                ax.fill_between(df_sorted[date_col], df_sorted[numeric_cols[0]], alpha=0.3, color='#2563eb')
                ax.set_title(f'Time Series: {numeric_cols[0]} over Time', fontsize=14, fontweight='bold')
                ax.set_xlabel(date_col, fontsize=12)
                ax.set_ylabel(numeric_cols[0], fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                chart_base64 = self._fig_to_base64(fig)
                plt.close(fig)
                return chart_base64
            return None
        except Exception as e:
            logger.error(f"Error creating line chart: {e}")
            return None
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        img = BytesIO()
        fig.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()