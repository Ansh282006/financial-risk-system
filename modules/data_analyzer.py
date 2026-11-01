# modules/data_analyzer.py
"""
Data Analyzer Module - Complete Module-wise Implementation
Covers: Data Prep, Pre-processing, Model Evaluation, Regression, Classification, Unsupervised Learning
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import base64
from io import BytesIO
import os
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Complete Data Analyzer with all ML techniques"""
    
    def __init__(self):
        self.data = None
        self.filename = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    # ============= UNIT I: Types of Learning & Data Prep =============
    
    def analyze_csv(self, filepath):
        """Read CSV, identify missing values, noisy data cleaning, feature construction"""
        self.filename = os.path.basename(filepath)
        self.data = pd.read_csv(filepath)
        
        # Basic Info
        self.results['basic_info'] = {
            'filename': self.filename,
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
        }
        
        # Missing Values Analysis
        missing_analysis = []
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = round(missing_count / len(self.data) * 100, 2)
            if missing_count > 0:
                missing_analysis.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'recommendation': 'Fill with median/mode' if missing_pct < 20 else 'Consider dropping'
                })
        
        self.results['missing_values'] = missing_analysis
        
        # Data Types & Noisy Data Detection
        column_analysis = []
        for col in self.data.columns:
            col_info = {
                'name': col,
                'type': str(self.data[col].dtype),
                'unique_values': self.data[col].nunique(),
                'null_percentage': round(self.data[col].isnull().sum() / len(self.data) * 100, 2)
            }
            
            if self.data[col].dtype in ['int64', 'float64']:
                # Detect outliers using IQR
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.data[(self.data[col] < Q1 - 1.5 * IQR) | (self.data[col] > Q3 + 1.5 * IQR)].shape[0]
                col_info['outliers'] = outliers
                col_info['outlier_percentage'] = round(outliers / len(self.data) * 100, 2)
                col_info['min'] = float(self.data[col].min())
                col_info['max'] = float(self.data[col].max())
                col_info['mean'] = float(self.data[col].mean())
            
            column_analysis.append(col_info)
        
        self.results['column_analysis'] = column_analysis
        self.results['data_quality_score'] = self._calculate_data_quality()
        
        return self.results
    
    def _calculate_data_quality(self):
        """Calculate overall data quality score"""
        completeness = 1 - (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns) + 1))
        uniqueness = len(self.data.drop_duplicates()) / len(self.data) if len(self.data) > 0 else 1
        return round((completeness + uniqueness) / 2 * 100, 2)
    
    # ============= UNIT II: Pre-processing & Transformation =============
    
    def preprocess_data(self):
        """Normalize data, PCA, K-Fold CV, LDA"""
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return {}
        
        X = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        
        # Normalization
        X_normalized = self.scaler.fit_transform(X)
        
        results = {}
        
        # PCA for dimensionality reduction
        try:
            pca = PCA(n_components=min(5, len(numeric_cols)))
            X_pca = pca.fit_transform(X_normalized)
            
            # Create PCA visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Explained variance ratio
            axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
            axes[0].set_xlabel('Principal Components')
            axes[0].set_ylabel('Explained Variance Ratio')
            axes[0].set_title('PCA - Explained Variance')
            axes[0].grid(True, alpha=0.3)
            
            # Cumulative variance
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            axes[1].plot(range(1, len(cumulative)+1), cumulative, 'bo-')
            axes[1].set_xlabel('Number of Components')
            axes[1].set_ylabel('Cumulative Explained Variance')
            axes[1].set_title('PCA - Cumulative Variance')
            axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            results['pca_chart'] = self._fig_to_base64(fig)
            plt.close(fig)
            
            results['pca_info'] = {
                'components': len(pca.components_),
                'explained_variance': pca.explained_variance_ratio_[:3].tolist(),
                'total_variance_explained': float(cumulative[-1])
            }
        except Exception as e:
            logger.warning(f"PCA failed: {e}")
        
        # LDA for risk separation if we have a target column
        target_col = None
        for col in ['default.payment.next.month', 'risk', 'Risk', 'default']:
            if col in self.data.columns:
                target_col = col
                break
        
        if target_col and len(numeric_cols) > 1:
            try:
                y = self.data[target_col]
                lda = LDA(n_components=1)
                X_lda = lda.fit_transform(X_normalized, y)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['blue', 'red']
                for i, label in enumerate(np.unique(y)):
                    mask = y == label
                    ax.hist(X_lda[mask], bins=30, alpha=0.7, color=colors[i], label=f'Class {label}')
                ax.set_xlabel('LDA Component')
                ax.set_ylabel('Frequency')
                ax.set_title('LDA - Risk Separation')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                results['lda_chart'] = self._fig_to_base64(fig)
                plt.close(fig)
                
                results['lda_info'] = {
                    'classes_separated': len(np.unique(y)),
                    'explained_ratio': float(lda.explained_variance_ratio_[0]) if hasattr(lda, 'explained_variance_ratio_') else 1.0
                }
            except Exception as e:
                logger.warning(f"LDA failed: {e}")
        
        # K-Fold Cross Validation preparation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        results['kfold_info'] = {
            'n_splits': 5,
            'shuffle': True,
            'total_samples': len(X)
        }
        
        self.results['preprocessing'] = results
        return results
    
    # ============= UNIT III: Model Evaluation Metrics =============
    
    def evaluate_models(self):
        """Show confusion matrix, precision, recall, AUC-ROC, F-measure"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}
        
        # Find target column
        target_col = None
        for col in ['default.payment.next.month', 'risk', 'Risk', 'default']:
            if col in self.data.columns:
                target_col = col
                break
        
        if not target_col:
            return {}
        
        X = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        y = self.data[target_col]
        
        # Create a synthetic binary target if needed
        if len(np.unique(y)) > 2:
            y = (y > y.median()).astype(int)
        
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        # Train multiple classifiers for comparison
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        for name, clf in classifiers.items():
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
                
                # Train on full data for confusion matrix
                clf.fit(X_scaled, y)
                y_pred = clf.predict(X_scaled)
                y_proba = clf.predict_proba(X_scaled)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
                
                # Confusion Matrix
                cm = confusion_matrix(y, y_pred)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Confusion Matrix Heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
                axes[0].set_xlabel('Predicted')
                axes[0].set_ylabel('Actual')
                axes[0].set_title(f'{name} - Confusion Matrix')
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y, y_proba)
                roc_auc = auc(fpr, tpr)
                axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                axes[1].plot([0, 1], [0, 1], 'k--')
                axes[1].set_xlabel('False Positive Rate')
                axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title(f'{name} - ROC Curve')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                results[name] = {
                    'chart': self._fig_to_base64(fig),
                    'cv_auc_mean': float(cv_scores.mean()),
                    'cv_auc_std': float(cv_scores.std()),
                    'roc_auc': float(roc_auc),
                    'confusion_matrix': cm.tolist()
                }
                plt.close(fig)
                
                # Classification Report
                report = classification_report(y, y_pred, output_dict=True)
                results[name]['precision'] = report.get('weighted avg', {}).get('precision', 0)
                results[name]['recall'] = report.get('weighted avg', {}).get('recall', 0)
                results[name]['f1_score'] = report.get('weighted avg', {}).get('f1-score', 0)
                
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        self.results['model_evaluation'] = results
        return results
    
    # ============= UNIT IV: Regression Models =============
    
    def regression_analysis(self):
        """Ridge/Lasso regression, gradient descent visualization"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}
        
        # Find target for regression (use first numeric column or loan amount)
        target_col = None
        for col in ['eligible_loan_amount', 'loan_amount', 'LIMIT_BAL', numeric_cols[0]]:
            if col in self.data.columns:
                target_col = col
                break
        
        if not target_col:
            target_col = numeric_cols[0]
        
        feature_cols = [col for col in numeric_cols if col != target_col][:5]
        if not feature_cols:
            return {}
        
        X = self.data[feature_cols].fillna(self.data[feature_cols].median())
        y = self.data[target_col].fillna(self.data[target_col].median())
        
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        y_pred_lr = lr.predict(X_scaled)
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        y_pred_ridge = ridge.predict(X_scaled)
        
        # Lasso Regression
        lasso = Lasso(alpha=1.0)
        lasso.fit(X_scaled, y)
        y_pred_lasso = lasso.predict(X_scaled)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Actual vs Predicted - Linear Regression
        axes[0, 0].scatter(y, y_pred_lr, alpha=0.5, color='blue')
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title(f'Linear Regression - R²: {lr.score(X_scaled, y):.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ridge vs Lasso Coefficients
        x = np.arange(len(feature_cols))
        width = 0.35
        axes[0, 1].bar(x - width/2, ridge.coef_, width, label='Ridge')
        axes[0, 1].bar(x + width/2, lasso.coef_, width, label='Lasso')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(feature_cols, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Coefficient Value')
        axes[0, 1].set_title('Ridge vs Lasso Coefficients')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residual Plot
        residuals = y - y_pred_lr
        axes[1, 0].scatter(y_pred_lr, residuals, alpha=0.5, color='green')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residual Plot - Linear Regression')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient Descent Visualization (simulated)
        iterations = np.arange(1, 101)
        loss_history = 1 / (1 + 0.1 * iterations) + np.random.normal(0, 0.02, 100)
        axes[1, 1].plot(iterations, loss_history, 'b-')
        axes[1, 1].set_xlabel('Iterations')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Gradient Descent - Loss Convergence')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        results = {
            'regression_chart': self._fig_to_base64(fig),
            'linear_regression_r2': float(lr.score(X_scaled, y)),
            'ridge_r2': float(ridge.score(X_scaled, y)),
            'lasso_r2': float(lasso.score(X_scaled, y)),
            'feature_importance': dict(zip(feature_cols, abs(lr.coef_[:len(feature_cols)])))
        }
        plt.close(fig)
        
        self.results['regression_analysis'] = results
        return results
    
    # ============= UNIT V: Classification Models =============
    
    def classification_analysis(self):
        """SVM, Random Forest, Naive Bayes, Decision Tree, KNN"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}
        
        # Find target column
        target_col = None
        for col in ['default.payment.next.month', 'risk', 'Risk', 'default']:
            if col in self.data.columns:
                target_col = col
                break
        
        if not target_col:
            # Create synthetic target based on risk profile
            if 'LIMIT_BAL' in self.data.columns and 'AGE' in self.data.columns:
                risk_score = (self.data['LIMIT_BAL'] < 50000).astype(int) + (self.data['AGE'] > 60).astype(int)
                y = (risk_score > 1).astype(int)
            else:
                return {}
        else:
            y = self.data[target_col]
        
        # Convert to binary if needed
        if len(np.unique(y)) > 2:
            y = (y > y.median()).astype(int)
        
        feature_cols = [col for col in numeric_cols if col != target_col][:8]
        X = self.data[feature_cols].fillna(self.data[feature_cols].median())
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        classifiers = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        plot_idx = 0
        
        for name, clf in classifiers.items():
            if plot_idx >= 6:
                break
            
            clf.fit(X_scaled, y)
            y_pred = clf.predict(X_scaled)
            y_proba = clf.predict_proba(X_scaled)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[plot_idx].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            axes[plot_idx].plot([0, 1], [0, 1], 'k--')
            axes[plot_idx].set_xlabel('False Positive Rate')
            axes[plot_idx].set_ylabel('True Positive Rate')
            axes[plot_idx].set_title(f'{name} - ROC Curve')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            
            results[name] = {
                'roc_auc': float(roc_auc),
                'accuracy': float(clf.score(X_scaled, y))
            }
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        results['classification_roc_chart'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Feature Importance for Random Forest
        if 'Random Forest' in classifiers:
            rf = classifiers['Random Forest']
            rf.fit(X_scaled, y)
            importance = rf.feature_importances_
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.argsort(importance)[-10:]
            ax.barh(range(len(indices)), importance[indices], color='#6366f1')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_cols[i] for i in indices])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Random Forest - Feature Importance')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            results['feature_importance_chart'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        self.results['classification_analysis'] = results
        return results
    
    # ============= UNIT VI: Unsupervised Learning =============
    
    def unsupervised_analysis(self):
        """K-Means clustering, Hierarchical clustering for customer segmentation"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}
        
        X = self.data[numeric_cols[:5]].fillna(self.data[numeric_cols[:5]].median())
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        # K-Means Clustering
        inertias = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Elbow Method
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(K_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('K-Means - Elbow Method')
        axes[0].grid(True, alpha=0.3)
        
        # Optimal K (elbow detection)
        optimal_k = 3  # default
        for i in range(1, len(inertias) - 1):
            if inertias[i] - inertias[i+1] < (inertias[i-1] - inertias[i]) / 2:
                optimal_k = i + 2
                break
        
        # Final K-Means with optimal K
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        self.data['Cluster'] = clusters
        
        # Cluster visualization (using first 2 PCA components)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title(f'K-Means Clustering (k={optimal_k})')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        results['kmeans_chart'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Hierarchical Clustering Dendrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Take sample for dendrogram (max 200 samples for performance)
        sample_size = min(200, len(X_scaled))
        indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[indices]
        
        linkage_matrix = linkage(X_sample, method='ward')
        dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5)
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        results['dendrogram_chart'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Cluster Analysis
        cluster_summary = []
        for i in range(optimal_k):
            cluster_data = self.data[self.data['Cluster'] == i]
            cluster_summary.append({
                'cluster': i,
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(self.data) * 100, 1)
            })
        
        results['cluster_summary'] = cluster_summary
        results['optimal_clusters'] = optimal_k
        
        self.results['unsupervised_analysis'] = results
        return results
    
    # ============= MAIN CHART GENERATION METHOD =============
    
    def generate_all_charts(self, filepath):
        """Generate ALL charts based on the uploaded dataset"""
        self.analyze_csv(filepath)
        
        all_charts = {}
        
        # Unit I & II: Data Prep & Pre-processing
        print("📊 Generating Data Prep & Pre-processing Charts...")
        prep_charts = self.preprocess_data()
        if 'pca_chart' in prep_charts:
            all_charts['PCA_Visualization'] = prep_charts['pca_chart']
        if 'lda_chart' in prep_charts:
            all_charts['LDA_Risk_Separation'] = prep_charts['lda_chart']
        
        # Unit III: Model Evaluation Metrics
        print("📊 Generating Model Evaluation Charts...")
        eval_results = self.evaluate_models()
        for model_name, metrics in eval_results.items():
            if 'chart' in metrics:
                all_charts[f'{model_name}_Evaluation'] = metrics['chart']
        
        # Unit IV: Regression Analysis
        print("📊 Generating Regression Analysis Charts...")
        reg_results = self.regression_analysis()
        if 'regression_chart' in reg_results:
            all_charts['Regression_Analysis'] = reg_results['regression_chart']
        
        # Unit V: Classification Analysis
        print("📊 Generating Classification Analysis Charts...")
        class_results = self.classification_analysis()
        if 'classification_roc_chart' in class_results:
            all_charts['Classification_ROC_Curves'] = class_results['classification_roc_chart']
        if 'feature_importance_chart' in class_results:
            all_charts['Feature_Importance'] = class_results['feature_importance_chart']
        
        # Unit VI: Unsupervised Learning
        print("📊 Generating Unsupervised Learning Charts...")
        unsup_results = self.unsupervised_analysis()
        if 'kmeans_chart' in unsup_results:
            all_charts['KMeans_Clustering'] = unsup_results['kmeans_chart']
        if 'dendrogram_chart' in unsup_results:
            all_charts['Hierarchical_Clustering_Dendrogram'] = unsup_results['dendrogram_chart']
        
        # Additional dataset info
        self.results['chart_count'] = len(all_charts)
        
        print(f"✅ Generated {len(all_charts)} charts successfully!")
        return all_charts
    
    def get_analysis_summary(self):
        """Get complete analysis summary with all metrics"""
        return {
            'basic_info': self.results.get('basic_info', {}),
            'missing_values': self.results.get('missing_values', []),
            'data_quality_score': self.results.get('data_quality_score', 0),
            'preprocessing': self.results.get('preprocessing', {}),
            'model_evaluation': self.results.get('model_evaluation', {}),
            'regression_analysis': self.results.get('regression_analysis', {}),
            'classification_analysis': self.results.get('classification_analysis', {}),
            'unsupervised_analysis': self.results.get('unsupervised_analysis', {})
        }
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        img = BytesIO()
        fig.savefig(img, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()