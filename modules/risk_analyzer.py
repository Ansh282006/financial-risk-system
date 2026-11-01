# modules/risk_analyzer.py
"""
Risk Analyzer Module - Analyzes Investment Risk and Profitability
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Analyzes investment risk and predicts profitability"""
    
    def __init__(self, loan_portfolio_path=None, operational_loss_path=None):
        self.loan_portfolio_path = loan_portfolio_path
        self.operational_loss_path = operational_loss_path
        self.models = {}
        self.scaler = None
        self.is_trained = False
        self.feature_cols = None
    
    def _create_synthetic_data(self, n_samples=5000):
        """Create synthetic investment data"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'investment_amount': np.random.uniform(10000, 10000000, n_samples),
            'portfolio_diversity': np.random.uniform(0.1, 1.0, n_samples),
            'market_volatility': np.random.uniform(0.05, 0.4, n_samples),
            'stock_allocation': np.random.uniform(0, 0.8, n_samples),
            'debt_allocation': np.random.uniform(0, 0.8, n_samples),
            'gold_allocation': np.random.uniform(0, 0.3, n_samples),
            'investment_horizon': np.random.choice([1, 3, 5, 10, 15, 20], n_samples),
            'expected_return': np.random.uniform(5, 20, n_samples)
        })
        
        # Calculate risk score
        data['risk_score'] = (
            data['market_volatility'] * 30 +
            data['stock_allocation'] * 20 +
            (1 - data['portfolio_diversity']) * 15
        ).clip(0, 100)
        
        # Add risk category
        data['risk_category'] = pd.cut(data['risk_score'], bins=3, labels=['Low', 'Medium', 'High'])
        
        return data
    
    def train_model(self):
        """Train risk analysis models"""
        logger.info("Training Risk Analyzer Model...")
        
        df = self._create_synthetic_data(5000)
        
        # Feature columns
        self.feature_cols = ['investment_amount', 'portfolio_diversity', 'market_volatility',
                            'stock_allocation', 'debt_allocation', 'gold_allocation', 'investment_horizon']
        
        X = df[self.feature_cols].fillna(0)
        y_score = df['risk_score']
        y_return = df['expected_return']
        y_category = df['risk_category'].astype('category').cat.codes
        
        # Split data - ensure same indices
        X_train, X_test, y_score_train, y_score_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
        _, _, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, random_state=42)
        _, _, y_category_train, y_category_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Risk Score Regressor
        self.models['risk_regressor'] = RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        )
        self.models['risk_regressor'].fit(X_train_scaled, y_score_train)
        
        # Train Return Predictor
        self.models['return_regressor'] = RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        )
        self.models['return_regressor'].fit(X_train_scaled, y_return_train)
        
        # Train Risk Classifier
        self.models['risk_classifier'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['risk_classifier'].fit(X_train_scaled, y_category_train)
        
        self.is_trained = True
        
        # Save models
        self._save_models()
        
        logger.info("✅ Risk Analyzer trained successfully!")
        
        return {'status': 'success'}
    
    def analyze(self, portfolio_data):
        """Analyze investment portfolio risk"""
        if not self.is_trained:
            self.train_model()
        
        # Extract portfolio allocation
        allocations = portfolio_data.get('portfolio', [])
        risk_tolerance = portfolio_data.get('risk_tolerance', 'Moderate')
        investment_amount = portfolio_data.get('investment_amount', 100000)
        horizon = portfolio_data.get('horizon', 5)
        
        # Calculate portfolio metrics
        stock_pct = 0
        debt_pct = 0
        gold_pct = 0
        
        for asset in allocations:
            pct = asset.get('percentage', 0) / 100
            asset_type = asset.get('type', '')
            if 'Stock' in asset_type or 'Equity' in asset_type:
                stock_pct += pct
            elif 'Mutual' in asset_type or 'Debt' in asset_type or 'Fixed' in asset_type:
                debt_pct += pct
            elif 'Gold' in asset_type:
                gold_pct += pct
        
        # Normalize
        total = stock_pct + debt_pct + gold_pct
        if total > 0:
            stock_pct = stock_pct / total
            debt_pct = debt_pct / total
            gold_pct = gold_pct / total
        
        # Prepare features for prediction
        input_data = {
            'investment_amount': investment_amount,
            'portfolio_diversity': min(1.0, len(allocations) / 10),
            'market_volatility': 0.15 + (stock_pct * 0.2),
            'stock_allocation': stock_pct,
            'debt_allocation': debt_pct,
            'gold_allocation': gold_pct,
            'investment_horizon': horizon
        }
        
        input_df = pd.DataFrame([input_data])[self.feature_cols]
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        risk_score = max(0, min(100, self.models['risk_regressor'].predict(input_scaled)[0]))
        expected_return = max(0, self.models['return_regressor'].predict(input_scaled)[0])
        risk_category_code = int(self.models['risk_classifier'].predict(input_scaled)[0])
        
        # Map risk category
        risk_categories = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_colors = ['success', 'warning', 'danger']
        risk_category = risk_categories[min(risk_category_code, 2)]
        risk_color = risk_colors[min(risk_category_code, 2)]
        
        # Calculate profit/loss
        expected_profit = investment_amount * (expected_return / 100)
        
        return {
            'risk_score': round(risk_score, 1),
            'risk_category': risk_category,
            'risk_color': risk_color,
            'expected_return': round(expected_return, 2),
            'expected_profit': round(expected_profit, 0),
            'formatted_profit': f"₹{expected_profit:,.0f}",
            'portfolio_analysis': {
                'diversification_score': round(min(100, len(allocations) * 12.5), 1),
                'stock_exposure': round(stock_pct * 100, 1),
                'safe_assets': round((debt_pct + gold_pct) * 100, 1)
            },
            'recommendations': self._generate_recommendations(risk_score, stock_pct, risk_tolerance),
            'risk_factors': self._identify_risk_factors(risk_score, stock_pct)
        }
    
    def _generate_recommendations(self, risk_score, stock_pct, risk_tolerance):
        """Generate risk-based recommendations"""
        recommendations = []
        
        if risk_score > 70:
            recommendations.append("⚠️ High risk detected - Consider reducing equity exposure")
            recommendations.append("📊 Add more debt instruments for stability")
        
        if stock_pct > 0.6 and risk_tolerance != 'Aggressive':
            recommendations.append("📉 Reduce stock allocation to 40-50% for better balance")
        
        if risk_score < 30:
            recommendations.append("✅ Well-diversified portfolio with good risk management")
        
        if not recommendations:
            recommendations.append("📈 Maintain current allocation with periodic rebalancing")
        
        return recommendations
    
    def _identify_risk_factors(self, risk_score, stock_pct):
        """Identify specific risk factors"""
        factors = []
        
        if stock_pct > 0.5:
            factors.append("High equity concentration increases market risk")
        
        if risk_score > 60:
            factors.append("Portfolio volatility above recommended levels")
        
        if not factors:
            factors.append("Risk levels within acceptable range")
        
        return factors
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.models, 'models/risk_models.pkl')
        joblib.dump(self.scaler, 'models/risk_scaler.pkl')
        joblib.dump(self.feature_cols, 'models/risk_features.pkl')
        logger.info("✅ Risk models saved")