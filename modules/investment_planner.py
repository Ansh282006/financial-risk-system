# modules/investment_planner.py
"""
Investment Planner Module - Future Investment Planning
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvestmentPlanner:
    """Plans future investments based on market conditions"""
    
    def __init__(self, market_data_path=None, economic_data_path=None):
        self.market_data_path = market_data_path
        self.economic_data_path = economic_data_path
        self.models = {}
        self.scaler = None
        self.is_trained = False
        self.feature_cols = None
        self.sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Real Estate']
    
    def load_data(self):
        """Create synthetic data for investment planning"""
        logger.info("Creating synthetic investment planning data...")
        return self._create_synthetic_data()
    
    def _create_synthetic_data(self, n_samples=5000):
        """Create synthetic market data"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'investment_amount': np.random.uniform(10000, 5000000, n_samples),
            'time_horizon': np.random.choice([5, 7, 10, 15, 20], n_samples),
            'risk_appetite_code': np.random.choice([0, 1, 2], n_samples),  # 0=Low, 1=Moderate, 2=High
            'market_volatility': np.random.uniform(0.05, 0.35, n_samples)
        })
        
        # Calculate returns for different sectors
        for sector in self.sectors:
            base_return = {
                'Technology': 14, 'Healthcare': 12, 'Finance': 10,
                'Energy': 8, 'Consumer': 11, 'Real Estate': 9
            }[sector]
            # Add noise based on volatility
            data[f'{sector}_return'] = base_return + np.random.normal(0, 2, n_samples)
        
        return data
    
    def train_model(self):
        """Train investment planning models"""
        logger.info("Training Investment Planner Model...")
        
        df = self.load_data()
        
        # Feature columns
        self.feature_cols = ['investment_amount', 'time_horizon', 'market_volatility', 'risk_appetite_code']
        
        X = df[self.feature_cols].fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models for each sector
        self.models = {}
        for sector in self.sectors:
            y = df[f'{sector}_return']
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            self.models[sector] = model
            logger.info(f"  {sector} Model trained")
        
        self.is_trained = True
        
        # Save models
        self._save_models()
        
        logger.info("✅ Investment Planner trained successfully!")
        
        return {'sectors_trained': len(self.sectors)}
    
    def plan(self, investment_data):
        """Create investment plan based on client preferences"""
        if not self.is_trained:
            self.train_model()
        
        amount = investment_data.get('amount', 100000)
        time_horizon = investment_data.get('time_horizon', 10)
        risk_appetite = investment_data.get('risk_appetite', 'Moderate')
        
        # Map risk appetite to code
        risk_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        risk_code = risk_map.get(risk_appetite, 1)
        
        # Prepare features for prediction
        input_data = {
            'investment_amount': amount,
            'time_horizon': time_horizon,
            'market_volatility': 0.15,
            'risk_appetite_code': risk_code
        }
        
        input_df = pd.DataFrame([input_data])[self.feature_cols]
        input_scaled = self.scaler.transform(input_df)
        
        # Predict returns for each sector
        sector_returns = {}
        for sector, model in self.models.items():
            predicted_return = model.predict(input_scaled)[0]
            sector_returns[sector] = max(2, min(25, predicted_return))
        
        # Create investment plan based on time horizon and risk
        if risk_appetite == 'High':
            recommended_sectors = ['Technology', 'Healthcare', 'Finance']
            allocation = {'Technology': 40, 'Healthcare': 35, 'Finance': 25}
        elif risk_appetite == 'Low':
            recommended_sectors = ['Real Estate', 'Energy', 'Consumer']
            allocation = {'Real Estate': 35, 'Energy': 35, 'Consumer': 30}
        else:
            recommended_sectors = ['Technology', 'Consumer', 'Real Estate']
            allocation = {'Technology': 35, 'Consumer': 35, 'Real Estate': 30}
        
        # Calculate projected returns
        total_projected = amount
        projected_returns = {}
        
        for sector, alloc_pct in allocation.items():
            annual_return = sector_returns.get(sector, 10)
            years = time_horizon
            sector_value = amount * (alloc_pct / 100) * ((1 + annual_return/100) ** years)
            total_projected += sector_value - (amount * (alloc_pct / 100))
            
            projected_returns[sector] = {
                'allocation': alloc_pct,
                'annual_return': round(annual_return, 1),
                'projected_value': round(sector_value, 0),
                'total_gain': round(sector_value - (amount * (alloc_pct / 100)), 0)
            }
        
        total_return = total_projected - amount
        
        # Determine plan type
        if time_horizon >= 15:
            plan_type = "Long-term Growth"
        elif time_horizon >= 10:
            plan_type = "Balanced Growth"
        else:
            plan_type = "Moderate Growth"
        
        # Generate explanation
        explanation = f"""
        Based on your {risk_appetite.lower()} risk appetite and {time_horizon}-year investment horizon, 
        we recommend a {plan_type.lower()} strategy.
        
        Technology sector offers high growth potential with some volatility.
        Healthcare provides stable demand and consistent returns.
        Finance sector benefits from economic growth cycles.
        """
        
        return {
            'plan_type': plan_type,
            'investment_amount': amount,
            'formatted_amount': f"₹{amount:,.0f}",
            'time_horizon': time_horizon,
            'risk_appetite': risk_appetite,
            'allocation': allocation,
            'sector_returns': sector_returns,
            'projected_returns': projected_returns,
            'total_projected_value': round(total_projected, 0),
            'formatted_final_value': f"₹{total_projected:,.0f}",
            'total_return': round(total_return, 0),
            'formatted_return': f"₹{total_return:,.0f}",
            'explanation': explanation,
            'recommendations': [
                "📈 Start with a Systematic Investment Plan (SIP)",
                "🔄 Review and rebalance portfolio annually",
                "📊 Maintain emergency fund of 6 months expenses",
                "📝 Track performance quarterly"
            ]
        }
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.models, 'models/investment_models.pkl')
        joblib.dump(self.scaler, 'models/investment_scaler.pkl')
        joblib.dump(self.feature_cols, 'models/investment_features.pkl')
        logger.info("✅ Investment models saved")