# modules/loan_predictor.py
"""
Loan Predictor Module - Predicts Loan Amount and Interest Rate
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoanPredictor:
    """Predicts loan eligibility, amount, and interest rate"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.models = {}
        self.scaler = None
        self.is_trained = False
        self.feature_columns = None
    
    def load_data(self):
        """Load and prepare the dataset"""
        if self.dataset_path and os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset: {df.shape}")
            # Create synthetic target variables based on existing features
            df = self._create_target_variables(df)
            return df
        else:
            logger.warning("Dataset not found. Creating synthetic data...")
            return self._create_synthetic_data()
    
    def _create_target_variables(self, df):
        """Create target variables from existing features"""
        # Use available numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create loan amount based on LIMIT_BAL if available
        if 'LIMIT_BAL' in df.columns:
            df['eligible_loan_amount'] = (df['LIMIT_BAL'] * np.random.uniform(0.3, 0.7, len(df))).astype(int)
        else:
            df['eligible_loan_amount'] = np.random.randint(50000, 500000, len(df))
        
        # Create interest rate based on risk profile
        if 'PAY_0' in df.columns:
            df['interest_rate'] = 8 + (df['PAY_0'].clip(0, 5) * 0.5)
        else:
            df['interest_rate'] = np.random.uniform(6, 15, len(df))
        
        # Create default risk
        if 'default.payment.next.month' in df.columns:
            df['default_risk'] = df['default.payment.next.month']
        else:
            df['default_risk'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
        
        return df
    
    def _create_synthetic_data(self, n_samples=5000):
        """Create synthetic credit data for training"""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'LIMIT_BAL': np.random.randint(10000, 1000000, n_samples),
            'AGE': np.random.randint(18, 80, n_samples),
            'PAY_0': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6], n_samples),
            'PAY_2': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6], n_samples),
            'PAY_3': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6], n_samples),
            'BILL_AMT1': np.random.randint(0, 500000, n_samples),
            'BILL_AMT2': np.random.randint(0, 500000, n_samples),
            'BILL_AMT3': np.random.randint(0, 500000, n_samples),
            'PAY_AMT1': np.random.randint(0, 200000, n_samples),
            'PAY_AMT2': np.random.randint(0, 200000, n_samples),
            'PAY_AMT3': np.random.randint(0, 200000, n_samples),
        })
        
        # Calculate default risk
        payment_delays = (df[['PAY_0', 'PAY_2', 'PAY_3']] > 2).sum(axis=1)
        df['default_risk'] = (payment_delays > 1).astype(int)
        
        # Calculate eligible loan amount
        df['eligible_loan_amount'] = (
            df['LIMIT_BAL'] * 0.6 * 
            (1 - df['default_risk'] * 0.5) * 
            (df['AGE'] / 50).clip(0.5, 1.5)
        ).astype(int)
        
        # Calculate interest rate
        df['interest_rate'] = 8 + (df['default_risk'] * 4) + (df['PAY_0'].clip(0, 5) * 0.3)
        df['interest_rate'] = df['interest_rate'].clip(6, 18)
        
        return df
    
    def train_model(self):
        """Train the loan prediction model"""
        logger.info("Training Loan Predictor Model...")
        
        df = self.load_data()
        
        # Define features (use columns that exist in the dataset)
        potential_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 
                              'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                              'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']
        
        self.feature_columns = [col for col in potential_features if col in df.columns]
        
        if not self.feature_columns:
            self.feature_columns = ['LIMIT_BAL', 'AGE'] if 'LIMIT_BAL' in df.columns else ['AGE']
        
        X = df[self.feature_columns].fillna(0)
        y_amount = df['eligible_loan_amount']
        y_rate = df['interest_rate']
        y_risk = df['default_risk']
        
        # Ensure same number of samples
        logger.info(f"Training samples: {len(X)}")
        
        # Split data
        X_train, X_test, y_amount_train, y_amount_test = train_test_split(X, y_amount, test_size=0.2, random_state=42)
        _, _, y_rate_train, y_rate_test = train_test_split(X, y_rate, test_size=0.2, random_state=42)
        _, _, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Loan Amount Regressor
        self.models['amount_regressor'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['amount_regressor'].fit(X_train_scaled, y_amount_train)
        
        # Train Interest Rate Regressor
        self.models['rate_regressor'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['rate_regressor'].fit(X_train_scaled, y_rate_train)
        
        # Train Risk Classifier
        self.models['risk_classifier'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['risk_classifier'].fit(X_train_scaled, y_risk_train)
        
        self.is_trained = True
        
        # Save models
        self._save_models()
        
        logger.info("✅ Loan Predictor trained successfully!")
        
        return {'status': 'success'}
    
    def predict(self, customer_data):
        """Predict loan amount and interest rate"""
        if not self.is_trained:
            self.train_model()
        
        # Prepare input features
        input_data = {}
        for col in self.feature_columns:
            if col in customer_data:
                input_data[col] = float(customer_data[col])
            else:
                input_data[col] = 0
        
        input_df = pd.DataFrame([input_data])[self.feature_columns]
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        loan_amount = max(0, self.models['amount_regressor'].predict(input_scaled)[0])
        interest_rate = max(5, min(18, self.models['rate_regressor'].predict(input_scaled)[0]))
        risk_prob = self.models['risk_classifier'].predict_proba(input_scaled)[0][1]
        
        # Calculate monthly EMI
        loan_term_years = customer_data.get('loan_term_years', 5)
        monthly_rate = interest_rate / 100 / 12
        months = loan_term_years * 12
        
        if monthly_rate > 0:
            monthly_emi = loan_amount * monthly_rate * (1 + monthly_rate)**months / ((1 + monthly_rate)**months - 1)
        else:
            monthly_emi = loan_amount / months
        
        # Determine risk category
        if risk_prob < 0.3:
            risk_category = "Low Risk"
            risk_color = "success"
        elif risk_prob < 0.6:
            risk_category = "Medium Risk"
            risk_color = "warning"
        else:
            risk_category = "High Risk"
            risk_color = "danger"
        
        return {
            'eligible_loan_amount': round(loan_amount, 0),
            'formatted_amount': f"₹{loan_amount:,.0f}",
            'interest_rate': round(interest_rate, 2),
            'monthly_emi': round(monthly_emi, 0),
            'formatted_emi': f"₹{monthly_emi:,.0f}",
            'risk_probability': round(risk_prob * 100, 1),
            'risk_category': risk_category,
            'risk_color': risk_color,
            'loan_term_years': loan_term_years,
            'total_repayment': round(monthly_emi * months, 0),
            'total_interest': round(monthly_emi * months - loan_amount, 0),
            'recommendations': self._generate_recommendations(customer_data, risk_prob)
        }
    
    def _generate_recommendations(self, customer_data, risk_prob):
        """Generate loan recommendations"""
        recommendations = []
        
        if risk_prob > 0.6:
            recommendations.append("⚠️ Consider providing collateral to reduce interest rate")
            recommendations.append("📋 Shorten the loan tenure to reduce risk")
        
        age = customer_data.get('AGE', 35)
        if age < 25:
            recommendations.append("👤 Young age - Consider adding a co-applicant")
        
        pay_0 = customer_data.get('PAY_0', 0)
        if pay_0 > 2:
            recommendations.append("📊 Improve payment history before applying")
        
        if not recommendations:
            recommendations.append("✅ You have a strong credit profile")
            recommendations.append("💰 You qualify for our best interest rates")
        
        return recommendations
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.models, 'models/loan_models.pkl')
        joblib.dump(self.scaler, 'models/loan_scaler.pkl')
        joblib.dump(self.feature_columns, 'models/loan_features.pkl')
        logger.info("✅ Loan models saved")