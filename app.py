# app.py
"""
Financial Risk Early Warning System
ML-Based Project using 7 Financial Datasets
"""

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Import modules
from modules.data_analyzer import DataAnalyzer
from modules.loan_predictor import LoanPredictor
from modules.risk_analyzer import RiskAnalyzer
from modules.investment_planner import InvestmentPlanner

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'financial_risk_system_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('charts', exist_ok=True)

# Dataset paths
DATASETS_DIR = r"E:\EDUCATIONAL COURSE\SECOND YEAR ENGINEERING\ML\CP\financial-risk-system"

DATASETS = {
    'uci_credit_card': os.path.join(DATASETS_DIR, 'UCI_Credit_Card.csv'),
    'loan_portfolio': os.path.join(DATASETS_DIR, 'loan_portfolio_2000.1.csv'),
    'market_data': os.path.join(DATASETS_DIR, 'market_data_2000.1.csv'),
    'economic_indicators': os.path.join(DATASETS_DIR, 'economic_indicators_2000.1.csv'),
    'operational_loss': os.path.join(DATASETS_DIR, 'operational_loss_2000.1.csv')
}

# Initialize modules
data_analyzer = DataAnalyzer()
loan_predictor = LoanPredictor(DATASETS['uci_credit_card'])
risk_analyzer = RiskAnalyzer(DATASETS['loan_portfolio'], DATASETS['operational_loss'])
investment_planner = InvestmentPlanner(DATASETS['market_data'], DATASETS['economic_indicators'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def train_all_models():
    """Train all ML models"""
    print("\n" + "=" * 60)
    print("Training Machine Learning Models...")
    print("=" * 60)
    
    results = {}
    
    # Train Loan Predictor
    print("\n📊 Training Loan Predictor Model...")
    try:
        results['loan'] = loan_predictor.train_model()
        print("✅ Loan Predictor trained successfully!")
    except Exception as e:
        print(f"❌ Loan Predictor error: {e}")
        results['loan'] = {'error': str(e)}
    
    # Train Risk Analyzer
    print("\n📈 Training Risk Analyzer Model...")
    try:
        results['risk'] = risk_analyzer.train_model()
        print("✅ Risk Analyzer trained successfully!")
    except Exception as e:
        print(f"❌ Risk Analyzer error: {e}")
        results['risk'] = {'error': str(e)}
    
    # Train Investment Planner
    print("\n💰 Training Investment Planner Model...")
    try:
        results['investment'] = investment_planner.train_model()
        print("✅ Investment Planner trained successfully!")
    except Exception as e:
        print(f"❌ Investment Planner error: {e}")
        results['investment'] = {'error': str(e)}
    
    print("\n" + "=" * 60)
    return results


# ============= ROUTES =============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/loan-prediction')
def loan_prediction_page():
    return render_template('loan_prediction.html')


@app.route('/risk-analysis')
def risk_analysis_page():
    return render_template('risk_analysis.html')


@app.route('/investment-planning')
def investment_planning_page():
    return render_template('investment_planning.html')


# ============= API ENDPOINTS =============

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and analyze CSV file with all ML techniques"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get complete analysis with all metrics
        analysis_result = data_analyzer.analyze_csv(filepath)
        
        # Generate all ML charts
        charts = data_analyzer.generate_all_charts(filepath)
        
        # Get complete summary
        full_analysis = data_analyzer.get_analysis_summary()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'analysis': full_analysis,
            'charts': charts,
            'chart_count': len(charts),
            'message': f'Successfully analyzed {filename}'
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-loan', methods=['POST'])
def predict_loan():
    try:
        data = request.json
        result = loan_predictor.predict(data)
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-risk', methods=['POST'])
def analyze_risk():
    try:
        data = request.json
        result = risk_analyzer.analyze(data)
        return jsonify({'success': True, 'analysis': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plan-investment', methods=['POST'])
def plan_investment():
    try:
        data = request.json
        result = investment_planner.plan(data)
        return jsonify({'success': True, 'plan': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-datasets-info')
def get_datasets_info():
    datasets_info = {}
    for name, path in DATASETS.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, nrows=1)
                datasets_info[name] = {
                    'exists': True,
                    'columns': len(df.columns),
                    'size_mb': round(os.path.getsize(path) / (1024 * 1024), 2)
                }
            except:
                datasets_info[name] = {'exists': False}
        else:
            datasets_info[name] = {'exists': False}
    return jsonify(datasets_info)


@app.route('/api/train-models', methods=['POST'])
def train_models():
    try:
        results = train_all_models()
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-sample-data')
def get_sample_data():
    """Get sample data for demonstration"""
    sample_data = {
        'loan': {
            'monthly_income': 75000,
            'credit_score': 720,
            'LIMIT_BAL': 200000,
            'AGE': 35,
            'employment_years': 5
        },
        'investment': {
            'amount': 500000,
            'time_horizon': 10,
            'risk_appetite': 'Moderate'
        }
    }
    return jsonify(sample_data)


if __name__ == '__main__':
    print("=" * 60)
    print("Financial Risk Early Warning System")
    print("ML-Based Project with 7 Financial Datasets")
    print("=" * 60)
    
    # Train models on startup
    train_all_models()
    
    print("\n🚀 Server starting at http://localhost:5000")
    print("📊 Available Pages:")
    print("   - Dashboard: http://localhost:5000/")
    print("   - Upload & Charts: http://localhost:5000/upload")
    print("   - Loan Prediction: http://localhost:5000/loan-prediction")
    print("   - Risk Analysis: http://localhost:5000/risk-analysis")
    print("   - Investment Planning: http://localhost:5000/investment-planning")
    print("=" * 60)
    
    app.run(debug=True, port=5000)