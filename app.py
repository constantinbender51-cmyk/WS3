"""
Trading ML System - Predict Optimal Positions
Trains multiple ML models and displays results via web dashboard
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from flask import Flask, render_template_string
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import gdown
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def download_data(google_drive_url):
    """Download CSV from Google Drive"""
    print("Downloading data from Google Drive...")
    
    # Extract file ID from the URL
    file_id = google_drive_url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    output = 'trading_data.csv'
    gdown.download(download_url, output, quiet=False)
    
    df = pd.read_csv(output)
    
    # Try to parse date column if it exists
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    return df


def create_baseline_features(df, lookback=30):
    """Create baseline OHLCV features"""
    features = df.copy()
    
    # Returns
    for col in ['open', 'high', 'low', 'close']:
        if col in features.columns:
            features[f'{col}_return'] = features[col].pct_change()
            features[f'{col}_return_5d'] = features[col].pct_change(5)
            features[f'{col}_return_10d'] = features[col].pct_change(10)
            features[f'{col}_return_20d'] = features[col].pct_change(20)
    
    # Volume changes
    if 'volume' in features.columns:
        features['volume_change'] = features['volume'].pct_change()
        features['volume_ma_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
    
    # Price momentum
    if 'close' in features.columns:
        features['momentum_5'] = features['close'] / features['close'].shift(5) - 1
        features['momentum_10'] = features['close'] / features['close'].shift(10) - 1
        features['momentum_20'] = features['close'] / features['close'].shift(20) - 1
        
        # Volatility
        if 'close_return' in features.columns:
            features['volatility_5'] = features['close_return'].rolling(5).std()
            features['volatility_10'] = features['close_return'].rolling(10).std()
            features['volatility_20'] = features['close_return'].rolling(20).std()
    
    # High-Low spread
    if 'high' in features.columns and 'low' in features.columns and 'close' in features.columns:
        features['hl_spread'] = (features['high'] - features['low']) / features['close']
        features['hl_spread_ma'] = features['hl_spread'].rolling(10).mean()
    
    return features


def create_technical_indicators(df):
    """Add technical indicators to features"""
    features = df.copy()
    
    if 'close' not in features.columns:
        return features
    
    # RSI
    delta = features['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = features['close'].ewm(span=12, adjust=False).mean()
    exp2 = features['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_diff'] = features['macd'] - features['macd_signal']
    
    # Bollinger Bands
    features['bb_ma'] = features['close'].rolling(20).mean()
    features['bb_std'] = features['close'].rolling(20).std()
    features['bb_upper'] = features['bb_ma'] + (features['bb_std'] * 2)
    features['bb_lower'] = features['bb_ma'] - (features['bb_std'] * 2)
    features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    # Moving Averages
    for window in [5, 10, 20, 50]:
        features[f'ma_{window}'] = features['close'].rolling(window).mean()
        features[f'price_to_ma_{window}'] = features['close'] / features[f'ma_{window}']
    
    # ATR (Average True Range)
    if 'high' in features.columns and 'low' in features.columns:
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - features['close'].shift())
        low_close = np.abs(features['low'] - features['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr'] = true_range.rolling(14).mean()
    
    return features


def prepare_data(df, use_technical=False, lookback=30):
    """Prepare features and target for ML"""
    print(f"\nPreparing data with{'out' if not use_technical else ''} technical indicators...")
    
    # Create features
    features = create_baseline_features(df, lookback)
    if use_technical:
        features = create_technical_indicators(features)
    
    # Drop rows with NaN
    features = features.dropna()
    
    # Define columns to exclude (target, dates, and other non-numeric columns)
    exclude_cols = ['optimal_position', 'date', 'Date', 'datetime', 'Datetime', 'Unnamed: 0', 'index']
    
    # Get all numeric columns that aren't excluded
    feature_cols = []
    for col in features.columns:
        # Skip if in exclude list
        if any(excl.lower() in col.lower() for excl in exclude_cols):
            continue
        # Only include numeric columns
        if pd.api.types.is_numeric_dtype(features[col]):
            feature_cols.append(col)
    
    # Remove 'optimal_position' from features if it somehow got included
    feature_cols = [col for col in feature_cols if 'optimal_position' not in col.lower()]
    
    X = features[feature_cols]
    y = features['optimal_position']
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Feature columns: {feature_cols}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, features


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train):
    """Train all models"""
    models = {}
    
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    # Naive Bayes
    print("\n1. Training Naive Bayes...")
    models['Naive Bayes'] = GaussianNB()
    models['Naive Bayes'].fit(X_train, y_train)
    
    # Logistic Regression
    print("2. Training Logistic Regression...")
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    models['Logistic Regression'].fit(X_train, y_train)
    
    # Random Forest
    print("3. Training Random Forest...")
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    models['Random Forest'].fit(X_train, y_train)
    
    # XGBoost - Map classes to [0, 1, 2] for compatibility
    print("4. Training XGBoost...")
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    # Convert y_train from [-1, 0, 1] to [0, 1, 2] for XGBoost
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    
    # Calculate sample weights for XGBoost
    sample_weights_xgb = compute_sample_weight(class_weight='balanced', y=y_train_mapped)
    
    models['XGBoost'].fit(X_train, y_train_mapped, sample_weight=sample_weights_xgb)
    
    # LightGBM
    print("5. Training LightGBM...")
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced')
    models['LightGBM'].fit(X_train, y_train)
    
    # Neural Network
    print("6. Training Neural Network...")
    models['Neural Network'] = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42, early_stopping=True)
    models['Neural Network'].fit(X_train, y_train)
    
    print("\nAll models trained!")
    return models


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Evaluate all models"""
    results = {}
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Predictions
        # For XGBoost, the prediction needs to be re-mapped from [0, 1, 2] back to [-1, 0, 1]
        if name == 'XGBoost':
            train_pred_mapped = model.predict(X_train)
            test_pred_mapped = model.predict(X_test)
            train_pred = pd.Series(train_pred_mapped).map({0: -1, 1: 0, 2: 1}).values
            test_pred = pd.Series(test_pred_mapped).map({0: -1, 1: 0, 2: 1}).values
        else:
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
        
        # Accuracy
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        
        results[name] = {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_report': classification_report(y_test, test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, test_pred)
        }
    
    return results


# ============================================================================
# CAPITAL SIMULATION
# ============================================================================

def simulate_trading(positions, returns, initial_capital=10000):
    """Simulate trading with given positions"""
    capital = initial_capital
    capital_history = [initial_capital]
    
    for pos, ret in zip(positions, returns):
        if pos == 1:  # Long
            capital *= (1 + ret)
        elif pos == -1:  # Short
            capital *= (1 - ret)
        # pos == 0: stay in cash, no change
        
        capital_history.append(capital)
    
    return np.array(capital_history)


def calculate_metrics(capital_history):
    """Calculate trading metrics"""
    returns = np.diff(capital_history) / capital_history[:-1]
    
    total_return = (capital_history[-1] / capital_history[0] - 1) * 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = np.min(capital_history / np.maximum.accumulate(capital_history) - 1) * 100
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_capital': capital_history[-1]
    }


def simulate_all_strategies(results, test_data, test_target):
    """Simulate trading for all strategies"""
    # Calculate returns
    returns = test_data['close'].pct_change().fillna(0).values[1:]  # Align with positions
    
    strategies = {}
    
    # Optimal strategy
    optimal_positions = test_target.values[:-1]  # Shift to align
    strategies['Optimal'] = simulate_trading(optimal_positions, returns)
    
    # Buy and Hold
    buy_hold_positions = np.ones(len(returns))
    strategies['Buy & Hold'] = simulate_trading(buy_hold_positions, returns)
    
    # Model predictions
    for name, result in results.items():
        pred_positions = result['test_pred'][:-1]  # Shift to align
        strategies[name] = simulate_trading(pred_positions, returns)
    
    # Calculate metrics
    metrics = {}
    for name, capital in strategies.items():
        metrics[name] = calculate_metrics(capital)
    
    return strategies, metrics


# ============================================================================
# WEB DASHBOARD
# ============================================================================

def create_dashboard(baseline_results, enhanced_results, 
                     baseline_strategies, enhanced_strategies,
                     baseline_metrics, enhanced_metrics,
                     test_data, test_target):
    """Create interactive web dashboard"""
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        # Create plots
        
        # 1. Model Accuracy Comparison
        fig_accuracy = go.Figure()
        
        models = list(baseline_results.keys())
        baseline_train = [baseline_results[m]['train_acc'] for m in models]
        baseline_test = [baseline_results[m]['test_acc'] for m in models]
        enhanced_train = [enhanced_results[m]['train_acc'] for m in models]
        enhanced_test = [enhanced_results[m]['test_acc'] for m in models]
        
        fig_accuracy.add_trace(go.Bar(name='Baseline Train', x=models, y=baseline_train))
        fig_accuracy.add_trace(go.Bar(name='Baseline Test', x=models, y=baseline_test))
        fig_accuracy.add_trace(go.Bar(name='Enhanced Train', x=models, y=enhanced_train))
        fig_accuracy.add_trace(go.Bar(name='Enhanced Test', x=models, y=enhanced_test))
        
        fig_accuracy.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            barmode='group',
            height=500
        )
        
        # 2. Capital Curves - Baseline
        fig_capital_baseline = go.Figure()
        for name, capital in baseline_strategies.items():
            fig_capital_baseline.add_trace(go.Scatter(
                y=capital,
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
        
        fig_capital_baseline.update_layout(
            title='Capital Evolution - Baseline Features',
            xaxis_title='Trading Day',
            yaxis_title='Capital ($)',
            height=500,
            hovermode='x unified'
        )
        
        # 3. Capital Curves - Enhanced
        fig_capital_enhanced = go.Figure()
        for name, capital in enhanced_strategies.items():
            fig_capital_enhanced.add_trace(go.Scatter(
                y=capital,
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
        
        fig_capital_enhanced.update_layout(
            title='Capital Evolution - Enhanced Features (with Technical Indicators)',
            xaxis_title='Trading Day',
            yaxis_title='Capital ($)',
            height=500,
            hovermode='x unified'
        )
        
        # 4. Performance Metrics Table
        metrics_df = pd.DataFrame({
            'Baseline': baseline_metrics,
            'Enhanced': enhanced_metrics
        }).T
        
        # 5. Predictions vs Actual - Best Model
        best_model_name = max(baseline_results.items(), key=lambda x: x[1]['test_acc'])[0]
        
        fig_predictions = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Baseline Predictions vs Actual', 'Enhanced Predictions vs Actual')
        )
        
        # Baseline
        test_indices = range(len(test_target))
        fig_predictions.add_trace(
            go.Scatter(x=list(test_indices), y=test_target.values, 
                      mode='lines', name='Actual', line=dict(color='black', width=1)),
            row=1, col=1
        )
        fig_predictions.add_trace(
            go.Scatter(x=list(test_indices), y=baseline_results[best_model_name]['test_pred'],
                      mode='markers', name='Baseline Pred', marker=dict(size=3)),
            row=1, col=1
        )
        
        # Enhanced
        fig_predictions.add_trace(
            go.Scatter(x=list(test_indices), y=test_target.values,
                      mode='lines', name='Actual', line=dict(color='black', width=1), showlegend=False),
            row=2, col=1
        )
        fig_predictions.add_trace(
            go.Scatter(x=list(test_indices), y=enhanced_results[best_model_name]['test_pred'],
                      mode='markers', name='Enhanced Pred', marker=dict(size=3)),
            row=2, col=1
        )
        
        fig_predictions.update_layout(height=700, title_text=f'Predictions vs Actual - {best_model_name}')
        fig_predictions.update_yaxes(title_text="Position", row=1, col=1)
        fig_predictions.update_yaxes(title_text="Position", row=2, col=1)
        fig_predictions.update_xaxes(title_text="Time", row=2, col=1)
        
        # HTML Template
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading ML Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                h2 {
                    color: #666;
                    border-bottom: 2px solid #333;
                    padding-bottom: 10px;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .metrics-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                .metrics-table th, .metrics-table td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }
                .metrics-table th {
                    background-color: #4CAF50;
                    color: white;
                }
                .metrics-table tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .plot-container {
                    margin: 30px 0;
                }
                .summary {
                    background-color: #e7f3fe;
                    border-left: 6px solid #2196F3;
                    padding: 15px;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Trading ML System Dashboard 📈</h1>
                
                <div class="summary">
                    <h3>Executive Summary</h3>
                    <p><strong>Best Model:</strong> {{ best_model }}</p>
                    <p><strong>Best Strategy ROI (Baseline):</strong> {{ best_baseline_roi }}%</p>
                    <p><strong>Best Strategy ROI (Enhanced):</strong> {{ best_enhanced_roi }}%</p>
                    <p><strong>Technical Indicators Improvement:</strong> {{ improvement }}</p>
                </div>
                
                <h2>📊 Model Accuracy Comparison</h2>
                <div class="plot-container">
                    {{ plot_accuracy | safe }}
                </div>
                
                <h2>📈 Performance Metrics</h2>
                <h3>Baseline Features</h3>
                {{ baseline_table | safe }}
                
                <h3>Enhanced Features (with Technical Indicators)</h3>
                {{ enhanced_table | safe }}
                
                <h2>💰 Capital Evolution</h2>
                <div class="plot-container">
                    {{ plot_capital_baseline | safe }}
                </div>
                
                <div class="plot-container">
                    {{ plot_capital_enhanced | safe }}
                </div>
                
                <h2>🎯 Predictions vs Actual</h2>
                <div class="plot-container">
                    {{ plot_predictions | safe }}
                </div>
                
            </div>
        </body>
        </html>
        '''
        
        # Find best model and strategies
        best_model = max(baseline_results.items(), key=lambda x: x[1]['test_acc'])[0]
        best_baseline = max(baseline_metrics.items(), key=lambda x: x[1]['total_return'])
        best_enhanced = max(enhanced_metrics.items(), key=lambda x: x[1]['total_return'])
        
        # Calculate improvement
        baseline_best_roi = best_baseline[1]['total_return']
        enhanced_best_roi = best_enhanced[1]['total_return']
        improvement = "✅ Improved" if enhanced_best_roi > baseline_best_roi else "❌ No improvement"
        
        # Create metrics tables HTML
        def create_metrics_table(metrics):
            html = '<table class="metrics-table"><tr><th>Strategy</th><th>Total Return (%)</th><th>Sharpe Ratio</th><th>Max Drawdown (%)</th><th>Final Capital ($)</th></tr>'
            for name, m in sorted(metrics.items(), key=lambda x: x[1]['total_return'], reverse=True):
                html += f'<tr><td>{name}</td><td>{m["total_return"]:.2f}</td><td>{m["sharpe_ratio"]:.2f}</td><td>{m["max_drawdown"]:.2f}</td><td>${m["final_capital"]:.2f}</td></tr>'
            html += '</table>'
            return html
        
        baseline_table = create_metrics_table(baseline_metrics)
        enhanced_table = create_metrics_table(enhanced_metrics)
        
        return render_template_string(
            html_template,
            plot_accuracy=fig_accuracy.to_html(full_html=False, include_plotlyjs=False),
            plot_capital_baseline=fig_capital_baseline.to_html(full_html=False, include_plotlyjs=False),
            plot_capital_enhanced=fig_capital_enhanced.to_html(full_html=False, include_plotlyjs=False),
            plot_predictions=fig_predictions.to_html(full_html=False, include_plotlyjs=False),
            baseline_table=baseline_table,
            enhanced_table=enhanced_table,
            best_model=best_model,
            best_baseline_roi=f"{baseline_best_roi:.2f}",
            best_enhanced_roi=f"{enhanced_best_roi:.2f}",
            improvement=improvement
        )
    
    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    GOOGLE_DRIVE_URL = "https://docs.google.com/spreadsheets/d/1bjfIzg2_I_zN95v5oQac671_4AV0D74x/edit?usp=drivesdk&ouid=114372102418564925207&rtpof=true&sd=true"
    
    # Download data
    df = download_data(GOOGLE_DRIVE_URL)
    
    # ========================================================================
    # BASELINE FEATURES
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE FEATURES (OHLCV Only)")
    print("="*80)
    
    X_baseline, y_baseline, data_baseline = prepare_data(df, use_technical=False)
    
    # Split data chronologically
    split_idx = int(len(X_baseline) * 0.8)
    X_train_base = X_baseline.iloc[:split_idx]
    X_test_base = X_baseline.iloc[split_idx:]
    y_train_base = y_baseline.iloc[:split_idx]
    y_test_base = y_baseline.iloc[split_idx:]
    test_data_base = data_baseline.iloc[split_idx:]
    
    # Normalize
    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)
    
    # Train and evaluate
    models_baseline = train_models(X_train_base_scaled, y_train_base)
    results_baseline = evaluate_models(models_baseline, X_train_base_scaled, y_train_base, 
                                       X_test_base_scaled, y_test_base)
    
    # Simulate trading
    strategies_baseline, metrics_baseline = simulate_all_strategies(
        results_baseline, test_data_base, y_test_base
    )
    
    # ========================================================================
    # ENHANCED FEATURES (with Technical Indicators)
    # ========================================================================
    print("\n" + "="*80)
    print("ENHANCED FEATURES (OHLCV + Technical Indicators)")
    print("="*80)
    
    X_enhanced, y_enhanced, data_enhanced = prepare_data(df, use_technical=True)
    
    # Split data chronologically
    split_idx = int(len(X_enhanced) * 0.8)
    X_train_enh = X_enhanced.iloc[:split_idx]
    X_test_enh = X_enhanced.iloc[split_idx:]
    y_train_enh = y_enhanced.iloc[:split_idx]
    y_test_enh = y_enhanced.iloc[split_idx:]
    test_data_enh = data_enhanced.iloc[split_idx:]
    
    # Normalize
    scaler_enh = StandardScaler()
    X_train_enh_scaled = scaler_enh.fit_transform(X_train_enh)
    X_test_enh_scaled = scaler_enh.transform(X_test_enh)
    
    # Train and evaluate
    models_enhanced = train_models(X_train_enh_scaled, y_train_enh)
    results_enhanced = evaluate_models(models_enhanced, X_train_enh_scaled, y_train_enh,
                                       X_test_enh_scaled, y_test_enh)
    
    # Simulate trading
    strategies_enhanced, metrics_enhanced = simulate_all_strategies(
        results_enhanced, test_data_enh, y_test_enh
    )
    
    # ========================================================================
    # COMPARISON AND DASHBOARD
    # ========================================================================
    print("\n" + "="*80)
    print("STARTING WEB DASHBOARD")
    print("="*80)
    print("\n🚀 Dashboard will be available at: http://0.0.0.0:8080")
    print("   (Also accessible via http://localhost:8080)")
    print("\nPress Ctrl+C to stop the server\n")
    
    app = create_dashboard(
        results_baseline, results_enhanced,
        strategies_baseline, strategies_enhanced,
        metrics_baseline, metrics_enhanced,
        test_data_base, y_test_base
    )
    
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == "__main__":
    main()
