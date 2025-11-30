import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template_string, jsonify
import json
import threading

class BinanceDataFetcher:
    """Fetch OHLCV data from Binance"""
    
    def __init__(self, symbol='BTCUSDT', interval='1d'):
        self.symbol = symbol
        self.interval = interval
        self.base_url = 'https://api.binance.com/api/v3/klines'
    
    def fetch_historical_data(self, start_date='2018-01-01', end_date=None):
        """Fetch historical klines data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        print(f"Fetching {self.symbol} data from {start_date} to {end_date}...")
        
        while current_ts < end_ts:
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'startTime': current_ts,
                'limit': 1000
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_ts = data[-1][0] + 1
                
                print(f"Fetched {len(all_data)} candles...", end='\r')
                
            except Exception as e:
                print(f"\nError fetching data: {e}")
                break
        
        print(f"\nTotal candles fetched: {len(all_data)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]


class FeatureEngine:
    """Engineer features for range prediction"""
    
    def __init__(self, atr_period=14):
        self.atr_period = atr_period
    
    def calculate_atr(self, df):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def engineer_features(self, df):
        """Create all features"""
        feat_df = df.copy()
        
        # Target: log of next day's range
        feat_df['range'] = feat_df['high'] - feat_df['low']
        feat_df['target'] = np.log(feat_df['range'].shift(-1))
        
        # Current range features
        feat_df['range_pct'] = feat_df['range'] / feat_df['close']
        
        # Range moving averages
        for window in [5, 10, 20, 30]:
            feat_df[f'range_ma_{window}'] = feat_df['range'].rolling(window).mean()
            feat_df[f'range_std_{window}'] = feat_df['range'].rolling(window).std()
        
        # Range patterns
        feat_df['range_momentum_5'] = feat_df['range'] / feat_df['range'].rolling(5).mean()
        feat_df['range_momentum_10'] = feat_df['range'] / feat_df['range'].rolling(10).mean()
        
        # Consecutive expanding/contracting ranges
        feat_df['range_change'] = feat_df['range'].pct_change()
        feat_df['range_expanding'] = (feat_df['range_change'] > 0).astype(int)
        feat_df['consec_expand'] = feat_df['range_expanding'].groupby(
            (feat_df['range_expanding'] != feat_df['range_expanding'].shift()).cumsum()
        ).cumsum()
        feat_df['consec_contract'] = (1 - feat_df['range_expanding']).groupby(
            (feat_df['range_expanding'] == feat_df['range_expanding'].shift()).cumsum()
        ).cumsum()
        
        # Range percentiles
        feat_df['range_percentile_20'] = feat_df['range'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        
        # Max/min ranges
        feat_df['max_range_20'] = feat_df['range'].rolling(20).max()
        feat_df['min_range_20'] = feat_df['range'].rolling(20).min()
        feat_df['range_vs_max'] = feat_df['range'] / feat_df['max_range_20']
        
        # ATR features
        feat_df['atr'] = self.calculate_atr(df)
        feat_df['range_to_atr'] = feat_df['range'] / feat_df['atr']
        feat_df['atr_change'] = feat_df['atr'].pct_change()
        feat_df['atr_momentum'] = feat_df['atr'] / feat_df['atr'].rolling(10).mean()
        
        # Volume dynamics
        feat_df['volume_ma_10'] = feat_df['volume'].rolling(10).mean()
        feat_df['volume_ma_30'] = feat_df['volume'].rolling(30).mean()
        feat_df['volume_ratio'] = feat_df['volume'] / feat_df['volume_ma_10']
        feat_df['volume_spike'] = (feat_df['volume'] > feat_df['volume_ma_10'] * 2).astype(int)
        feat_df['volume_trend'] = feat_df['volume'].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        )
        
        # Volume-range relationship
        feat_df['volume_range_corr'] = feat_df['volume'].rolling(20).corr(feat_df['range'])
        
        # Temporal patterns
        feat_df['day_of_week'] = feat_df.index.dayofweek
        feat_df['day_of_month'] = feat_df.index.day
        feat_df['week_of_month'] = (feat_df.index.day - 1) // 7 + 1
        feat_df['month'] = feat_df.index.month
        
        # Market regime: Bull (1) or Bear (0) based on 365 SMA
        feat_df['sma_365'] = feat_df['close'].rolling(365).mean()
        feat_df['market_regime'] = (feat_df['close'] > feat_df['sma_365']).astype(int)
        feat_df['distance_from_sma'] = (feat_df['close'] - feat_df['sma_365']) / feat_df['sma_365']
        
        # Days in current regime
        feat_df['regime_change'] = feat_df['market_regime'] != feat_df['market_regime'].shift(1)
        feat_df['days_in_regime'] = feat_df.groupby(
            feat_df['regime_change'].cumsum()
        ).cumcount() + 1
        
        # Historical range lags
        for lag in [1, 2, 3, 5, 7, 14]:
            feat_df[f'range_lag_{lag}'] = feat_df['range'].shift(lag)
        
        return feat_df
    
    def get_feature_columns(self):
        """Return list of feature column names"""
        features = []
        
        # Range MAs
        features.extend([f'range_ma_{w}' for w in [5, 10, 20, 30]])
        features.extend([f'range_std_{w}' for w in [5, 10, 20, 30]])
        
        # Range patterns
        features.extend([
            'range_pct', 'range_momentum_5', 'range_momentum_10',
            'consec_expand', 'consec_contract', 'range_percentile_20',
            'range_vs_max'
        ])
        
        # ATR features
        features.extend([
            'atr', 'range_to_atr', 'atr_change', 'atr_momentum'
        ])
        
        # Volume dynamics
        features.extend([
            'volume_ratio', 'volume_spike', 'volume_trend', 'volume_range_corr'
        ])
        
        # Temporal
        features.extend([
            'day_of_week', 'day_of_month', 'week_of_month', 'month'
        ])
        
        # Market regime
        features.extend([
            'market_regime', 'distance_from_sma', 'days_in_regime'
        ])
        
        # Lags
        features.extend([f'range_lag_{lag}' for lag in [1, 2, 3, 5, 7, 14]])
        
        return features


class RangePredictor:
    """XGBoost model for range prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.feature_engine = FeatureEngine()
        self.evaluation_data = {}
    
    def train(self, df, test_size=0.15, val_size=0.15):
        """Train the model with proper time-series split"""
        
        # Engineer features
        print("\nEngineering features...")
        df_features = self.feature_engine.engineer_features(df)
        
        # Get feature columns
        self.feature_cols = self.feature_engine.get_feature_columns()
        
        # Drop NaN rows (due to rolling windows and lags)
        df_features = df_features.dropna()
        
        print(f"Total samples after feature engineering: {len(df_features)}")
        print(f"Features: {len(self.feature_cols)}")
        
        # Time-series split
        n = len(df_features)
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size))
        
        train_df = df_features.iloc[:val_start]
        val_df = df_features.iloc[val_start:test_start]
        test_df = df_features.iloc[test_start:]
        
        print(f"\nData split:")
        print(f"Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
        print(f"Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
        
        X_train, y_train = train_df[self.feature_cols], train_df['target']
        X_val, y_val = val_df[self.feature_cols], val_df['target']
        X_test, y_test = test_df[self.feature_cols], test_df['target']
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'learning_rate': 0.01,
            'max_depth': 4,
            'min_child_weight': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'gamma': 0.5,
            'alpha': 0.5,
            'lambda': 3.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        print("\nTraining XGBoost model...")
        
        self.model = xgb.XGBRegressor(**params, n_estimators=1000)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=50
        )
        
        # Evaluate and store results
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Store all predictions for visualization
        self.evaluation_data = {
            'train': {'X': X_train, 'y': y_train, 'df': train_df},
            'val': {'X': X_val, 'y': y_val, 'df': val_df},
            'test': {'X': X_test, 'y': y_test, 'df': test_df}
        }
        
        for name, X, y in [('Train', X_train, y_train), 
                            ('Val', X_val, y_val), 
                            ('Test', X_test, y_test)]:
            pred = self.model.predict(X)
            
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)
            
            # Convert from log space for interpretation
            actual_range = np.exp(y)
            pred_range = np.exp(pred)
            mape = np.mean(np.abs((actual_range - pred_range) / actual_range)) * 100
            
            # Store predictions
            self.evaluation_data[name.lower()]['pred'] = pred
            self.evaluation_data[name.lower()]['pred_range'] = pred_range
            self.evaluation_data[name.lower()]['actual_range'] = actual_range
            
            print(f"\n{name} Set:")
            print(f"  MAE (log space):  {mae:.4f}")
            print(f"  RMSE (log space): {rmse:.4f}")
            print(f"  R² Score:         {r2:.4f}")
            print(f"  MAPE (% range):   {mape:.2f}%")
        
        # Feature importance
        print("\n" + "="*60)
        print("TOP 15 FEATURE IMPORTANCES")
        print("="*60)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.evaluation_data['feature_importance'] = importance_df
        
        print(importance_df.head(15).to_string(index=False))
        
        return test_df, X_test, y_test
    
    def save_model(self, filepath='range_predictor.pkl'):
        """Save model and feature columns"""
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'feature_engine': self.feature_engine,
            'evaluation_data': self.evaluation_data
        }, filepath)
        print(f"\nModel saved to {filepath}")


# Flask Web Application
app = Flask(__name__)

# Global variable to store predictor
predictor = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BTCUSDT Range Prediction - Model Evaluation</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .metric-card .value {
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }
        .metric-card .subvalue {
            font-size: 14px;
            margin-top: 5px;
            opacity: 0.8;
        }
        .chart {
            margin-bottom: 30px;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .insights {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .insights h3 {
            margin-top: 0;
            color: #856404;
        }
        .insights ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .insights li {
            margin: 8px 0;
            color: #856404;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            color: #666;
            transition: all 0.3s;
        }
        .tab.active {
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 BTCUSDT Range Prediction - Model Evaluation</h1>
        <p class="subtitle">XGBoost Model Performance Analysis</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">Overview</button>
            <button class="tab" onclick="showTab('predictions')">Predictions</button>
            <button class="tab" onclick="showTab('features')">Features</button>
            <button class="tab" onclick="showTab('insights')">Insights</button>
        </div>
        
        <div id="overview" class="tab-content active">
            <div class="metrics" id="metrics"></div>
            <div class="chart" id="r2-chart"></div>
        </div>
        
        <div id="predictions" class="tab-content">
            <div class="chart" id="predictions-train"></div>
            <div class="chart" id="predictions-val"></div>
            <div class="chart" id="predictions-test"></div>
            <div class="chart" id="error-dist"></div>
        </div>
        
        <div id="features" class="tab-content">
            <div class="chart" id="feature-importance"></div>
            <div class="chart" id="feature-analysis"></div>
        </div>
        
        <div id="insights" class="tab-content">
            <div class="insights" id="insights-content"></div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        // Fetch data and render
        fetch('/api/evaluation')
            .then(response => response.json())
            .then(data => {
                renderMetrics(data);
                renderPredictions(data);
                renderFeatures(data);
                renderInsights(data);
            });
        
        function renderMetrics(data) {
            const metricsHtml = `
                <div class="metric-card">
                    <h3>Test R² Score</h3>
                    <p class="value">${data.test.r2.toFixed(3)}</p>
                    <p class="subvalue">Variance explained</p>
                </div>
                <div class="metric-card">
                    <h3>Test MAPE</h3>
                    <p class="value">${data.test.mape.toFixed(1)}%</p>
                    <p class="subvalue">Mean error percentage</p>
                </div>
                <div class="metric-card">
                    <h3>Val R² Score</h3>
                    <p class="value">${data.val.r2.toFixed(3)}</p>
                    <p class="subvalue">Validation performance</p>
                </div>
                <div class="metric-card">
                    <h3>Train R² Score</h3>
                    <p class="value">${data.train.r2.toFixed(3)}</p>
                    <p class="subvalue">Training performance</p>
                </div>
            `;
            document.getElementById('metrics').innerHTML = metricsHtml;
            
            // R2 comparison chart
            Plotly.newPlot('r2-chart', [{
                x: ['Train', 'Validation', 'Test'],
                y: [data.train.r2, data.val.r2, data.test.r2],
                type: 'bar',
                marker: {
                    color: ['#4CAF50', '#FFC107', '#F44336']
                }
            }], {
                title: 'R² Score Across Datasets',
                yaxis: { title: 'R² Score', range: [0, 1] }
            });
        }
        
        function renderPredictions(data) {
            ['train', 'val', 'test'].forEach(split => {
                const splitData = data[split];
                Plotly.newPlot(`predictions-${split}`, [
                    {
                        x: splitData.dates,
                        y: splitData.actual,
                        name: 'Actual Range',
                        mode: 'lines',
                        line: { color: '#667eea', width: 2 }
                    },
                    {
                        x: splitData.dates,
                        y: splitData.predicted,
                        name: 'Predicted Range',
                        mode: 'lines',
                        line: { color: '#f50057', width: 2, dash: 'dot' }
                    }
                ], {
                    title: `${split.charAt(0).toUpperCase() + split.slice(1)} Set: Actual vs Predicted Range`,
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Range ($)' }
                });
            });
            
            // Error distribution
            const testErrors = data.test.actual.map((a, i) => 
                ((data.test.predicted[i] - a) / a * 100)
            );
            Plotly.newPlot('error-dist', [{
                x: testErrors,
                type: 'histogram',
                nbinsx: 30,
                marker: { color: '#764ba2' }
            }], {
                title: 'Test Set: Error Distribution',
                xaxis: { title: 'Error (%)' },
                yaxis: { title: 'Frequency' }
            });
        }
        
        function renderFeatures(data) {
            const top15 = data.feature_importance.slice(0, 15);
            Plotly.newPlot('feature-importance', [{
                y: top15.map(f => f.feature),
                x: top15.map(f => f.importance),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#667eea' }
            }], {
                title: 'Top 15 Feature Importances',
                xaxis: { title: 'Importance' },
                margin: { l: 150 }
            });
        }
        
        function renderInsights(data) {
            const html = `
                <h3>🔍 Key Insights from Model Evaluation</h3>
                
                <h4>1. Overfitting Detected</h4>
                <ul>
                    <li><strong>Train R² = ${data.train.r2.toFixed(3)}</strong> (excellent) vs <strong>Test R² = ${data.test.r2.toFixed(3)}</strong> (poor)</li>
                    <li>Large gap indicates the model memorized training patterns that don't generalize</li>
                    <li>The model learned training-specific noise rather than true predictive patterns</li>
                </ul>
                
                <h4>2. Why Test Performance is Low</h4>
                <ul>
                    <li><strong>Regime Change:</strong> Test period (Nov 2024 - Nov 2025) may have different volatility patterns than training data (2018-2023)</li>
                    <li><strong>Market Evolution:</strong> Crypto markets evolved significantly - 2024/2025 patterns differ from 2018-2023</li>
                    <li><strong>Feature Limitations:</strong> Current features may not capture recent market dynamics</li>
                    <li><strong>MAPE = ${data.test.mape.toFixed(1)}%</strong> means predictions are off by ~40% on average</li>
                </ul>
                
                <h4>3. Model Behavior Analysis</h4>
                <ul>
                    <li><strong>Conservative Predictions:</strong> Model tends to predict around mean range values</li>
                    <li><strong>Misses Extremes:</strong> Struggles with high volatility spikes (Nov 20: predicted $2489, actual $6899)</li>
                    <li><strong>Reasonable on Normal Days:</strong> Performs better on typical volatility days (Nov 21: -1.4% error)</li>
                </ul>
                
                <h4>4. Top Predictive Features</h4>
                <ul>
                    <li><strong>range_ma_5 (28.5%):</strong> Recent average range is most predictive</li>
                    <li><strong>atr (21.1%):</strong> ATR captures volatility well</li>
                    <li><strong>range_ma_10 (17.7%):</strong> 10-day average also important</li>
                    <li>Temporal and regime features have minimal impact (<3% each)</li>
                </ul>
                
                <h4>5. Recommendations for Improvement</h4>
                <ul>
                    <li><strong>Stronger Regularization:</strong> Increase alpha/lambda to reduce overfitting</li>
                    <li><strong>Add Recent Features:</strong> Include features from 2024+ data patterns</li>
                    <li><strong>Regime Detection:</strong> Train separate models for high/low volatility regimes</li>
                    <li><strong>Ensemble Approach:</strong> Combine with GARCH or other volatility models</li>
                    <li><strong>Rolling Retraining:</strong> Retrain monthly to adapt to market changes</li>
                    <li><strong>Quantile Prediction:</strong> Predict range distributions instead of point estimates</li>
                </ul>
                
                <h4>6. Practical Use Despite Low R²</h4>
                <ul>
                    <li>Even with R² = ${data.test.r2.toFixed(3)}, predictions provide directional guidance</li>
                    <li>Use predictions as one signal among many, not standalone</li>
                    <li>Best for risk management (setting wide stop-losses) rather than precise trading</li>
                    <li>Consider prediction + 1 standard deviation for conservative estimates</li>
                </ul>
            `;
            document.getElementById('insights-content').innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/evaluation')
def get_evaluation():
    if predictor is None or not predictor.evaluation_data:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    eval_data = predictor.evaluation_data
    
    def prepare_split_data(split_name):
        split_data = eval_data[split_name]
        df = split_data['df']
        actual = split_data['actual_range']
        predicted = split_data['pred_range']
        
        mae = mean_absolute_error(split_data['y'], split_data['pred'])
        rmse = np.sqrt(mean_squared_error(split_data['y'], split_data['pred']))
        r2 = r2_score(split_data['y'], split_data['pred'])
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'actual': actual.tolist(),
            'predicted': predicted.tolist(),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }
    
    response_data = {
        'train': prepare_split_data('train'),
        'val': prepare_split_data('val'),
        'test': prepare_split_data('test'),
        'feature_importance': eval_data['feature_importance'].head(20).to_dict('records')
    }
    
    return jsonify(response_data)


def train_and_start_server():
    """Train model and start Flask server"""
    global predictor
    
    print("="*60)
    print("BTCUSDT RANGE PREDICTION SYSTEM")
    print("="*60)
    
    # Fetch data
    fetcher = BinanceDataFetcher(symbol='BTCUSDT', interval='1d')
    df = fetcher.fetch_historical_data(start_date='2018-01-01')
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Train model
    predictor = RangePredictor()
    test_df, X_test, y_test = predictor.train(df)
    
    # Save model
    predictor.save_model('range_predictor.pkl')
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (Last 10 days of test set)")
    print("="*60)
    
    test_predictions_log = predictor.model.predict(X_test[-10:])
    test_predictions = np.exp(test_predictions_log)
    actual_log = y_test.iloc[-10:].values
    actual = np.exp(actual_log)
    
    results_df = pd.DataFrame({
        'Date': X_test.index[-10:],
        'Predicted Range': test_predictions,
        'Actual Range': actual,
        'Error': test_predictions - actual,
        'Error %': ((test_predictions - actual) / actual * 100)
    })
    
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\n🌐 Starting web server on http://0.0.0.0:8080")
    print("📊 Open your browser to view interactive visualizations")
    print("="*60 + "\n")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == "__main__":
    train_and_start_server()
