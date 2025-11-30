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
            'learning_rate': 0.03,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'gamma': 0.1,
            'alpha': 0.1,
            'lambda': 1.0,
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
        
        # Evaluate
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
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
        
        print(importance_df.head(15).to_string(index=False))
        
        return test_df, X_test, y_test
    
    def predict(self, df):
        """Make predictions on new data"""
        df_features = self.feature_engine.engineer_features(df)
        df_features = df_features.dropna()
        
        X = df_features[self.feature_cols]
        predictions_log = self.model.predict(X)
        predictions = np.exp(predictions_log)
        
        return predictions, df_features.index
    
    def save_model(self, filepath='range_predictor.pkl'):
        """Save model and feature columns"""
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'feature_engine': self.feature_engine
        }, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='range_predictor.pkl'):
        """Load model and feature columns"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.feature_engine = data['feature_engine']
        print(f"Model loaded from {filepath}")


def main():
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
    
    # Example prediction on test set
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


if __name__ == "__main__":
    main()
