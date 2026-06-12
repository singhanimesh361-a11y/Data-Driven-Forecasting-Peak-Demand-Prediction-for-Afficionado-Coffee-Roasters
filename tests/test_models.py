"""Unit tests for ADIP forecasting models."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.baseline import NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster
from src.models.sarima_model import SARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgb_model import XGBoostForecaster
from src.models.ensemble import EnsembleForecaster, EnsembleQualityError

@pytest.fixture
def sample_df():
    dates = pd.date_range("2025-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "store_id": [3] * 100,
        "total_revenue": np.random.normal(1000, 100, 100)
    })
    return df

@pytest.fixture
def feature_df(sample_df):
    df = sample_df.copy()
    df['lag_1d'] = df['total_revenue'].shift(1).fillna(1000)
    df['lag_7d'] = df['total_revenue'].shift(7).fillna(1000)
    df['day_of_week'] = df['date'].dt.dayofweek
    return df

# --- SARIMA Tests ---

@patch('src.models.sarima_model.pm')
def test_sarima_fit(mock_pm, sample_df):
    model = SARIMAForecaster()
    mock_model = MagicMock()
    mock_pm.auto_arima.return_value = mock_model
    
    model.fit(sample_df)
    assert 3 in model.models
    mock_pm.auto_arima.assert_called_once()

@patch('src.models.sarima_model.pm')
def test_sarima_predict(mock_pm, sample_df):
    model = SARIMAForecaster()
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array([1100, 1200]), np.array([[1000, 1200], [1100, 1300]]))
    mock_pm.auto_arima.return_value = mock_model
    
    model.fit(sample_df)
    preds = model.predict(horizon=2, store_id=3)
    
    assert len(preds) == 2
    assert 'predicted_value' in preds.columns
    assert 'lower_95' in preds.columns
    assert (preds['predicted_value'] >= 0).all()

def test_sarima_annual_seasonality_reliable():
    model = SARIMAForecaster()
    assert not model.is_annual_seasonality_reliable

# --- Prophet Tests ---

@patch('prophet.Prophet')
def test_prophet_fit(mock_prophet, sample_df):
    model = ProphetForecaster()
    mock_instance = MagicMock()
    mock_prophet.return_value = mock_instance
    
    mock_instance.predict.return_value = pd.DataFrame({'yhat': [1000] * 100})
    
    model.fit(sample_df)
    assert 3 in model.models
    mock_instance.fit.assert_called_once()

@patch('prophet.Prophet')
def test_prophet_predict(mock_prophet, sample_df):
    model = ProphetForecaster()
    mock_instance = MagicMock()
    mock_prophet.return_value = mock_instance
    
    future_df = pd.DataFrame({'ds': pd.date_range('2025-01-01', periods=2)})
    mock_instance.make_future_dataframe.return_value = future_df
    
    pred_df = pd.DataFrame({
        'ds': future_df['ds'],
        'yhat': [1100] * len(future_df),
        'yhat_lower': [1000] * len(future_df),
        'yhat_upper': [1200] * len(future_df)
    })
    mock_instance.predict.return_value = pd.DataFrame({'yhat': [1100] * 100})
    model.fit(sample_df)
    
    mock_instance.predict.return_value = pred_df
    preds = model.predict(horizon=2, store_id=3)
    
    assert len(preds) == 2
    assert (preds['predicted_value'] >= 0).all()
    assert (preds['lower_80'] <= preds['predicted_value']).all()

# --- XGBoost Tests ---

def test_xgb_fit(feature_df):
    model = XGBoostForecaster(n_optuna_trials=1) # Fast test
    # Disable optuna logging to stdout
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    model.fit(feature_df, feature_cols=['lag_1d', 'lag_7d', 'day_of_week'])
    assert 3 in model.models
    assert not model.get_feature_importance(store_id=3).empty

def test_xgb_predict(feature_df):
    model = XGBoostForecaster(n_optuna_trials=1)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    model.fit(feature_df, feature_cols=['lag_1d', 'lag_7d', 'day_of_week'])
    
    test_X = feature_df.tail(5)
    preds = model.predict(test_X, store_id=3)
    
    assert len(preds) == 5
    assert 'predicted_value' in preds.columns
    assert (preds['predicted_value'] >= 0).all()

def test_xgb_feature_importance(feature_df):
    model = XGBoostForecaster(n_optuna_trials=1)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    model.fit(feature_df, feature_cols=['lag_1d', 'lag_7d', 'day_of_week'])
    fi = model.get_feature_importance(store_id=3)
    
    assert len(fi) == 3
    assert 'feature_name' in fi.columns

# --- Ensemble Tests ---

def test_ensemble_quality_gate_pass(feature_df):
    # Mock constituents
    prophet = MagicMock()
    xgb = MagicMock()
    
    # Mock predict returning something reasonable
    prophet_preds = pd.DataFrame({'predicted_value': feature_df['total_revenue'] * 0.9, 'lower_80': 0, 'upper_80': 0, 'lower_95': 0, 'upper_95': 0})
    xgb_preds = pd.DataFrame({'predicted_value': feature_df['total_revenue'] * 1.1})
    
    prophet.predict.return_value = prophet_preds
    xgb.predict.return_value = xgb_preds
    
    model = EnsembleForecaster(prophet=prophet, xgb=xgb, quality_threshold_mape=25.0)
    model.fit(feature_df, feature_cols=['lag_1d'])
    
    assert 3 in model.meta_learners
    assert len(model.meta_learners[3].coef_) == 2

def test_ensemble_quality_gate_fail_both(feature_df):
    prophet = MagicMock()
    xgb = MagicMock()
    
    # Mock predict returning terrible predictions (MAPE > 25)
    prophet_preds = pd.DataFrame({'predicted_value': feature_df['total_revenue'] * 0.1, 'lower_80': 0, 'upper_80': 0, 'lower_95': 0, 'upper_95': 0})
    xgb_preds = pd.DataFrame({'predicted_value': feature_df['total_revenue'] * 2.0})
    
    prophet.predict.return_value = prophet_preds
    xgb.predict.return_value = xgb_preds
    
    model = EnsembleForecaster(prophet=prophet, xgb=xgb, quality_threshold_mape=25.0)
    
    with pytest.raises(EnsembleQualityError):
        model.fit(feature_df, feature_cols=['lag_1d'])

def test_ensemble_weights(feature_df):
    prophet = MagicMock()
    xgb = MagicMock()
    prophet.predict.return_value = pd.DataFrame({'predicted_value': feature_df['total_revenue'], 'lower_80': 0, 'upper_80': 0, 'lower_95': 0, 'upper_95': 0})
    xgb.predict.return_value = pd.DataFrame({'predicted_value': feature_df['total_revenue'] * 1.05})
    
    model = EnsembleForecaster(prophet=prophet, xgb=xgb, quality_threshold_mape=25.0)
    model.fit(feature_df, feature_cols=['lag_1d'])
    
    weights = model.get_weights()
    assert 3 in weights
    assert 'prophet' in weights[3]
    assert 'xgb' in weights[3]
