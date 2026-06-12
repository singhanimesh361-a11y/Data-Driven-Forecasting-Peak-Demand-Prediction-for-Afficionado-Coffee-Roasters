"""PyTorch LSTM Forecaster for ADIP."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List
import optuna
from sklearn.preprocessing import StandardScaler

from src.models.prophet_model import BaseForecaster

logger = logging.getLogger(__name__)

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMForecaster(BaseForecaster):
    def __init__(self, n_optuna_trials: int = 10, epochs: int = 50, batch_size: int = 32, seq_length: int = 14):
        self.n_optuna_trials = n_optuna_trials
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.models: Dict[str, nn.Module] = {}
        self.scalers_X: Dict[str, StandardScaler] = {}
        self.scalers_y: Dict[str, StandardScaler] = {}
        self.feature_cols: List[str] = []
        self.best_params: Dict[str, Dict[str, Any]] = {}

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        Xs, ys = [], []
        for i in range(len(X) - self.seq_length):
            Xs.append(X[i:(i + self.seq_length)])
            ys.append(y[i + self.seq_length])
        return np.array(Xs), np.array(ys)

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "total_revenue", **kwargs: Any) -> "LSTMForecaster":
        self.feature_cols = feature_cols
        store_ids = df['store_id'].unique()
        
        for sid in store_ids:
            store_df = df[df['store_id'] == sid].sort_values('date').reset_index(drop=True)
            store_df = store_df.dropna(subset=self.feature_cols + [target_col])
            
            if len(store_df) < self.seq_length + 2:
                logger.warning(f"Not enough data for store {sid} after dropping NaNs.")
                continue
                
            # Scale data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(store_df[self.feature_cols].values)
            y_scaled = scaler_y.fit_transform(store_df[[target_col]].values).flatten()
            
            self.scalers_X[sid] = scaler_X
            self.scalers_y[sid] = scaler_y
            
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
            if len(X_seq) == 0:
                logger.warning(f"Not enough data for store {sid} to create sequences of length {self.seq_length}.")
                continue
                
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Optuna objective
            def objective(trial):
                hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
                num_layers = trial.suggest_int("num_layers", 1, 3)
                dropout = trial.suggest_float("dropout", 0.1, 0.5)
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                
                # Split for validation
                val_size = int(0.2 * len(dataset))
                train_size = len(dataset) - val_size
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
                
                model = LSTMNet(input_size=len(self.feature_cols), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                for epoch in range(10): # Quick train for optuna
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        out = model(batch_X)
                        loss = criterion(out, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        out = model(batch_X)
                        val_loss += criterion(out, batch_y).item()
                return val_loss / len(val_loader)
            
            if self.n_optuna_trials > 0 and len(dataset) >= 14:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.ERROR)
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=self.n_optuna_trials)
                best_params = study.best_params
            else:
                best_params = {"hidden_size": 64, "num_layers": 2, "dropout": 0.2, "lr": 0.001}
                
            self.best_params[sid] = best_params
            logger.info(f"[{sid}] Best LSTM params: {best_params}")
            
            # Final train
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            model = LSTMNet(input_size=len(self.feature_cols), hidden_size=best_params["hidden_size"], 
                           num_layers=best_params["num_layers"], dropout=best_params["dropout"])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
            
            model.train()
            for epoch in range(self.epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    out = model(batch_X)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
            
            self.models[sid] = model
            
        return self

    def predict(self, X: pd.DataFrame, store_id: str, **kwargs: Any) -> pd.DataFrame:
        if store_id not in self.models:
            raise KeyError(f"No fitted model for store_id='{store_id}'.")
            
        model = self.models[store_id]
        scaler_X = self.scalers_X[store_id]
        scaler_y = self.scalers_y[store_id]
        
        # We need sequences, so we assume X has the necessary history prepended
        X_clean = X[self.feature_cols].fillna(0)
        X_scaled = scaler_X.transform(X_clean.values)
        
        # Pad with zeros if X is shorter than seq_length
        if len(X_scaled) < self.seq_length:
            padding = np.zeros((self.seq_length - len(X_scaled), len(self.feature_cols)))
            X_scaled = np.vstack([padding, X_scaled])
            
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # If we just want to predict the exact rows in X, we need to create sliding windows ending at each row in X
        # For simplicity, let's just create sequences by padding the start
        padded_X = np.vstack([np.zeros((self.seq_length - 1, len(self.feature_cols))), X_scaled])
        Xs = []
        for i in range(len(X_scaled)):
            Xs.append(padded_X[i:(i + self.seq_length)])
        
        X_tensor = torch.tensor(np.array(Xs), dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            out_scaled = model(X_tensor).numpy()
            
        out_unscaled = scaler_y.inverse_transform(out_scaled).flatten()
        out_unscaled = np.maximum(out_unscaled, 0.0)
        
        # Simple confidence intervals (LSTM doesn't natively output them without MC Dropout)
        lower_80 = out_unscaled * 0.9
        upper_80 = out_unscaled * 1.1
        lower_95 = out_unscaled * 0.8
        upper_95 = out_unscaled * 1.2
        
        if "date" in X.columns:
            dates = X["date"]
        elif "forecast_date" in X.columns:
            dates = X["forecast_date"]
        else:
            dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=len(out_unscaled))
            
        return pd.DataFrame({
            "forecast_date": dates,
            "store_id": store_id,
            "predicted_value": out_unscaled,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "model_name": "LSTM"
        })
