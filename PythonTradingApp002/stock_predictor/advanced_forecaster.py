import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Any
from stock_predictor.data_classes import ForecastResults
from stock_predictor.model import BidirectionalForecastModel

class AdvancedForecaster:
    def __init__(self, app_config, data_handler, np_module, tf_module):
        """
        Initialize the advanced forecaster with a single model approach.

        Args:
            app_config: Application configuration.
            data_handler: Data handler instance.
            np_module: Reference to NumPy.
            tf_module: Reference to TensorFlow.
        """
        self.app_config = app_config
        self.data_handler = data_handler
        self.np = np_module
        self.tf = tf_module
        self.forecast_results = ForecastResults(
            sequence_length=app_config.learning.timestamp,
            forecast_horizon=app_config.prediction.predict_days
        )
        # Validate that initial_data_period matches timestamp when set_initial_data is True
        if app_config.prediction.set_initial_data and app_config.prediction.initial_data_period != app_config.learning.timestamp:
            raise ValueError(
                f"When set_initial_data=True, initial_data_period ({app_config.prediction.initial_data_period}) "
                f"must equal app_config.learning.timestamp ({app_config.learning.timestamp})"
            )

    def generate_monte_carlo_path(self, base_prediction: np.ndarray, drift_mult: float = 1.0, 
                                 vol_mult: float = 1.0, last_known_value: float = None, horizon: int = 30) -> np.ndarray:
        """
        Generate a Monte Carlo simulation path with scenario-specific parameters.

        Args:
            base_prediction: Base prediction array to derive drift and volatility.
            drift_mult: Multiplier for drift (e.g., 1.5 for bull scenario).
            vol_mult: Multiplier for volatility (e.g., 2.0 for volatile scenario).
            last_known_value: Starting value for the path (defaults to first base prediction value).
            horizon: Number of days to simulate.

        Returns:
            np.ndarray: Simulated price path.
        """
        if last_known_value is None:
            last_known_value = base_prediction[0]
        returns = self.np.diff(base_prediction) / base_prediction[:-1]
        base_drift = self.np.mean(returns) if len(returns) > 0 else 0.0
        base_vol = self.np.std(returns) if len(returns) > 0 else 0.01  # Avoid zero volatility
        drift = base_drift * drift_mult
        volatility = base_vol * vol_mult
        z = self.np.random.standard_normal(horizon - 1)
        increments = self.np.exp((drift - 0.5 * volatility ** 2) + volatility * z)
        path = last_known_value * self.np.concatenate(([1.0], self.np.cumprod(increments)))
        return path

    def calculate_feature_importance(self, train_data: np.ndarray, features: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance using a Random Forest model.

        Args:
            train_data: Training data array.
            features: List of feature names.

        Returns:
            Dict[str, float]: Feature importance scores.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        try:
            close_idx = features.index('close')
            X = train_data[:, [i for i in range(len(features)) if i != close_idx]]
            y = train_data[:, close_idx]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_scaled, y)
            importance = rf_model.feature_importances_
            return {feat: float(imp) for feat, imp in zip([f for f in features if f != 'close'], importance)}
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return {feat: 0.0 for feat in features if feat != 'close'}

    def create_base_model(self, num_features: int) -> tf.keras.Model:
        """
        Create a Bidirectional LSTM model for forecasting.

        Args:
            num_features: Number of input features.

        Returns:
            tf.keras.Model: Compiled Bidirectional LSTM model.
        """
        model = BidirectionalForecastModel(
            num_layers=self.app_config.learning.num_layers,
            size_layer=min(64, max(32, num_features * self.app_config.learning.timestamp // 8)),
            output_size=1,
            dropout_rate=self.app_config.learning.dropout_rate,
            l2_reg=self.app_config.learning.l2_reg
        )
        model.compile(
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate),
            loss='mae'
        )
        return model

    def create_dataset(self, data: np.ndarray, sequence_length: int, batch_size: int) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from a NumPy array.

        Args:
            data: Input data array.
            sequence_length: Length of input sequences.
            batch_size: Batch size for training.

        Returns:
            tf.data.Dataset: Prepared dataset.
        """
        close_idx = self.data_handler.stock_data.feature_list.index('close')
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, close_idx])
        X = self.np.array(X)
        y = self.np.array(y)
        dataset = self.tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size).prefetch(self.tf.data.AUTOTUNE)
        return dataset

    def train_base_model(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str]) -> Dict[str, Any]:
        """
        Train the base model on training data.

        Args:
            model: Model to train.
            train_data: Training data array.
            features: List of feature names.

        Returns:
            Dict[str, Any]: Training diagnostics.
        """
        dataset = self.create_dataset(train_data, self.app_config.learning.timestamp, self.app_config.learning.batch_size)
        val_data = train_data[int(len(train_data) * self.app_config.tf_config.split_ratio):]
        val_dataset = None
        if len(val_data) > self.app_config.learning.timestamp:
            val_dataset = self.create_dataset(val_data, self.app_config.learning.timestamp, self.app_config.learning.batch_size)
        
        history = model.fit(
            dataset,
            epochs=self.app_config.learning.epoch,
            validation_data=val_dataset,
            verbose=0
        )
        diagnostics = {
            'final_loss': float(history.history['loss'][-1]),
            'epoch_history': [
                {
                    'avg_train_loss': float(history.history['loss'][i]),
                    'avg_val_loss': float(history.history['val_loss'][i]) if 'val_loss' in history.history else None
                }
                for i in range(len(history.history['loss']))
            ]
        }
        return diagnostics

    def predict_base(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Generate base predictions using the trained model.

        Args:
            model: Trained BidirectionalForecastModel.
            train_data: Scaled training data array with shape (num_samples, num_features).
            features: List of feature names.

        Returns:
            np.ndarray: Array of scaled predictions with length equal to predict_days.
        """
        if 'close' not in features:
            raise ValueError("'close' feature missing in features list")
        
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None
        horizon = self.app_config.prediction.predict_days
        sequence_length = self.app_config.learning.timestamp

        # Validate input data
        if len(train_data) < sequence_length:
            raise ValueError(f"Training data has {len(train_data)} rows, but must have at least {sequence_length} rows for sequence length")

        # Use the last sequence_length days as the starting point
        # The constructor ensures initial_data_period == sequence_length when set_initial_data=True
        working_sequence = train_data[-sequence_length:].copy()

        # Iteratively predict for the forecast horizon
        future_predictions = []
        for i in range(horizon):
            seq_reshaped = working_sequence.reshape(1, sequence_length, len(features))
            pred_close = model.predict(seq_reshaped, verbose=0)[-1, 0]
            future_predictions.append(pred_close)
            # Create a new row with the predicted close value
            new_row = working_sequence[-1].copy()
            new_row[close_idx] = pred_close
            if self.app_config.prediction.use_previous_close and prev_close_idx is not None:
                new_row[prev_close_idx] = pred_close if i == 0 else future_predictions[i - 1]
            working_sequence = self.np.vstack([working_sequence[1:], new_row])
        
        return self.np.array(future_predictions)

    def run_forecast(self) -> tuple:
        """
        Run an advanced forecast using a single Bidirectional LSTM model with optional Monte Carlo simulations or rolling window validation.

        Returns:
            Tuple: (ensemble_predictions, diagnostics_list, feature_importance)
                - ensemble_predictions: np.ndarray of shape (num_simulations, predict_days) containing scaled predictions.
                - diagnostics_list: List of diagnostics dictionaries from training or rolling window validation.
                - feature_importance: Dictionary of feature importance scores.
        """
        # Validate input data
        if not hasattr(self.data_handler.stock_data, 'get_training_array') or not self.data_handler.stock_data.get_training_array().size:
            raise ValueError("Training data is empty or not available in DataHandler")
        
        train_data = self.data_handler.stock_data.get_training_array()
        features = self.data_handler.stock_data.feature_list
        if not features:
            raise ValueError("Feature list is empty in StockData")
        
        horizon = self.app_config.prediction.predict_days
        if horizon <= 0:
            raise ValueError(f"Prediction horizon must be positive, got {horizon}")
        
        simulation_size = self.app_config.prediction_advanced.num_monte_carlo_simulations
        if simulation_size <= 0:
            raise ValueError(f"Number of Monte Carlo simulations must be positive, got {simulation_size}")

        # Set random seed for reproducibility
        if hasattr(self.app_config.prediction_advanced, 'monte_carlo_seed'):
            self.np.random.seed(self.app_config.prediction_advanced.monte_carlo_seed)

        ensemble_predictions = []
        diagnostics_list = []

        if self.app_config.prediction_advanced.enable_rolling_window:
            from stock_predictor.rolling_window_forecaster import RollingWindowForecaster
            forecaster = RollingWindowForecaster(self.app_config, self.data_handler, np=self.np, tf=self.tf)
            
            for layers in self.app_config.prediction_advanced.lstm_layers:
                for units in self.app_config.prediction_advanced.lstm_units:
                    for dropout in self.app_config.prediction_advanced.dropout_rates:
                        self.app_config.learning.num_layers = layers
                        self.app_config.learning.size_layer = units
                        self.app_config.learning.dropout_rate = dropout
                        # Note: Ensure RollingWindowForecaster.run_rolling_validation respects set_initial_data
                        results, diagnostics, _ = forecaster.run_rolling_validation()
                        if results is None or len(results) == 0:
                            raise ValueError("Rolling window validation returned no predictions")
                        ensemble_predictions.extend(results)
                        diagnostics_list.append(diagnostics)
        else:
            model = self.create_base_model(len(features))
            diagnostics = self.train_base_model(model, train_data, features)
            base_prediction_scaled = self.predict_base(model, train_data, features)
            if len(base_prediction_scaled) != horizon:
                raise ValueError(f"Base prediction length {len(base_prediction_scaled)} does not match horizon {horizon}")
            
            base_prediction = self.data_handler.stock_data.unscale_close_price(base_prediction_scaled)
            
            if self.app_config.prediction_advanced.enable_monte_carlo:
                last_known_value = self.data_handler.stock_data.csv_data_train_period['close'].values[-1]
                scenarios = self.app_config.prediction_advanced.monte_carlo_scenarios
                sims_per_scenario = max(1, simulation_size // len(scenarios))
                for scenario in scenarios:
                    drift_mult, vol_mult = self.app_config.prediction_advanced.scenario_parameters.get(scenario, [1.0, 1.0])
                    for _ in range(sims_per_scenario):
                        sim_path = self.generate_monte_carlo_path(
                            base_prediction, drift_mult, vol_mult, last_known_value, horizon)
                        sim_path_scaled = self.data_handler.stock_data.scale_close_price(sim_path)
                        ensemble_predictions.append(sim_path_scaled)
                total_sims = len(ensemble_predictions)
                if total_sims < simulation_size:
                    for _ in range(simulation_size - total_sims):
                        sim_path = self.generate_monte_carlo_path(
                            base_prediction, 1.0, 1.0, last_known_value, horizon)
                        sim_path_scaled = self.data_handler.stock_data.scale_close_price(sim_path)
                        ensemble_predictions.append(sim_path_scaled)
            else:
                ensemble_predictions.append(base_prediction_scaled)
            diagnostics_list.append(diagnostics)

        ensemble_predictions = self.np.array(ensemble_predictions)
        if ensemble_predictions.size == 0:
            raise ValueError("No predictions generated")

        self.forecast_results.simulation_predictions = ensemble_predictions
        self.forecast_results.ensemble_mean = self.np.mean(ensemble_predictions, axis=0)
        self.forecast_results.ensemble_std = self.np.std(ensemble_predictions, axis=0)
        self.forecast_results.evaluate_predictions(self.data_handler.stock_data, 
                                                   validation_metrics=self.app_config.prediction_advanced.validation_metrics)
        self.forecast_results.calculate_confidence_intervals()

        feature_importance = self.calculate_feature_importance(train_data, features)
        self.forecast_results.feature_importance_scores = feature_importance

        return (ensemble_predictions, diagnostics_list, feature_importance)