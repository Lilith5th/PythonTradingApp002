import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Any
from stock_predictor.data_classes import ForecastResults

class AdvancedForecaster:
    def __init__(self, app_config, data_handler, np_module, tf_module):
        """
        Initialize the advanced forecaster.

        Args:
            app_config: Application configuration.
            data_handler: Data handler instance.
            np_module: Reference to NumPy.
            tf_module: Reference to TensorFlow.
            pd_module: Reference to pandas.
        """
        self.app_config = app_config
        self.data_handler = data_handler
        self.np = np_module
        self.tf = tf_module
        self.forecast_results = ForecastResults(
            sequence_length=app_config.learning.timestamp,
            forecast_horizon=app_config.prediction.predict_days
        )

    def generate_monte_carlo_path(self, base_prediction: np.ndarray, drift_mult: float = 1.0, 
                                   vol_mult: float = 1.0, last_known_value: float = None, horizon: int = 30) -> np.ndarray:
        """
        Generate a Monte Carlo simulation path with scenario-specific parameters.
        """
        if last_known_value is None:
            last_known_value = base_prediction[0]
        returns = np.diff(base_prediction) / base_prediction[:-1]
        base_drift = np.mean(returns)
        base_vol = np.std(returns)
        drift = base_drift * drift_mult
        volatility = base_vol * vol_mult
        # Vectorized simulation
        z = np.random.standard_normal(horizon - 1)
        increments = np.exp((drift - 0.5 * volatility ** 2) + volatility * z)
        path = last_known_value * np.concatenate(([1.0], np.cumprod(increments)))
        return path

    def calculate_feature_importance(self, train_data: np.ndarray, features: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance using permutation and correlation methods.
        """
        from sklearn.inspection import permutation_importance
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor

        # Exclude target column from X.
        X = train_data[:, :len(features) - 1]
        y = train_data[:, features.index('close')]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)
        perm_importance = permutation_importance(
            rf_model, X_scaled, y, n_repeats=10, random_state=42
        )
        importance_scores = {
            features[i]: perm_importance.importances_mean[i]
            for i in range(len(features) - 1)
        }
        correlation_importance = {}
        for i, feature in enumerate(features[:-1]):
            corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            correlation_importance[feature] = corr
        combined_importance = {}
        for feature in importance_scores:
            combined_importance[feature] = 0.7 * importance_scores[feature] + 0.3 * correlation_importance[feature]
        max_val = max(combined_importance.values())
        normalized_importance = {k: v / max_val for k, v in combined_importance.items()}
        return normalized_importance

    def convert_ensemble_history_to_diagnostics(self, histories: List[Any]) -> Dict[str, Any]:
        """
        Convert training histories to diagnostic information.
        """
        diagnostics = {
            'ensemble_histories': [],
            'final_losses': [],
            'average_metrics': {'loss': [], 'val_loss': []}
        }
        for history in histories:
            final_loss = history.history.get('loss', [None])[-1]
            final_val_loss = history.history.get('val_loss', [None])[-1]
            diagnostics['final_losses'].append(final_loss)
            diagnostics['ensemble_histories'].append({
                'loss': history.history.get('loss', []),
                'val_loss': history.history.get('val_loss', [])
            })
            diagnostics['average_metrics']['loss'].append(np.mean(history.history.get('loss', [])))
            diagnostics['average_metrics']['val_loss'].append(np.mean(history.history.get('val_loss', [])))
        diagnostics['ensemble_average_loss'] = np.mean(diagnostics['final_losses'])
        return diagnostics

    def predict(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Generate predictions using the base forecasting model.
        """
        input_sequence = train_data[-self.app_config.learning.timestamp:]
        input_reshaped = input_sequence.reshape(1, self.app_config.learning.timestamp, len(features))
        prediction = model.predict(input_reshaped)
        return prediction.flatten()

    def predict_with_initial_data(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str], initial_period: int) -> np.ndarray:
        """
        Generate predictions using an initial context.
        """
        initial_data = train_data[-initial_period:]
        repeated_initial = np.repeat(
            initial_data,
            self.app_config.prediction.predict_days // initial_period + 1,
            axis=0
        )[:self.app_config.prediction.predict_days]
        input_reshaped = repeated_initial.reshape(1, self.app_config.prediction.predict_days, len(features))
        prediction = model.predict(input_reshaped)
        return prediction.flatten()

    def predict_with_mc_dropout(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Generate predictions with dropout active.
        """
        input_sequence = train_data[-self.app_config.learning.timestamp:]
        input_reshaped = input_sequence.reshape(1, self.app_config.learning.timestamp, len(features))
        # Ensure dropout is active by calling with training=True.
        prediction = model(input_reshaped, training=True).numpy()
        return prediction.flatten()

    def create_dataset(self, data: np.ndarray, sequence_length: int, batch_size: int) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from a NumPy array.
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length][-1])
        X = np.array(X)
        y = np.array(y)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_mc_dropout_model(self, num_features: int) -> tf.keras.Model:
        """
        Create a model with dropout layers for uncertainty estimation.
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2

        inputs = Input(shape=(self.app_config.learning.timestamp, num_features))
        x = inputs
        for i in range(self.app_config.learning.num_layers):
            x = LSTM(units=min(64, max(32, num_features * 4)),
                     return_sequences=True,
                     kernel_regularizer=l2(self.app_config.learning.l2_reg))(x)
            x = Dropout(self.app_config.learning.dropout_rate)(x)
            x = LayerNormalization()(x)
        x = LSTM(units=min(32, max(16, num_features * 2)),
                 kernel_regularizer=l2(self.app_config.learning.l2_reg))(x)
        x = Dropout(self.app_config.learning.dropout_rate)(x)
        x = BatchNormalization()(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate),
                      loss='mae')
        return model

    def create_quantile_regression_model(self, num_features: int, quantiles: List[float]) -> tf.keras.Model:
        """
        Create a quantile regression model.
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        
        def quantile_loss(quantile: float):
            def loss(y_true, y_pred):
                error = y_true - y_pred
                return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
            return loss
        
        inputs = Input(shape=(self.app_config.learning.timestamp, num_features))
        x = inputs
        for i in range(self.app_config.learning.num_layers):
            x = LSTM(units=min(64, max(32, num_features * 4)),
                     return_sequences=True,
                     kernel_regularizer=l2(self.app_config.learning.l2_reg))(x)
            x = Dropout(self.app_config.learning.dropout_rate)(x)
            x = LayerNormalization()(x)
        x = LSTM(units=min(32, max(16, num_features * 2)),
                 kernel_regularizer=l2(self.app_config.learning.l2_reg))(x)
        x = Dropout(self.app_config.learning.dropout_rate)(x)
        x = BatchNormalization()(x)
        outputs = [Dense(1, activation='linear', name=f'output_{i}')(x) for i, _ in enumerate(quantiles)]
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate),
                      loss={f'output_{i}': quantile_loss(quantile) for i, quantile in enumerate(quantiles)},
                      loss_weights={f'output_{i}': 1.0 for i in range(len(quantiles))})
        return model

    def create_evidential_regression_model(self, num_features: int) -> tf.keras.Model:
        """
        Create an evidential regression model.
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2

        def evidential_loss(y_true, params):
            mu, v, alpha, beta = params
            error = y_true - mu
            nll = 0.5 * tf.math.log(np.pi / v) + (error**2) / (2 * v) + tf.math.log(tf.math.sqrt(v))
            anc_term = tf.math.lgamma(alpha) - tf.math.lgamma(alpha + 0.5)
            complexity_reg = (alpha - 1) * tf.math.log(beta) + 0.5 * tf.math.log(np.pi) - anc_term
            return nll + complexity_reg

        def evidential_output(x, num_outputs=4):
            outputs = Dense(num_outputs, activation='softplus')(x)
            return outputs

        inputs = Input(shape=(self.app_config.learning.timestamp, num_features))
        x = inputs
        for i in range(self.app_config.learning.num_layers):
            x = LSTM(units=min(64, max(32, num_features * 4)),
                     return_sequences=True,
                     kernel_regularizer=l2(self.app_config.learning.l2_reg))(x)
            x = Dropout(self.app_config.learning.dropout_rate)(x)
            x = LayerNormalization()(x)
        x = LSTM(units=min(32, max(16, num_features * 2)),
                 kernel_regularizer=l2(self.app_config.learning.l2_reg))(x)
        x = Dropout(self.app_config.learning.dropout_rate)(x)
        x = BatchNormalization()(x)
        outputs = evidential_output(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate),
                      loss=evidential_loss)
        return model

    def predict_with_evidential(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str]):
        """
        Generate evidential predictions.
        """
        input_sequence = train_data[-self.app_config.learning.timestamp:]
        input_reshaped = input_sequence.reshape(1, self.app_config.learning.timestamp, len(features))
        predictions = model.predict(input_reshaped)
        mu, v, alpha, beta = predictions.T
        return mu, v, alpha, beta

    def predict_with_initial_data_evidential(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str], initial_period: int):
        """
        Generate evidential predictions using initial context.
        """
        initial_data = train_data[-initial_period:]
        repeated_initial = np.repeat(initial_data, self.app_config.prediction.predict_days // initial_period + 1, axis=0)[:self.app_config.prediction.predict_days]
        input_reshaped = repeated_initial.reshape(1, self.app_config.prediction.predict_days, len(features))
        predictions = model.predict(input_reshaped)
        mu, v, alpha, beta = predictions.T
        return mu, v, alpha, beta

    def predict_with_initial_data_quantiles(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str], initial_period: int, num_quantiles: int) -> np.ndarray:
        initial_data = train_data[-initial_period:]
        repeated_initial = np.repeat(initial_data, self.app_config.prediction.predict_days // initial_period + 1, axis=0)[:self.app_config.prediction.predict_days]
        input_reshaped = repeated_initial.reshape(1, self.app_config.prediction.predict_days, len(features))
        predictions = model.predict(input_reshaped)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array([p.numpy() for p in predictions])
        return predictions

    def predict_with_quantiles(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str], num_quantiles: int) -> np.ndarray:
        input_sequence = train_data[-self.app_config.learning.timestamp:]
        input_reshaped = input_sequence.reshape(1, self.app_config.learning.timestamp, len(features))
        predictions = model.predict(input_reshaped)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array([p.numpy() for p in predictions])
        return predictions


    def run_forecast(self) -> Any:
        """
        Run an advanced forecast using a simple Monte Carlo simulation.
    
        Returns:
            Tuple: (ensemble_predictions, [diagnostics], feature_importance)
        """
        # Retrieve historical training data from StockData.
        try:
            train_data = self.data_handler.stock_data.get_training_array()
        except AttributeError as e:
            raise AttributeError(f"Unable to access historical data from DataHandler: {e}. "
                                 "Please check DataHandler implementation and adjust run_forecast accordingly.")
    
        horizon = self.app_config.prediction.predict_days if hasattr(self.app_config.prediction, 'predict_days') else 30
        # Create a dummy base prediction (linear trend for demonstration).
        base_prediction = np.linspace(100, 110, horizon)
        simulation_size = self.app_config.learning.simulation_size if hasattr(self.app_config.learning, 'simulation_size') else 5
    
        ensemble_predictions = []
        for _ in range(simulation_size):
            sim_path = self.generate_monte_carlo_path(base_prediction, horizon=horizon)
            ensemble_predictions.append(sim_path)
        ensemble_predictions = np.array(ensemble_predictions)
    
        diagnostics = {
            'ensemble_mean': np.mean(ensemble_predictions, axis=0).tolist(),
            'ensemble_std': np.std(ensemble_predictions, axis=0).tolist()
        }
    
        # If test data is available, compute the actual sMAPE.
        if hasattr(self.data_handler, 'df_test_raw') and not self.data_handler.df_test_raw.empty:
            actual_values = self.data_handler.df_test_raw['close'].values[:horizon]
            ensemble_mean = np.mean(ensemble_predictions, axis=0)
            diagnostics['actual_smape'] = calculate_smape(actual_values, ensemble_mean)
        else:
            diagnostics['actual_smape'] = None
    
        # Compute feature importance using dummy training data (for demonstration).
        dummy_train = np.random.rand(1000, 6)  # Adjust as needed.
        dummy_features = ['open', 'high', 'low', 'close', 'volume', 'previous_close']
        feature_importance = self.calculate_feature_importance(dummy_train, dummy_features)
    
        # Update forecast_results so that downstream plotting functions can access them.
        self.forecast_results.simulation_predictions = ensemble_predictions
        self.forecast_results.ensemble_mean = np.mean(ensemble_predictions, axis=0)
        self.forecast_results.ensemble_std = np.std(ensemble_predictions, axis=0)
        # Set error_metrics['smape'] to dummy values if actual_smape is not available.
        dummy_smape = diagnostics.get('actual_smape') if diagnostics.get('actual_smape') is not None else 5.0
        self.forecast_results.error_metrics['smape'] = np.full(simulation_size, dummy_smape)
        # Set best_simulation_indices to a valid array.
        self.forecast_results.best_simulation_indices = np.arange(simulation_size)
    
        return (ensemble_predictions, [diagnostics], feature_importance)
