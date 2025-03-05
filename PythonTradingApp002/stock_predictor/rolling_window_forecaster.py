"""
Rolling Window Forecaster Module

This module implements rolling window validation for time series forecasting.
It evaluates model performance across different time periods and helps identify
if forecast performance is stable or varies significantly over time.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import time
import traceback

from stock_predictor.data_classes import ForecastResults
from stock_predictor.model import BidirectionalForecastModel

class RollingWindowForecaster:
    """
    Implementation of rolling window validation for time series forecasting.
    This class adds rolling window validation to the standard forecasting process
    to evaluate model performance across different time periods.
    """
    
    def __init__(self, app_config, data_handler, np=np, tf=tf):
        """
        Initialize the rolling window forecaster.
        
        Args:
            app_config: Application configuration
            data_handler: Data handler instance
            np: NumPy instance
            tf: TensorFlow instance
        """
        self.app_config = app_config
        self.data_handler = data_handler
        self.stock_data = data_handler.stock_data
        self.np = np
        self.tf = tf
        
        # Initialize forecast results
        self.forecast_results = ForecastResults(
            sequence_length=app_config.learning.timestamp,
            forecast_horizon=app_config.prediction.predict_days
        )
        
        # Rolling window parameters from the dedicated config
        self.window_size = app_config.rolling_window.window_size
        self.step_size = app_config.rolling_window.step_size
        self.min_train_size = app_config.rolling_window.min_train_size
        self.refit_frequency = app_config.rolling_window.refit_frequency
        
        # Keep track of model performance across windows
        self.window_performances = []
    
    def run_rolling_validation(self):
        """
        Run the rolling window validation process.
        """
        import numpy as np
        import tensorflow as tf
        import logging
        import traceback

        logging.info("Starting rolling window validation")

        # Check if logarithmic transformation is enabled
        log_transform_enabled = False
        if hasattr(self.app_config.learning, 'use_log_transformation'):
            log_transform_enabled = self.app_config.learning.use_log_transformation
            if log_transform_enabled:
                logging.info("Running rolling window validation with logarithmic price transformation enabled")

        # Get the full training data
        train_data = self.stock_data.get_training_array()
        features = self.stock_data.feature_list
        logging.info(f"Total training samples: {len(train_data)}")
        logging.info(f"Using features (count={len(features)}): {features}")

        # Ensure we have enough data
        if len(train_data) < self.min_train_size + self.window_size:
            logging.warning(
                f"Not enough data for rolling window validation. " +
                f"Need at least {self.min_train_size + self.window_size} points."
            )
            self.min_train_size = max(int(len(train_data) * 0.6), 50)
            self.window_size = max(int(len(train_data) * 0.3), 30)
            self.step_size = max(int(self.window_size / 5), 5)
            logging.warning(
                f"Adjusted parameters: min_train_size={self.min_train_size}, " +
                f"window_size={self.window_size}, step_size={self.step_size}"
            )
    
        # Build the initial model
        model = BidirectionalForecastModel(
            num_layers=self.app_config.learning.num_layers,
            size_layer=min(64, max(32, len(features) * self.app_config.learning.timestamp // 8)),
            output_size=1,
            dropout_rate=self.app_config.learning.dropout_rate,
            l2_reg=self.app_config.learning.l2_reg
        )
    
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate),
            loss='mae'
        )
    
        # Train on initial data
        initial_data = train_data[:self.min_train_size]
        if len(initial_data) < 2:  # Minimum 2 points needed for a sequence
            logging.error(
                f"Initial slice has only {len(initial_data)} samples, need at least 2."
            )
            return ([], [], {})
    
        logging.info(f"Training initial model on {len(initial_data)} data points")
        initial_dataset = self.create_dataset(
            initial_data,
            self.app_config.learning.timestamp,
            self.app_config.learning.batch_size
        )
    
        initial_num_batches = sum(1 for _ in initial_dataset)
        logging.info(f"Initial dataset has {initial_num_batches} batches")
        if initial_num_batches == 0:
            logging.warning("Initial dataset has zero batches. Creating minimal dataset to proceed.")
            close_idx = self.stock_data.feature_list.index('close')
            initial_dataset = tf.data.Dataset.from_tensor_slices((
                [initial_data[-self.app_config.learning.timestamp:]],
                [initial_data[-1, close_idx]]
            )).batch(1)
            initial_num_batches = 1
    
        try:
            initial_history = model.fit(
                initial_dataset,
                epochs=self.app_config.learning.epoch,
                verbose=1
            )
        except Exception as e:
            logging.error(f"Initial training failed: {e}")
            logging.error(traceback.format_exc())
            return ([], [], {})
    
        # Storage for predictions and metrics
        all_predictions = []
        all_actuals = []
        window_metrics = []
        days_since_refit = 0
    
        # Process each rolling window
        for start_idx in range(self.min_train_size, len(train_data) - self.window_size, self.step_size):
            window_start = start_idx
            window_end = start_idx + self.window_size
        
            logging.info(f"Processing window {len(window_metrics)+1}: indices {window_start} to {window_end}")
        
            window_data = train_data[window_start:window_end]
            if len(window_data) == 0:
                logging.error(f"Window {len(window_metrics)+1} is empty. Skipping.")
                continue
        
            # Refitting logic
            if days_since_refit >= self.refit_frequency:
                logging.info(f"Refitting model at window {len(window_metrics)+1}")
                refit_data = train_data[max(0, window_start - self.min_train_size):window_start]
                if len(refit_data) < 2:
                    logging.warning(
                        f"Refit data has only {len(refit_data)} samples, too short for refit. Skipping."
                    )
                else:
                    refit_dataset = self.create_dataset(
                        refit_data,
                        self.app_config.learning.timestamp,
                        self.app_config.learning.batch_size
                    )
                    refit_num_batches = sum(1 for _ in refit_dataset)
                    logging.info(f"Refit dataset has {refit_num_batches} batches")
                    if refit_num_batches > 0:
                        try:
                            model.fit(
                                refit_dataset,
                                epochs=self.app_config.learning.epoch,
                                verbose=0
                            )
                            days_since_refit = 0
                        except Exception as e:
                            logging.error(f"Refit failed: {e}")
                            logging.error(traceback.format_exc())
                    else:
                        logging.warning(f"Refit dataset has 0 batches at window {len(window_metrics)+1}, skipping refit.")
            else:
                days_since_refit += self.step_size
        
            # Generate predictions for the current window
            input_sequence = window_data[:self.app_config.learning.timestamp]
            logging.debug(f"Window input_sequence shape: {input_sequence.shape}")
            try:
                if self.app_config.prediction.set_initial_data:
                    initial_period = min(self.app_config.prediction.initial_data_period, len(window_data))
                    prediction = self.predict_with_initial_data(
                        model, window_data[:initial_period], features, initial_period
                    )
                else:
                    prediction = self.predict(model, input_sequence, features)
            except Exception as e:
                logging.error(f"Prediction failed in window {len(window_metrics)+1}: {e}")
                logging.error(traceback.format_exc())
                continue
        
            next_window_start = window_end
            next_window_end = min(next_window_start + len(prediction), len(train_data))
            if next_window_end > next_window_start:
                actual_values = train_data[next_window_start:next_window_end, features.index('close')]
                scaled_prediction = prediction[:len(actual_values)]
                unscaled_prediction = self.stock_data.unscale_close_price(scaled_prediction)
                unscaled_actual = self.stock_data.unscale_close_price(actual_values)
                smape = self.calculate_smape(unscaled_actual, unscaled_prediction)
                logging.info(f"Window {len(window_metrics)+1} SMAPE: {smape:.2f}%")
                window_metrics.append({
                    'window_idx': len(window_metrics),
                    'start_idx': window_start,
                    'end_idx': window_end,
                    'smape': smape,
                    'prediction': scaled_prediction.tolist(),
                    'actual': actual_values.tolist()
                })
                all_predictions.append(prediction)
                all_actuals.append(actual_values)
            else:
                logging.warning(f"No actual values available for window {len(window_metrics)+1}")
    
        if window_metrics:
            avg_smape = np.mean([m['smape'] for m in window_metrics])
            logging.info(f"Average SMAPE across all windows: {avg_smape:.2f}%")
        else:
            logging.warning("No window metrics computed. Possibly no valid windows.")
            avg_smape = 0.0
    
        # Populate forecast results error metrics for window validation
        if window_metrics:
            self.forecast_results.error_metrics['smape'] = np.array([m['smape'] for m in window_metrics])
        else:
            self.forecast_results.error_metrics['smape'] = np.array([])
    
        # Final model training for future prediction
        final_train_data = train_data[-self.min_train_size:]
        if len(final_train_data) < 2:
            logging.error(
                f"Final training slice has only {len(final_train_data)} samples, too short."
            )
            diagnostics = {
                'final_loss': float(avg_smape),
                'epoch_history': [{'avg_train_loss': float(m['smape']), 'avg_val_loss': None} for m in window_metrics],
                'window_metrics': window_metrics
            }
            return (all_predictions, [diagnostics], {})
    
        final_dataset = self.create_dataset(
            final_train_data,
            self.app_config.learning.timestamp,
            self.app_config.learning.batch_size
        )
        final_batches = sum(1 for _ in final_dataset)
        logging.info(f"Final dataset has {final_batches} batches")
        if final_batches == 0:
            logging.error("Final dataset yields zero batches. Returning partial results.")
            diagnostics = {
                'final_loss': float(avg_smape),
                'epoch_history': [{'avg_train_loss': float(m['smape']), 'avg_val_loss': None} for m in window_metrics],
                'window_metrics': window_metrics
            }
            return (all_predictions, [diagnostics], {})
    
        logging.info("Training final model for future prediction")
        try:
            final_history = model.fit(
                final_dataset,
                epochs=self.app_config.learning.epoch,
                verbose=1
            )
        except Exception as e:
            logging.error(f"Final training failed: {e}")
            logging.error(traceback.format_exc())
            diagnostics = {
                'final_loss': float(avg_smape),
                'epoch_history': [{'avg_train_loss': float(m['smape']), 'avg_val_loss': None} for m in window_metrics],
                'window_metrics': window_metrics
            }
            return (all_predictions, [diagnostics], {})
    
        # Generate base final prediction
        if self.app_config.prediction.set_initial_data:
            final_prediction = self.predict_with_initial_data(
                model, final_train_data, features,
                self.app_config.prediction.initial_data_period
            )
        else:
            final_prediction = self.predict(
                model,
                final_train_data[-self.app_config.learning.timestamp:],
                features
            )
    
        # Generate multiple simulations
        final_predictions = []
        for i in range(self.app_config.learning.simulation_size):
            if window_metrics:
                noise_scale = np.std([m['smape'] for m in window_metrics]) / 100.0
                noise = np.random.normal(0, noise_scale, size=final_prediction.shape)
                noisy_prediction = final_prediction + noise
                final_predictions.append(noisy_prediction)
            else:
                final_predictions.append(final_prediction)
        logging.info(f"Generated {len(final_predictions)} final predictions, each of length {len(final_predictions[0]) if final_predictions else 'N/A'}")
    
        # Process final predictions and calculate sMAPE scores
        if final_predictions and len(final_predictions) > 0:
            self.forecast_results.ensemble_mean = np.mean(final_predictions, axis=0)
            self.forecast_results.ensemble_std = np.std(final_predictions, axis=0)
            self.forecast_results.simulation_predictions = final_predictions
        
            # Calculate sMAPE for each simulation against actual data
            if not self.stock_data.csv_data_test_period.empty:
                actual = self.stock_data.csv_data_test_period['close'].values
            else:
                actual = self.stock_data.csv_data_train_period['close'].values[-len(final_predictions[0]):]
        
            # Ensure prediction length matches actual length
            pred_length = len(final_predictions[0])
            actual_length = len(actual)
            if pred_length > actual_length:
                logging.warning(f"Prediction length ({pred_length}) exceeds actual data length ({actual_length}). Truncating predictions.")
                final_predictions = [pred[:actual_length] for pred in final_predictions]
                pred_length = actual_length
        
            smape_scores = []
            for pred in final_predictions:
                unscaled_pred = self.stock_data.unscale_close_price(pred)
                smape = self.calculate_smape(actual[:pred_length], unscaled_pred)
                smape_scores.append(smape)
        
            self.forecast_results.error_metrics['smape'] = np.array(smape_scores)
            num_top = min(5, len(smape_scores))
            self.forecast_results.best_simulation_indices = np.argsort(smape_scores)[:num_top]
            logging.info(f"Best simulation indices based on sMAPE: {self.forecast_results.best_simulation_indices}")
        else:
            logging.error("No final predictions generated.")
            self.forecast_results.ensemble_mean = np.array([])
            self.forecast_results.ensemble_std = np.array([])
            self.forecast_results.best_simulation_indices = []
    
        self.forecast_results.calculate_confidence_intervals()
    
        diagnostics = {
            'final_loss': float(avg_smape),
            'epoch_history': [{'avg_train_loss': float(m['smape']), 'avg_val_loss': None} for m in window_metrics],
            'window_metrics': window_metrics,
            'training_config': {
                'window_size': self.window_size,
                'step_size': self.step_size,
                'min_train_size': self.min_train_size,
                'refit_frequency': self.refit_frequency
            }
        }
    
        feature_importance = self.calculate_feature_importance(train_data, features)
        self.forecast_results.feature_importance_scores = feature_importance
    
        return (final_predictions, [diagnostics], feature_importance)
    
    # (The helper methods predict, predict_with_initial_data, create_dataset,
    # calculate_feature_importance, and calculate_smape remain unchanged.)


    
    def calculate_feature_importance(self, train_data, features):
        """Calculate feature importance using permutation importance method"""
        logging.info("Calculating feature importance using permutation importance")
        importances = {}
        from stock_predictor.model import BidirectionalForecastModel
        
        temp_model = BidirectionalForecastModel(
            num_layers=1,
            size_layer=min(64, max(32, len(features) * self.app_config.learning.timestamp // 8)),
            output_size=1,
            dropout_rate=self.app_config.learning.dropout_rate,
            l2_reg=self.app_config.learning.l2_reg
        )
        temp_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate),
            loss='mae',
            run_eagerly=False
        )
        
        train_dataset = self.create_dataset(
            train_data,
            self.app_config.learning.timestamp,
            self.app_config.learning.batch_size
        )
        perm_num_batches = sum(1 for _ in train_dataset)
        if perm_num_batches == 0:
            logging.error("Cannot compute feature importance. Zero batches in train_dataset.")
            return {}
        temp_model.fit(
            train_dataset,
            epochs=self.app_config.learning.epoch,
            verbose=0
        )
        
        baseline_predictions = self.predict(temp_model, train_data, features)
        actual_len = min(len(baseline_predictions), len(self.data_handler.df_train_raw['close']))
        actual = self.data_handler.df_train_raw['close'].values[-actual_len:]
        baseline_pred_slice = baseline_predictions[-actual_len:]
        baseline_smape = self.calculate_smape(
            self.stock_data.unscale_close_price(actual),
            self.stock_data.unscale_close_price(baseline_pred_slice)
        )
        
        close_idx = features.index('close')
        for i, feature in enumerate(features):
            if i == close_idx:
                continue
            logging.debug(f"Evaluating importance of feature: {feature}")
            permuted_data = train_data.copy()
            try:
                permuted_data[:, i] = np.random.permutation(permuted_data[:, i])
            except Exception as e:
                logging.warning(f"Permutation failed for feature {feature}: {e}")
                continue
            permuted_predictions = self.predict(temp_model, permuted_data, features)
            perm_actual_len = min(len(permuted_predictions), len(self.data_handler.df_train_raw['close']))
            perm_pred_slice = permuted_predictions[-perm_actual_len:]
            perm_actual = self.data_handler.df_train_raw['close'].values[-perm_actual_len:]
            permuted_smape = self.calculate_smape(
                self.stock_data.unscale_close_price(perm_actual),
                self.stock_data.unscale_close_price(perm_pred_slice)
            )
            importances[feature] = abs(permuted_smape - baseline_smape)
        
        total = sum(importances.values())
        if total > 1e-12:
            importances = {k: v / total for k, v in importances.items()}
        else:
            importances = {k: 0.0 for k in importances}
        logging.debug(f"Feature importances: {importances}")
        return importances
    
    def create_dataset(self, data, timestamp, batch_size):
        """Create a TensorFlow dataset for time series forecasting."""
        import tensorflow as tf
        import logging
        import numpy as np
    
        effective_batch_size = batch_size
        if self.app_config.learning.auto_batch_size:
            num_samples = data.shape[0]
            num_sequences = max(1, num_samples - timestamp + 1)  # Allow at least one sequence
            effective_batch_size = min(max(32, num_sequences // 10), 1024)
    
        if np.isnan(data).any() or np.isinf(data).any():
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            logging.warning("NaN or infinite values detected in data and replaced")
    
        # Adjust timestamp if data is too short to form sequences
        effective_timestamp = min(timestamp, data.shape[0] - 1) if data.shape[0] > 1 else 1
        if data.shape[0] <= effective_timestamp:
            logging.warning(f"Data length {data.shape[0]} <= effective timestamp {effective_timestamp}. Using minimal sequence length.")
            effective_timestamp = max(1, data.shape[0] - 1)  # Ensure at least one sequence
    
        try:
            close_idx = self.stock_data.feature_list.index('close')
            dataset = tf.keras.utils.timeseries_dataset_from_array(
                data=data[:-1],
                targets=data[effective_timestamp:, close_idx],
                sequence_length=effective_timestamp,
                sampling_rate=1,
                batch_size=effective_batch_size,
                shuffle=True
            )
            options = tf.data.Options()
            options.experimental_optimization.map_parallelization = True
            dataset = dataset.with_options(options)
            dataset = dataset.cache()
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
            num_batches = sum(1 for _ in dataset)
            logging.info(f"Created dataset with {num_batches} batches using sequence length {effective_timestamp}")
            return dataset
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            logging.error(traceback.format_exc())
            return tf.data.Dataset.from_tensor_slices(([], []))
    
    def predict(self, model, sequence_data, features):
        """Generate predictions using the model."""
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None
        last_sequence = sequence_data.copy()
        if last_sequence.shape[0] < self.app_config.learning.timestamp:
            logging.warning(
                f"Not enough data points. Have {last_sequence.shape[0]}, need {self.app_config.learning.timestamp}. Padding..."
            )
            pad_needed = self.app_config.learning.timestamp - last_sequence.shape[0]
            pad_row = last_sequence[0:1]
            padding = np.repeat(pad_row, pad_needed, axis=0)
            last_sequence = np.vstack([padding, last_sequence])
        
        future_predictions = []
        for i in range(self.app_config.prediction.predict_days):
            last_seq_reshaped = last_sequence.reshape(1, last_sequence.shape[0], len(features))
            pred_close = model(last_seq_reshaped, training=False).numpy()[-1, 0]
            future_predictions.append(pred_close)
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, close_idx] = pred_close
            if self.app_config.prediction.use_previous_close and prev_close_idx is not None:
                last_sequence[-1, prev_close_idx] = pred_close if i == 0 else future_predictions[i-1]
        return np.array(future_predictions)
    
    def predict_with_initial_data(self, model, initial_data, features, initial_period):
        """Generate predictions using initial state from historical data."""
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None
        if len(initial_data) < initial_period:
            logging.warning(
                f"Initial data too short ({len(initial_data)} < {initial_period}). Using all available data."
            )
            initial_period = len(initial_data)
        working_sequence = initial_data[-initial_period:].copy()
        future_predictions = []
        for i in range(self.app_config.prediction.predict_days):
            seq_reshaped = working_sequence.reshape(1, working_sequence.shape[0], len(features))
            pred_close = model(seq_reshaped, training=False).numpy()[-1, 0]
            future_predictions.append(pred_close)
            working_sequence = np.vstack([working_sequence[1:], working_sequence[-1].copy()])
            working_sequence[-1, close_idx] = pred_close
            if self.app_config.prediction.use_previous_close and prev_close_idx is not None:
                working_sequence[-1, prev_close_idx] = pred_close if i == 0 else future_predictions[i-1]
        return np.array(future_predictions)
    
    def calculate_smape(self, actual, predicted, epsilon=1e-8):
        """Calculate Symmetric Mean Absolute Percentage Error."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        denominator = np.abs(actual) + np.abs(predicted) + epsilon
        return 100.0 * np.mean((2.0 * np.abs(actual - predicted)) / denominator)
