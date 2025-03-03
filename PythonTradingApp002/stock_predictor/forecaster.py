import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import logging
import time
import pandas as pd
import sys
from .model import BidirectionalForecastModel
from .data_classes import StockData, ForecastResults

class Forecaster:
    def __init__(self, app_config, data_handler):
        self.app_config = app_config
        self.data_handler = data_handler
        self.stock_data = data_handler.stock_data
        
        # Initialize config-based properties
        self.start_forecast_from_backtest = app_config.prediction.start_forecast_from_backtest
        self.backtesting_start_date = app_config.learning_pref.backtesting_start_date
        self.timestamp = app_config.learning.timestamp
        self.predict_days = app_config.prediction.predict_days
        self.use_previous_close = app_config.prediction.use_previous_close
        self.initial_data_period = app_config.prediction.initial_data_period
        
        # Initialize forecast results
        self.forecast_results = ForecastResults(
            sequence_length=self.timestamp,
            forecast_horizon=self.predict_days
        )
    
    def run_simulations(self):
        """Run all forecast simulations"""
        start_time = time.time()
        
        train_data = self.stock_data.get_training_array()
        features = self.stock_data.feature_list
        
        # Clear previous results
        self.forecast_results.simulation_predictions = []
        self.forecast_results.learning_curves = []
        
        # Run simulations
        for sim_index in range(1, self.app_config.learning.simulation_size + 1):
            logging.info(f"Starting simulation {sim_index}/{self.app_config.learning.simulation_size}")
            result = self.run_single_forecast(sim_index, train_data, features)
            self.forecast_results.simulation_predictions.append(result['predictions'])
            
            # Store diagnostics from first simulation
            if sim_index == 1:
                self.forecast_results.learning_curves = result['diagnostics']['epoch_history']
                self.forecast_results.final_model_loss = result['diagnostics']['final_loss']
        
        # Calculate feature importance
        self.forecast_results.feature_importance_scores = self.calculate_feature_importance(train_data, features)
        self.forecast_results.model_training_time = time.time() - start_time
        
        # Evaluate predictions
        self.forecast_results.evaluate_predictions(self.stock_data)
        
        logging.info(f"Learning phase completed in {self.forecast_results.model_training_time:.2f} seconds")
        
        return (
            self.forecast_results.simulation_predictions,
            [{'epoch_history': self.forecast_results.learning_curves, 'final_loss': self.forecast_results.final_model_loss}],
            self.forecast_results.feature_importance_scores
        )
    
    def evaluate(self, results_array):
        """Legacy method that now uses ForecastResults internally"""
        start_time = time.time()
        self.forecast_results.simulation_predictions = results_array
        self.forecast_results.evaluate_predictions(self.stock_data)
        logging.info(f"Prediction phase completed in {time.time() - start_time:.2f} seconds")
        return (
            self.forecast_results.ensemble_mean,
            self.forecast_results.ensemble_std,
            self.forecast_results.error_metrics.get('smape', np.array([])),
            self.forecast_results.best_simulation_indices
        )
    
    def calculate_smape(self, actual, predicted, epsilon=1e-8):
        """Legacy method that delegates to ForecastResults"""
        return self.forecast_results.calculate_smape(actual, predicted, epsilon)

    def calculate_feature_importance(self, train_data, features):
        logging.info("Calculating feature importance using permutation importance")
        importances = {}
        model = BidirectionalForecastModel(
            num_layers=1,
            size_layer=min(64, max(32, len(features) * self.timestamp // 8)),
            output_size=1,
            dropout_rate=self.app_config.learning.dropout_rate,
            l2_reg=self.app_config.learning.l2_reg
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate), loss='mae')
        train_dataset = self.create_dataset(train_data, self.timestamp, self.app_config.learning.batch_size)
        model.fit(train_dataset, epochs=self.app_config.learning.epoch, verbose=0)
        
        baseline_predictions = self.predict(model, train_data, features)
        actual = self.data_handler.df_train_raw['close'].values[-len(baseline_predictions):]
        baseline_smape = self.calculate_smape(actual, baseline_predictions)
        
        close_idx = features.index('close')
        for i, feature in enumerate(features):
            if i != close_idx:
                logging.debug(f"Evaluating importance of feature: {feature}")
                permuted_data = train_data.copy()
                permuted_data[:, i] = np.random.permutation(permuted_data[:, i])
                permuted_predictions = self.predict(model, permuted_data, features)
                permuted_smape = self.calculate_smape(actual, permuted_predictions)
                importances[feature] = abs(permuted_smape - baseline_smape)
        
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        else:
            importances = {k: 0.0 for k in importances}
        logging.debug(f"Feature importances: {importances}")
        return importances

    def run_single_forecast(self, sim_index, train_data, features):
        """Run a single forecast simulation"""
        logging.info(f"Starting simulation {sim_index}")
        train_data = np.array(train_data)
        logging.debug(f"train_data shape before training: {train_data.shape}")
        
        start_time = time.time()
        history, model, diagnostics = self.train(train_data)
        train_time = time.time() - start_time
        logging.info(f"Simulation {sim_index} training completed in {train_time:.2f} seconds")
        
        start_time = time.time()
        
        if self.app_config.prediction.set_initial_data:
            logging.info(f"Using prediction with initial historical data (period: {self.initial_data_period} days)")
            predictions = self.predict_with_initial_data(model, train_data, features)
        else:
            predictions = self.predict(model, train_data, features)
        
        predict_time = time.time() - start_time
        logging.info(f"Simulation {sim_index} prediction completed in {predict_time:.2f} seconds")
        
        logging.info(f"Finished simulation {sim_index}")
        return {'predictions': predictions, 'diagnostics': diagnostics}

    def train(self, train_data):
        logging.info(f"Training data shape: {train_data.shape}")
        logging.info(f"Timestamp: {self.timestamp}")
        logging.info(f"Batch size: {self.app_config.learning.batch_size}")

        train_dataset = self.create_dataset(train_data, self.timestamp, self.app_config.learning.batch_size)
        
        val_data = train_data[int(len(train_data) * self.app_config.tf_config.split_ratio):]
        
        val_dataset = None
        validation_steps = None
        
        if len(val_data) > self.timestamp:
            try:
                val_dataset = self.create_dataset(val_data, self.timestamp, self.app_config.learning.batch_size)
                validation_steps = max(1, sum(1 for _ in val_dataset))
            except Exception as e:
                logging.warning(f"Could not create validation dataset: {e}")
                val_dataset = None
                validation_steps = None

        num_batches = sum(1 for _ in train_dataset)
        logging.info(f"Total batches in dataset: {num_batches}")

        steps_per_epoch = max(1, num_batches)
        logging.info(f"Adjusted steps_per_epoch: {steps_per_epoch}")

        size_layer = min(64, max(32, len(self.data_handler.features) * self.timestamp // 8))
        logging.info(f"Dynamically set size_layer to {size_layer}")

        model = BidirectionalForecastModel(
            num_layers=1,
            size_layer=size_layer,
            output_size=1,
            dropout_rate=self.app_config.learning.dropout_rate,
            l2_reg=self.app_config.learning.l2_reg
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate),
            loss='mae'
        )

        fit_params = {
            'epochs': self.app_config.learning.epoch,
            'steps_per_epoch': steps_per_epoch,
            'verbose': 1
        }

        if val_dataset is not None and validation_steps is not None:
            fit_params['validation_data'] = val_dataset
            fit_params['validation_steps'] = validation_steps

        try:
            history = model.fit(train_dataset, **fit_params)
            diagnostics = {
                'final_loss': float(history.history['loss'][-1]),
                'epoch_history': [
                    {
                        'avg_train_loss': float(history.history['loss'][i]),
                        'avg_val_loss': float(history.history['val_loss'][i]) if 'val_loss' in history.history and i < len(history.history['val_loss']) else None
                    }
                    for i in range(len(history.history['loss']))
                ]
            }
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

        return history, model, diagnostics

    @tf.autograph.experimental.do_not_convert
    def create_dataset(self, data, timestamp, batch_size):
        """
        Create a TensorFlow dataset for time series forecasting with advanced logging and batch size management.
        """
        logging.info(f"Input data shape: {data.shape}")
        logging.info(f"Data type before processing: {data.dtype}")
        logging.info(f"Timestamp (sequence length): {timestamp}")
        logging.info(f"Original batch size input: {batch_size}")
    
        # Ensure data is numeric by converting to float32, replacing non-numeric with 0
        try:
            data = np.array(data, dtype=np.float32)
        except (ValueError, TypeError) as e:
            logging.warning(f"Data contains non-numeric values: {e}. Attempting to convert with NaN handling.")
            # Convert problematic entries to NaN, then to 0
            data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
            logging.info(f"Converted data shape: {data.shape}, dtype: {data.dtype}")
    
        # Check for invalid values after conversion
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning("Data contains NaN or infinite values after conversion. Replacing with 0.")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
        if self.app_config.learning.auto_batch_size:
            num_samples = data.shape[0]
            num_sequences = num_samples - timestamp
            effective_batch_size = min(max(32, num_sequences // 10), 1024)
            logging.info("Auto batch size enabled")
            logging.info(f"Dynamically calculated batch size: {effective_batch_size}")
        else:
            effective_batch_size = self.app_config.learning.manual_batch_size
            logging.info("Manual batch size used")
            logging.info(f"Manual batch size: {effective_batch_size}")

        if len(data.shape) != 2:
            logging.error(f"Expected 2D data (samples, features), got {data.shape}")
            raise ValueError(f"Expected 2D data (samples, features), got {data.shape}")

        num_samples = data.shape[0]
        num_features = data.shape[1]
    
        if num_samples < timestamp + 1:
            logging.error(f"Data length {num_samples} is insufficient for timestamp {timestamp}")
            raise ValueError(f"Data length {num_samples} is insufficient for timestamp {timestamp}")

        num_sequences = num_samples - timestamp
        logging.debug(f"Total possible sequences: {num_sequences}")

        try:
            close_idx = self.data_handler.features.index('close')
            logging.debug(f"Close price index: {close_idx}")
            logging.debug(f"Feature list: {self.data_handler.features}")
            if close_idx >= num_features or close_idx < 0:
                logging.error(f"Invalid close index {close_idx} for {num_features} features")
                raise ValueError(f"Invalid close index {close_idx} for {num_features} features")
        except ValueError as e:
            logging.error(f"Failed to find 'close' feature: {e}")
            raise

        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=data[:-1],
            targets=data[timestamp:, close_idx],
            sequence_length=timestamp,
            sampling_rate=1,
            batch_size=effective_batch_size,
            shuffle=True
        )

        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        dataset = dataset.with_options(options)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        dataset_samples = 0
        for i, batch in enumerate(dataset.take(3)):
            logging.debug(f"Batch {i}: inputs={batch[0].shape}, targets={batch[1].shape}")
            dataset_samples += batch[0].shape[0]
    
        logging.info(f"Total samples in first 3 batches: {dataset_samples}")
        logging.info(f"Final effective batch size: {effective_batch_size}")

        return dataset

    def predict(self, model, train_data, features):
        if 'close' not in features:
            raise ValueError("'close' feature is missing in features list")
    
        if self.start_forecast_from_backtest:
            forecast_date = self.backtesting_start_date
            idx_backtest = self.data_handler.df_train_raw['datetime'].searchsorted(pd.to_datetime(forecast_date))
            start_idx = max(0, idx_backtest - self.timestamp)
            last_sequence = train_data[start_idx:idx_backtest + 1][-self.timestamp:].copy()
        else:
            last_sequence = train_data[-self.timestamp:].copy()
    
        logging.debug(f"last_sequence shape: {last_sequence.shape}")
    
        if last_sequence.shape[0] < self.timestamp:
            logging.warning(f"Not enough data points. Have {last_sequence.shape[0]}, need {self.timestamp}")
            padding_needed = self.timestamp - last_sequence.shape[0]
            padding = np.zeros((padding_needed, len(features)))
            last_sequence = np.vstack([padding, last_sequence])
            logging.info(f"Padded last_sequence to shape: {last_sequence.shape}")
    
        future_predictions = []
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None
    
        # Use self.stock_data.future_dates instead of self.data_handler.future_dates
        future_length = min(self.predict_days, len(self.stock_data.future_dates))
    
        for i in range(future_length):
            logging.debug(f"Prediction step {i+1}/{future_length}, last_sequence shape: {last_sequence.shape}")
            last_seq_reshaped = last_sequence.reshape(1, self.timestamp, len(features))
            pred_close = model(last_seq_reshaped, training=False).numpy()[-1, 0]
            future_predictions.append(pred_close)
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, close_idx] = pred_close
            if self.use_previous_close and prev_close_idx is not None:
                last_sequence[-1, prev_close_idx] = pred_close if i == 0 else future_predictions[i - 1]
    
        return np.array(future_predictions)

    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, model, sequences):
        """Make predictions for a batch of sequences using TF's graph execution"""
        return model(sequences, training=False)

    def optimize_model_for_inference(self, model):
        """Convert model for optimized inference"""
        optimized_model = tf.function(
            lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec(shape=[None, None, len(self.data_handler.features)], dtype=tf.float32)]
        )
        return optimized_model

    @tf.function
    def predict_sequence(self, model, initial_sequence, num_steps, close_idx):
        """Predict sequence of values using TensorFlow operations for better GPU utilization"""
        future_preds = tf.TensorArray(tf.float32, size=num_steps)
        sequence = tf.identity(initial_sequence)
    
        for i in tf.range(num_steps):
            reshaped = tf.reshape(sequence, [1, tf.shape(sequence)[0], tf.shape(sequence)[1]])
            pred = model(reshaped, training=False)[0, 0]
            future_preds = future_preds.write(i, pred)
            sequence = tf.roll(sequence, shift=-1, axis=0)
            new_last_row = tf.tensor_scatter_nd_update(sequence[-1], [[close_idx]], [pred])
            sequence = tf.tensor_scatter_nd_update(sequence, [[tf.shape(sequence)[0]-1]], [new_last_row])
    
        return future_preds.stack()

    def predict_with_initial_data(self, model, train_data, features):
        """
        Generate predictions using the trained model with initial state from historical data.
        """
        if 'close' not in features:
            raise ValueError("'close' feature is missing in features list")
    
        logging.info("==== USING PREDICT WITH INITIAL DATA METHOD ====")
        start_time = time.time()
    
        if self.start_forecast_from_backtest:
            forecast_date = self.backtesting_start_date
            backtest_date = pd.to_datetime(forecast_date)
            idx_backtest = self.data_handler.df_train_raw['datetime'].searchsorted(backtest_date)
            start_idx = max(0, idx_backtest - self.initial_data_period)
            initial_sequence = train_data[start_idx:idx_backtest].copy()
            logging.info(f"Using initial data from {self.data_handler.df_train_raw['datetime'].iloc[start_idx].date()} "
                         f"to {self.data_handler.df_train_raw['datetime'].iloc[idx_backtest-1].date()} "
                         f"({len(initial_sequence)} days)")
        else:
            initial_sequence = train_data[-self.initial_data_period:].copy()
            logging.info(f"Using last {len(initial_sequence)} days of training data as initial state")
    
        if len(initial_sequence) < 5:
            logging.warning(f"Initial sequence too short ({len(initial_sequence)} days). Need at least 5 days.")
            return self.predict(model, train_data, features)
    
        logging.info(f"Initial sequence shape: {initial_sequence.shape}")
    
        future_predictions = []
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None
    
        working_sequence = initial_sequence.copy()
    
        logging.info(f"Starting prediction for {self.predict_days} days")
    
        for i in range(self.predict_days):
            seq_reshaped = working_sequence.reshape(1, working_sequence.shape[0], len(features))
            if i % 5 == 0:
                logging.debug(f"Prediction step {i+1}/{self.predict_days}, working sequence shape: {seq_reshaped.shape}")
            pred_close = model(seq_reshaped, training=False).numpy()[-1, 0]
            future_predictions.append(pred_close)
            working_sequence = np.vstack([working_sequence[1:], working_sequence[-1].copy()])
            working_sequence[-1, close_idx] = pred_close
            if self.use_previous_close and prev_close_idx is not None:
                working_sequence[-1, prev_close_idx] = pred_close if i == 0 else future_predictions[i - 1]
    
        elapsed_time = time.time() - start_time
        logging.info(f"==== PREDICTION WITH INITIAL DATA COMPLETED in {elapsed_time:.2f} seconds ====")
        logging.info(f"Predicted {len(future_predictions)} days with values ranging from "
                     f"{min(future_predictions):.4f} to {max(future_predictions):.4f}")
    
        return np.array(future_predictions)