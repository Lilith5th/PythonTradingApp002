# forecaster.py
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import logging
import time
import pandas as pd
import sys
from .model import BidirectionalForecastModel

class Forecaster:
    def __init__(self, app_config, data_handler, np=np, tf=tf):
        self.app_config = app_config
        self.data_handler = data_handler
        self.np = np
        self.tf = tf
        self.num_cores = multiprocessing.cpu_count()  # 16 cores on 7950X3D

    @tf.function
    def create_dataset(self, data, timestamp, batch_size):
        import tensorflow as tf
        
        # Ensure data is a tensor and cast to float32
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            data = tf.cast(data, tf.float32)

        num_samples = tf.shape(data)[0]
        num_sequences = num_samples - timestamp
        
        # Check if there’s enough data for at least one sequence
        if num_sequences < 1:
            raise ValueError(f"Data length {num_samples} is insufficient for timestamp {timestamp}")

        close_idx = self.data_handler.features.index('close')
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=data[:-1],
            targets=data[timestamp:],
            sequence_length=timestamp,
            sampling_rate=1,
            batch_size=batch_size,
            shuffle=True
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(self, model, optimizer, x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.mean_absolute_error(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train(self, train_data):
        import tensorflow as tf
        import logging
        import numpy as np

        batch_size = max(128, self.app_config.learning.batch_size)  # Large batch for threading
        train_dataset = self.create_dataset(
            train_data,
            self.app_config.learning.timestamp,
            batch_size
        )

        val_data = train_data[int(len(train_data) * self.app_config.tf_config.split_ratio):]
        val_dataset = (
            self.create_dataset(
                val_data,
                self.app_config.learning.timestamp,
                batch_size
            ) if len(val_data) > self.app_config.learning.timestamp else None
        )

        num_sequences = len(train_data) - self.app_config.learning.timestamp
        steps_per_epoch = max(1, num_sequences // batch_size)
        logging.debug(f"Steps per epoch: {steps_per_epoch}")

        num_batches = sum(1 for _ in train_dataset)
        logging.debug(f"Total batches in dataset: {num_batches}")
        steps_per_epoch = min(steps_per_epoch, num_batches)
        logging.debug(f"Adjusted steps_per_epoch: {steps_per_epoch}")

        for batch in train_dataset.take(1):
            logging.info(f"Train batch shapes: inputs={batch[0].shape}, targets={batch[1].shape}")
            break
        else:
            logging.error("Train dataset is empty")
            raise ValueError("Train dataset is empty")

        size_layer = min(64, max(32, len(self.data_handler.features) * self.app_config.learning.timestamp // 8))
        logging.info(f"Dynamically set size_layer to {size_layer}")

        model = BidirectionalForecastModel(
            num_layers=1,
            size_layer=size_layer,
            output_size=1,
            dropout_rate=self.app_config.learning.dropout_rate,
            l2_reg=self.app_config.learning.l2_reg
        )

        tf.keras.backend.clear_session()

        gpus = tf.config.list_physical_devices('GPU')
        is_cpu_only = not gpus or not self.app_config.learning_pref.use_gpu_if_available
        if is_cpu_only:
            logging.info("Running on CPU: Disabling mixed precision.")
            tf.keras.mixed_precision.set_global_policy('float32')
        else:
            logging.info("Running on GPU: Keeping mixed precision.")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        logging.info(f"Threading config - intra_op: {tf.config.threading.get_intra_op_parallelism_threads()}, inter_op: {tf.config.threading.get_inter_op_parallelism_threads()}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.app_config.learning.learning_rate)

        # Custom training loop
        epochs = self.app_config.learning.epoch
        diagnostics = {'epoch_history': []}
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}/{epochs}")
            print(f"Starting epoch {epoch + 1}/{epochs} - direct print")
            sys.stdout.flush()

            epoch_loss = tf.keras.metrics.Mean()
            for step, (x_batch, y_batch) in enumerate(train_dataset.take(steps_per_epoch)):
                loss = self.train_step(model, optimizer, x_batch, y_batch)
                epoch_loss.update_state(loss)

                loss_value = float(loss) if loss is not None else 'N/A'
                logging.info(f"Batch {step + 1}/{steps_per_epoch} - loss: {loss_value:.4f}")
                print(f"Batch {step + 1}/{steps_per_epoch} completed - loss: {loss_value:.4f}")
                sys.stdout.flush()

            avg_loss = epoch_loss.result().numpy()
            val_loss = 'N/A'
            if val_dataset:
                val_loss_metric = tf.keras.metrics.Mean()
                for x_val, y_val in val_dataset:
                    val_pred = model(x_val, training=False)
                    val_loss_metric.update_state(tf.keras.losses.mean_absolute_error(y_val, val_pred))
                val_loss = val_loss_metric.result().numpy()

            diagnostics['epoch_history'].append({'avg_train_loss': float(avg_loss)})
            logging.info(f"Epoch {epoch + 1}/{epochs} completed - loss: {avg_loss:.4f}, val_loss: {val_loss:.4f}")
            print(f"Epoch {epoch + 1}/{epochs} completed - loss: {avg_loss:.4f}, val_loss: {val_loss:.4f}")
            sys.stdout.flush()

        logging.info("Training ended.")
        print("Training ended - direct print")
        sys.stdout.flush()
        diagnostics['final_loss'] = diagnostics['epoch_history'][-1]['avg_train_loss']
        return None, model, diagnostics

    def run_simulations(self):
        import logging
        logging.info(f"Starting {self.app_config.learning.simulation_size} simulations")
        train_data = self.data_handler.df_train_scaled.values
        if len(train_data.shape) != 2:
            logging.error(f"Train data is not 2D: {train_data.shape}")
            raise ValueError(f"Train data must be 2D, got {train_data.shape}")
        logging.debug(f"Train data shape: {train_data.shape}")
        features = self.data_handler.features
        timestamp = self.app_config.learning.timestamp
        logging.debug(f"Features: {features}, Timestamp: {timestamp}")
        predictions = []
        diagnostics = []
        for sim_index in range(1, self.app_config.learning.simulation_size + 1):
            result = self.run_single_forecast(sim_index, train_data, features)
            predictions.append(result['predictions'])
            diagnostics.append(result['diagnostics'])
        logging.info("Finished all simulations")
        feature_importance = self.calculate_feature_importance(train_data, features)
        return predictions, diagnostics, feature_importance

    def run_single_forecast(self, sim_index, train_data, features):
        logging.info(f"Starting simulation {sim_index}")
        train_data = np.array(train_data)
        logging.debug(f"train_data shape before training: {train_data.shape}")
        start_time = time.time()
        _, model, diagnostics = self.train(train_data)
        train_time = time.time() - start_time
        logging.info(f"Simulation {sim_index} training completed in {train_time:.2f} seconds")
        start_time = time.time()
        predictions = self.predict(model, train_data, features)
        predict_time = time.time() - start_time
        logging.info(f"Simulation {sim_index} prediction completed in {predict_time:.2f} seconds")
        logging.info(f"Finished simulation {sim_index}")
        return {'predictions': predictions, 'diagnostics': diagnostics}

    @tf.function
    def predict(self, model, train_data, features):
        if self.app_config.prediction.start_forecast_from_backtest:
            forecast_date = self.app_config.learning_pref.backtesting_start_date
            idx_backtest = self.data_handler.df_train_raw['datetime'].searchsorted(pd.to_datetime(forecast_date))
            start_idx = max(0, idx_backtest - self.app_config.learning.timestamp)
            last_sequence = train_data[start_idx:idx_backtest + 1][-self.app_config.learning.timestamp:].copy()
        else:
            last_sequence = train_data[-self.app_config.learning.timestamp:].copy()

        if not isinstance(last_sequence, tf.Tensor):
            last_sequence = tf.convert_to_tensor(last_sequence, dtype=tf.float32)
        else:
            last_sequence = tf.cast(last_sequence, tf.float32)

        future_predictions = []
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None

        for i in tf.range(self.app_config.prediction.predict_days):
            last_seq_reshaped = tf.reshape(last_sequence, [1, self.app_config.learning.timestamp, len(features)])
            pred_close = model(last_seq_reshaped, training=False)[-1, 0]
            future_predictions.append(pred_close)
            last_sequence = tf.roll(last_sequence, shift=-1, axis=0)
            last_sequence = tf.tensor_scatter_nd_update(
                last_sequence, [[tf.shape(last_sequence)[0] - 1, close_idx]], [pred_close]
            )
            if self.app_config.prediction.use_previous_close and prev_close_idx is not None:
                update_value = pred_close if i == 0 else future_predictions[i - 1]
                last_sequence = tf.tensor_scatter_nd_update(
                    last_sequence, [[tf.shape(last_sequence)[0] - 1, prev_close_idx]], [update_value]
                )

        return tf.stack(future_predictions)

    def evaluate(self, results_array):
        import numpy as np
        if isinstance(results_array, list):
            results_array = np.array(results_array)
        smape_scores = []
        for result in results_array:
            if self.app_config.prediction.start_forecast_from_backtest and not self.data_handler.df_test_raw.empty:
                actual = self.data_handler.df_test_raw['close'].values[:len(result)]
                pred = self.data_handler.close_scaler.inverse_transform(result.reshape(-1, 1)).flatten()[:len(actual)]
            else:
                actual = self.data_handler.df_train_raw['close'].values[:len(result)]
                pred = self.data_handler.close_scaler.inverse_transform(result.reshape(-1, 1)).flatten()[:len(actual)]
            smape_scores.append(self.calculate_smape(actual, pred))
        smape_scores = np.array(smape_scores)
        top_indices = np.argsort(smape_scores)[:self.app_config.plot.num_predictions_shown]
        top_predictions = results_array[top_indices]
        predictions_mean = np.mean(top_predictions, axis=0)
        predictions_std = np.std(top_predictions, axis=0)
        return predictions_mean, predictions_std, smape_scores, top_indices

    def calculate_smape(self, actual, predicted, epsilon=1e-8):
        actual = self.np.array(actual)
        predicted = self.np.array(predicted)
        numerator = 2.0 * self.np.abs(actual - predicted)
        denominator = self.np.abs(actual) + self.np.abs(predicted) + epsilon
        return 100.0 * self.np.mean(numerator / denominator)

    def calculate_feature_importance(self, train_data, features):
        importances = {}
        close_idx = features.index('close')
        for i, feature in enumerate(features):
            if i != close_idx:
                correlation = self.np.corrcoef(train_data[:, i], train_data[:, close_idx])[0, 1]
                importances[feature] = abs(correlation) if not self.np.isnan(correlation) else 0.0
        return importances