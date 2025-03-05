"""
Forecaster Module for Stock Prediction Application

This module handles the training and prediction of stock prices
using various machine learning models and approaches.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import time
import traceback
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from typing import Dict, List, Tuple, Any, Union, Optional

from .data_classes import ForecastResults
from stock_predictor.dataset_factory import DatasetFactory
from .model_factory import ModelFactory
from utils.error_handler import (
    handle_errors, log_errors, catch_and_log_errors, 
    ErrorAwareThread, get_error_handler
)


class Forecaster:
    """Forecaster class for training models and generating predictions"""
    
    def __init__(self, app_config, data_handler):
        """
        Initialize the forecaster
        
        Args:
            app_config: Application configuration
            data_handler: Data handler with stock data
        """
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
        
        # Set up error handler
        self.error_handler = get_error_handler()
    
    @handle_errors(context="run_simulations")
    def run_simulations(self):
        """
        Run all forecast simulations
        
        Returns:
            Tuple: (predictions, diagnostics, feature_importance)
        """
        start_time = time.time()
        
        # Get training data from StockData
        train_data = self.stock_data.get_training_array()
        features = self.stock_data.feature_list
        
        # Log whether we're using logarithmic transformation
        log_transform_enabled = False
        if hasattr(self.app_config.learning, 'use_log_transformation'):
            log_transform_enabled = self.app_config.learning.use_log_transformation
            if log_transform_enabled:
                logging.info("Running forecast with logarithmic price transformation enabled")
        
        # Clear previous results
        self.forecast_results.simulation_predictions = []
        self.forecast_results.learning_curves = []
        
        # Run simulations
        simulation_size = self.app_config.learning.simulation_size
        logging.info(f"Starting {simulation_size} simulations")
        
        # Parallel execution if multiple simulations
        if simulation_size > 1 and self.app_config.learning_pref.use_gpu_if_available:
            results = self._run_parallel_simulations(train_data, features, simulation_size)
        else:
            # Sequential execution
            results = []
            for sim_index in range(1, simulation_size + 1):
                logging.info(f"Starting simulation {sim_index}/{simulation_size}")
                result = self._run_single_forecast(sim_index, train_data, features)
                results.append(result)
        
        # Process results
        predictions = []
        all_diagnostics = []
        
        for result in results:
            if result and 'predictions' in result:
                predictions.append(result['predictions'])
                if 'diagnostics' in result:
                    all_diagnostics.append(result['diagnostics'])
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(train_data, features)
        
        # Update forecast results
        self.forecast_results.simulation_predictions = predictions
        if all_diagnostics and 'epoch_history' in all_diagnostics[0]:
            self.forecast_results.learning_curves = all_diagnostics[0]['epoch_history']
            self.forecast_results.final_model_loss = all_diagnostics[0].get('final_loss')
        self.forecast_results.feature_importance_scores = feature_importance
        
        # Evaluate predictions
        self.forecast_results.evaluate_predictions(self.stock_data)
        
        # Record training time
        self.forecast_results.model_training_time = time.time() - start_time
        logging.info(f"Learning phase completed in {self.forecast_results.model_training_time:.2f} seconds")
        
        return (
            self.forecast_results.simulation_predictions,
            all_diagnostics,
            self.forecast_results.feature_importance_scores
        )
    
    @log_errors(context="run_parallel_simulations")
    def _run_parallel_simulations(self, train_data, features, simulation_size):
        """
        Run simulations in parallel using ThreadPoolExecutor
        
        Args:
            train_data: Training data
            features: Feature list
            simulation_size: Number of simulations to run
            
        Returns:
            list: List of simulation results
        """
        results = []
        max_workers = min(simulation_size, multiprocessing.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sim = {
                executor.submit(self._run_single_forecast, sim_index, train_data, features): sim_index
                for sim_index in range(1, simulation_size + 1)
            }
            
            for future in as_completed(future_to_sim):
                sim_index = future_to_sim[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Completed simulation {sim_index}/{simulation_size}")
                except Exception as e:
                    logging.error(f"Simulation {sim_index} failed: {e}")
                    logging.error(traceback.format_exc())
        
        return results
    
    @catch_and_log_errors(context="run_single_forecast")
    def _run_single_forecast(self, sim_index, train_data, features):
        """
        Run a single forecast simulation
        
        Args:
            sim_index: Simulation index
            train_data: Training data
            features: Feature list
            
        Returns:
            dict: Simulation results
        """
        logging.info(f"Starting simulation {sim_index}")
        train_data = np.array(train_data)
        logging.debug(f"train_data shape before training: {train_data.shape}")
        
        start_time = time.time()
        history, model, diagnostics = self._train(train_data, features)
        train_time = time.time() - start_time
        logging.info(f"Simulation {sim_index} training completed in {train_time:.2f} seconds")
        
        start_time = time.time()
        
        if self.app_config.prediction.set_initial_data:
            logging.info(f"Using prediction with initial historical data (period: {self.initial_data_period} days)")
            predictions = self._predict_with_initial_data(model, train_data, features)
        else:
            predictions = self._predict(model, train_data, features)
        
        predict_time = time.time() - start_time
        logging.info(f"Simulation {sim_index} prediction completed in {predict_time:.2f} seconds")
        
        logging.info(f"Finished simulation {sim_index}")
        return {'predictions': predictions, 'diagnostics': diagnostics}
    
    @handle_errors(context="train_model")
    def _train(self, train_data, features):
        """
        Train a model using the specified parameters
        
        Args:
            train_data: Training data
            features: Feature list
            
        Returns:
            tuple: (history, model, diagnostics)
        """
        logging.info(f"Training data shape: {train_data.shape}")
        logging.info(f"Timestamp: {self.timestamp}")
        logging.info(f"Batch size: {self.app_config.learning.batch_size}")
        
        # Find the target index (close price)
        close_idx = features.index('close')
        
        # Create dataset using DatasetFactory
        train_dataset = DatasetFactory.create_dataset(
            train_data,
            self.timestamp,
            self.app_config.learning.batch_size,
            target_idx=close_idx,
            auto_batch_size=self.app_config.learning.auto_batch_size
        )
        
        # Create validation dataset if possible
        val_dataset = None
        
        # Split training data for validation if ratio > 0
        split_ratio = self.app_config.tf_config.split_ratio if hasattr(self.app_config, 'tf_config') else 0.8
        if split_ratio < 1.0:
            train_dataset, val_dataset, _ = DatasetFactory.split_train_val_test(
                train_dataset, 
                val_ratio=1.0 - split_ratio, 
                test_ratio=0.0
            )
        
        # Create model using ModelFactory
        input_shape = (self.timestamp, len(features))
        model_type = getattr(self.app_config.learning, 'model_type', 'bilstm')
        
        model = ModelFactory.create_model(
            model_type,
            self.app_config.learning,
            input_shape
        )
        
        # Prepare training parameters
        fit_params = {
            'epochs': self.app_config.learning.epoch,
            'verbose': 1
        }
        
        if val_dataset is not None:
            fit_params['validation_data'] = val_dataset
        
        # Train the model
        start_time = time.time()
        history = model.fit(train_dataset, **fit_params)
        training_time = time.time() - start_time
        
        # Prepare diagnostics
        diagnostics = {
            'training_time': training_time,
            'final_loss': float(history.history['loss'][-1]),
            'epoch_history': [
                {
                    'avg_train_loss': float(history.history['loss'][i]),
                    'avg_val_loss': float(history.history['val_loss'][i]) if 'val_loss' in history.history and i < len(history.history['val_loss']) else None
                }
                for i in range(len(history.history['loss']))
            ]
        }
        
        return history, model, diagnostics
    
    @catch_and_log_errors(context="predict")
    def _predict(self, model, train_data, features):
        """
        Generate predictions using the trained model
        
        Args:
            model: Trained model
            train_data: Training data
            features: Feature list
            
        Returns:
            np.ndarray: Predictions
        """
        if 'close' not in features:
            raise ValueError("'close' feature is missing in features list")
        
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None
        
        # Get the appropriate sequence based on forecast start
        if self.start_forecast_from_backtest:
            forecast_date = self.backtesting_start_date
            idx_backtest = self.data_handler.df_train_raw['datetime'].searchsorted(pd.to_datetime(forecast_date))
            # Ensure we have enough data for the sequence
            start_idx = max(0, idx_backtest - self.timestamp)
            last_sequence = train_data[start_idx:idx_backtest + 1][-self.timestamp:].copy()
        else:
            last_sequence = train_data[-self.timestamp:].copy()
        
        # Create sequence for prediction
        if last_sequence.shape[0] < self.timestamp:
            logging.warning(f"Not enough data points. Have {last_sequence.shape[0]}, need {self.timestamp}")
            padding_needed = self.timestamp - last_sequence.shape[0]
            padding = np.zeros((padding_needed, len(features)))
            last_sequence = np.vstack([padding, last_sequence])
            logging.info(f"Padded last_sequence to shape: {last_sequence.shape}")
        
        # Generate predictions
        future_predictions = []
        
        # Get number of prediction days, limited by available future dates
        future_length = min(self.predict_days, len(self.stock_data.future_dates))
        
        for i in range(future_length):
            # Reshape for prediction
            last_seq_reshaped = last_sequence.reshape(1, self.timestamp, len(features))
            
            # Make prediction
            pred_close = model(last_seq_reshaped, training=False).numpy()[-1, 0]
            future_predictions.append(pred_close)
            
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, close_idx] = pred_close
            
            # Update previous_close if needed
            if self.use_previous_close and prev_close_idx is not None:
                last_sequence[-1, prev_close_idx] = pred_close if i == 0 else future_predictions[i - 1]
        
        return np.array(future_predictions)
    
    @catch_and_log_errors(context="predict_with_initial_data")
    def _predict_with_initial_data(self, model, train_data, features):
        """
        Generate predictions using historical data as initial state
        
        Args:
            model: Trained model
            train_data: Training data
            features: Feature list
            
        Returns:
            np.ndarray: Predictions
        """
        if 'close' not in features:
            raise ValueError("'close' feature is missing in features list")
        
        logging.info("==== USING PREDICT WITH INITIAL DATA METHOD ====")
        start_time = time.time()
        
        close_idx = features.index('close')
        prev_close_idx = features.index('previous_close') if 'previous_close' in features else None
        
        # Get initial sequence based on forecast start
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
        
        # Check if initial sequence is too short
        if len(initial_sequence) < 5:
            logging.warning(f"Initial sequence too short ({len(initial_sequence)} days). Need at least 5 days.")
            return self._predict(model, train_data, features)
        
        logging.info(f"Initial sequence shape: {initial_sequence.shape}")
        
        # Generate predictions
        future_predictions = []
        working_sequence = initial_sequence.copy()
        
        for i in range(self.predict_days):
            # Reshape for prediction
            seq_reshaped = working_sequence.reshape(1, working_sequence.shape[0], len(features))
            
            if i % 5 == 0:
                logging.debug(f"Prediction step {i+1}/{self.predict_days}, working sequence shape: {seq_reshaped.shape}")
            
            # Make prediction
            pred_close = model(seq_reshaped, training=False).numpy()[-1, 0]
            future_predictions.append(pred_close)
            
            # Update sequence for next prediction
            working_sequence = np.vstack([working_sequence[1:], working_sequence[-1].copy()])
            working_sequence[-1, close_idx] = pred_close
            
            # Update previous_close if needed
            if self.use_previous_close and prev_close_idx is not None:
                working_sequence[-1, prev_close_idx] = pred_close if i == 0 else future_predictions[i - 1]
        
        elapsed_time = time.time() - start_time
        logging.info(f"==== PREDICTION WITH INITIAL DATA COMPLETED in {elapsed_time:.2f} seconds ====")
        logging.info(f"Predicted {len(future_predictions)} days with values ranging from "
                     f"{min(future_predictions):.4f} to {max(future_predictions):.4f}")
        
        return np.array(future_predictions)
    
    @log_errors(context="calculate_feature_importance")
    def _calculate_feature_importance(self, train_data, features):
        """
        Calculate feature importance using permutation importance
        
        Args:
            train_data: Training data
            features: Feature list
            
        Returns:
            dict: Feature importance scores
        """
        logging.info("Calculating feature importance using permutation importance")
        
        importances = {}
        close_idx = features.index('close')
        
        # Skip if only one feature (close)
        if len(features) <= 1:
            logging.info("No additional features besides 'close', skipping feature importance calculation")
            return importances
        
        try:
            # Create a small model for feature importance
            input_shape = (self.timestamp, len(features))
            model = ModelFactory.create_model('lstm', self.app_config.learning, input_shape)
            
            # Create a minimal dataset
            train_dataset = DatasetFactory.create_dataset(
                train_data,
                self.timestamp,
                batch_size=32,
                target_idx=close_idx,
                shuffle=True
            )
            
            # Train the model with fewer epochs
            model.fit(train_dataset, epochs=min(5, self.app_config.learning.epoch), verbose=0)
            
            # Generate baseline predictions
            baseline_predictions = self._predict(model, train_data, features)
            actual_values = self.data_handler.df_train_raw['close'].values[-len(baseline_predictions):]
            baseline_error = self.calculate_smape(
                self.stock_data.unscale_close_price(actual_values),
                self.stock_data.unscale_close_price(baseline_predictions)
            )
            
            # Calculate importance for each feature
            for i, feature in enumerate(features):
                if i != close_idx:
                    logging.debug(f"Evaluating importance of feature: {feature}")
                    
                    # Permute the feature values
                    permuted_data = train_data.copy()
                    permuted_data[:, i] = np.random.permutation(permuted_data[:, i])
                    
                    # Generate predictions with permuted data
                    permuted_predictions = self._predict(model, permuted_data, features)
                    permuted_error = self.calculate_smape(
                        self.stock_data.unscale_close_price(actual_values),
                        self.stock_data.unscale_close_price(permuted_predictions)
                    )
                    
                    # Importance is the increase in error
                    importances[feature] = max(0, permuted_error - baseline_error)
            
            # Normalize importance scores
            total = sum(importances.values())
            if total > 0:
                importances = {k: v / total for k, v in importances.items()}
            
            logging.info(f"Feature importance calculation complete: {len(importances)} features analyzed")
            
        except Exception as e:
            logging.error(f"Error calculating feature importance: {e}")
            logging.error(traceback.format_exc())
            return {}
        
        return importances
    
    def calculate_smape(self, actual, predicted, epsilon=1e-8):
        """
        Calculate Symmetric Mean Absolute Percentage Error
        
        Args:
            actual: Actual values
            predicted: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            float: SMAPE value
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        numerator = 2.0 * np.abs(actual - predicted)
        denominator = np.abs(actual) + np.abs(predicted) + epsilon
        return 100.0 * np.mean(numerator / denominator)