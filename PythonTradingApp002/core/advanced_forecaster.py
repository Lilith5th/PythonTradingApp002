"""
Advanced Forecaster Module for Stock Prediction

This module provides advanced forecasting capabilities including:
- Uncertainty quantification
- Monte Carlo simulations
- Feature importance analysis
- Ensemble methods

It leverages the uncertainty module and model factory for standardized
implementation and proper error handling.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import time
import traceback
from typing import List, Dict, Any, Union, Optional, Tuple

from .data_classes import ForecastResults
from .model_factory import ModelFactory
from .uncertainty import UncertaintyFactory
from .dataset_factory import DatasetFactory
from .error_handler import handle_errors, catch_and_log_errors, get_error_handler


class AdvancedForecaster:
    """Advanced forecaster with uncertainty quantification and ensemble methods"""
    
    def __init__(self, app_config, data_handler):
        """
        Initialize the advanced forecaster

        Args:
            app_config: Application configuration
            data_handler: Data handler instance
        """
        self.app_config = app_config
        self.data_handler = data_handler
        self.stock_data = data_handler.stock_data
        
        # Create forecast results container
        self.forecast_results = ForecastResults(
            sequence_length=app_config.learning.timestamp,
            forecast_horizon=app_config.prediction.predict_days
        )
        
        # Initialize models
        self.models = {}
        self.uncertainty_handler = None
        
        # Error handler
        self.error_handler = get_error_handler()
    
    @handle_errors(context="monte_carlo_path")
    def generate_monte_carlo_path(self, base_prediction: np.ndarray, drift_mult: float = 1.0, 
                                 vol_mult: float = 1.0, last_known_value: float = None, 
                                 horizon: int = 30) -> np.ndarray:
        """
        Generate a Monte Carlo simulation path with scenario-specific parameters.
        
        Args:
            base_prediction: Base prediction to build Monte Carlo path from
            drift_mult: Multiplier for drift (trend)
            vol_mult: Multiplier for volatility
            last_known_value: Last known value (default: first value of base_prediction)
            horizon: Forecast horizon
            
        Returns:
            np.ndarray: Monte Carlo simulation path
        """
        if last_known_value is None:
            last_known_value = base_prediction[0]
            
        # Calculate returns from base prediction
        returns = np.diff(base_prediction) / base_prediction[:-1]
        base_drift = np.mean(returns)
        base_vol = np.std(returns)
        
        # Apply multipliers
        drift = base_drift * drift_mult
        volatility = base_vol * vol_mult
        
        # Generate random normal samples
        z = np.random.standard_normal(horizon - 1)
        
        # Vectorized simulation
        increments = np.exp((drift - 0.5 * volatility ** 2) + volatility * z)
        path = last_known_value * np.concatenate(([1.0], np.cumprod(increments)))
        
        return path
    
    @handle_errors(context="feature_importance")
    def calculate_feature_importance(self, train_data: np.ndarray, features: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance using permutation and correlation methods.
        
        Args:
            train_data: Training data
            features: List of feature names
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        from sklearn.inspection import permutation_importance
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor

        # Exclude target column from X
        X = train_data[:, :len(features) - 1]
        y = train_data[:, features.index('close')]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a random forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            rf_model, X_scaled, y, n_repeats=10, random_state=42
        )
        
        # Create dictionary of importance scores
        importance_scores = {
            features[i]: perm_importance.importances_mean[i]
            for i in range(len(features) - 1)
        }
        
        # Calculate correlation importance
        correlation_importance = {}
        for i, feature in enumerate(features[:-1]):
            corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            correlation_importance[feature] = corr
        
        # Combine importance scores with weights
        combined_importance = {}
        for feature in importance_scores:
            combined_importance[feature] = 0.7 * importance_scores[feature] + 0.3 * correlation_importance[feature]
        
        # Normalize to [0, 1] range
        max_val = max(combined_importance.values())
        normalized_importance = {k: v / max_val for k, v in combined_importance.items()}
        
        return normalized_importance
    
    @handle_errors(context="diagnostics_conversion")
    def convert_ensemble_history_to_diagnostics(self, histories: List[Any]) -> Dict[str, Any]:
        """
        Convert training histories to diagnostic information.
        
        Args:
            histories: List of training histories
            
        Returns:
            Dict[str, Any]: Diagnostic information
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
    
    @handle_errors(context="base_predict")
    def predict(self, model: tf.keras.Model, train_data: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Generate predictions using the base forecasting model.
        
        Args:
            model: Trained model
            train_data: Training data
            features: List of feature names
            
        Returns:
            np.ndarray: Predictions
        """
        input_sequence = train_data[-self.app_config.learning.timestamp:]
        input_reshaped = input_sequence.reshape(1, self.app_config.learning.timestamp, len(features))
        prediction = model.predict(input_reshaped)
        
        # Handle different output formats
        if isinstance(prediction, list):
            # For quantile regression
            return prediction[1].flatten()  # Return median (quantile 0.5)
        
        return prediction.flatten()
    
    @handle_errors(context="predict_with_initial_data")
    def predict_with_initial_data(self, model: tf.keras.Model, train_data: np.ndarray, 
                                 features: List[str], initial_period: int) -> np.ndarray:
        """
        Generate predictions using an initial context.
        
        Args:
            model: Trained model
            train_data: Training data
            features: List of feature names
            initial_period: Number of initial data points to use
            
        Returns:
            np.ndarray: Predictions
        """
        initial_data = train_data[-initial_period:]
        
        # Repeat initial data to match prediction horizon
        repeated_initial = np.repeat(
            initial_data,
            self.app_config.prediction.predict_days // initial_period + 1,
            axis=0
        )[:self.app_config.prediction.predict_days]
        
        # Reshape for model input
        input_reshaped = repeated_initial.reshape(1, self.app_config.prediction.predict_days, len(features))
        
        # Generate prediction
        prediction = model.predict(input_reshaped)
        
        # Handle different output formats
        if isinstance(prediction, list):
            # For quantile regression
            return prediction[1].flatten()  # Return median (quantile 0.5)
        
        return prediction.flatten()
    
    @handle_errors(context="uncertainty_predict")
    def predict_with_uncertainty(self, train_data: np.ndarray, features: List[str]) -> Dict[str, Any]:
        """
        Generate predictions with uncertainty quantification.
        
        Args:
            train_data: Training data
            features: List of feature names
            
        Returns:
            Dict[str, Any]: Predictions with uncertainty information
        """
        # Get uncertainty configuration
        uncertainty_method = self.app_config.prediction_advanced.uncertainty_method
        
        # Ensure we have an uncertainty handler
        if self.uncertainty_handler is None:
            # Create and initialize the uncertainty handler
            if uncertainty_method == 'bootstrap' and 'bootstrap_models' in self.models:
                # Use pre-trained bootstrap models
                self.uncertainty_handler = UncertaintyFactory.create(
                    uncertainty_method, self.models['bootstrap_models'], self.app_config.prediction_advanced
                )
            else:
                # Use the base model with MC dropout, quantile, or evidential
                base_model = self.get_or_create_model(uncertainty_method, features)
                self.uncertainty_handler = UncertaintyFactory.create(
                    uncertainty_method, base_model, self.app_config.prediction_advanced
                )
        
        # Generate predictions with uncertainty
        input_sequence = train_data[-self.app_config.learning.timestamp:]
        uncertainty_results = self.uncertainty_handler.quantify_uncertainty(input_sequence, features)
        
        return uncertainty_results
    
    @handle_errors(context="create_dataset")
    def create_dataset(self, data: np.ndarray, sequence_length: int, batch_size: int) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from a NumPy array.
        
        Args:
            data: Input data
            sequence_length: Length of input sequences
            batch_size: Batch size
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Use dataset factory
        return DatasetFactory.create_dataset(
            data=data,
            timestamp=sequence_length,
            batch_size=batch_size,
            auto_batch_size=self.app_config.learning.auto_batch_size
        )
    
    @catch_and_log_errors(context="get_or_create_model")
    def get_or_create_model(self, model_type: str, features: List[str]) -> tf.keras.Model:
        """
        Get an existing model or create a new one.
        
        Args:
            model_type: Type of model to create
            features: List of feature names
            
        Returns:
            tf.keras.Model: Model
        """
        # Check if we already have this model type
        if model_type in self.models:
            return self.models[model_type]
        
        # Create the model using the model factory
        model = ModelFactory.create_model(
            model_type=model_type,
            config=self.app_config.learning,
            input_shape=(self.app_config.learning.timestamp, len(features))
        )
        
        # Store the model
        self.models[model_type] = model
        
        # Train the model if it's not an ensemble
        if model_type not in ['ensemble']:
            train_data = self.stock_data.get_training_array()
            dataset = self.create_dataset(
                train_data, 
                self.app_config.learning.timestamp, 
                self.app_config.learning.batch_size
            )
            
            model.fit(
                dataset, 
                epochs=self.app_config.learning.epoch, 
                verbose=1
            )
        
        return model
    
    @catch_and_log_errors(context="create_bootstrap_models")
    def create_bootstrap_models(self, features: List[str], num_models: int = 5) -> List[tf.keras.Model]:
        """
        Create and train bootstrap models.
        
        Args:
            features: List of feature names
            num_models: Number of bootstrap models to create
            
        Returns:
            List[tf.keras.Model]: List of trained bootstrap models
        """
        # Get training data
        train_data = self.stock_data.get_training_array()
        
        # Create bootstrap datasets
        bootstrap_datasets = DatasetFactory.create_bootstrap_datasets(
            data=train_data,
            num_datasets=num_models,
            sample_ratio=0.8,
            sequence_length=self.app_config.learning.timestamp,
            batch_size=self.app_config.learning.batch_size,
            seed=42
        )
        
        # Train models on bootstrap datasets
        models = []
        for i, dataset in enumerate(bootstrap_datasets):
            model = ModelFactory.create_model(
                model_type='lstm',  # Use basic LSTM for bootstrap models
                config=self.app_config.learning,
                input_shape=(self.app_config.learning.timestamp, len(features))
            )
            
            model.fit(
                dataset, 
                epochs=self.app_config.learning.epoch, 
                verbose=0
            )
            
            models.append(model)
            logging.info(f"Bootstrap model {i+1}/{num_models} trained")
        
        # Store the models
        self.models['bootstrap_models'] = models
        
        return models
    
    @handle_errors(context="run_forecast")
    def run_forecast(self) -> Any:
        """
        Run an advanced forecast with uncertainty quantification.
    
        Returns:
            Tuple: (ensemble_predictions, [diagnostics], feature_importance)
        """
        # Start timing
        start_time = time.time()
        
        # Retrieve historical training data
        try:
            train_data = self.stock_data.get_training_array()
            features = self.stock_data.feature_list
        except AttributeError as e:
            raise AttributeError(f"Unable to access historical data from DataHandler: {e}. "
                                 "Please check DataHandler implementation.")
        
        logging.info(f"Running advanced forecast with method: {self.app_config.prediction_advanced.uncertainty_method}")
        
        # Get prediction horizon
        horizon = self.app_config.prediction.predict_days if hasattr(self.app_config.prediction, 'predict_days') else 30
        
        # Initialize arrays to store predictions
        ensemble_predictions = []
        
        # Check which advanced method to use
        if self.app_config.prediction_advanced.enable_uncertainty_quantification:
            # Use uncertainty quantification
            uncertainty_method = self.app_config.prediction_advanced.uncertainty_method
            
            if uncertainty_method == 'bootstrap' and len(self.models.get('bootstrap_models', [])) == 0:
                # Create bootstrap models if needed
                self.create_bootstrap_models(features)
            
            # Generate predictions with uncertainty
            uncertainty_results = self.predict_with_uncertainty(train_data, features)
            
            # Extract predictions and confidence intervals
            mean_prediction = uncertainty_results['mean']
            lower_ci = uncertainty_results['lower']
            upper_ci = uncertainty_results['upper']
            std_prediction = uncertainty_results['std']
            
            # Use multiple samples if available
            if 'predictions' in uncertainty_results and len(uncertainty_results['predictions']) > 1:
                ensemble_predictions = uncertainty_results['predictions']
            else:
                # If only mean prediction is available, generate variations with noise
                for _ in range(self.app_config.learning.simulation_size):
                    # Add noise based on standard deviation
                    noise = np.random.normal(0, std_prediction)
                    noisy_prediction = mean_prediction + noise
                    ensemble_predictions.append(noisy_prediction)
            
            # Store confidence intervals
            self.forecast_results.confidence_intervals = {
                'lower': lower_ci,
                'upper': upper_ci,
                'mean': mean_prediction
            }
            
        elif self.app_config.prediction_advanced.enable_monte_carlo:
            # Use Monte Carlo simulations with scenarios
            scenarios = self.app_config.prediction_advanced.monte_carlo_scenarios
            scenario_params = self.app_config.prediction_advanced.scenario_parameters
            
            # Create base model if needed
            base_model = self.get_or_create_model('lstm', features)
            
            # Generate base prediction
            base_prediction = self.predict(base_model, train_data, features)
            
            # Generate Monte Carlo paths for each scenario
            for scenario in scenarios:
                if scenario in scenario_params:
                    drift_mult, vol_mult = scenario_params[scenario]
                    
                    # Generate multiple paths per scenario
                    paths_per_scenario = max(1, self.app_config.learning.simulation_size // len(scenarios))
                    
                    for _ in range(paths_per_scenario):
                        mc_path = self.generate_monte_carlo_path(
                            base_prediction=base_prediction,
                            drift_mult=drift_mult,
                            vol_mult=vol_mult,
                            horizon=horizon
                        )
                        ensemble_predictions.append(mc_path)
            
        else:
            # Fallback to simple ensemble of models
            model_types = ['lstm', 'gru', 'bilstm']
            
            for model_type in model_types:
                model = self.get_or_create_model(model_type, features)
                prediction = self.predict(model, train_data, features)
                ensemble_predictions.append(prediction)
            
            # Generate additional simulations to meet simulation_size
            additional_sims = max(0, self.app_config.learning.simulation_size - len(model_types))
            
            if additional_sims > 0:
                # Use the base model for additional simulations with noise
                base_model = self.models['lstm']
                base_prediction = self.predict(base_model, train_data, features)
                
                # Estimate prediction variance
                prediction_std = np.std([p for p in ensemble_predictions], axis=0)
                
                # Generate additional simulations with noise
                for _ in range(additional_sims):
                    noise = np.random.normal(0, prediction_std)
                    noisy_prediction = base_prediction + noise
                    ensemble_predictions.append(noisy_prediction)
        
        # Convert to numpy array
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        ensemble_std = np.std(ensemble_predictions, axis=0)
        
        # Calculate feature importance
        feature_importance = self.calculate_feature_importance(train_data, features)
        
        # Update forecast results
        self.forecast_results.simulation_predictions = ensemble_predictions
        self.forecast_results.ensemble_mean = ensemble_mean
        self.forecast_results.ensemble_std = ensemble_std
        self.forecast_results.feature_importance_scores = feature_importance
        
        # Set dummy SMAPE values if actual data is not available
        dummy_smape = 5.0
        self.forecast_results.error_metrics['smape'] = np.full(len(ensemble_predictions), dummy_smape)
        
        # Set best simulation indices
        self.forecast_results.best_simulation_indices = np.arange(min(5, len(ensemble_predictions)))
        
        # Update forecast time
        self.forecast_results.prediction_generation_time = time.time() - start_time
        
        # If test data is available, compute actual SMAPE
        if hasattr(self.data_handler, 'df_test_raw') and not self.data_handler.df_test_raw.empty:
            actual_values = self.data_handler.df_test_raw['close'].values[:horizon]
            ensemble_mean_unscaled = self.stock_data.unscale_close_price(ensemble_mean)
            
            # Calculate SMAPE if we have enough test data
            if len(actual_values) > 0:
                from sklearn.metrics import mean_absolute_percentage_error
                actual_smape = mean_absolute_percentage_error(actual_values, ensemble_mean_unscaled[:len(actual_values)]) * 100
                self.forecast_results.error_metrics['smape'][:] = actual_smape
        
        # Prepare diagnostics
        diagnostics = {
            'ensemble_mean': ensemble_mean.tolist(),
            'ensemble_std': ensemble_std.tolist(),
            'model_types': list(self.models.keys()),
            'feature_importance': feature_importance,
            'forecast_time': self.forecast_results.prediction_generation_time
        }
        
        logging.info(f"Advanced forecast completed in {self.forecast_results.prediction_generation_time:.2f} seconds")
        
        return (ensemble_predictions, [diagnostics], feature_importance)
    
    def __del__(self):
        """Cleanup resources"""
        # Clear any TensorFlow sessions
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass