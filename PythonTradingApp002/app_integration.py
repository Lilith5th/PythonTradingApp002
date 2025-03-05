"""
Application Integration Module for Stock Prediction Application

This module integrates all components of the stock prediction application,
providing high-level functions to run forecasts with different methods.
It serves as the main entry point for programmatic use of the system.
"""

import os
import logging
import tempfile
import pickle
import threading
import traceback
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional

# Import core modules
from stock_predictor.config import AppConfig
from stock_predictor.data_handler import DataHandler
from stock_predictor.data_classes import StockData, ForecastResults
from stock_predictor.model_factory import ModelFactory
from stock_predictor.dataset_factory import DatasetFactory
from stock_predictor.error_handler import (
    get_error_handler, handle_errors, catch_and_log_errors,
    ErrorAwareThread, validate_gpu_availability
)
from stock_predictor.config_utils import update_config_from_gui, validate_config

# Import specialized forecasters
from stock_predictor.forecaster import Forecaster
from stock_predictor.advanced_forecaster import AdvancedForecaster
from stock_predictor.rolling_window_forecaster import RollingWindowForecaster

# Optional imports for uncertainty quantification
try:
    from stock_predictor.uncertainty import UncertaintyFactory
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False
    logging.warning("Uncertainty module not available. Uncertainty quantification will be disabled.")


class AppIntegration:
    """
    Main integration class for the stock prediction application
    """
    
    def __init__(self, app_config=None, data_handler=None):
        """
        Initialize the application integration
        
        Args:
            app_config: Application configuration (optional)
            data_handler: Data handler (optional)
        """
        self.app_config = app_config or AppConfig()
        self.data_handler = data_handler
        self.error_handler = get_error_handler()
        
        # Initialize GPU settings
        self._configure_gpu_settings()
    
    @handle_errors(context="GPU Configuration")
    def _configure_gpu_settings(self):
        """Configure GPU settings"""
        if self.app_config.learning_pref.use_gpu_if_available:
            # Check GPU availability
            gpus_available, status = validate_gpu_availability()
            logging.info(f"GPU status: {status}")
            
            if gpus_available:
                # Configure GPUs
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logging.info(f"Configured memory growth for {gpu.name}")
                    except Exception as e:
                        logging.warning(f"Could not configure memory growth for {gpu.name}: {e}")
            else:
                if self.app_config.learning_pref.use_gpu_if_available:
                    logging.warning("GPUs requested but not available. Using CPU instead.")
        else:
            # Disable GPU
            logging.info("GPU usage disabled in configuration. Using CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    @handle_errors(context="Load Data")
    def load_data(self, csv_file_path=None):
        """
        Load data for prediction
        
        Args:
            csv_file_path: Path to CSV file (optional)
            
        Returns:
            DataHandler: Initialized data handler
        """
        if csv_file_path:
            self.app_config.csv.file_path = csv_file_path
        
        # Create data handler
        self.data_handler = DataHandler(self.app_config)
        
        # Check if data was loaded properly
        if self.data_handler.stock_data.csv_data_raw is None or self.data_handler.stock_data.csv_data_raw.empty:
            raise ValueError(f"Failed to load data from {self.app_config.csv.file_path}")
        
        return self.data_handler
    
    @handle_errors(context="Validate Configuration")
    def validate_configuration(self):
        """
        Validate the application configuration
        
        Returns:
            Tuple[bool, List[str]]: Validation status and errors
        """
        # Validate the configuration
        errors = validate_config(self.app_config)
        
        if errors:
            logging.warning(f"Configuration validation failed with {len(errors)} errors")
            for error in errors:
                logging.warning(f" - {error}")
            return False, errors
        
        # Check if data handler is initialized
        if self.data_handler is None:
            logging.warning("Data handler not initialized. Call load_data() first.")
            return False, ["Data handler not initialized"]
        
        return True, []
    
    @handle_errors(context="Standard Forecast")
    def run_standard_forecast(self):
        """
        Run a standard forecast
        
        Returns:
            ForecastResults: Forecast results
        """
        # Validate configuration
        valid, errors = self.validate_configuration()
        if not valid:
            errors_str = "; ".join(errors)
            raise ValueError(f"Invalid configuration: {errors_str}")
        
        # Create forecaster
        forecaster = Forecaster(self.app_config, self.data_handler)
        
        # Run simulations
        results_array, diagnostics_list, feature_importance = forecaster.run_simulations()
        
        # Evaluate results
        _, _, _, _ = forecaster.evaluate(results_array)
        
        return forecaster.forecast_results
    
    @handle_errors(context="Advanced Forecast")
    def run_advanced_forecast(self):
        """
        Run an advanced forecast with ensemble methods and uncertainty quantification
        
        Returns:
            ForecastResults: Forecast results
        """
        # Validate configuration
        valid, errors = self.validate_configuration()
        if not valid:
            errors_str = "; ".join(errors)
            raise ValueError(f"Invalid configuration: {errors_str}")
        
        # Check if advanced prediction features are enabled
        adv_config = self.app_config.prediction_advanced
        if not (adv_config.use_ensemble_methods or 
                adv_config.enable_uncertainty_quantification or 
                adv_config.enable_monte_carlo):
            logging.warning("No advanced forecasting methods enabled in configuration")
        
        # Create forecaster
        forecaster = AdvancedForecaster(
            self.app_config, 
            self.data_handler, 
            np_module=np, 
            tf_module=tf
        )
        
        # Run forecast
        results = forecaster.run_forecast()
        
        return forecaster.forecast_results
    
    @handle_errors(context="Rolling Window Forecast")
    def run_rolling_window_forecast(self):
        """
        Run a forecast with rolling window validation
        
        Returns:
            ForecastResults: Forecast results
        """
        # Validate configuration
        valid, errors = self.validate_configuration()
        if not valid:
            errors_str = "; ".join(errors)
            raise ValueError(f"Invalid configuration: {errors_str}")
        
        # Check if rolling window validation is enabled
        if not self.app_config.rolling_window.use_rolling_window:
            logging.warning("Rolling window validation not enabled in configuration")
        
        # Create forecaster
        forecaster = RollingWindowForecaster(
            self.app_config, 
            self.data_handler, 
            np_module=np, 
            tf_module=tf
        )
        
        # Run rolling validation
        results_array, diagnostics_list, feature_importance = forecaster.run_rolling_validation()
        
        return forecaster.forecast_results
    
    @handle_errors(context="Uncertainty Forecast")
    def run_uncertainty_forecast(self, method=None):
        """
        Run a forecast with uncertainty quantification
        
        Args:
            method: Uncertainty quantification method (optional)
            
        Returns:
            Dict: Forecast results with uncertainty
        """
        if not UNCERTAINTY_AVAILABLE:
            raise ImportError("Uncertainty module not available")
        
        # Validate configuration
        valid, errors = self.validate_configuration()
        if not valid:
            errors_str = "; ".join(errors)
            raise ValueError(f"Invalid configuration: {errors_str}")
        
        # Use configured method if not specified
        if method is None:
            method = self.app_config.prediction_advanced.uncertainty_method
        
        # Get input data
        train_data = self.data_handler.stock_data.get_training_array()
        features = self.data_handler.stock_data.feature_list
        sequence_length = self.app_config.learning.timestamp
        
        # Create base model
        model = ModelFactory.create_model(
            'mc_dropout' if method == 'mc_dropout' else 'lstm',
            self.app_config.learning,
            (sequence_length, len(features))
        )
        
        # Create dataset
        dataset = DatasetFactory.create_dataset(
            train_data,
            sequence_length,
            self.app_config.learning.batch_size,
            features.index('close') if 'close' in features else 0,
            auto_batch_size=self.app_config.learning.auto_batch_size
        )
        
        # Train the model
        model.fit(
            dataset,
            epochs=self.app_config.learning.epoch,
            verbose=1
        )
        
        # Create uncertainty quantifier
        uncertainty = UncertaintyFactory.create(
            method,
            model,
            self.app_config.prediction_advanced
        )
        
        # Generate prediction with uncertainty
        input_sequence = train_data[-sequence_length:]
        result = uncertainty.quantify_uncertainty(input_sequence, features)
        
        # Update forecast results
        self.forecast_results = ForecastResults(
            sequence_length=sequence_length,
            forecast_horizon=len(result['mean'])
        )
        self.forecast_results.ensemble_mean = result['mean']
        self.forecast_results.ensemble_std = result.get('std', np.zeros_like(result['mean']))
        self.forecast_results.simulation_predictions = result.get('predictions', [result['mean']])
        self.forecast_results.confidence_intervals = {
            'lower': result['lower'],
            'upper': result['upper'],
            'mean': result['mean']
        }
        
        return result
    
    @handle_errors(context="Parallel Forecasts")
    def run_parallel_forecasts(self, scenarios):
        """
        Run multiple forecast scenarios in parallel
        
        Args:
            scenarios: List of scenario configurations
            
        Returns:
            List[ForecastResults]: List of forecast results
        """
        # Validate configuration
        valid, errors = self.validate_configuration()
        if not valid:
            errors_str = "; ".join(errors)
            raise ValueError(f"Invalid configuration: {errors_str}")
        
        # Check if scenarios is a list
        if not isinstance(scenarios, list):
            raise ValueError("Scenarios must be a list of configuration dictionaries")
        
        # Create temporary files for results
        temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in scenarios]
        
        # Configure device resources
        devices = self._configure_device_resources(len(scenarios))
        
        # Create processes
        processes = []
        for i, (scenario_config, temp_file) in enumerate(zip(scenarios, temp_files)):
            # Create a copy of the configuration
            config_copy = pickle.loads(pickle.dumps(self.app_config))
            
            # Apply scenario configuration
            for key, value in scenario_config.items():
                sections = key.split('.')
                if len(sections) == 1:
                    # Top-level setting
                    setattr(config_copy, key, value)
                elif len(sections) == 2:
                    # Section setting
                    section, setting = sections
                    if hasattr(config_copy, section):
                        setattr(getattr(config_copy, section), setting, value)
            
            # Create process
            p = multiprocessing.Process(
                target=self._run_forecast_in_subprocess,
                args=(config_copy, temp_file, devices[i % len(devices)])
            )
            processes.append(p)
            p.start()
        
        # Wait for processes to complete
        results = []
        for p, temp_file in zip(processes, temp_files):
            p.join()
            if p.exitcode == 0 and os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    data = pickle.load(f)
                os.unlink(temp_file)
                if isinstance(data, dict) and 'error' in data:
                    logging.error(f"Scenario failed: {data['error']}")
                else:
                    results.append(data)
        
        return results
    
    def _run_forecast_in_subprocess(self, config, output_file, device):
        """
        Run a forecast in a subprocess
        
        Args:
            config: Configuration for the forecast
            output_file: Path to output file
            device: Device to use
        """
        import os
        import sys
        import logging
        import traceback
        import tensorflow as tf
        import pickle
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Set device
        with tf.device(device):
            try:
                # Create data handler
                data_handler = DataHandler(config)
                
                # Choose appropriate forecaster
                if config.rolling_window.use_rolling_window:
                    forecaster = RollingWindowForecaster(config, data_handler, np_module=np, tf_module=tf)
                    result = forecaster.run_rolling_validation()
                elif (config.prediction_advanced.use_ensemble_methods or 
                      config.prediction_advanced.enable_uncertainty_quantification or 
                      config.prediction_advanced.enable_monte_carlo):
                    forecaster = AdvancedForecaster(config, data_handler, np_module=np, tf_module=tf)
                    result = forecaster.run_forecast()
                else:
                    forecaster = Forecaster(config, data_handler)
                    result = forecaster.run_simulations()
                
                # Calculate confidence intervals
                forecaster.forecast_results.calculate_confidence_intervals()
                
                # Save results
                with open(output_file, 'wb') as f:
                    pickle.dump(forecaster.forecast_results, f)
            
            except Exception as e:
                logging.error(f"Forecast failed on {device}: {e}")
                logging.error(traceback.format_exc())
                with open(output_file, 'wb') as f:
                    pickle.dump({"error": str(e)}, f)
                sys.exit(1)
    
    def _configure_device_resources(self, num_forecasts=1):
        """
        Configure device resources for parallel forecasting
        
        Args:
            num_forecasts: Number of parallel forecasts
            
        Returns:
            List[str]: List of device strings
        """
        try:
            # Check if GPU is enabled
            if not self.app_config.learning_pref.use_gpu_if_available:
                return ['/CPU:0'] * num_forecasts
            
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logging.warning("No GPUs found, using CPU")
                return ['/CPU:0'] * num_forecasts
            
            # Configure memory growth for all GPUs
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logging.warning(f"Could not set memory growth for {gpu.name}: {e}")
            
            # Create device strings
            devices = []
            for i in range(num_forecasts):
                gpu_idx = i % len(gpus)
                devices.append(f'/GPU:{gpu_idx}')
            
            return devices
        
        except Exception as e:
            logging.error(f"Error configuring device resources: {e}")
            logging.error(traceback.format_exc())
            return ['/CPU:0'] * num_forecasts


# Create high-level functions for easy use
def run_forecast(app_config=None, csv_file_path=None, forecast_type='standard'):
    """
    Run a forecast with the specified configuration
    
    Args:
        app_config: Application configuration (optional)
        csv_file_path: Path to CSV file (optional)
        forecast_type: Type of forecast ('standard', 'advanced', 'rolling_window', 'uncertainty')
        
    Returns:
        ForecastResults: Forecast results
    """
    # Create app integration
    app = AppIntegration(app_config)
    
    # Load data
    app.load_data(csv_file_path)
    
    # Run forecast
    if forecast_type == 'standard':
        return app.run_standard_forecast()
    elif forecast_type == 'advanced':
        return app.run_advanced_forecast()
    elif forecast_type == 'rolling_window':
        return app.run_rolling_window_forecast()
    elif forecast_type == 'uncertainty':
        return app.run_uncertainty_forecast()
    else:
        raise ValueError(f"Unknown forecast type: {forecast_type}")

def run_parallel_scenarios(app_config=None, csv_file_path=None, scenarios=None):
    """
    Run multiple forecast scenarios in parallel
    
    Args:
        app_config: Application configuration (optional)
        csv_file_path: Path to CSV file (optional)
        scenarios: List of scenario configurations
        
    Returns:
        List[ForecastResults]: List of forecast results
    """
    # Create app integration
    app = AppIntegration(app_config)
    
    # Load data
    app.load_data(csv_file_path)
    
    # Default scenarios if none provided
    if scenarios is None:
        scenarios = [
            {'learning.learning_rate': 0.001, 'learning.epoch': 100},
            {'learning.learning_rate': 0.01, 'learning.epoch': 100},
            {'learning.learning_rate': 0.001, 'learning.epoch': 200}
        ]
    
    # Run parallel forecasts
    return app.run_parallel_forecasts(scenarios)

def run_gui(app_config=None):
    """
    Run the GUI application
    
    Args:
        app_config: Application configuration (optional)
    """
    from stock_predictor.gui_wrapper_gpu import create_gui
    
    if app_config is None:
        app_config = AppConfig()
    
    create_gui(app_config)