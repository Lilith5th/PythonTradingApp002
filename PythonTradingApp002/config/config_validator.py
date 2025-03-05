"""
Configuration Validator for Stock Prediction Application

This module provides specialized validation for the application configuration,
ensuring that all components have valid and compatible settings before execution.
"""

import logging
import os
from typing import Dict, List, Any, Tuple, Union, Optional


class ConfigValidator:
    """Class for validating application configuration"""
    
    def __init__(self, app_config):
        """
        Initialize the configuration validator
        
        Args:
            app_config: Application configuration object
        """
        self.app_config = app_config
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """
        Validate all configuration settings
        
        Returns:
            bool: Whether the configuration is valid
        """
        self.errors = []
        self.warnings = []
        
        # Core configuration
        self._validate_csv_config()
        self._validate_learning_config()
        self._validate_prediction_config()
        self._validate_features_config()
        
        # Advanced features
        self._validate_rolling_window_config()
        self._validate_prediction_advanced_config()
        self._validate_strategy_config()
        
        # Check cross-module compatibility
        self._validate_cross_module_compatibility()
        
        # Log validation results
        if self.errors:
            logging.error(f"Configuration validation failed with {len(self.errors)} errors")
            for i, error in enumerate(self.errors):
                logging.error(f"  Error {i+1}: {error}")
        
        if self.warnings:
            logging.warning(f"Configuration validation generated {len(self.warnings)} warnings")
            for i, warning in enumerate(self.warnings):
                logging.warning(f"  Warning {i+1}: {warning}")
        
        return len(self.errors) == 0
    
    def get_results(self) -> Tuple[List[str], List[str]]:
        """
        Get validation results
        
        Returns:
            Tuple[List[str], List[str]]: Lists of errors and warnings
        """
        return self.errors.copy(), self.warnings.copy()
    
    def _validate_csv_config(self):
        """Validate CSV configuration"""
        csv = getattr(self.app_config, 'csv', None)
        if csv is None:
            self.errors.append("CSV configuration not found")
            return
        
        # Validate file path
        if not hasattr(csv, 'file_path') or not csv.file_path:
            self.errors.append("CSV file path is empty")
        elif not os.path.isabs(csv.file_path):
            # Relative path - check if valid relative to current directory and script directory
            if not os.path.exists(csv.file_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                script_relative_path = os.path.join(script_dir, csv.file_path)
                if not os.path.exists(script_relative_path):
                    self.warnings.append(f"CSV file not found: {csv.file_path}")
        elif not os.path.exists(csv.file_path):
            self.warnings.append(f"CSV file not found: {csv.file_path}")
        
        # Validate parse_dates
        if not hasattr(csv, 'parse_dates') or not csv.parse_dates:
            self.warnings.append("CSV parse_dates is empty, date parsing may fail")
        elif 'datetime' not in csv.parse_dates:
            self.warnings.append("'datetime' column not included in parse_dates")
    
    def _validate_learning_config(self):
        """Validate learning configuration"""
        learning = getattr(self.app_config, 'learning', None)
        if learning is None:
            self.errors.append("Learning configuration not found")
            return
        
        # Validate numeric parameters
        self._validate_positive_int('learning.timestamp', getattr(learning, 'timestamp', None), min_val=2)
        self._validate_positive_int('learning.simulation_size', getattr(learning, 'simulation_size', None), min_val=1)
        self._validate_positive_int('learning.epoch', getattr(learning, 'epoch', None), min_val=1)
        self._validate_positive_int('learning.batch_size', getattr(learning, 'batch_size', None), min_val=1)
        self._validate_positive_int('learning.num_layers', getattr(learning, 'num_layers', None), min_val=1, max_val=10)
        self._validate_positive_int('learning.size_layer', getattr(learning, 'size_layer', None), min_val=16)
        
        # Validate floating point parameters
        self._validate_probability('learning.dropout_rate', getattr(learning, 'dropout_rate', None))
        self._validate_positive_float('learning.learning_rate', getattr(learning, 'learning_rate', None))
        self._validate_positive_float('learning.l2_reg', getattr(learning, 'l2_reg', None))
        
        # Validate boolean parameters
        if not isinstance(getattr(learning, 'use_features', False), bool):
            self.errors.append("learning.use_features must be a boolean")
        
        if not isinstance(getattr(learning, 'use_log_transformation', False), bool):
            self.errors.append("learning.use_log_transformation must be a boolean")
        
        # Validate preset
        valid_presets = ["gpu-high-performance", "high-performance", "high", "medium", "low"]
        preset = getattr(learning, 'preset', None)
        if preset is not None and preset not in valid_presets:
            self.errors.append(f"learning.preset '{preset}' is not valid. Must be one of: {', '.join(valid_presets)}")
        
        # Check auto_batch_size and manual_batch_size
        auto_batch_size = getattr(learning, 'auto_batch_size', True)
        if not auto_batch_size:
            # If auto_batch_size is disabled, manual_batch_size must be valid
            self._validate_positive_int('learning.manual_batch_size', getattr(learning, 'manual_batch_size', None), min_val=1)
    
    def _validate_prediction_config(self):
        """Validate prediction configuration"""
        prediction = getattr(self.app_config, 'prediction', None)
        if prediction is None:
            self.errors.append("Prediction configuration not found")
            return
        
        # Validate numeric parameters
        self._validate_positive_int('prediction.predict_days', getattr(prediction, 'predict_days', None), min_val=1, max_val=365)
        
        # Validate boolean parameters
        if not isinstance(getattr(prediction, 'start_forecast_from_backtest', False), bool):
            self.errors.append("prediction.start_forecast_from_backtest must be a boolean")
        
        if not isinstance(getattr(prediction, 'use_previous_close', False), bool):
            self.errors.append("prediction.use_previous_close must be a boolean")
        
        # Validate initial data settings
        set_initial_data = getattr(prediction, 'set_initial_data', False)
        if not isinstance(set_initial_data, bool):
            self.errors.append("prediction.set_initial_data must be a boolean")
        
        if set_initial_data:
            learning = getattr(self.app_config, 'learning', None)
            initial_data_period = getattr(prediction, 'initial_data_period', None)
            timestamp = getattr(learning, 'timestamp', None) if learning else None
            
            self._validate_positive_int('prediction.initial_data_period', initial_data_period, min_val=1)
            
            # Check that initial_data_period matches timestamp
            if initial_data_period is not None and timestamp is not None and initial_data_period != timestamp:
                self.warnings.append(f"prediction.initial_data_period ({initial_data_period}) does not match learning.timestamp ({timestamp})")
    
    def _validate_features_config(self):
        """Validate feature selection configuration"""
        features = getattr(self.app_config, 'feature_selection', None)
        if features is None:
            self.errors.append("Feature selection configuration not found")
            return
        
        # Validate auto selection settings
        if not isinstance(getattr(features, 'auto_select_features', True), bool):
            self.errors.append("feature_selection.auto_select_features must be a boolean")
        
        # Validate number of features to select
        self._validate_positive_int('feature_selection.num_features_to_select', getattr(features, 'num_features_to_select', None), min_val=1)
        
        # Validate method
        valid_methods = ["importance", "mutual_info"]
        feature_method = getattr(features, 'feature_selection_method', None)
        if feature_method is not None and feature_method not in valid_methods:
            self.errors.append(f"feature_selection.feature_selection_method '{feature_method}' is not valid. Must be one of: {', '.join(valid_methods)}")
    
    def _validate_rolling_window_config(self):
        """Validate rolling window configuration"""
        rolling = getattr(self.app_config, 'rolling_window', None)
        if rolling is None:
            self.errors.append("Rolling window configuration not found")
            return
        
        # Only validate further if rolling window is enabled
        if not getattr(rolling, 'use_rolling_window', False):
            return
        
        # Validate numeric parameters
        self._validate_positive_int('rolling_window.window_size', getattr(rolling, 'window_size', None), min_val=5)
        self._validate_positive_int('rolling_window.step_size', getattr(rolling, 'step_size', None), min_val=1)
        self._validate_positive_int('rolling_window.min_train_size', getattr(rolling, 'min_train_size', None), min_val=10)
        self._validate_positive_int('rolling_window.refit_frequency', getattr(rolling, 'refit_frequency', None), min_val=1)
        
        # Check relationships between parameters
        window_size = getattr(rolling, 'window_size', 0)
        step_size = getattr(rolling, 'step_size', 0)
        min_train_size = getattr(rolling, 'min_train_size', 0)
        
        if window_size > 0 and step_size > 0 and step_size > window_size:
            self.warnings.append(f"rolling_window.step_size ({step_size}) is larger than window_size ({window_size})")
        
        if min_train_size > 0 and window_size > 0 and min_train_size < window_size:
            self.warnings.append(f"rolling_window.min_train_size ({min_train_size}) is smaller than window_size ({window_size})")
    
    def _validate_prediction_advanced_config(self):
        """Validate advanced prediction configuration"""
        adv = getattr(self.app_config, 'prediction_advanced', None)
        if adv is None:
            self.errors.append("Advanced prediction configuration not found")
            return
        
        # Validate ensemble methods
        use_ensemble = getattr(adv, 'use_ensemble_methods', False)
        if use_ensemble:
            # Validate ensemble method
            valid_methods = ["voting", "stacking", "bagging", "boosting"]
            ensemble_method = getattr(adv, 'ensemble_method', None)
            if ensemble_method is not None and ensemble_method not in valid_methods:
                self.errors.append(f"prediction_advanced.ensemble_method '{ensemble_method}' is not valid. Must be one of: {', '.join(valid_methods)}")
            
            # Validate ensemble models
            ensemble_models = getattr(adv, 'ensemble_models', [])
            valid_models = ["lstm", "gru", "bilstm", "transformer", "tcn"]
            invalid_models = [model for model in ensemble_models if model not in valid_models]
            if invalid_models:
                self.errors.append(f"Invalid ensemble models: {', '.join(invalid_models)}. Must be one of: {', '.join(valid_models)}")
            
            # Validate ensemble weights
            weights = getattr(adv, 'ensemble_weights', [])
            if weights:
                if sum(weights) <= 0:
                    self.errors.append("prediction_advanced.ensemble_weights sum must be positive")
                if len(weights) != len(ensemble_models):
                    self.warnings.append(f"prediction_advanced.ensemble_weights length ({len(weights)}) does not match ensemble_models length ({len(ensemble_models)})")
        
        # Validate uncertainty quantification
        uncertainty = getattr(adv, 'enable_uncertainty_quantification', False)
        if uncertainty:
            # Validate uncertainty method
            valid_methods = ["mc_dropout", "bootstrap", "quantile", "evidential"]
            uncertainty_method = getattr(adv, 'uncertainty_method', None)
            if uncertainty_method is not None and uncertainty_method not in valid_methods:
                self.errors.append(f"prediction_advanced.uncertainty_method '{uncertainty_method}' is not valid. Must be one of: {', '.join(valid_methods)}")
            
            # Validate method-specific parameters
            if uncertainty_method == "mc_dropout":
                self._validate_positive_int('prediction_advanced.mc_dropout_samples', getattr(adv, 'mc_dropout_samples', None), min_val=10)
            
            # Validate confidence level
            self._validate_probability('prediction_advanced.confidence_level', getattr(adv, 'confidence_level', None))
        
        # Validate Monte Carlo simulations
        monte_carlo = getattr(adv, 'enable_monte_carlo', False)
        if monte_carlo:
            self._validate_positive_int('prediction_advanced.num_monte_carlo_simulations', getattr(adv, 'num_monte_carlo_simulations', None), min_val=10)
            
            # Validate scenarios
            scenarios = getattr(adv, 'monte_carlo_scenarios', [])
            if not scenarios:
                self.warnings.append("prediction_advanced.monte_carlo_scenarios is empty")
            
            # Validate scenario parameters
            scenario_params = getattr(adv, 'scenario_parameters', {})
            for scenario in scenarios:
                if scenario not in scenario_params:
                    self.warnings.append(f"No parameters defined for scenario '{scenario}'")
                else:
                    params = scenario_params[scenario]
                    if len(params) != 2:
                        self.errors.append(f"Scenario '{scenario}' parameters must have 2 values (drift, volatility)")
                    elif params[0] <= 0 or params[1] <= 0:
                        self.errors.append(f"Scenario '{scenario}' parameters must be positive")
        
        # Validate rolling window in advanced prediction
        if getattr(adv, 'enable_rolling_window', False):
            # Check if main rolling window is also enabled
            if getattr(self.app_config, 'rolling_window', None) and getattr(self.app_config.rolling_window, 'use_rolling_window', False):
                self.warnings.append("Both prediction_advanced.enable_rolling_window and rolling_window.use_rolling_window are enabled")
            
            # Validate window parameters
            self._validate_positive_int('prediction_advanced.window_size', getattr(adv, 'window_size', None), min_val=5)
            self._validate_positive_int('prediction_advanced.step_size', getattr(adv, 'step_size', None), min_val=1)
            self._validate_positive_int('prediction_advanced.min_train_size', getattr(adv, 'min_train_size', None), min_val=10)
            self._validate_positive_int('prediction_advanced.refit_frequency', getattr(adv, 'refit_frequency', None), min_val=1)
    
    def _validate_strategy_config(self):
        """Validate strategy configuration"""
        strategy = getattr(self.app_config, 'strategy', None)
        if strategy is None:
            self.errors.append("Strategy configuration not found")
            return
        
        # Validate strategy type
        valid_strategies = [
            "buy_and_hold", "moving_average_crossover", "rsi_based", "macd_based", 
            "bollinger_bands", "trend_following", "mean_reversion", "breakout", "ml_optimized"
        ]
        strategy_type = getattr(strategy, 'strategy_type', None)
        if strategy_type is not None and strategy_type not in valid_strategies:
            self.errors.append(f"strategy.strategy_type '{strategy_type}' is not valid. Must be one of: {', '.join(valid_strategies)}")
        
        # Validate numeric parameters
        self._validate_positive_float('strategy.initial_capital', getattr(strategy, 'initial_capital', None))
        self._validate_positive_float('strategy.position_size_pct', getattr(strategy, 'position_size_pct', None))
        self._validate_positive_float('strategy.take_profit_pct', getattr(strategy, 'take_profit_pct', None))
        self._validate_positive_float('strategy.stop_loss_pct', getattr(strategy, 'stop_loss_pct', None))
        self._validate_positive_float('strategy.trailing_stop_pct', getattr(strategy, 'trailing_stop_pct', None))
        self._validate_positive_int('strategy.max_positions', getattr(strategy, 'max_positions', None), min_val=1)
        self._validate_positive_int('strategy.backtest_period', getattr(strategy, 'backtest_period', None), min_val=1)
        
        # Validate ML optimization settings
        ml_optimization = getattr(strategy, 'enable_ml_optimization', False)
        if ml_optimization:
            # Check ML model path if using saved model
            if getattr(strategy, 'use_saved_ml_model', False):
                ml_model_path = getattr(strategy, 'ml_model_path', None)
                if not ml_model_path:
                    self.errors.append("strategy.ml_model_path is empty but use_saved_ml_model is enabled")
                elif not os.path.exists(ml_model_path):
                    self.warnings.append(f"ML model file not found: {ml_model_path}")
            
            # Check ML features
            ml_features = getattr(strategy, 'ml_features', None)
            if not ml_features:
                self.warnings.append("strategy.ml_features is empty")
        
        # Check for strategy type and ML optimization compatibility
        if strategy_type == "ml_optimized" and not ml_optimization:
            self.warnings.append("strategy.strategy_type is 'ml_optimized' but enable_ml_optimization is False")
        
        # Validate optimization metric
        valid_metrics = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown", "total_return", "win_rate", "profit_factor"]
        optimization_metric = getattr(strategy, 'optimization_metric', None)
        if optimization_metric is not None and optimization_metric not in valid_metrics:
            self.errors.append(f"strategy.optimization_metric '{optimization_metric}' is not valid. Must be one of: {', '.join(valid_metrics)}")
    
    def _validate_cross_module_compatibility(self):
        """Validate compatibility between different modules"""
        # Check learning and prediction compatibility
        learning = getattr(self.app_config, 'learning', None)
        prediction = getattr(self.app_config, 'prediction', None)
        if learning is not None and prediction is not None:
            # Check timestamp and initial_data_period compatibility
            if getattr(prediction, 'set_initial_data', False):
                timestamp = getattr(learning, 'timestamp', 0)
                initial_period = getattr(prediction, 'initial_data_period', 0)
                if timestamp != initial_period:
                    self.warnings.append(f"learning.timestamp ({timestamp}) does not match prediction.initial_data_period ({initial_period})")
        
        # Check feature selection and learning compatibility
        features = getattr(self.app_config, 'feature_selection', None)
        if learning is not None and features is not None:
            use_features = getattr(learning, 'use_features', False)
            if not use_features:
                # If features are not used, warn about feature selection settings
                auto_select = getattr(features, 'auto_select_features', False)
                if auto_select:
                    self.warnings.append("feature_selection.auto_select_features is enabled but learning.use_features is disabled")
        
        # Check rolling window configurations for conflicts
        rolling = getattr(self.app_config, 'rolling_window', None)
        adv_pred = getattr(self.app_config, 'prediction_advanced', None)
        if rolling is not None and adv_pred is not None:
            if getattr(rolling, 'use_rolling_window', False) and getattr(adv_pred, 'enable_rolling_window', False):
                self.warnings.append("Both rolling_window.use_rolling_window and prediction_advanced.enable_rolling_window are enabled")
        
        # Check strategy and advanced prediction compatibility
        strategy = getattr(self.app_config, 'strategy', None)
        if strategy is not None and adv_pred is not None:
            if getattr(strategy, 'enable_ml_optimization', False) and getattr(adv_pred, 'enable_uncertainty_quantification', False):
                self.warnings.append("Both strategy.enable_ml_optimization and prediction_advanced.enable_uncertainty_quantification are enabled")
    
    def _validate_positive_int(self, name, value, min_val=None, max_val=None):
        """Validate that a value is a positive integer within the specified range"""
        if value is None:
            self.errors.append(f"{name} is missing")
            return False
        
        try:
            value = int(value)
            if value <= 0:
                self.errors.append(f"{name} must be positive, got {value}")
                return False
            
            if min_val is not None and value < min_val:
                self.errors.append(f"{name} must be at least {min_val}, got {value}")
                return False
            
            if max_val is not None and value > max_val:
                self.errors.append(f"{name} must be at most {max_val}, got {value}")
                return False
            
            return True
        except (ValueError, TypeError):
            self.errors.append(f"{name} must be an integer, got {value}")
            return False
    
    def _validate_positive_float(self, name, value, min_val=None, max_val=None):
        """Validate that a value is a positive float within the specified range"""
        if value is None:
            self.errors.append(f"{name} is missing")
            return False
        
        try:
            value = float(value)
            if value <= 0:
                self.errors.append(f"{name} must be positive, got {value}")
                return False
            
            if min_val is not None and value < min_val:
                self.errors.append(f"{name} must be at least {min_val}, got {value}")
                return False
            
            if max_val is not None and value > max_val:
                self.errors.append(f"{name} must be at most {max_val}, got {value}")
                return False
            
            return True
        except (ValueError, TypeError):
            self.errors.append(f"{name} must be a number, got {value}")
            return False
    
    def _validate_probability(self, name, value):
        """Validate that a value is a probability (between 0 and 1)"""
        if value is None:
            self.errors.append(f"{name} is missing")
            return False
        
        try:
            value = float(value)
            if value < 0 or value > 1:
                self.errors.append(f"{name} must be between 0 and 1, got {value}")
                return False
            
            return True
        except (ValueError, TypeError):
            self.errors.append(f"{name} must be a number, got {value}")
            return False


def validate_config(app_config) -> Tuple[bool, List[str], List[str]]:
    """
    Validate application configuration
    
    Args:
        app_config: Application configuration object
        
    Returns:
        Tuple[bool, List[str], List[str]]: (success, errors, warnings)
    """
    validator = ConfigValidator(app_config)
    success = validator.validate_all()
    errors, warnings = validator.get_results()
    
    return success, errors, warnings


def validate_model_input(inputs, sequence_length, num_features):
    """
    Validate model input shape
    
    Args:
        inputs: Model inputs
        sequence_length: Expected sequence length
        num_features: Expected number of features
        
    Returns:
        bool: Whether the input is valid
        
    Raises:
        ValueError: If input shape is invalid
    """
    import numpy as np
    
    # Check for None
    if inputs is None:
        raise ValueError("Model input is None")
    
    # Convert to numpy array if needed
    if not isinstance(inputs, np.ndarray):
        try:
            inputs = np.array(inputs)
        except:
            raise ValueError(f"Cannot convert input to numpy array: {type(inputs)}")
    
    # Check shape
    if len(inputs.shape) != 3:
        raise ValueError(f"Expected 3D input (batch, sequence, features), got shape {inputs.shape}")
    
    batch_size, actual_sequence_length, actual_features = inputs.shape
    
    if actual_sequence_length != sequence_length:
        raise ValueError(f"Expected sequence length {sequence_length}, got {actual_sequence_length}")
    
    if actual_features != num_features:
        raise ValueError(f"Expected {num_features} features, got {actual_features}")
    
    # Check for NaN or infinity
    if np.isnan(inputs).any() or np.isinf(inputs).any():
        raise ValueError("Input contains NaN or infinite values")
    
    return True


def check_gpu_compatibility(app_config):
    """
    Check GPU compatibility for the application configuration
    
    Args:
        app_config: Application configuration object
        
    Returns:
        Tuple[bool, str]: Whether GPUs are compatible and status message
    """
    try:
        import tensorflow as tf
        
        # Check if GPU is required
        gpu_required = getattr(app_config, 'learning_pref', None) and getattr(app_config.learning_pref, 'use_gpu_if_available', False)
        
        if not gpu_required:
            return True, "GPU not required"
        
        # List physical devices
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            return False, "No GPUs found but GPU is required"
        
        # Check if GPUs can be used
        for gpu in gpus:
            try:
                # Try to configure memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                return False, f"GPU {gpu.name} initialization failed: {e}"
        
        # Advanced prediction settings that might need more GPU memory
        advanced_enabled = False
        high_memory_usage = False
        
        if hasattr(app_config, 'prediction_advanced'):
            adv = app_config.prediction_advanced
            if getattr(adv, 'use_ensemble_methods', False):
                advanced_enabled = True
                ensemble_size = len(getattr(adv, 'ensemble_models', []))
                if ensemble_size > 2:
                    high_memory_usage = True
            
            if getattr(adv, 'enable_uncertainty_quantification', False):
                advanced_enabled = True
                uncertainty_method = getattr(adv, 'uncertainty_method', '')
                if uncertainty_method in ['mc_dropout', 'bootstrap']:
                    high_memory_usage = True
            
            if getattr(adv, 'enable_monte_carlo', False):
                advanced_enabled = True
                num_simulations = getattr(adv, 'num_monte_carlo_simulations', 0)
                if num_simulations > 500:
                    high_memory_usage = True
        
        memory_warning = ""
        if advanced_enabled and high_memory_usage:
            memory_warning = " (Advanced features may require significant GPU memory)"
        
        return True, f"Found {len(gpus)} GPUs: {', '.join(gpu.name for gpu in gpus)}{memory_warning}"
    
    except Exception as e:
        return False, f"GPU compatibility check failed: {e}"