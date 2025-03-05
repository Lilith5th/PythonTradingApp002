"""
Configuration Utilities for Stock Prediction Application

This module provides standardized functions for:
- Updating configuration from GUI components
- Validating configuration values
- Synchronizing related configuration parameters
- Converting between different data types
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import traceback
from dataclasses import fields, is_dataclass
from typing import Dict, Any, List, Tuple, Union, Optional

def parse_bool_str(value):
    """
    Convert various 'true/false' string forms to boolean
    
    Args:
        value: String or boolean value
        
    Returns:
        bool: Boolean equivalent of value
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == 'true'

def validate_numeric_range(value, min_val=None, max_val=None, name="Value"):
    """
    Validate that a numeric value is within the specified range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        name: Name of the value for error messages
        
    Returns:
        float: Validated value
        
    Raises:
        ValueError: If value is outside the allowed range
    """
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a number, got '{value}'")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be at least {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be at most {max_val}, got {value}")
    
    return value

def validate_int_range(value, min_val=None, max_val=None, name="Value"):
    """
    Validate that an integer value is within the specified range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        name: Name of the value for error messages
        
    Returns:
        int: Validated value
        
    Raises:
        ValueError: If value is outside the allowed range
    """
    # First validate as numeric
    value = validate_numeric_range(value, min_val, max_val, name)
    
    # Then check that it's an integer
    if int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value}")
    
    return int(value)

def validate_positive(value, name="Value"):
    """
    Validate that a value is positive
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        float: Validated value
        
    Raises:
        ValueError: If value is not positive
    """
    return validate_numeric_range(value, min_val=0, name=name)

def validate_probability(value, name="Probability"):
    """
    Validate that a value is a valid probability (between 0 and 1)
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        float: Validated value
        
    Raises:
        ValueError: If value is not a valid probability
    """
    return validate_numeric_range(value, min_val=0, max_val=1, name=name)

def validate_list_values(values, validator_func, name="Values"):
    """
    Validate each value in a list using the specified validator function
    
    Args:
        values: List of values to validate
        validator_func: Function to validate each value
        name: Name of the values for error messages
        
    Returns:
        list: List of validated values
        
    Raises:
        ValueError: If any value fails validation
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple, got {type(values)}")
    
    return [validator_func(value, f"{name}[{i}]") for i, value in enumerate(values)]

def parse_comma_separated_values(text, converter=float, validator=None):
    """
    Parse a comma-separated string into a list of values
    
    Args:
        text: Comma-separated string
        converter: Function to convert each value
        validator: Function to validate each value (optional)
        
    Returns:
        list: List of converted values
        
    Raises:
        ValueError: If any value fails conversion or validation
    """
    if not text:
        return []
    
    try:
        values = [converter(item.strip()) for item in text.split(',')]
        if validator:
            values = [validator(value) for value in values]
        return values
    except ValueError as e:
        raise ValueError(f"Invalid value in list: {e}")

def update_config_from_gui(entries, app_config, validate=True):
    """
    Update configuration from GUI entries with validation
    
    Args:
        entries: Dictionary of GUI entries
        app_config: Application configuration object
        validate: Whether to validate values before updating
        
    Returns:
        bool: True if update was successful, False otherwise
        list: List of validation errors
    """
    errors = []
    
    try:
        # Process each section
        for section, entries_dict in entries.items():
            # Skip sections with no entries
            if not entries_dict:
                continue
                
            # Get the corresponding config object
            try:
                config_obj = getattr(app_config, section)
            except AttributeError:
                errors.append(f"No '{section}' section in configuration")
                continue
                
            # Skip configuration if overridden
            if section == "features" and hasattr(app_config, "learning") and getattr(app_config.learning, "use_features", False):
                logging.info(f"Skipping '{section}' configuration (override enabled)")
                continue
                
            # Handle special tabs
            if section == "advanced_prediction" and hasattr(entries_dict, "update_config"):
                try:
                    entries_dict.update_config(config_obj)
                except Exception as e:
                    errors.append(f"Error updating '{section}' configuration: {str(e)}")
                continue
                
            # Update each field
            if hasattr(entries_dict, "get_config") or hasattr(entries_dict, "update_config"):
                # The tab module provides a method to get/update config
                try:
                    if hasattr(entries_dict, "update_config"):
                        entries_dict.update_config(config_obj)
                    elif hasattr(entries_dict, "get_config"):
                        new_config = entries_dict.get_config()
                        for key, value in vars(new_config).items():
                            if hasattr(config_obj, key):
                                setattr(config_obj, key, value)
                except Exception as e:
                    errors.append(f"Error updating '{section}' configuration: {str(e)}")
            else:
                # Default field-by-field update with validation
                try:
                    update_config_object(config_obj, entries_dict, validate=validate)
                except Exception as e:
                    errors.append(f"Error updating '{section}' configuration: {str(e)}")
        
        # Process special fields that need synchronization
        synchronize_config(app_config)
        
        if errors:
            logging.error(f"Configuration update completed with {len(errors)} errors")
            return False, errors
        else:
            logging.info("Configuration updated successfully")
            return True, []
    except Exception as e:
        logging.error(f"Error updating configuration: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        errors.append(f"Unexpected error: {str(e)}")
        return False, errors

def update_config_object(obj, entries_dict, validate=True):
    """
    Update a configuration object from GUI entries
    
    Args:
        obj: Configuration object to update
        entries_dict: Dictionary of GUI entries
        validate: Whether to validate values before updating
        
    Raises:
        ValueError: If a value fails validation
    """
    if not is_dataclass(obj):
        raise ValueError(f"Object {obj} is not a dataclass")
    
    for f in fields(obj):
        if f.name not in entries_dict:
            continue
            
        value = entries_dict[f.name]
        current_val = getattr(obj, f.name)
        
        # Handle different widget types
        if isinstance(value, tk.BooleanVar):
            setattr(obj, f.name, value.get())
        elif isinstance(value, ttk.Combobox):
            new_val = value.get()
            if validate:
                new_val = validate_field_value(new_val, current_val, f.name)
            setattr(obj, f.name, new_val)
        elif isinstance(value, ttk.Entry):
            new_val = value.get()
            if validate:
                new_val = validate_field_value(new_val, current_val, f.name)
            setattr(obj, f.name, new_val)
        elif isinstance(value, dict) and f.name in ["ml_features", "model_vars", "metric_vars"]:
            # Handle special case of feature checkboxes
            selected = [k for k, v in value.items() if v.get()]
            setattr(obj, f.name, selected)
        elif hasattr(value, 'get') and callable(value.get):
            try:
                new_val = value.get()
                if validate:
                    new_val = validate_field_value(new_val, current_val, f.name)
                setattr(obj, f.name, new_val)
            except Exception as e:
                logging.warning(f"Error getting value for {f.name}: {str(e)}")
        else:
            logging.warning(f"Couldn't update {f.name}: Unsupported widget type {type(value)}")

def validate_field_value(new_val, current_val, field_name):
    """
    Validate and convert a field value based on the current value's type
    
    Args:
        new_val: New value from GUI
        current_val: Current value in configuration
        field_name: Name of the field
        
    Returns:
        Any: Validated and converted value
        
    Raises:
        ValueError: If validation fails
    """
    if new_val is None or new_val == "":
        # Allow empty strings for string fields
        if isinstance(current_val, str):
            return new_val
        else:
            raise ValueError(f"{field_name} cannot be empty")
    
    # Convert based on the type of the current value
    if isinstance(current_val, bool):
        return parse_bool_str(new_val)
    elif isinstance(current_val, int):
        # Special validation for specific fields
        if field_name in ["timestamp", "predict_days", "simulation_size", "epoch", "batch_size", "window_size"]:
            return validate_int_range(new_val, min_val=1, name=field_name)
        elif field_name in ["num_layers", "size_layer"]:
            return validate_int_range(new_val, min_val=1, max_val=10, name=field_name)
        else:
            return int(new_val)
    elif isinstance(current_val, float):
        # Special validation for specific fields
        if field_name in ["learning_rate", "l2_reg"]:
            return validate_positive(new_val, name=field_name)
        elif field_name in ["dropout_rate", "confidence_level"]:
            return validate_probability(new_val, name=field_name)
        else:
            return float(new_val)
    elif isinstance(current_val, list):
        # Parse comma-separated list
        if isinstance(new_val, str):
            if field_name in ["ensemble_weights"]:
                weights = parse_comma_separated_values(new_val, float, validate_positive)
                # Normalize weights to sum to 1
                if weights:
                    weights_sum = sum(weights)
                    if weights_sum > 0:
                        return [w / weights_sum for w in weights]
                return weights
            else:
                return parse_comma_separated_values(new_val)
        return new_val
    else:
        # Default: return as is
        return new_val

def synchronize_config(app_config):
    """
    Synchronize related configuration parameters
    
    Args:
        app_config: Application configuration object
    """
    # Synchronize Learning and Prediction tabs
    if hasattr(app_config, "learning") and hasattr(app_config, "prediction"):
        learning = app_config.learning
        prediction = app_config.prediction
        
        # Synchronize timestamp and initial_data_period
        if getattr(prediction, "set_initial_data", False):
            prediction.initial_data_period = learning.timestamp
        
    # Synchronize ML optimization settings
    if hasattr(app_config, "strategy") and hasattr(app_config, "prediction_advanced"):
        strategy = app_config.strategy
        pred_adv = app_config.prediction_advanced
        
        # Enable advanced features if ML optimization is enabled
        if strategy.strategy_type == "ml_optimized":
            strategy.enable_ml_optimization = True
            
        # Ensure ML optimization is enabled for ML-based strategies
        if strategy.strategy_type == "ml_optimized" and not strategy.enable_ml_optimization:
            logging.warning("ML strategy selected but ML optimization disabled. Enabling ML optimization.")
            strategy.enable_ml_optimization = True
            
    # Synchronize batch size settings
    if hasattr(app_config, "learning"):
        learning = app_config.learning
        
        # If auto batch size is enabled, ignore manual batch size
        if learning.auto_batch_size:
            # Don't change manual_batch_size, just log
            logging.debug("Auto batch size enabled, manual batch size will be ignored")

def validate_config(app_config):
    """
    Validate the entire configuration
    
    Args:
        app_config: Application configuration object
        
    Returns:
        list: List of validation errors
    """
    errors = []
    
    # Validate learning configuration
    if hasattr(app_config, "learning"):
        learning = app_config.learning
        
        try:
            validate_int_range(learning.timestamp, min_val=1, name="timestamp")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(learning.simulation_size, min_val=1, name="simulation_size")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(learning.epoch, min_val=1, name="epoch")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(learning.batch_size, min_val=1, name="batch_size")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(learning.num_layers, min_val=1, max_val=10, name="num_layers")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(learning.size_layer, min_val=1, name="size_layer")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_probability(learning.dropout_rate, name="dropout_rate")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_positive(learning.learning_rate, name="learning_rate")
        except ValueError as e:
            errors.append(str(e))
    
    # Validate prediction configuration
    if hasattr(app_config, "prediction"):
        prediction = app_config.prediction
        
        try:
            validate_int_range(prediction.predict_days, min_val=1, max_val=365, name="predict_days")
        except ValueError as e:
            errors.append(str(e))
            
        if prediction.set_initial_data:
            try:
                validate_int_range(prediction.initial_data_period, min_val=1, name="initial_data_period")
            except ValueError as e:
                errors.append(str(e))
    
    # Validate rolling window configuration
    if hasattr(app_config, "rolling_window") and app_config.rolling_window.use_rolling_window:
        rolling = app_config.rolling_window
        
        try:
            validate_int_range(rolling.window_size, min_val=10, name="window_size")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(rolling.step_size, min_val=1, name="step_size")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(rolling.min_train_size, min_val=10, name="min_train_size")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(rolling.refit_frequency, min_val=1, name="refit_frequency")
        except ValueError as e:
            errors.append(str(e))
    
    # Validate advanced prediction configuration
    if hasattr(app_config, "prediction_advanced"):
        adv = app_config.prediction_advanced
        
        if adv.enable_uncertainty_quantification:
            try:
                validate_probability(adv.confidence_level, name="confidence_level")
            except ValueError as e:
                errors.append(str(e))
                
            if adv.uncertainty_method == "mc_dropout":
                try:
                    validate_int_range(adv.mc_dropout_samples, min_val=10, name="mc_dropout_samples")
                except ValueError as e:
                    errors.append(str(e))
        
        if adv.enable_monte_carlo:
            try:
                validate_int_range(adv.num_monte_carlo_simulations, min_val=10, name="num_monte_carlo_simulations")
            except ValueError as e:
                errors.append(str(e))
    
    # Validate strategy configuration
    if hasattr(app_config, "strategy"):
        strategy = app_config.strategy
        
        try:
            validate_positive(strategy.initial_capital, name="initial_capital")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_positive(strategy.position_size_pct, name="position_size_pct")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_positive(strategy.take_profit_pct, name="take_profit_pct")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_positive(strategy.stop_loss_pct, name="stop_loss_pct")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_positive(strategy.trailing_stop_pct, name="trailing_stop_pct")
        except ValueError as e:
            errors.append(str(e))
            
        try:
            validate_int_range(strategy.max_positions, min_val=1, name="max_positions")
        except ValueError as e:
            errors.append(str(e))
    
    return errors

def update_gui_from_config(entries, app_config):
    """
    Update GUI components from configuration
    
    Args:
        entries: Dictionary of GUI entries
        app_config: Application configuration object
    """
    # Update each section
    for section, entries_dict in entries.items():
        # Get the corresponding config object
        try:
            config_obj = getattr(app_config, section)
        except AttributeError:
            logging.warning(f"No '{section}' section in configuration")
            continue
            
        # Update each field
        if is_dataclass(config_obj):
            for f in fields(config_obj):
                if f.name not in entries_dict:
                    continue
                    
                value = getattr(config_obj, f.name)
                widget = entries_dict[f.name]
                
                if isinstance(widget, tk.BooleanVar):
                    widget.set(value)
                elif isinstance(widget, ttk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(value))
                elif isinstance(widget, ttk.Combobox):
                    widget.set(value)
                elif isinstance(widget, dict) and f.name in ["ml_features", "model_vars", "metric_vars"]:
                    # Handle special case of feature checkboxes
                    for k, v in widget.items():
                        v.set(k in value)
                elif hasattr(widget, 'set') and callable(widget.set):
                    try:
                        widget.set(value)
                    except Exception as e:
                        logging.warning(f"Error setting value for {f.name}: {str(e)}")
        else:
            logging.warning(f"Config object for '{section}' is not a dataclass")