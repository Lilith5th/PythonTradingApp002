# Migration Guide for Stock Prediction Application

This guide documents the changes made to the stock prediction application and provides guidance on how to migrate existing code to use the new improved components.

## Table of Contents

- [Overview](#overview)
- [Key Issues Addressed](#key-issues-addressed)
- [New Components](#new-components)
- [Migration Steps](#migration-steps)
- [Before and After Code Examples](#before-and-after-code-examples)
- [Testing Changes](#testing-changes)
- [FAQ](#faq)

## Overview

The stock prediction application has been refactored to address several inconsistencies, code duplications, and potential issues. The changes focus on:

1. Standardizing configuration updates
2. Improving tab synchronization
3. Enhancing error handling
4. Consolidating duplicate functionality
5. Creating robust model and dataset factories
6. Implementing proper uncertainty quantification

These improvements make the application more maintainable, reliable, and easier to extend without breaking existing functionality.

## Key Issues Addressed

### 1. GUI Structure Inconsistencies

- **Tab Synchronization**: Different ways of handling linked parameters across tabs (e.g., `timestamp` in Learning and `initial_data_period` in Prediction)
- **Inconsistent Tab States**: Inconsistent logic for enabling/disabling tabs based on settings
- **Feature Override**: Different implementations for handling feature engineering override

### 2. Configuration Duplication and Conflicts

- **Duplicate Settings**: Parameters like `batch_size` defined and managed in multiple ways
- **Conflicting Logic**: Different methods for checking the same conditions (e.g., feature override)

### 3. Machine Learning Implementation Issues

- **Empty Implementation Files**: Referenced functionality with no implementation
- **Inconsistent Model Architecture**: Different model creation methods across files
- **Inconsistent Feature Importance**: Multiple implementations calculating the same metric

### 4. Error Handling and Validation Issues

- **Inconsistent GPU Configuration**: Different GPU handling in various parts
- **Path Handling Issues**: Insufficient validation of file paths
- **Limited Fallback Mechanisms**: Inadequate handling of failures

### 5. Logic Disconnects

- **Configuration and Implementation Gaps**: UI components for features not implemented
- **ML Feature Selection Issues**: Lack of validation for selected features

## New Components

### 1. Configuration Utilities (`config_utils.py`)

A standardized system for:
- Updating configuration from GUI components
- Validating configuration values
- Synchronizing related parameters
- Converting between data types

### 2. Tab Synchronizer (`tab_synchronizer.py`)

Manages synchronization between tabs with features for:
- Registering related fields across tabs
- Automatically updating related settings
- Handling state changes in tabs

### 3. Error Handler (`error_handler.py`)

Comprehensive error handling with:
- Standardized error reporting
- Decorators for function-level error handling
- Error-aware threads
- Validation utilities

### 4. Model Factory (`model_factory.py`)

A factory pattern for model creation:
- Standardized model creation for all types
- Consistent parameter handling
- Support for ensemble models

### 5. Dataset Factory (`dataset_factory.py`)

Standardized dataset creation:
- Consistent dataset creation across components
- Validation of input data
- Support for various dataset types (time series, bootstrap, sliding window)

### 6. Uncertainty Quantification (`uncertainty.py`)

Robust implementation of uncertainty methods:
- Monte Carlo Dropout
- Bootstrap Sampling
- Quantile Regression
- Evidential Regression

## Migration Steps

### Step 1: Replace Manual Configuration Updates

**Old approach:**
```python
def update_config_from_gui(entries, app_config):
    try:
        # Process each configuration section
        for section, config_obj in {
            "preferences": app_config.learning_pref,
            "learning": app_config.learning,
            # ...
        }.items():
            if section not in entries:
                continue
            # Update each field
            update_section_config(entries[section], config_obj)
        return True
    except Exception as e:
        logging.error(f"Error updating configuration: {str(e)}")
        messagebox.showerror("Configuration Error", f"Invalid input: {e}")
        return False
```

**New approach:**
```python
from config_utils import update_config_from_gui

success, errors = update_config_from_gui(entries, app_config)
if not success:
    # Handle errors
    for error in errors:
        logging.error(f"Configuration error: {error}")
    if errors:
        messagebox.showerror("Configuration Error", "\n".join(errors[:5]))
```

### Step 2: Implement Tab Synchronization

**Old approach:**
```python
# Manual synchronization in different places
def update_initial_data_period(*args):
    try:
        timestamp_value = int(widget.get())
        app_config.learning.timestamp = timestamp_value
        app_config.prediction.initial_data_period = timestamp_value
        # Update Prediction tab UI
        if 'prediction' in entries:
            entries['prediction']['initial_data_period'].delete(0, tk.END)
            entries['prediction']['initial_data_period'].insert(0, str(timestamp_value))
    except ValueError as e:
        logging.warning(f"Invalid timestamp value: {e}")
```

**New approach:**
```python
from tab_synchronizer import SynchronizedTabManager

# Initialize the tab manager
tab_manager = SynchronizedTabManager(root, app_config)
notebook = tab_manager.create_notebook()

# Add tabs
tab_manager.add_tab("learning", learning_tab, app_config.learning)
tab_manager.add_tab("prediction", prediction_tab, app_config.prediction)
# ... add other tabs

# Set up synchronization
tab_manager.setup_synchronization()

# Update config from GUI
success, errors = tab_manager.update_config_from_gui()
```

### Step 3: Use Error Handling

**Old approach:**
```python
def run_forecast_thread():
    try:
        # Run the forecast
        result = run_forecast_func(app_config, temp_file_path)
        # Process the result
        root.after(0, lambda: handle_forecast_result(result, temp_file_path))
    except Exception as e:
        logging.error(f"Error running forecast: {e}")
        root.after(0, lambda: messagebox.showerror("Error", f"Forecast failed: {e}"))
```

**New approach:**
```python
from error_handler import ErrorAwareThread, get_error_handler

def run_forecast_thread():
    # Initialize error handler
    error_handler = get_error_handler(root)
    
    # Create and start error-aware thread
    thread = ErrorAwareThread(
        target=run_forecast_func,
        args=(app_config, temp_file_path),
        show_dialog=True,
        context="Forecast execution"
    )
    thread.start()
    
    # Schedule result handling
    def check_thread():
        if not thread.is_alive():
            if thread.error:
                # Error already handled by ErrorAwareThread
                status_var.set("Forecast failed")
            else:
                result = thread.get_result()
                handle_forecast_result(result, temp_file_path)
        else:
            # Check again later
            root.after(100, check_thread)
    
    root.after(100, check_thread)
```

### Step 4: Use Model Factory

**Old approach:**
```python
def create_mc_dropout_model(self, num_features: int) -> tf.keras.Model:
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
```

**New approach:**
```python
from model_factory import ModelFactory

# Create a model
model = ModelFactory.create_model(
    model_type='mc_dropout',
    config=self.app_config.learning,
    input_shape=(self.app_config.learning.timestamp, len(features))
)

# For ensemble models
ensemble_models = ModelFactory.create_ensemble_models(
    model_types=['lstm', 'gru', 'bilstm'],
    config=self.app_config.learning,
    input_shape=(self.app_config.learning.timestamp, len(features)),
    weights=[0.5, 0.3, 0.2]
)
```

### Step 5: Use Dataset Factory

**Old approach:**
```python
def create_dataset(self, data, timestamp, batch_size):
    X, y = [], []
    for i in range(len(data) - timestamp):
        X.append(data[i:i+timestamp])
        y.append(data[i+timestamp][-1])
    X = np.array(X)
    y = np.array(y)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
```

**New approach:**
```python
from dataset_factory import DatasetFactory

# Create a dataset
dataset = DatasetFactory.create_dataset(
    data=train_data,
    timestamp=self.app_config.learning.timestamp,
    batch_size=self.app_config.learning.batch_size,
    target_idx=self.data_handler.features.index('close'),
    auto_batch_size=self.app_config.learning.auto_batch_size
)

# Split into train/validation
train_dataset, val_dataset, _ = DatasetFactory.split_train_val_test(
    dataset=dataset,
    val_ratio=0.2,
    test_ratio=0.0
)
```

### Step 6: Use Uncertainty Module

**Old approach:**
```python
def predict_with_mc_dropout(self, model, train_data, features):
    input_sequence = train_data[-self.app_config.learning.timestamp:]
    input_reshaped = input_sequence.reshape(1, self.app_config.learning.timestamp, len(features))
    prediction = model(input_reshaped, training=True).numpy()
    return prediction.flatten()
```

**New approach:**
```python
from uncertainty import UncertaintyFactory

# Create uncertainty quantifier
uncertainty_method = self.app_config.prediction_advanced.uncertainty_method
uncertainty = UncertaintyFactory.create(
    method=uncertainty_method,
    model=model,
    config=self.app_config.prediction_advanced
)

# Quantify uncertainty
result = uncertainty.quantify_uncertainty(
    sequence_data=train_data[-self.app_config.learning.timestamp:],
    features=features
)

# Use the results
mean_prediction = result['mean']
lower_bound = result['lower']
upper_bound = result['upper']
```

## Before and After Code Examples

### Example 1: Running a Forecast

**Before:**
```python
def on_run():
    if not update_config_from_gui(entries, app_config):
        return
    btn_run.config(state="disabled")
    status_var.set("Preparing forecast...")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
    devices = configure_device_resources(app_config, num_forecasts=1)
    device = devices[0]
    def run_forecast():
        process = multiprocessing.Process(
            target=run_forecast_in_subprocess,
            args=(app_config, temp_file_path, device)
        )
        process.start()
        process.join()
        root.after(0, lambda: handle_forecast_result(process.exitcode, temp_file_path))
    threading.Thread(target=run_forecast, daemon=True).start()
```

**After:**
```python
from error_handler import ErrorAwareThread, get_error_handler
from config_utils import update_config_from_gui

def on_run():
    # Initialize error handler
    error_handler = get_error_handler(root)
    
    # Update configuration with validation
    success, errors = update_config_from_gui(entries, app_config)
    if not success:
        error_handler.show_error(
            title="Configuration Error",
            message="Invalid configuration settings",
            detail="\n".join(errors)
        )
        return
    
    btn_run.config(state="disabled")
    status_var.set("Preparing forecast...")
    
    # Validate GPU availability
    from error_handler import validate_gpu_availability
    try:
        gpu_available, message = validate_gpu_availability()
        if not gpu_available:
            logging.warning(f"GPU not available: {message}")
            status_var.set("GPU not available, using CPU fallback")
    except Exception as e:
        error_handler.log_error(e, "GPU validation")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    # Run forecast in error-aware thread
    thread = ErrorAwareThread(
        target=run_forecast_func,
        args=(app_config, temp_file_path),
        show_dialog=True,
        context="Forecast execution"
    )
    thread.start()
    
    # Schedule result handling
    def check_thread():
        if not thread.is_alive():
            if thread.error:
                status_var.set("Forecast failed")
                btn_run.config(state="normal")
            else:
                result = thread.get_result()
                handle_forecast_result(result, temp_file_path)
        else:
            # Check again later
            root.after(100, check_thread)
    
    root.after(100, check_thread)
```

### Example 2: Creating a Model

**Before:**
```python
def _create_bidirectional_lstm_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
    inputs = Input(shape=input_shape)
    x = inputs
    
    for i in range(num_layers):
        x = Bidirectional(LSTM(
            units=size_layer // 2,
            return_sequences=(i < num_layers - 1),
            kernel_regularizer=l2(l2_reg_val)
        ))(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    return Model(inputs=inputs, outputs=outputs)
```

**After:**
```python
from model_factory import ModelFactory

# Create model
model = ModelFactory.create_model(
    model_type='bilstm',
    config=app_config.learning,
    input_shape=(app_config.learning.timestamp, len(features))
)
```

### Example 3: Error Handling

**Before:**
```python
try:
    prediction = model.predict(input_reshaped)
except Exception as e:
    logging.error(f"Prediction failed: {e}")
    return np.zeros(self.app_config.prediction.predict_days)
```

**After:**
```python
from error_handler import catch_and_log_errors

@catch_and_log_errors(show_dialog=True, context="Model prediction")
def predict_sequence(model, input_sequence):
    reshaped = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])
    prediction = model.predict(reshaped)
    return prediction.flatten()

# Call the protected function
prediction = predict_sequence(model, input_sequence)
if prediction is None:  # Error occurred
    # Return fallback prediction
    prediction = np.zeros(self.app_config.prediction.predict_days)
```

## Testing Changes

After migrating your code, test the following scenarios:

1. **Basic Functionality**: Ensure core forecasting still works
2. **Tab Synchronization**: Verify related parameters stay in sync
3. **Error Handling**: Test error conditions to ensure proper handling
4. **Model Creation**: Test different model types
5. **Dataset Creation**: Verify datasets are created correctly
6. **Uncertainty Quantification**: Test different uncertainty methods

## FAQ

### How do I know which parts need migration?

Start with the main entry points (`gui_wrapper_gpu.py`, `forecaster.py`, `advanced_forecaster.py`) and work through the codebase following dependencies.

### Can I partially migrate my code?

Yes, you can start by using individual components like the `ErrorHandler` or `ModelFactory` before implementing the full migration.

### How do I handle custom model types?

You can extend `ModelFactory` by adding your custom model creation method:

```python
class CustomModelFactory(ModelFactory):
    @staticmethod
    def create_model(model_type, config, input_shape):
        if model_type == 'my_custom_model':
            return CustomModelFactory._create_custom_model(input_shape, config)
        return ModelFactory.create_model(model_type, config, input_shape)
    
    @staticmethod
    def _create_custom_model(input_shape, config):
        # Your custom model implementation
        pass
```

### My code uses different dataset creation logic. How should I migrate?

Use the `DatasetFactory` as a base and extend it if needed:

```python
class CustomDatasetFactory(DatasetFactory):
    @staticmethod
    def create_custom_dataset(data, config):
        # Your custom dataset creation logic
        pass
```

### How do I handle legacy code that isn't migrated yet?

Use adapter patterns to bridge between old and new code:

```python
def legacy_adapter(old_function, new_parameters):
    # Convert new parameters to old format
    old_parameters = convert_params(new_parameters)
    
    # Call legacy function
    result = old_function(old_parameters)
    
    # Convert result to new format
    return convert_result(result)
```
