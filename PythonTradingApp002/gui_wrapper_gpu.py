import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
import logging
import sys
import threading
import multiprocessing
import pickle
import tempfile
import traceback
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gui import plot_gui
from core.data_classes import StockData, ForecastResults
from config.config import (
    AppConfig, RootWidgetsConfig, PreferencesConfig, 
    FeaturesConfig, AdvancedPredictionConfig
)

# Import necessary modules for GUI tabs
from gui.tabs import (
    preferences_tab, learning_tab, prediction_tab,
    plot_tab, features_tab, rolling_window_tab, 
    advanced_prediction_tab, strategy_tab
)

# Import utility modules
from config.config_utils import update_config_from_gui, validate_config
from gui.tab_synchronizer import SynchronizedTabManager
from utils.error_handler import (
    get_error_handler, handle_errors, catch_and_log_errors, 
    ErrorAwareThread, validate_gpu_availability
)
from stock_predictor.dataset_factory import DatasetFactory
from core.model_factory import ModelFactory

matplotlib.use('TkAgg')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

@handle_errors(context="GPU Configuration")
def configure_device_resources(app_config, num_forecasts=1):
    """
    GPU-optimized resource allocation for forecasting with improved error handling.
    
    Args:
        app_config: Application configuration
        num_forecasts: Number of parallel forecasts to run
        
    Returns:
        List of device strings to use for parallel processing
        
    Raises:
        RuntimeError: If GPU initialization fails
    """
    # First validate GPU availability
    gpu_available, status_message = validate_gpu_availability()
    if not gpu_available:
        if app_config.learning_pref.use_gpu_if_available:
            raise RuntimeError(f"GPU initialization failed: {status_message}. Use CPU module instead.")
        else:
            logging.warning(f"GPU not available: {status_message}. Using CPU anyway as specified in preferences.")
            return ['/CPU:0'] * num_forecasts
    
    gpus = tf.config.list_physical_devices('GPU')
    devices = []
    
    # Configure GPUs
    for i, gpu in enumerate(gpus):
        # Configure memory growth to avoid allocating all memory at once
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Configured memory growth for {gpu.name}")
        except RuntimeError as e:
            logging.warning(f"Failed to set memory growth for {gpu.name}: {e}")
        
        # Try to get memory info for better allocation
        try:
            device_name = f'GPU:{i}'
            memory_info = tf.config.experimental.get_memory_info(device_name)
            # Limit to 80% of available memory
            memory_limit = 0.8 * (memory_info.get('peak', 8192 * 1024 * 1024))
            
            # Set virtual device configuration if memory limit is available
            if memory_limit:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
                logging.info(f"Set memory limit of {memory_limit/1024/1024:.0f}MB for {gpu.name}")
        except Exception as e:
            logging.warning(f"Could not set memory limit for {gpu.name}: {e}. Using dynamic growth.")
        
        # Add device to list
        devices.append(f'/GPU:{i % len(gpus)}')
    
    # Distribute forecasts across available GPUs
    return [devices[i % len(devices)] for i in range(num_forecasts)]

@catch_and_log_errors(context="Run Forecast Subprocess")
def run_forecast_in_subprocess(app_config, output_file_path, device):
    """Run a forecast in a subprocess with improved error handling"""
    import os
    import logging
    import sys
    import tensorflow as tf
    import pickle
    from stock_predictor import forecast_module

    # Configure logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    # Run forecast with device placement
    with tf.device(device):
        try:
            logging.info(f"Running forecast on {device}")
            plot_data = forecast_module.run_program(app_config)
            logging.info("Forecast completed successfully")
            
            # Calculate confidence intervals
            confidence_intervals = calculate_confidence_intervals({'predictions': plot_data[5]})
            enhanced_plot_data = plot_data + (confidence_intervals,)
            
            # Save results
            with open(output_file_path, 'wb') as f:
                pickle.dump(enhanced_plot_data, f)
                
            return 0  # Success
        except Exception as e:
            logging.error(f"Forecast failed on {device}: {e}")
            logging.error(traceback.format_exc())
            
            # Save error information
            with open(output_file_path, 'wb') as f:
                pickle.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
                
            return 1  # Error

def calculate_confidence_intervals(plot_data, confidence=0.95):
    """
    Calculate confidence intervals for predictions
    
    Args:
        plot_data: Dictionary with predictions
        confidence: Confidence level (0-1)
        
    Returns:
        dict: Confidence intervals
    """
    predictions = np.array(plot_data['predictions'])
    
    # Calculate statistics
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    # Calculate z-score based on confidence level
    if confidence == 0.95:
        z_score = 1.96
    elif confidence == 0.99:
        z_score = 2.576
    elif confidence == 0.90:
        z_score = 1.645
    else:
        # For other confidence levels, approximate using normal distribution
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate margin of error
    margin_error = z_score * std / np.sqrt(len(predictions))
    
    return {
        'lower': mean - margin_error,
        'upper': mean + margin_error,
        'mean': mean,
        'std': std
    }

@handle_errors(context="Run Parallel Forecasts")
def run_parallel_forecasts(app_config, scenarios, status_var, btn_run):
    """
    Run multiple forecast scenarios in parallel
    
    Args:
        app_config: Application configuration
        scenarios: List of scenario configurations
        status_var: Status variable for UI updates
        btn_run: Run button to disable during processing
        
    Returns:
        tuple: (processes, temp_files) for result handling
    """
    # Disable run button and update status
    btn_run.config(state="disabled")
    status_var.set(f"Running {len(scenarios)} forecast scenarios on GPU...")
    
    # Create temporary files for results
    temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in scenarios]
    
    # Configure devices for parallel processing
    devices = configure_device_resources(app_config, len(scenarios))
    
    # Create and start processes
    processes = []
    for i, (scenario_config, temp_file) in enumerate(zip(scenarios, temp_files)):
        # Create a copy of the configuration
        config_copy = pickle.loads(pickle.dumps(app_config))
        
        # Apply scenario-specific configuration
        for key, value in scenario_config.items():
            # Handle nested attributes (e.g., "learning.learning_rate")
            if "." in key:
                obj_name, attr_name = key.split(".", 1)
                if hasattr(config_copy, obj_name):
                    obj = getattr(config_copy, obj_name)
                    if hasattr(obj, attr_name):
                        setattr(obj, attr_name, value)
            else:
                # Direct attribute
                if hasattr(config_copy.learning, key):
                    setattr(config_copy.learning, key, value)
        
        # Create and start process
        p = multiprocessing.Process(
            target=run_forecast_in_subprocess,
            args=(config_copy, temp_file, devices[i])
        )
        processes.append(p)
        p.start()
    
    return processes, temp_files

@handle_errors(context="Handle Parallel Results")
def handle_parallel_results(processes, temp_files, app_config, status_var, btn_run):
    """
    Handle results from parallel forecasts
    
    Args:
        processes: List of running processes
        temp_files: List of temporary files with results
        app_config: Application configuration
        status_var: Status variable for UI updates
        btn_run: Run button to re-enable after processing
        
    Returns:
        tuple: (results, forecast_results_list) processed results
    """
    results = []
    forecast_results_list = []
    
    for p, temp_file in zip(processes, temp_files):
        p.join()
        
        if p.exitcode == 0 and os.path.exists(temp_file):
            # Load results
            with open(temp_file, 'rb') as f:
                data = pickle.load(f)
            
            # Clean up temporary file
            os.unlink(temp_file)
            
            # Check for error
            if isinstance(data, dict) and 'error' in data:
                logging.error(f"Scenario failed: {data['error']}")
                if 'traceback' in data:
                    logging.error(f"Traceback: {data['traceback']}")
            else:
                # Process successful results
                results.append(data)
                
                # Create ForecastResults object
                fr = ForecastResults()
                fr.ensemble_mean = data[3]
                fr.ensemble_std = data[4]
                fr.simulation_predictions = data[5]
                fr.error_metrics['smape'] = data[6]
                fr.best_simulation_indices = data[7]
                fr.calculate_confidence_intervals()
                forecast_results_list.append(fr)
    
    # Re-enable run button and update status
    btn_run.config(state="normal")
    status_var.set(f"Completed {len(results)} forecasts")
    
    # If we have results, create plot
    if results:
        plot_gui.create_parallel_plot_gui(app_config, results)
    
    return results, forecast_results_list

@handle_errors(context="Auto-tune Hyperparameters")
def auto_tune_hyperparameters(app_config, param_grid=None):
    """
    Automatically tune hyperparameters using grid search
    
    Args:
        app_config: Application configuration
        param_grid: Grid of parameters to search
        
    Returns:
        tuple: (best_params, best_score) Best parameters and score
    """
    import itertools
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'epoch': [50, 100, 200],
            'l2_reg': [0.001, 0.005, 0.01],
            'num_layers': [2, 3, 4]
        }
    
    # Initialize best parameters
    best_score = float('-inf')
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), values)) 
                          for values in itertools.product(*param_grid.values())]
    
    # Create temporary files and configure devices
    temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in param_combinations]
    devices = configure_device_resources(app_config, len(param_combinations))
    
    # Run forecasts for each parameter combination
    processes = []
    for i, params in enumerate(param_combinations):
        # Create configuration copy with current parameters
        config_copy = pickle.loads(pickle.dumps(app_config))
        for key, value in params.items():
            setattr(config_copy.learning, key, value)
        
        # Start process
        process = multiprocessing.Process(
            target=run_forecast_in_subprocess,
            args=(config_copy, temp_files[i], devices[i])
        )
        processes.append(process)
        process.start()
    
    # Process results
    for i, process in enumerate(processes):
        process.join()
        
        if process.exitcode == 0 and os.path.exists(temp_files[i]):
            # Load results
            with open(temp_files[i], 'rb') as f:
                result = pickle.load(f)
            
            # Evaluate if valid result
            if not isinstance(result, dict) or 'error' not in result:
                score = evaluate_forecast(result)
                
                # Update best parameters if better
                if score > best_score:
                    best_score = score
                    best_params = param_combinations[i]
                    logging.info(f"New best score: {best_score:.4f} with params: {best_params}")
    
    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    # Update configuration with best parameters
    if best_params:
        for key, value in best_params.items():
            setattr(app_config.learning, key, value)
            
        logging.info(f"Best parameters: {best_params}, Score: {best_score:.4f}")
    else:
        logging.warning("No valid results found during auto-tuning")
    
    return best_params, best_score

def evaluate_forecast(result):
    """
    Evaluate a forecast result
    
    Args:
        result: Forecast result data
        
    Returns:
        float: Negative mean SMAPE (higher is better)
    """
    # Extract SMAPE scores (lower is better)
    smape_scores = result[6]
    
    # Return negative mean (so higher is better for optimization)
    return -np.mean(smape_scores)

@handle_errors(context="Create GUI")
def create_gui(app_config):
    """
    Create the main GUI with improved synchronization and error handling
    
    Args:
        app_config: Application configuration
    """
    # First, ensure all required config sections exist
    if not hasattr(app_config, 'root_widgets'):
        from config.config import RootWidgetsConfig
        app_config.root_widgets = RootWidgetsConfig()
    
    if not hasattr(app_config, 'preferences'):
        from config.config import PreferencesConfig
        app_config.preferences = PreferencesConfig()
    
    if not hasattr(app_config, 'features'):
        from config.config import FeaturesConfig
        app_config.features = FeaturesConfig()
    
    if not hasattr(app_config, 'advanced_prediction'):
        from config.config import AdvancedPredictionConfig
        app_config.advanced_prediction = AdvancedPredictionConfig()
    
    # Make sure learning.size_layer is within valid range
    if app_config.learning.size_layer > 10:
        app_config.learning.size_layer = 10
        logging.warning("learning.size_layer adjusted to maximum value of 10")
    
    # Initialize root window
    root = tk.Tk()
    root.title("Dynamic Forecast Configuration (GPU)")
    root.geometry("900x650")
    
    # Initialize error handler with root window
    error_handler = get_error_handler(root)
    
    # Create data handler instance
    from core.data_handler import DataHandler
    data_handler = DataHandler(app_config)
    
    # Configure grid layout
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=0)
    root.grid_columnconfigure(0, weight=1)
    
    # Create synchronized tab manager
    tab_manager = SynchronizedTabManager(root, app_config)
    notebook = tab_manager.create_notebook()
    notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # Create frames in desired order
    tab_manager.add_tab("preferences", preferences_tab, app_config.preferences)  # Fixed: using preferences instead of learning_pref
    tab_manager.add_tab("learning", learning_tab, app_config.learning)
    tab_manager.add_tab("features", features_tab, app_config.feature_selection)
    tab_manager.add_tab("rolling_window", rolling_window_tab, app_config.rolling_window)
    tab_manager.add_tab("prediction", prediction_tab, app_config.prediction)
    tab_manager.add_tab("plot", plot_tab, app_config.plot)
    tab_manager.add_tab("advanced_prediction", advanced_prediction_tab, app_config.prediction_advanced)
    tab_manager.add_tab("strategy", strategy_tab, app_config.strategy)
    
    # Set up tab synchronization
    tab_manager.setup_synchronization()
    
    # Create button frame
    button_frame = ttk.Frame(root)
    button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    button_frame.grid_propagate(False)
    button_frame.configure(height=40)
    
    # Configure button layout
    button_count = 9
    for i in range(button_count):
        button_frame.grid_columnconfigure(i, weight=1)
    btn_width = 18
    
    # Create buttons with error handling wrappers
    btn_run = ttk.Button(
        button_frame, 
        text="Run Forecast", 
        width=btn_width, 
        command=lambda: on_run(app_config, tab_manager, data_handler)
    )
    btn_run.grid(row=0, column=0, padx=3, pady=5)
    
    btn_parallel = ttk.Button(
        button_frame, 
        text="Run Parallel Scenarios", 
        width=btn_width, 
        command=lambda: run_parallel_scenarios_handler(app_config, tab_manager, data_handler)
    )
    btn_parallel.grid(row=0, column=1, padx=3, pady=5)
    
    btn_auto_tune = ttk.Button(
        button_frame, 
        text="Auto-tune Parameters", 
        width=btn_width, 
        command=lambda: run_auto_tune_handler(app_config, tab_manager, data_handler)
    )
    btn_auto_tune.grid(row=0, column=2, padx=3, pady=5)
    
    btn_compare = ttk.Button(
        button_frame, 
        text="Compare Models", 
        width=btn_width, 
        command=lambda: run_model_comparison(app_config, tab_manager, data_handler)
    )
    btn_compare.grid(row=0, column=3, padx=3, pady=5)
    
    btn_backtest = ttk.Button(
        button_frame, 
        text="Run Backtesting", 
        width=btn_width, 
        command=lambda: run_backtesting(app_config, tab_manager, data_handler)
    )
    btn_backtest.grid(row=0, column=4, padx=3, pady=5)
    
    # Add strategy buttons
    btn_strategy = ttk.Button(
        button_frame, 
        text="Run Strategy Backtest", 
        width=btn_width, 
        command=lambda: handle_strategy_backtest(app_config, data_handler)
    )
    btn_strategy.grid(row=0, column=5, padx=3, pady=5)
    
    btn_compare_strat = ttk.Button(
        button_frame, 
        text="Compare Strategies", 
        width=btn_width, 
        command=lambda: handle_strategy_comparison(app_config, data_handler)
    )
    btn_compare_strat.grid(row=0, column=6, padx=3, pady=5)
    
    btn_ml = ttk.Button(
        button_frame, 
        text="ML Optimization", 
        width=btn_width, 
        command=lambda: handle_ml_optimization(app_config, data_handler)
    )
    btn_ml.grid(row=0, column=7, padx=3, pady=5)
    
    btn_close = ttk.Button(
        button_frame, 
        text="Close", 
        width=btn_width//2, 
        command=lambda: on_closing(root)
    )
    btn_close.grid(row=0, column=8, padx=3, pady=5)
    
    # Status bar
    status_var = tk.StringVar(value="Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=2, column=0, sticky="ew")
    
    # Set up close handler
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    
    # Store button_frame for later reference
    root.button_frame = button_frame
    
    # Store important references
    root.btn_run = btn_run
    root.btn_parallel = btn_parallel
    root.btn_auto_tune = btn_auto_tune
    root.status_var = status_var
    root.tab_manager = tab_manager
    root.data_handler = data_handler
    root.app_config = app_config  # Store app_config reference
    
    # Start the main loop
    root.mainloop()

@catch_and_log_errors(context="Run Forecast")
def on_run(app_config, tab_manager, data_handler):
    """
    Run the forecast with error handling
    
    Args:
        app_config: Application configuration
        tab_manager: Tab manager instance
        data_handler: Data handler instance
    """
    # Update configuration from GUI
    success, errors = tab_manager.update_config_from_gui()
    if not success:
        messagebox.showerror("Configuration Error", 
                            f"Configuration errors: {', '.join(errors)}")
        return
    
    # Validate configuration
    config_errors = validate_config(app_config)
    if config_errors:
        messagebox.showerror("Validation Error", 
                            f"Configuration validation errors: {', '.join(config_errors)}")
        return
    
    # Disable the run button and update status
    root = tab_manager.root
    btn_run = root.btn_run
    status_var = root.status_var
    
    btn_run.config(state="disabled")
    status_var.set("Preparing forecast...")
    
    # Create temporary file for results
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    # Configure devices
    devices = configure_device_resources(app_config, num_forecasts=1)
    device = devices[0]
    
    # Run forecast in a separate thread
    def run_forecast():
        process = multiprocessing.Process(
            target=run_forecast_in_subprocess,
            args=(app_config, temp_file_path, device)
        )
        process.start()
        process.join()
        root.after(0, lambda: handle_forecast_result(process.exitcode, temp_file_path, root))
    
    threading.Thread(target=run_forecast, daemon=True).start()

@catch_and_log_errors(context="Handle Forecast Result")
def handle_forecast_result(exitcode, temp_file_path, root):
    """
    Handle the forecast result
    
    Args:
        exitcode: Process exit code
        temp_file_path: Path to temporary file with results
        root: Root window
    """
    # Re-enable run button
    btn_run = root.btn_run
    status_var = root.status_var
    btn_run.config(state="normal")
    
    if exitcode == 0 and os.path.exists(temp_file_path):
        # Load results
        with open(temp_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Check for error
        if isinstance(data, dict) and 'error' in data:
            messagebox.showerror("Error", f"Forecast simulation failed: {data['error']}")
            status_var.set("Forecast failed")
            
            # Log traceback if available
            if 'traceback' in data:
                logging.error(f"Forecast traceback: {data['traceback']}")
        else:
            status_var.set("Forecast complete - Displaying results")
            
            # Display results
            root.after(10, lambda: plot_gui.create_plot_gui_with_data(root.app_config, data[:-1]))
    else:
        messagebox.showerror("Error", "Forecast simulation failed.")
        status_var.set("Forecast failed")

@catch_and_log_errors(context="Run Parallel Scenarios")
def run_parallel_scenarios_handler(app_config, tab_manager, data_handler):
    """
    Run parallel forecast scenarios
    
    Args:
        app_config: Application configuration
        tab_manager: Tab manager instance
        data_handler: Data handler instance
    """
    # Update configuration from GUI
    success, errors = tab_manager.update_config_from_gui()
    if not success:
        messagebox.showerror("Configuration Error", 
                            f"Configuration errors: {', '.join(errors)}")
        return
    
    # Define scenarios (more sophisticated combinations could be implemented)
    scenarios = [
        {'learning_rate': 0.001, 'epoch': 100},
        {'learning_rate': 0.01, 'epoch': 150},
        {'learning_rate': 0.1, 'epoch': 200},
    ]
    
    # Run parallel forecasts
    root = tab_manager.root
    btn_parallel = root.btn_parallel
    status_var = root.status_var
    
    processes, temp_files = run_parallel_forecasts(app_config, scenarios, status_var, btn_parallel)
    
    # Handle results in the main thread
    root.after(100, lambda: handle_parallel_results(processes, temp_files, app_config, status_var, btn_parallel))

@catch_and_log_errors(context="Run Auto-tune")
def run_auto_tune_handler(app_config, tab_manager, data_handler):
    """
    Run auto-tuning of hyperparameters
    
    Args:
        app_config: Application configuration
        tab_manager: Tab manager instance
        data_handler: Data handler instance
    """
    # Update configuration from GUI
    success, errors = tab_manager.update_config_from_gui()
    if not success:
        messagebox.showerror("Configuration Error", 
                            f"Configuration errors: {', '.join(errors)}")
        return
    
    # Get references
    root = tab_manager.root
    btn_auto_tune = root.btn_auto_tune
    status_var = root.status_var
    
    # Update status and disable button
    status_var.set("Running hyperparameter optimization...")
    btn_auto_tune.config(state="disabled")
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [32, 64, 128],
        'epoch': [50, 100],
        'dropout_rate': [0.2, 0.3, 0.4],
        'l2_reg': [0.001, 0.01]
    }
    
    # Run auto-tuning in a separate thread
    def tune_callback():
        try:
            best_params, score = auto_tune_hyperparameters(app_config, param_grid)
            root.after(0, lambda: handle_auto_tune_result(best_params, score, tab_manager, root))
        except Exception as e:
            root.after(0, lambda: handle_auto_tune_error(e, root))
    
    threading.Thread(target=tune_callback, daemon=True).start()

@catch_and_log_errors(context="Handle Auto-tune Result")
def handle_auto_tune_result(best_params, score, tab_manager, root):
    """
    Handle auto-tuning results
    
    Args:
        best_params: Best parameters found
        score: Best score
        tab_manager: Tab manager instance
        root: Root window
    """
    # Update status and re-enable button
    status_var = root.status_var
    btn_auto_tune = root.btn_auto_tune
    
    status_var.set(f"Optimization complete. Best score: {score:.4f}")
    btn_auto_tune.config(state="normal")
    
    # Update GUI with best parameters
    tab_manager.update_gui_from_config()
    
    # Show message
    if best_params:
        messagebox.showinfo("Tuning Complete", 
                           f"Best parameters: {best_params}\nScore: {score:.4f}")
    else:
        messagebox.showwarning("Tuning Incomplete", 
                              "No valid parameter set found. Try different parameter ranges.")

@catch_and_log_errors(context="Handle Auto-tune Error")
def handle_auto_tune_error(error, root):
    """
    Handle auto-tuning error
    
    Args:
        error: Error that occurred
        root: Root window
    """
    # Update status and re-enable button
    status_var = root.status_var
    btn_auto_tune = root.btn_auto_tune
    
    status_var.set("Optimization failed")
    btn_auto_tune.config(state="normal")
    
    # Show error message
    messagebox.showerror("Tuning Error", f"Hyperparameter optimization failed: {str(error)}")

@catch_and_log_errors(context="Run Model Comparison")
def run_model_comparison(app_config, tab_manager, data_handler):
    """
    Run comparison of different model architectures
    
    Args:
        app_config: Application configuration
        tab_manager: Tab manager instance
        data_handler: Data handler instance
    """
    # Update configuration from GUI
    success, errors = tab_manager.update_config_from_gui()
    if not success:
        messagebox.showerror("Configuration Error", 
                            f"Configuration errors: {', '.join(errors)}")
        return
# Get references
    root = tab_manager.root
    status_var = root.status_var
    
    # Update status
    status_var.set("Running model comparison...")
    
    # Define model configurations
    model_configs = [
        {"model_type": "lstm", "learning_rate": 0.001},
        {"model_type": "gru", "learning_rate": 0.001},
        {"model_type": "transformer", "learning_rate": 0.001}
    ]
    
    # Run comparison in a thread
    def comparison_callback():
        try:
            # Create temporary files
            temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in model_configs]
            
            # Configure devices
            devices = configure_device_resources(app_config, len(model_configs))
            
            # Run processes
            processes = []
            for i, (model_config, temp_file) in enumerate(zip(model_configs, temp_files)):
                # Create configuration copy
                config_copy = pickle.loads(pickle.dumps(app_config))
                
                # Apply model configuration
                for key, value in model_config.items():
                    if key == "model_type":
                        # Set model type in advanced prediction config
                        config_copy.prediction_advanced.ensemble_models = [value]
                        config_copy.prediction_advanced.use_ensemble_methods = True
                    else:
                        # Set other parameters in learning config
                        setattr(config_copy.learning, key, value)
                
                # Start process
                p = multiprocessing.Process(
                    target=run_forecast_in_subprocess,
                    args=(config_copy, temp_file, devices[i])
                )
                processes.append(p)
                p.start()
            
            # Process results
            results = []
            for i, (p, temp_file) in enumerate(zip(processes, temp_files)):
                p.join()
                
                if p.exitcode == 0 and os.path.exists(temp_file):
                    with open(temp_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    os.unlink(temp_file)
                    
                    if not isinstance(data, dict) or 'error' not in data:
                        model_name = model_configs[i]["model_type"]
                        results.append((model_name, data))
            
            # Display results
            root.after(0, lambda: handle_model_comparison_result(results, app_config, root))
            
        except Exception as e:
            root.after(0, lambda: handle_model_comparison_error(e, root))
    
    threading.Thread(target=comparison_callback, daemon=True).start()

@catch_and_log_errors(context="Handle Model Comparison Result")
def handle_model_comparison_result(results, app_config, root):
    """
    Handle model comparison results
    
    Args:
        results: List of (model_name, data) tuples
        app_config: Application configuration
        root: Root window
    """
    # Update status
    status_var = root.status_var
    
    if results:
        status_var.set("Model comparison complete")
        plot_gui.create_model_comparison_gui(app_config, results)
    else:
        status_var.set("Model comparison failed - no valid results")
        messagebox.showerror("Error", "Failed to complete model comparison")

@catch_and_log_errors(context="Handle Model Comparison Error")
def handle_model_comparison_error(error, root):
    """
    Handle model comparison error
    
    Args:
        error: Error that occurred
        root: Root window
    """
    # Update status
    status_var = root.status_var
    status_var.set("Model comparison failed")
    
    # Show error message
    messagebox.showerror("Comparison Error", f"Model comparison failed: {str(error)}")

@catch_and_log_errors(context="Run Backtesting")
def run_backtesting(app_config, tab_manager, data_handler):
    """
    Run backtesting on historical data
    
    Args:
        app_config: Application configuration
        tab_manager: Tab manager instance
        data_handler: Data handler instance
    """
    # Update configuration from GUI
    success, errors = tab_manager.update_config_from_gui()
    if not success:
        messagebox.showerror("Configuration Error", 
                            f"Configuration errors: {', '.join(errors)}")
        return
    
    # Get references
    root = tab_manager.root
    status_var = root.status_var
    btn_backtest = next((child for child in root.button_frame.winfo_children() 
                         if isinstance(child, ttk.Button) and child['text'] == "Run Backtesting"), None)
    
    if not btn_backtest:
        btn_backtest = root.btn_run  # Fallback
    
    # Update status and disable button
    status_var.set("Running backtesting...")
    btn_backtest.config(state="disabled")
    
    # Run backtesting in a thread
    def backtest_callback():
        try:
            # Create configuration copy with backtesting enabled
            config_copy = pickle.loads(pickle.dumps(app_config))
            config_copy.backtest.enable_backtesting = True
            config_copy.backtest.test_periods = 30
            config_copy.backtest.walk_forward = True
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Configure device
            devices = configure_device_resources(config_copy, num_forecasts=1)
            device = devices[0]
            
            # Run forecast
            process = multiprocessing.Process(
                target=run_forecast_in_subprocess,
                args=(config_copy, temp_file_path, device)
            )
            process.start()
            process.join()
            
            # Handle result
            root.after(0, lambda: handle_backtest_result(process.exitcode, temp_file_path, app_config, root, btn_backtest))
            
        except Exception as e:
            root.after(0, lambda: handle_backtest_error(e, root, btn_backtest))
    
    threading.Thread(target=backtest_callback, daemon=True).start()

@catch_and_log_errors(context="Handle Backtest Result")
def handle_backtest_result(exitcode, temp_file_path, app_config, root, btn_backtest):
    """
    Handle backtesting results
    
    Args:
        exitcode: Process exit code
        temp_file_path: Path to temporary file with results
        app_config: Application configuration
        root: Root window
        btn_backtest: Backtest button
    """
    # Update status and re-enable button
    status_var = root.status_var
    btn_backtest.config(state="normal")
    
    if exitcode == 0 and os.path.exists(temp_file_path):
        # Load results
        with open(temp_file_path, 'rb') as f:
            backtest_data = pickle.load(f)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Check for error
        if isinstance(backtest_data, dict) and 'error' in backtest_data:
            messagebox.showerror("Error", f"Backtesting failed: {backtest_data['error']}")
            status_var.set("Backtesting failed")
        else:
            status_var.set("Backtesting complete - Displaying results")
            plot_gui.create_backtest_plot_gui(app_config, backtest_data)
    else:
        messagebox.showerror("Error", "Backtesting failed.")
        status_var.set("Backtesting failed")

@catch_and_log_errors(context="Handle Backtest Error")
def handle_backtest_error(error, root, btn_backtest):
    """
    Handle backtesting error
    
    Args:
        error: Error that occurred
        root: Root window
        btn_backtest: Backtest button
    """
    # Update status and re-enable button
    status_var = root.status_var
    btn_backtest.config(state="normal")
    status_var.set("Backtesting failed")
    
    # Show error message
    messagebox.showerror("Backtesting Error", f"Backtesting failed: {str(error)}")

@catch_and_log_errors(context="Handle Strategy Backtest")
def handle_strategy_backtest(app_config, data_handler):
    """
    Run strategy backtest
    
    Args:
        app_config: Application configuration
        data_handler: Data handler instance
    """
    # Import the strategy GUI functions
    from stock_predictor.strategy_gui_integration import run_strategy_backtest
    
    # Run the backtest
    run_strategy_backtest(app_config, data_handler)

@catch_and_log_errors(context="Handle Strategy Comparison")
def handle_strategy_comparison(app_config, data_handler):
    """
    Run strategy comparison
    
    Args:
        app_config: Application configuration
        data_handler: Data handler instance
    """
    # Import the strategy GUI functions
    from stock_predictor.strategy_gui_integration import run_strategy_comparison
    
    # Run the comparison
    run_strategy_comparison(app_config, data_handler)

@catch_and_log_errors(context="Handle ML Optimization")
def handle_ml_optimization(app_config, data_handler):
    """
    Run ML strategy optimization
    
    Args:
        app_config: Application configuration
        data_handler: Data handler instance
    """
    # Import the strategy GUI functions
    from stock_predictor.strategy_gui_integration import run_ml_optimization
    
    # Run the optimization
    run_ml_optimization(app_config, data_handler)

def on_closing(root):
    """
    Handle window closing
    
    Args:
        root: Root window
    """
    root.quit()
    root.destroy()


def plot_rolling_window_performance(app_config, diagnostics, plt=plt, np=np):
    """
    Plot rolling window validation performance
    
    Args:
        app_config: Application configuration
        diagnostics: Dictionary of performance metrics
        plt: Matplotlib pyplot module
        np: NumPy module
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig = plt.figure(figsize=getattr(app_config.gui, 'figure_size', (10, 8)))
    
    if 'window_metrics' not in diagnostics or not diagnostics['window_metrics']:
        plt.text(0.5, 0.5, "No rolling window metrics available", ha='center', va='center', fontsize=12)
        plt.tight_layout()
        return fig
    
    window_metrics = diagnostics['window_metrics']
    window_indices = [m['window_idx'] for m in window_metrics]
    smape_values = [m['smape'] for m in window_metrics]
    
    # Plot SMAPE values
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(window_indices, smape_values, marker='o', linestyle='-', color='blue')
    ax1.set_title('Rolling Window Validation Performance')
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('SMAPE (%)')
    ax1.grid(True)
    
    # Plot average line
    avg_smape = np.mean(smape_values)
    ax1.axhline(y=avg_smape, color='red', linestyle='--', label=f'Average SMAPE: {avg_smape:.2f}%')
    ax1.legend()
    
    # Plot histogram
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(smape_values, bins=min(10, len(smape_values)), color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of SMAPE Values')
    ax2.set_xlabel('SMAPE (%)')
    ax2.set_ylabel('Frequency')
    
    # Add statistics
    min_smape = np.min(smape_values)
    max_smape = np.max(smape_values)
    median_smape = np.median(smape_values)
    std_smape = np.std(smape_values)
    
    stats_text = f"Min: {min_smape:.2f}%  Max: {max_smape:.2f}%\nMedian: {median_smape:.2f}%  StdDev: {std_smape:.2f}%"
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

@handle_errors(context="Main Function")
def main():
    """Main function to run the application"""
    try:
        # Import required modules
        from stock_predictor import forecast_module
        
        # Create application configuration
        app_config = AppConfig()  # Using the direct import from config instead
        
        # Initialize all required sections
        # Already initialized in constructor, but just to be safe:
        if not hasattr(app_config, 'root_widgets'):
            app_config.root_widgets = RootWidgetsConfig()
        
        if not hasattr(app_config, 'preferences'):
            app_config.preferences = PreferencesConfig()
        
        if not hasattr(app_config, 'features'):
            app_config.features = FeaturesConfig()
        
        if not hasattr(app_config, 'advanced_prediction'):
            app_config.advanced_prediction = AdvancedPredictionConfig()
        
        # Ensure size_layer is valid
        if app_config.learning.size_layer > 10:
            app_config.learning.size_layer = 10
            logging.warning("learning.size_layer adjusted to maximum value of 10")
        
        # Create GUI
        create_gui(app_config)
        
    except RuntimeError as e:
        if "GPU" in str(e):
            logging.error(f"GPU initialization failed: {e}")
            print("Please use gui_wrapper_cpu.py if no GPU is available.")
        else:
            logging.error(f"Runtime error: {e}")
            print(f"Runtime error: {e}")
    except ImportError as e:
        logging.error(f"Failed to import required modules: {e}")
        print(f"Error: Missing required modules. {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        print(f"Unexpected error occurred: {e}")

if __name__ == '__main__':
    main()