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
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from stock_predictor import plot_gui
from stock_predictor.data_classes import StockData, ForecastResults
from stock_predictor.config import AppConfig

# Import necessary modules for GUI tabs
from stock_predictor.gui import preferences_tab, learning_tab, prediction_tab
from stock_predictor.gui import plot_tab, features_tab, rolling_window_tab, advanced_prediction_tab, strategy_tab

matplotlib.use('TkAgg')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_bool_str(value):
    """Converts various 'true/false' string forms to boolean."""
    if isinstance(value, bool):
        return value
    return value.strip().lower() == 'true'

def configure_device_resources(app_config, num_forecasts=1):
    """
    GPU-optimized resource allocation for forecasting.
    
    Args:
        app_config: Application configuration
        num_forecasts: Number of parallel forecasts to run
        
    Returns:
        List of device strings to use for parallel processing
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("No GPUs found. Use CPU module instead.")
    
    devices = []
    try:
        for i, gpu in enumerate(gpus):
            tf.config.experimental.set_memory_growth(gpu, True)
            try:
                device_name = f'GPU:{i}'
                memory_info = tf.config.experimental.get_memory_info(device_name)
                logging.debug(f"Memory info for {device_name}: {memory_info}")
                memory_limit = 0.8 * (memory_info.get('peak', 8192 * 1024 * 1024))
            except ValueError as e:
                logging.warning(f"Could not get memory info for {device_name}: {e}. Using dynamic growth.")
                memory_limit = None
            
            if memory_limit:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
            devices.append(f'/GPU:{i % len(gpus)}')
        
        return [devices[i % len(gpus)] for i in range(num_forecasts)]
    except RuntimeError as e:
        logging.error(f"GPU configuration failed: {e}")
        raise

def update_config_from_gui(entries, app_config):
    """
    Update the app_config from GUI entries.
    
    Args:
        entries: Dictionary containing the GUI entries
        app_config: The application configuration to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Process each configuration section
        for section, config_obj in {
            "preferences": app_config.learning_pref,
            "learning": app_config.learning,
            "prediction": app_config.prediction,
            "plot": app_config.plot,
            "features": app_config.feature_selection,
            "rolling_window": app_config.rolling_window,
            "advanced_prediction": app_config.prediction_advanced,
            "strategy": app_config.strategy
        }.items():
            if section not in entries:
                logging.warning(f"Section {section} not found in entries")
                continue
            
            # Check if Feature tab should be used based on Learning tab setting
            # (Assume that if learning.use_features is checked, then advanced features are used)
            if section == "features" and "learning" in entries and "use_features" in entries["learning"]:
                # If Learning Features is checked, we want the Features tab enabled.
                # So here, we simply skip updating the features section because it's controlled by learning.
                logging.info("Learning Features enabled: Features tab settings will be used")
                continue
            
            # Update config from the tab's entries
            if section == "advanced_prediction":
                try:
                    if hasattr(entries[section], 'update_config'):
                        entries[section].update_config(config_obj)
                    elif hasattr(entries[section], 'get_config'):
                        new_config = entries[section].get_config()
                        for key, value in vars(new_config).items():
                            if hasattr(config_obj, key):
                                setattr(config_obj, key, value)
                    else:
                        if isinstance(entries[section], dict):
                            update_section_config(entries[section], config_obj)
                        else:
                            logging.warning(f"Could not update {section} config: incompatible type")
                except Exception as e:
                    logging.error(f"Error updating {section} config: {e}")
            else:
                update_section_config(entries[section], config_obj)
        
        logging.info("Configuration updated from GUI.")
        return True
    except Exception as e:
        logging.error(f"Error updating configuration: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        messagebox.showerror("Configuration Error", f"Invalid input: {e}")
        return False

def update_section_config(entries, config_obj):
    from dataclasses import fields
    for f in fields(config_obj):
        if f.name in entries:
            value = entries[f.name]
            current_val = getattr(config_obj, f.name)
            if isinstance(value, tk.BooleanVar):
                setattr(config_obj, f.name, value.get())
            elif isinstance(value, ttk.Combobox):
                setattr(config_obj, f.name, value.get())
            elif isinstance(value, ttk.Entry):
                if isinstance(current_val, bool):
                    setattr(config_obj, f.name, parse_bool_str(value.get()))
                elif isinstance(current_val, int):
                    setattr(config_obj, f.name, int(value.get()))
                elif isinstance(current_val, float):
                    setattr(config_obj, f.name, float(value.get()))
                else:
                    setattr(config_obj, f.name, value.get())
            else:
                try:
                    # Check if the value has a get method that requires no arguments
                    if hasattr(value, 'get') and callable(value.get):
                        # Inspect the method signature
                        import inspect
                        sig = inspect.signature(value.get)
                        # Check if all parameters have default values (optional)
                        required_params = [p for p in sig.parameters.values() 
                                           if p.default is inspect.Parameter.empty and 
                                           p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
                        
                        # If there are no required parameters, we can call get()
                        if not required_params or len(required_params) == 0:
                            widget_value = value.get()
                            if isinstance(current_val, bool):
                                setattr(config_obj, f.name, parse_bool_str(widget_value))
                            elif isinstance(current_val, int):
                                setattr(config_obj, f.name, int(widget_value))
                            elif isinstance(current_val, float):
                                setattr(config_obj, f.name, float(widget_value))
                            else:
                                setattr(config_obj, f.name, widget_value)
                except Exception as e:
                    logging.warning(f"Could not get value for {f.name}: {e}")

def run_forecast_in_subprocess(app_config, output_file_path, device):
    import os
    import logging
    import sys
    import tensorflow as tf
    import pickle
    from stock_predictor import forecast_module

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    with tf.device(device):
        try:
            logging.info("About to run forecast_module.run_program")
            plot_data = forecast_module.run_program(app_config)
            logging.info("Forecast program completed successfully")
            confidence_intervals = calculate_confidence_intervals({'predictions': plot_data[5]})
            enhanced_plot_data = plot_data + (confidence_intervals,)
            with open(output_file_path, 'wb') as f:
                pickle.dump(enhanced_plot_data, f)
        except Exception as e:
            logging.error(f"Forecast failed on {device}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            with open(output_file_path, 'wb') as f:
                pickle.dump({"error": str(e)}, f)
            sys.exit(1)

def calculate_confidence_intervals(plot_data, confidence=0.95):
    predictions = np.array(plot_data['predictions'])
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    z_score = 1.96  # for 95% confidence
    margin_error = z_score * std / np.sqrt(len(predictions))
    return {
        'lower': mean - margin_error,
        'upper': mean + margin_error,
        'mean': mean
    }

def run_parallel_forecasts(app_config, scenarios, status_var, btn_run):
    btn_run.config(state="disabled")
    status_var.set(f"Running {len(scenarios)} forecast scenarios on GPU...")
    
    temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in scenarios]
    devices = configure_device_resources(app_config, len(scenarios))
    
    processes = []
    for i, (scenario_config, temp_file) in enumerate(zip(scenarios, temp_files)):
        config_copy = pickle.loads(pickle.dumps(app_config))
        for key, value in scenario_config.items():
            setattr(config_copy.learning, key, value)
        
        p = multiprocessing.Process(
            target=run_forecast_in_subprocess,
            args=(config_copy, temp_file, devices[i])
        )
        processes.append(p)
        p.start()
    
    return processes, temp_files

def handle_parallel_results(processes, temp_files, app_config, status_var, btn_run):
    results = []
    forecast_results_list = []  # For ForecastResults objects
    
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
                fr = ForecastResults()
                fr.ensemble_mean = data[3]
                fr.ensemble_std = data[4]
                fr.simulation_predictions = data[5]
                fr.error_metrics['smape'] = data[6]
                fr.best_simulation_indices = data[7]
                fr.calculate_confidence_intervals()
                forecast_results_list.append(fr)
    
    btn_run.config(state="normal")
    status_var.set(f"Completed {len(results)} forecasts")
    
    if results:
        plot_gui.create_parallel_plot_gui(app_config, results)
    
    return results, forecast_results_list

def auto_tune_hyperparameters(app_config, param_grid=None):
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'epoch': [50, 100, 200],
            'l2_reg': [0.001, 0.005, 0.01],
            'num_layers': [2, 3, 4]
        }
    
    best_score = float('-inf')
    best_params = None
    
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in param_combinations]
    devices = configure_device_resources(app_config, len(param_combinations))
    
    processes = []
    for i, params in enumerate(param_combinations):
        config_copy = pickle.loads(pickle.dumps(app_config))
        for key, value in params.items():
            setattr(config_copy.learning, key, value)
        
        process = multiprocessing.Process(
            target=run_forecast_in_subprocess,
            args=(config_copy, temp_files[i], devices[i])
        )
        processes.append(process)
        process.start()
    
    for i, process in enumerate(processes):
        process.join()
        if process.exitcode == 0:
            with open(temp_files[i], 'rb') as f:
                result = pickle.load(f)
            if not isinstance(result, dict) or 'error' not in result:
                score = evaluate_forecast(result)
                if score > best_score:
                    best_score = score
                    best_params = param_combinations[i]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    if best_params:
        for key, value in best_params.items():
            setattr(app_config.learning, key, value)
    return best_params, best_score

def evaluate_forecast(result):
    smape_scores = result[6]
    return -np.mean(smape_scores)

def create_gui(app_config):
    root = tk.Tk()
    root.title("Dynamic Forecast Configuration (GPU)")
    root.geometry("900x650")
    
    from stock_predictor.thread_safe_gui import start_queue_processor, stop_queue_processor

    # Start the queue processor
    start_queue_processor()

    # Make sure it stops when the app closes
    def on_closing():
        stop_queue_processor()
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)



    # Create data handler instance
    from stock_predictor.data_handler import DataHandler
    data_handler = DataHandler(app_config)
    
    # Configure grid layout
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=0)
    root.grid_columnconfigure(0, weight=1)
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # Create frames in desired order:
    # Preferences, Learning, Features, Rolling Window, Prediction, Plot, Advanced Prediction, Strategy
    frames = {
        "preferences": ttk.Frame(notebook),
        "learning": ttk.Frame(notebook),
        "features": ttk.Frame(notebook),
        "rolling_window": ttk.Frame(notebook),
        "prediction": ttk.Frame(notebook),
        "plot": ttk.Frame(notebook),
        "advanced_prediction": ttk.Frame(notebook),
        "strategy": ttk.Frame(notebook)  # Added strategy tab
    }
    
    for key, frame in frames.items():
        tab_title = " ".join(word.capitalize() for word in key.split("_"))
        notebook.add(frame, text=tab_title)
    
    # Create tab contents
    entries = {}
    entries["preferences"] = preferences_tab.create_tab(frames["preferences"], app_config.learning_pref)
    entries["learning"] = learning_tab.create_tab(frames["learning"], app_config.learning)
    entries["features"] = features_tab.create_tab(frames["features"], app_config.feature_selection)
    entries["rolling_window"] = rolling_window_tab.create_tab(frames["rolling_window"], app_config.rolling_window)
    entries["prediction"] = prediction_tab.create_tab(frames["prediction"], app_config.prediction)
    entries["plot"] = plot_tab.create_tab(frames["plot"], app_config.plot)
    entries["advanced_prediction"] = advanced_prediction_tab.create_tab(frames["advanced_prediction"], app_config.prediction_advanced)
    entries["strategy"] = strategy_tab.create_tab(frames["strategy"], app_config.strategy)  # Added strategy tab creation
    
    # Set up event handlers for tab interactions
    learning_tab.setup_events(entries["learning"], notebook)
    
    # Toggle Features Tab based on the "Learning Features" checkbox.
    # Rename the checkbox in the learning tab to "Learning Features"
    if "use_features" in entries["learning"]:
        def toggle_features_tab(*args):
            try:
                use_features = entries["learning"]["use_features"].get()
                features_tab_index = -1
                for i, tab_id in enumerate(notebook.tabs()):
                    # Check for the "Features" tab by its title.
                    if notebook.tab(tab_id, "text") == "Features":
                        features_tab_index = i
                        break
                if features_tab_index >= 0:
                    if use_features:
                        # When checked, enable the Features tab.
                        notebook.tab(features_tab_index, state="normal")
                    else:
                        # When unchecked, disable (gray out) the Features tab.
                        notebook.tab(features_tab_index, state="disabled")
            except Exception as e:
                logging.error(f"Error toggling features tab: {e}")
        entries["learning"]["use_features"].trace_add('write', toggle_features_tab)
        # Initialize state: if the checkbox is checked by default, the Features tab will be enabled.
        toggle_features_tab()
    
    # Forecast execution functions
    def handle_forecast_result(exitcode, temp_file_path):
        from stock_predictor.thread_safe_gui import run_in_main_thread
    
        # Update GUI elements
        run_in_main_thread(btn_run.config, state="normal")
    
        if exitcode == 0 and os.path.exists(temp_file_path):
            with open(temp_file_path, 'rb') as f:
                data = pickle.load(f)
            os.unlink(temp_file_path)
        
            if isinstance(data, dict) and 'error' in data:
                run_in_main_thread(messagebox.showerror, "Error", f"Forecast simulation failed: {data['error']}")
                run_in_main_thread(status_var.set, "Forecast failed")
            else:
                run_in_main_thread(status_var.set, "Forecast complete - Displaying results")
            
                # Import at function level to avoid circular imports
                from stock_predictor import plot_gui
                run_in_main_thread(plot_gui.create_plot_gui_with_data, app_config, data[:-1])
        else:
            run_in_main_thread(messagebox.showerror, "Error", "Forecast simulation failed.")
            run_in_main_thread(status_var.set, "Forecast failed")
            
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
        
            # Queue the result handling for the main thread
            from stock_predictor.thread_safe_gui import run_in_main_thread
            run_in_main_thread(handle_forecast_result, process.exitcode, temp_file_path)
        
        threading.Thread(target=run_forecast, daemon=True).start()

    def run_parallel_scenarios():
        scenarios = [
            {'learning_rate': 0.001, 'epoch': 100},
            {'learning_rate': 0.01, 'epoch': 150},
            {'learning_rate': 0.1, 'epoch': 200},
        ]
        if update_config_from_gui(entries, app_config):
            processes, temp_files = run_parallel_forecasts(app_config, scenarios, status_var, btn_parallel)
            root.after(100, lambda: handle_parallel_results(processes, temp_files, app_config, status_var, btn_parallel))

    def run_auto_tune():
        if update_config_from_gui(entries, app_config):
            status_var.set("Running hyperparameter optimization...")
            btn_auto_tune.config(state="disabled")
            def tune_callback():
                best_params, score = auto_tune_hyperparameters(app_config)
                status_var.set(f"Optimization complete. Best score: {score:.4f}")
                btn_auto_tune.config(state="normal")
                learning_tab.update_fields_from_config(entries["learning"], app_config.learning)
                messagebox.showinfo("Tuning Complete", f"Best parameters: {best_params}")
            threading.Thread(target=tune_callback, daemon=True).start()

    def run_comparison():
        if update_config_from_gui(entries, app_config):
            status_var.set("Running model comparison...")
            btn_compare.config(state="disabled")
            def comparison_callback():
                try:
                    model_configs = [
                        {"model_type": "lstm", "learning_rate": 0.001},
                        {"model_type": "gru", "learning_rate": 0.001},
                        {"model_type": "transformer", "learning_rate": 0.001}
                    ]
                    temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in model_configs]
                    devices = configure_device_resources(app_config, len(model_configs))
                    processes = []
                    for i, (model_config, temp_file) in enumerate(zip(model_configs, temp_files)):
                        config_copy = pickle.loads(pickle.dumps(app_config))
                        for key, value in model_config.items():
                            setattr(config_copy.learning, key, value)
                        p = multiprocessing.Process(
                            target=run_forecast_in_subprocess,
                            args=(config_copy, temp_file, devices[i])
                        )
                        processes.append(p)
                        p.start()
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
                    if results:
                        plot_gui.create_model_comparison_gui(app_config, results)
                        status_var.set("Model comparison complete")
                    else:
                        status_var.set("Model comparison failed - no valid results")
                        messagebox.showerror("Error", "Failed to complete model comparison")
                except Exception as e:
                    status_var.set("Model comparison failed")
                    logging.error(f"Error in model comparison: {e}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    messagebox.showerror("Error", f"Model comparison failed: {e}")
                finally:
                    btn_compare.config(state="normal")
            threading.Thread(target=comparison_callback, daemon=True).start()

    def run_backtesting():
        if update_config_from_gui(entries, app_config):
            status_var.set("Running backtesting...")
            btn_backtest.config(state="disabled")
            def backtest_callback():
                try:
                    config_copy = pickle.loads(pickle.dumps(app_config))
                    config_copy.backtest.enable_backtesting = True
                    config_copy.backtest.test_periods = 30
                    config_copy.backtest.walk_forward = True
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file_path = temp_file.name
                    devices = configure_device_resources(config_copy, num_forecasts=1)
                    device = devices[0]
                    process = multiprocessing.Process(
                        target=run_forecast_in_subprocess,
                        args=(config_copy, temp_file_path, device)
                    )
                    process.start()
                    process.join()
                    if process.exitcode == 0 and os.path.exists(temp_file_path):
                        with open(temp_file_path, 'rb') as f:
                            backtest_data = pickle.load(f)
                        os.unlink(temp_file_path)
                        if isinstance(backtest_data, dict) and 'error' in backtest_data:
                            messagebox.showerror("Error", f"Backtesting failed: {backtest_data['error']}")
                            status_var.set("Backtesting failed")
                        else:
                            status_var.set("Backtesting complete - Displaying results")
                            plot_gui.create_backtest_plot_gui(app_config, backtest_data)
                    else:
                        messagebox.showerror("Error", "Backtesting failed.")
                        status_var.set("Backtesting failed")
                except Exception as e:
                    status_var.set("Backtesting failed")
                    logging.error(f"Error in backtesting: {e}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    messagebox.showerror("Error", f"Backtesting failed: {e}")
                finally:
                    btn_backtest.config(state="normal")
            threading.Thread(target=backtest_callback, daemon=True).start()

    # Add strategy button functions
    def handle_strategy_backtest():
        """Run strategy backtest"""
        if update_config_from_gui(entries, app_config):
            # Import the strategy GUI functions
            from stock_predictor.strategy_gui_integration import run_strategy_backtest
            # Run the backtest
            run_strategy_backtest(app_config, data_handler)

    def handle_strategy_comparison():
        """Run strategy comparison"""
        if update_config_from_gui(entries, app_config):
            # Import the strategy GUI functions
            from stock_predictor.strategy_gui_integration import run_strategy_comparison
            # Run the comparison
            run_strategy_comparison(app_config, data_handler)

    def handle_ml_optimization():
        """Run ML strategy optimization"""
        if update_config_from_gui(entries, app_config):
            # Import the strategy GUI functions
            from stock_predictor.strategy_gui_integration import run_ml_optimization
            # Run the optimization
            run_ml_optimization(app_config, data_handler)

    def on_closing():
        root.quit()
        root.destroy()
    
    # Create button frame - fixed position at row 1
    button_frame = ttk.Frame(root)
    button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    button_frame.grid_propagate(False)
    button_frame.configure(height=40)
    button_count = 9  # Total number of buttons (including new strategy buttons)
    for i in range(button_count):
        button_frame.grid_columnconfigure(i, weight=1)
    btn_width = 18
    
    btn_run = ttk.Button(button_frame, text="Run Forecast", width=btn_width, command=on_run)
    btn_run.grid(row=0, column=0, padx=3, pady=5)
    btn_parallel = ttk.Button(button_frame, text="Run Parallel Scenarios", width=btn_width, command=run_parallel_scenarios)
    btn_parallel.grid(row=0, column=1, padx=3, pady=5)
    btn_auto_tune = ttk.Button(button_frame, text="Auto-tune Parameters", width=btn_width, command=run_auto_tune)
    btn_auto_tune.grid(row=0, column=2, padx=3, pady=5)
    btn_compare = ttk.Button(button_frame, text="Compare Models", width=btn_width, command=run_comparison)
    btn_compare.grid(row=0, column=3, padx=3, pady=5)
    btn_backtest = ttk.Button(button_frame, text="Run Backtesting", width=btn_width, command=run_backtesting)
    btn_backtest.grid(row=0, column=4, padx=3, pady=5)
    
    # Add strategy buttons
    btn_strategy = ttk.Button(button_frame, text="Run Strategy Backtest", width=btn_width, command=handle_strategy_backtest)
    btn_strategy.grid(row=0, column=5, padx=3, pady=5)
    
    btn_compare_strat = ttk.Button(button_frame, text="Compare Strategies", width=btn_width, command=handle_strategy_comparison)
    btn_compare_strat.grid(row=0, column=6, padx=3, pady=5)
    
    btn_ml = ttk.Button(button_frame, text="ML Optimization", width=btn_width, command=handle_ml_optimization)
    btn_ml.grid(row=0, column=7, padx=3, pady=5)
    
    btn_close = ttk.Button(button_frame, text="Close", width=btn_width//2, command=on_closing)
    btn_close.grid(row=0, column=8, padx=3, pady=5)
    
    status_var = tk.StringVar(value="Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=2, column=0, sticky="ew")
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

def plot_rolling_window_performance(app_config, diagnostics, plt=plt, np=np):
    fig = plt.figure(figsize=getattr(app_config.gui, 'figure_size', (10, 8)))
    if 'window_metrics' not in diagnostics or not diagnostics['window_metrics']:
        plt.text(0.5, 0.5, "No rolling window metrics available", ha='center', va='center', fontsize=12)
        plt.tight_layout()
        return fig
    window_metrics = diagnostics['window_metrics']
    window_indices = [m['window_idx'] for m in window_metrics]
    smape_values = [m['smape'] for m in window_metrics]
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(window_indices, smape_values, marker='o', linestyle='-', color='blue')
    ax1.set_title('Rolling Window Validation Performance')
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('SMAPE (%)')
    ax1.grid(True)
    avg_smape = np.mean(smape_values)
    ax1.axhline(y=avg_smape, color='red', linestyle='--', label=f'Average SMAPE: {avg_smape:.2f}%')
    ax1.legend()
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(smape_values, bins=min(10, len(smape_values)), color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of SMAPE Values')
    ax2.set_xlabel('SMAPE (%)')
    ax2.set_ylabel('Frequency')
    min_smape = np.min(smape_values)
    max_smape = np.max(smape_values)
    median_smape = np.median(smape_values)
    std_smape = np.std(smape_values)
    stats_text = f"Min: {min_smape:.2f}%  Max: {max_smape:.2f}%\nMedian: {median_smape:.2f}%  StdDev: {std_smape:.2f}%"
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig





def main():
    try:
        from stock_predictor import forecast_module
        app_config = forecast_module.AppConfig()
        create_gui(app_config)
    except RuntimeError as e:
        logging.error(f"GPU initialization failed: {e}")
        print("Please use gui_wrapper_cpu.py if no GPU is available.")
    except ImportError as e:
        logging.error(f"Failed to import required modules: {e}")
        print(f"Error: Missing required modules. {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        print(f"Unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
