"""
Improved GUI Wrapper Module for Stock Prediction Application

This module provides a consistent and robust GUI wrapper for the stock prediction
application, implementing improved error handling, tab synchronization, and
configuration management.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
import multiprocessing
import pickle
import tempfile
import traceback
import sys
import time

# Import configuration and utility modules
from stock_predictor.config import AppConfig
from stock_predictor.config_utils import update_config_from_gui, validate_config
from stock_predictor.tab_synchronizer import SynchronizedTabManager
from stock_predictor.error_handler import get_error_handler, ErrorAwareThread, catch_and_log_errors
from stock_predictor.data_handler import DataHandler

# Import GUI tab modules
from stock_predictor.gui import (
    preferences_tab, learning_tab, prediction_tab, plot_tab, 
    features_tab, rolling_window_tab, advanced_prediction_tab, strategy_tab
)

def configure_device_resources(app_config, num_forecasts=1):
    """
    GPU-optimized resource allocation for forecasting with improved error handling
    
    Args:
        app_config: Application configuration
        num_forecasts: Number of parallel forecasts to run
        
    Returns:
        List[str]: List of device strings to use for parallel processing
        
    Raises:
        RuntimeError: If no suitable devices are found
    """
    try:
        import tensorflow as tf
        
        # List available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logging.warning("No GPUs found. Will attempt to use CPU if available.")
            
            # Check if CPU is available
            cpus = tf.config.list_physical_devices('CPU')
            if not cpus:
                raise RuntimeError("No GPUs or CPUs found. Cannot proceed.")
            
            # Return CPU devices
            return ['/CPU:0'] * num_forecasts
        
        # Configure GPU memory growth
        devices = []
        for i, gpu in enumerate(gpus):
            try:
                # Enable memory growth to avoid allocating all memory at once
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Try to get memory info
                try:
                    device_name = f'GPU:{i}'
                    memory_info = tf.config.experimental.get_memory_info(device_name)
                    logging.debug(f"Memory info for {device_name}: {memory_info}")
                    # Limit to 80% of available memory
                    memory_limit = 0.8 * (memory_info.get('peak', 8192 * 1024 * 1024))
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Could not get memory info for {device_name}: {e}. Using dynamic growth.")
                    memory_limit = None
                
                # Set memory limit if available
                if memory_limit:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
                
                # Add device to list
                devices.append(f'/GPU:{i % len(gpus)}')
                
            except RuntimeError as e:
                logging.error(f"Could not configure GPU {i}: {e}")
        
        # If no GPUs could be configured, try CPU
        if not devices:
            logging.warning("Could not configure any GPUs. Falling back to CPU.")
            return ['/CPU:0'] * num_forecasts
        
        # Distribute forecasts across available devices
        return [devices[i % len(devices)] for i in range(num_forecasts)]
        
    except ImportError as e:
        logging.error(f"TensorFlow import error: {e}")
        raise RuntimeError("TensorFlow not available. Cannot configure devices.")
    except Exception as e:
        logging.error(f"Unexpected error configuring devices: {e}")
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Error configuring devices: {e}")


class ImprovedGUIWrapper:
    """
    Improved GUI wrapper for stock prediction application with robust error handling
    and synchronized configuration management
    """
    
    def __init__(self):
        """Initialize the GUI wrapper"""
        # Initialize configuration
        self.app_config = AppConfig()
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Dynamic Forecast Configuration (Improved)")
        self.root.geometry("900x650")
        
        # Initialize error handler
        self.error_handler = get_error_handler(self.root)
        
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create synchronized tab manager
        self.tab_manager = SynchronizedTabManager(self.root, self.app_config)
        self.notebook = self.tab_manager.create_notebook()
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create data handler instance
        try:
            self.data_handler = DataHandler(self.app_config)
        except Exception as e:
            self.error_handler.log_error(e, "DataHandler initialization")
            self.data_handler = None
            messagebox.showwarning(
                "Data Loading Error",
                f"Error initializing data handler: {str(e)}\n\nPlease check your data files."
            )
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, sticky="ew")
        
        # Create button frame
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.button_frame.grid_propagate(False)
        self.button_frame.configure(height=40)
        
        # Create tabs in desired order
        self._create_tabs()
        
        # Create buttons
        self._create_buttons()
        
        # Set up synchronization
        self.tab_manager.setup_synchronization()
        
        # Set up close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_tabs(self):
        """Create all tabs for the application"""
        try:
            # Order: Preferences, Learning, Features, Rolling Window, Prediction, Plot, Advanced Prediction, Strategy
            self.tab_manager.add_tab("learning_pref", preferences_tab, self.app_config.learning_pref)
            self.tab_manager.add_tab("learning", learning_tab, self.app_config.learning)
            self.tab_manager.add_tab("features", features_tab, self.app_config.feature_selection)
            self.tab_manager.add_tab("rolling_window", rolling_window_tab, self.app_config.rolling_window)
            self.tab_manager.add_tab("prediction", prediction_tab, self.app_config.prediction)
            self.tab_manager.add_tab("plot", plot_tab, self.app_config.plot)
            self.tab_manager.add_tab("advanced_prediction", advanced_prediction_tab, self.app_config.prediction_advanced)
            self.tab_manager.add_tab("strategy", strategy_tab, self.app_config.strategy)
            
            # Set up initial tab states
            # The SynchronizedTabManager handles disabling the Features tab if use_features is not checked
            
        except Exception as e:
            self.error_handler.log_error(e, "Tab creation")
            messagebox.showerror(
                "Initialization Error",
                f"Failed to create UI tabs: {str(e)}"
            )
    
    def _create_buttons(self):
        """Create buttons for the application"""
        # Configure button layout
        button_count = 9  # Total buttons
        for i in range(button_count):
            self.button_frame.grid_columnconfigure(i, weight=1)
        
        btn_width = 18
        
        # Create buttons
        self.btn_run = ttk.Button(
            self.button_frame,
            text="Run Forecast",
            width=btn_width,
            command=self.on_run
        )
        self.btn_run.grid(row=0, column=0, padx=3, pady=5)
        
        self.btn_parallel = ttk.Button(
            self.button_frame,
            text="Run Parallel Scenarios",
            width=btn_width,
            command=self.run_parallel_scenarios
        )
        self.btn_parallel.grid(row=0, column=1, padx=3, pady=5)
        
        self.btn_auto_tune = ttk.Button(
            self.button_frame,
            text="Auto-tune Parameters",
            width=btn_width,
            command=self.run_auto_tune
        )
        self.btn_auto_tune.grid(row=0, column=2, padx=3, pady=5)
        
        self.btn_compare = ttk.Button(
            self.button_frame,
            text="Compare Models",
            width=btn_width,
            command=self.run_comparison
        )
        self.btn_compare.grid(row=0, column=3, padx=3, pady=5)
        
        self.btn_backtest = ttk.Button(
            self.button_frame,
            text="Run Backtesting",
            width=btn_width,
            command=self.run_backtesting
        )
        self.btn_backtest.grid(row=0, column=4, padx=3, pady=5)
        
        # Strategy buttons
        self.btn_strategy = ttk.Button(
            self.button_frame,
            text="Run Strategy Backtest",
            width=btn_width,
            command=self.handle_strategy_backtest
        )
        self.btn_strategy.grid(row=0, column=5, padx=3, pady=5)
        
        self.btn_compare_strat = ttk.Button(
            self.button_frame,
            text="Compare Strategies",
            width=btn_width,
            command=self.handle_strategy_comparison
        )
        self.btn_compare_strat.grid(row=0, column=6, padx=3, pady=5)
        
        self.btn_ml = ttk.Button(
            self.button_frame,
            text="ML Optimization",
            width=btn_width,
            command=self.handle_ml_optimization
        )
        self.btn_ml.grid(row=0, column=7, padx=3, pady=5)
        
        self.btn_close = ttk.Button(
            self.button_frame,
            text="Close",
            width=btn_width//2,
            command=self.on_closing
        )
        self.btn_close.grid(row=0, column=8, padx=3, pady=5)
    
    def update_config_from_gui(self):
        """
        Update configuration from GUI with validation
        
        Returns:
            bool: True if update was successful
        """
        # Update configuration with validation
        success, errors = self.tab_manager.update_config_from_gui(validate=True)
        
        if not success:
            # Show error message with all validation errors
            error_message = "Configuration validation failed:\n\n"
            error_message += "\n".join(f"� {error}" for error in errors)
            messagebox.showerror("Validation Error", error_message)
            return False
        
        # Additional validation checks
        config_errors = validate_config(self.app_config)
        if config_errors:
            # Show error message with all validation errors
            error_message = "Configuration validation failed:\n\n"
            error_message += "\n".join(f"� {error}" for error in config_errors)
            messagebox.showerror("Validation Error", error_message)
            return False
        
        return True
    
    @catch_and_log_errors(context="Run forecast")
    def on_run(self):
        """Run the forecast"""
        if not self.update_config_from_gui():
            return
        
        # Disable the run button and update status
        self.btn_run.config(state="disabled")
        self.status_var.set("Preparing forecast...")
        
        # Create a temporary file for the results
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        # Configure devices
        try:
            devices = configure_device_resources(self.app_config, num_forecasts=1)
            device = devices[0]
        except Exception as e:
            self.btn_run.config(state="normal")
            self.status_var.set("Device configuration failed")
            self.error_handler.show_error(
                "Device Error",
                f"Failed to configure GPU/CPU: {str(e)}",
                "Please check your hardware and TensorFlow installation."
            )
            return
        
        # Run forecast in a separate thread
        forecast_thread = ErrorAwareThread(
            target=self.run_forecast_in_subprocess,
            args=(self.app_config, temp_file_path, device),
            show_dialog=True,
            context="Forecast execution"
        )
        forecast_thread.start()
        
        # Schedule result handling
        self.check_forecast_progress(forecast_thread, temp_file_path)
    
    def check_forecast_progress(self, thread, temp_file_path):
        """
        Check progress of forecast thread and handle results when complete
        
        Args:
            thread: Thread running the forecast
            temp_file_path: Path to temporary results file
        """
        if thread.is_alive():
            # Still running, check again in 100ms
            self.root.after(100, lambda: self.check_forecast_progress(thread, temp_file_path))
            return
        
        # Thread completed, handle result
        self.handle_forecast_result(thread.get_error() is None, temp_file_path)
    
    def run_forecast_in_subprocess(self, app_config, output_file_path, device):
        """
        Run a forecast in a subprocess
        
        Args:
            app_config: Application configuration
            output_file_path: Path to output file
            device: Device to use for forecast
        """
        from stock_predictor.forecast_module import run_program
        
        try:
            # Configure environment
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
            # Run the forecast
            with tf.device(device):
                plot_data = run_program(app_config)
                
                # Add confidence intervals
                try:
                    from stock_predictor.uncertainty import UncertaintyBase
                    confidence_intervals = UncertaintyBase().calculate_confidence_intervals(
                        plot_data[5]  # predictions
                    )
                except Exception as e:
                    logging.warning(f"Failed to calculate confidence intervals: {e}")
                    confidence_intervals = {
                        'mean': plot_data[3],  # ensemble_mean
                        'lower': plot_data[3] - plot_data[4],  # mean - std
                        'upper': plot_data[3] + plot_data[4]   # mean + std
                    }
                
                # Add confidence intervals to plot data
                enhanced_plot_data = plot_data + (confidence_intervals,)
                
                # Save results
                with open(output_file_path, 'wb') as f:
                    pickle.dump(enhanced_plot_data, f)
        
        except Exception as e:
            logging.error(f"Forecast failed on {device}: {e}")
            logging.error(traceback.format_exc())
            
            # Save error
            with open(output_file_path, 'wb') as f:
                pickle.dump({"error": str(e)}, f)
    
    def handle_forecast_result(self, success, temp_file_path):
        """
        Handle forecast result
        
        Args:
            success: Whether the forecast was successful
            temp_file_path: Path to temporary results file
        """
        self.btn_run.config(state="normal")
        
        if not success:
            self.status_var.set("Forecast failed")
            return
        
        if os.path.exists(temp_file_path):
            try:
                with open(temp_file_path, 'rb') as f:
                    data = pickle.load(f)
                
                os.unlink(temp_file_path)
                
                if isinstance(data, dict) and 'error' in data:
                    messagebox.showerror("Error", f"Forecast simulation failed: {data['error']}")
                    self.status_var.set("Forecast failed")
                else:
                    self.status_var.set("Forecast complete - Displaying results")
                    
                    # Import here to avoid circular imports
                    from stock_predictor import plot_gui
                    plot_gui.create_plot_gui_with_data(self.app_config, data)
            
            except Exception as e:
                self.error_handler.log_error(e, "Processing forecast results")
                self.status_var.set("Error processing forecast results")
                messagebox.showerror(
                    "Error",
                    f"Failed to process forecast results: {str(e)}"
                )
        else:
            self.status_var.set("Forecast failed - No results file")
            messagebox.showerror(
                "Error",
                "Forecast simulation failed: No results file found"
            )
    
    @catch_and_log_errors(context="Parallel scenarios")
    def run_parallel_scenarios(self):
        """Run parallel forecast scenarios"""
        if not self.update_config_from_gui():
            return
        
        # Disable button and update status
        self.btn_parallel.config(state="disabled")
        self.status_var.set("Running parallel forecast scenarios...")
        
        # Define scenarios
        scenarios = [
            {'learning_rate': 0.001, 'epoch': 100},
            {'learning_rate': 0.01, 'epoch': 150},
            {'learning_rate': 0.1, 'epoch': 200},
        ]
        
        # Create temporary files for results
        temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in scenarios]
        
        # Configure devices
        try:
            devices = configure_device_resources(self.app_config, num_forecasts=len(scenarios))
        except Exception as e:
            self.btn_parallel.config(state="normal")
            self.status_var.set("Device configuration failed")
            self.error_handler.show_error(
                "Device Error",
                f"Failed to configure GPU/CPU: {str(e)}",
                "Please check your hardware and TensorFlow installation."
            )
            return
        
        # Run parallel forecasts
        processes = []
        for i, (scenario_config, temp_file) in enumerate(zip(scenarios, temp_files)):
            # Create a copy of the configuration with scenario-specific settings
            config_copy = pickle.loads(pickle.dumps(self.app_config))
            for key, value in scenario_config.items():
                setattr(config_copy.learning, key, value)
            
            # Start process for this scenario
            p = multiprocessing.Process(
                target=self.run_forecast_in_subprocess,
                args=(config_copy, temp_file, devices[i % len(devices)])
            )
            processes.append(p)
            p.start()
        
        # Schedule result handling
        self.check_parallel_progress(processes, temp_files, 0)
    
    def check_parallel_progress(self, processes, temp_files, completed):
        """
        Check progress of parallel processes
        
        Args:
            processes: List of processes
            temp_files: List of temporary files
            completed: Number of completed processes
        """
        # Check for newly completed processes
        new_completed = sum(1 for p in processes if not p.is_alive())
        
        if new_completed > completed:
            # Update status with progress
            self.status_var.set(f"Running parallel scenarios: {new_completed}/{len(processes)} completed")
            completed = new_completed
        
        if completed < len(processes):
            # Still running, check again in 100ms
            self.root.after(100, lambda: self.check_parallel_progress(processes, temp_files, completed))
        else:
            # All processes completed, handle results
            self.handle_parallel_results(processes, temp_files)
    
    def handle_parallel_results(self, processes, temp_files):
        """
        Handle results from parallel forecast scenarios
        
        Args:
            processes: List of processes
            temp_files: List of temporary files
        """
        results = []
        
        # Wait for all processes to complete (should already be done)
        for p in processes:
            p.join()
        
        # Process results
        for p, temp_file in zip(processes, temp_files):
            if p.exitcode == 0 and os.path.exists(temp_file):
                try:
                    with open(temp_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict) and 'error' in data:
                        logging.error(f"Scenario failed: {data['error']}")
                    else:
                        results.append(data)
                
                except Exception as e:
                    logging.error(f"Error processing scenario result: {e}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        
        # Re-enable button and update status
        self.btn_parallel.config(state="normal")
        self.status_var.set(f"Completed {len(results)}/{len(processes)} parallel forecasts")
        
        # Display results if any
        if results:
            from stock_predictor import plot_gui
            plot_gui.create_parallel_plot_gui(self.app_config, results)
        else:
            messagebox.showwarning(
                "Parallel Forecast",
                "No valid results from parallel forecasts. Check log for details."
            )
    
    @catch_and_log_errors(context="Auto-tune")
    def run_auto_tune(self):
        """Run hyperparameter auto-tuning"""
        if not self.update_config_from_gui():
            return
        
        # Disable button and update status
        self.btn_auto_tune.config(state="disabled")
        self.status_var.set("Running hyperparameter optimization...")
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'epoch': [50, 100, 200],
            'l2_reg': [0.001, 0.005, 0.01],
            'num_layers': [2, 3, 4]
        }
        
        # Run optimization in a separate thread
        tune_thread = ErrorAwareThread(
            target=self.auto_tune_hyperparameters,
            args=(self.app_config, param_grid),
            show_dialog=True,
            context="Hyperparameter tuning"
        )
        tune_thread.start()
        
        # Schedule result handling
        self.check_tuning_progress(tune_thread)
    
    def check_tuning_progress(self, thread):
        """
        Check progress of auto-tuning thread
        
        Args:
            thread: Thread running the auto-tuning
        """
        if thread.is_alive():
            # Still running, check again in 100ms
            self.root.after(100, lambda: self.check_tuning_progress(thread))
            return
        
        # Thread completed, handle result
        result = thread.get_result()
        error = thread.get_error()
        
        # Re-enable button
        self.btn_auto_tune.config(state="normal")
        
        if error:
            self.status_var.set("Hyperparameter optimization failed")
            return
        
        # Unpack result
        best_params, score = result
        
        # Update status
        self.status_var.set(f"Optimization complete. Best score: {score:.4f}")
        
        # Update GUI with best parameters
        if best_params:
            self.tab_manager.update_gui_from_config()
            
            # Show results
            messagebox.showinfo(
                "Tuning Complete",
                f"Best parameters: {best_params}\nScore: {score:.4f}"
            )
    
    def auto_tune_hyperparameters(self, app_config, param_grid):
        """
        Run hyperparameter auto-tuning
        
        Args:
            app_config: Application configuration
            param_grid: Parameter grid to search
            
        Returns:
            Tuple: (best_params, best_score)
        """
        import itertools
        from stock_predictor.forecast_module import run_program
        
        best_score = float('-inf')
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), values)) 
                             for values in itertools.product(*param_grid.values())]
        
        # Create temporary files for results
        temp_files = [tempfile.NamedTemporaryFile(delete=False).name 
                     for _ in param_combinations]
        
        # Configure devices
        devices = configure_device_resources(app_config, num_forecasts=len(param_combinations))
        
        # Start processes for each parameter combination
        processes = []
        for i, (params, temp_file) in enumerate(zip(param_combinations, temp_files)):
            # Create a copy of the configuration with these parameters
            config_copy = pickle.loads(pickle.dumps(app_config))
            for key, value in params.items():
                setattr(config_copy.learning, key, value)
            
            # Start process
            process = multiprocessing.Process(
                target=self.run_forecast_in_subprocess,
                args=(config_copy, temp_file, devices[i % len(devices)])
            )
            processes.append((process, params, temp_file))
            process.start()
        
        # Wait for all processes to complete
        for process, params, temp_file in processes:
            process.join()
            
            # Check result
            if process.exitcode == 0 and os.path.exists(temp_file):
                try:
                    with open(temp_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Evaluate result if valid
                    if not isinstance(result, dict) or 'error' not in result:
                        score = self.evaluate_forecast(result)
                        if score > best_score:
                            best_score = score
                            best_params = params
                            logging.info(f"New best parameters: {params}, score: {score:.4f}")
                
                except Exception as e:
                    logging.error(f"Error processing tuning result: {e}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        
        # Update configuration with best parameters
        if best_params:
            for key, value in best_params.items():
                setattr(app_config.learning, key, value)
        
        return best_params, best_score
    
    def evaluate_forecast(self, result):
        """
        Evaluate a forecast result
        
        Args:
            result: Forecast result
            
        Returns:
            float: Evaluation score (negative mean SMAPE)
        """
        # Extract SMAPE scores from result
        smape_scores = result[6]
        
        # Return negative mean (so higher is better)
        return -np.mean(smape_scores)
    
    @catch_and_log_errors(context="Model comparison")
    def run_comparison(self):
        """Run model architecture comparison"""
        if not self.update_config_from_gui():
            return
        
        # Disable button and update status
        self.btn_compare.config(state="disabled")
        self.status_var.set("Running model comparison...")
        
        # Run comparison in a separate thread
        compare_thread = ErrorAwareThread(
            target=self.compare_models,
            args=(self.app_config,),
            show_dialog=True,
            context="Model comparison"
        )
        compare_thread.start()
        
        # Schedule result handling
        self.check_comparison_progress(compare_thread)
    
    def check_comparison_progress(self, thread):
        """
        Check progress of model comparison thread
        
        Args:
            thread: Thread running the comparison
        """
        if thread.is_alive():
            # Still running, check again in 100ms
            self.root.after(100, lambda: self.check_comparison_progress(thread))
            return
        
        # Thread completed, handle result
        result = thread.get_result()
        error = thread.get_error()
        
        # Re-enable button
        self.btn_compare.config(state="normal")
        
        if error:
            self.status_var.set("Model comparison failed")
            return
        
        # Update status
        self.status_var.set("Model comparison complete")
        
        # Display results
        if result:
            from stock_predictor import plot_gui
            plot_gui.create_model_comparison_gui(self.app_config, result)
        else:
            messagebox.showwarning(
                "Model Comparison",
                "No valid results from model comparison. Check log for details."
            )
    
    def compare_models(self, app_config):
        """
        Compare different model architectures
        
        Args:
            app_config: Application configuration
            
        Returns:
            List: List of (model_name, result) tuples
        """
        from stock_predictor.model_factory import ModelFactory
        
        # Define models to compare
        model_configs = [
            {"model_type": "lstm", "learning_rate": 0.001},
            {"model_type": "gru", "learning_rate": 0.001},
            {"model_type": "bilstm", "learning_rate": 0.001}
        ]
        
        # Advanced models if enabled
        if app_config.prediction_advanced.use_ensemble_methods:
            model_configs.append({"model_type": "transformer", "learning_rate": 0.001})
        
        if app_config.prediction_advanced.enable_uncertainty_quantification:
            model_configs.append({"model_type": app_config.prediction_advanced.uncertainty_method, "learning_rate": 0.001})
        
        # Create temporary files for results
        temp_files = [tempfile.NamedTemporaryFile(delete=False).name for _ in model_configs]
        
        # Configure devices
        devices = configure_device_resources(app_config, num_forecasts=len(model_configs))
        
        # Start processes for each model
        processes = []
        for i, (model_config, temp_file) in enumerate(zip(model_configs, temp_files)):
            # Create a copy of the configuration with this model type
            config_copy = pickle.loads(pickle.dumps(app_config))
            for key, value in model_config.items():
                setattr(config_copy.learning, key, value)
            
            # Start process
            p = multiprocessing.Process(
                target=self.run_forecast_in_subprocess,
                args=(config_copy, temp_file, devices[i % len(devices)])
            )
            processes.append((p, model_config, temp_file))
            p.start()
        
        # Wait for all processes to complete
        results = []
        for p, model_config, temp_file in processes:
            p.join()
            
            # Check result
            if p.exitcode == 0 and os.path.exists(temp_file):
                try:
                    with open(temp_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if not isinstance(data, dict) or 'error' not in data:
                        model_name = model_config["model_type"]
                        results.append((model_name, data))
                        logging.info(f"Model {model_name} comparison completed successfully")
                
                except Exception as e:
                    logging.error(f"Error processing model result: {e}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        
        return results
    
    @catch_and_log_errors(context="Backtesting")
    def run_backtesting(self):
        """Run backtesting"""
        if not self.update_config_from_gui():
            return
        
        # Disable button and update status
        self.btn_backtest.config(state="disabled")
        self.status_var.set("Running backtesting...")
        
        # Run backtesting in a separate thread
        backtest_thread = ErrorAwareThread(
            target=self.run_backtest,
            args=(self.app_config,),
            show_dialog=True,
            context="Backtesting"
        )
        backtest_thread.start()
        
        # Schedule result handling
        self.check_backtest_progress(backtest_thread)
    
    def check_backtest_progress(self, thread):
        """
        Check progress of backtesting thread
        
        Args:
            thread: Thread running the backtesting
        """
        if thread.is_alive():
            # Still running, check again in 100ms
            self.root.after(100, lambda: self.check_backtest_progress(thread))
            return
        
        # Thread completed, handle result
        result = thread.get_result()
        error = thread.get_error()
        
        # Re-enable button
        self.btn_backtest.config(state="normal")
        
        if error:
            self.status_var.set("Backtesting failed")
            return
        
        # Update status
        self.status_var.set("Backtesting complete")
        
        # Display results
        if result:
            from stock_predictor import plot_gui
            plot_gui.create_backtest_plot_gui(self.app_config, result)
        else:
            messagebox.showwarning(
                "Backtesting",
                "No valid results from backtesting. Check log for details."
            )
    
    def run_backtest(self, app_config):
        """
        Run backtesting
        
        Args:
            app_config: Application configuration
            
        Returns:
            Any: Backtesting result
        """
        # Create a copy of the configuration with backtesting enabled
        config_copy = pickle.loads(pickle.dumps(app_config))
        config_copy.backtest = {
            'enable_backtesting': True,
            'test_periods': 30,
            'walk_forward': True
        }
        
        # Create temporary file for result
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        # Configure device
        devices = configure_device_resources(config_copy, num_forecasts=1)
        device = devices[0]
        
        # Run backtesting
        process = multiprocessing.Process(
            target=self.run_forecast_in_subprocess,
            args=(config_copy, temp_file_path, device)
        )
        process.start()
        process.join()
        
        # Check result
        if process.exitcode == 0 and os.path.exists(temp_file_path):
            try:
                with open(temp_file_path, 'rb') as f:
                    backtest_data = pickle.load(f)
                
                os.unlink(temp_file_path)
                
                if isinstance(backtest_data, dict) and 'error' in backtest_data:
                    logging.error(f"Backtesting failed: {backtest_data['error']}")
                    return None
                
                return backtest_data
            
            except Exception as e:
                logging.error(f"Error processing backtesting result: {e}")
                return None
        
        return None
    
    @catch_and_log_errors(context="Strategy backtest")
    def handle_strategy_backtest(self):
        """Run strategy backtest"""
        if not self.update_config_from_gui():
            return
        
        # Check if data handler is available
        if self.data_handler is None:
            messagebox.showerror(
                "Error",
                "Data handler not available. Please check your data files."
            )
            return
        
        # Import strategy module
        try:
            from stock_predictor.strategy_gui_integration import run_strategy_backtest
            run_strategy_backtest(self.app_config, self.data_handler)
        except ImportError as e:
            messagebox.showerror(
                "Module Error",
                f"Strategy module not available: {str(e)}"
            )
        except Exception as e:
            self.error_handler.log_error(e, "Strategy backtest")
            messagebox.showerror(
                "Strategy Error",
                f"Error running strategy backtest: {str(e)}"
            )
    
    @catch_and_log_errors(context="Strategy comparison")
    def handle_strategy_comparison(self):
        """Run strategy comparison"""
        if not self.update_config_from_gui():
            return
        
        # Check if data handler is available
        if self.data_handler is None:
            messagebox.showerror(
                "Error",
                "Data handler not available. Please check your data files."
            )
            return
        
        # Import strategy module
        try:
            from stock_predictor.strategy_gui_integration import run_strategy_comparison
            run_strategy_comparison(self.app_config, self.data_handler)
        except ImportError as e:
            messagebox.showerror(
                "Module Error",
                f"Strategy module not available: {str(e)}"
            )
        except Exception as e:
            self.error_handler.log_error(e, "Strategy comparison")
            messagebox.showerror(
                "Strategy Error",
                f"Error running strategy comparison: {str(e)}"
            )
    
    @catch_and_log_errors(context="ML optimization")
    def handle_ml_optimization(self):
        """Run ML strategy optimization"""
        if not self.update_config_from_gui():
            return
        
        # Check if ML optimization is enabled
        if not self.app_config.strategy.enable_ml_optimization:
            messagebox.showinfo(
                "ML Optimization",
                "Machine learning optimization is not enabled. "
                "Please enable it in the Strategy tab first."
            )
            return
        
        # Check if data handler is available
        if self.data_handler is None:
            messagebox.showerror(
                "Error",
                "Data handler not available. Please check your data files."
            )
            return
        
        # Import strategy module
        try:
            from stock_predictor.strategy_gui_integration import run_ml_optimization
            run_ml_optimization(self.app_config, self.data_handler)
        except ImportError as e:
            messagebox.showerror(
                "Module Error",
                f"Strategy module not available: {str(e)}"
            )
        except Exception as e:
            self.error_handler.log_error(e, "ML optimization")
            messagebox.showerror(
                "Strategy Error",
                f"Error running ML optimization: {str(e)}"
            )
    
    def on_closing(self):
        """Handle window closing"""
        try:
            # Close any open threads
            logging.info("Closing application")
            
            # Quit and destroy
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logging.error(f"Error closing application: {e}")
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            logging.error(traceback.format_exc())


def main():
    """Main function to run the application"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        # Create and run the improved GUI wrapper
        app = ImprovedGUIWrapper()
        app.run()
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        
        # Show error dialog
        try:
            messagebox.showerror(
                "Critical Error",
                f"An unexpected error occurred: {str(e)}\n\n"
                f"Please check the logs for details."
            )
        except:
            print(f"Critical error: {e}")
            print(traceback.format_exc())


if __name__ == '__main__':
    main()