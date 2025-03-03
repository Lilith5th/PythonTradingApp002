import os
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import fields
import plot_gui
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
from sklearn.preprocessing import MinMaxScaler
import itertools

from stock_predictor import forecast_module

matplotlib.use('TkAgg')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_bool_str(value):
    if isinstance(value, bool):
        return value
    return value.strip().lower() == 'true'

def configure_device_resources(app_config, num_forecasts=1):
    """CPU-only resource allocation"""
    cpus = tf.config.list_physical_devices('CPU')
    tf.config.threading.set_inter_op_parallelism_threads(max(1, multiprocessing.cpu_count() // 2))
    tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
    tf.config.set_visible_devices(cpus, 'CPU')
    return ['/CPU:0'] * num_forecasts

def update_fields_for_preset(entries, preset):
    try:
        if "learning" not in entries:
            logging.error("Entries dictionary missing 'learning' key")
            messagebox.showwarning("Configuration Error", "Error: 'learning' tab data not found.")
            return
        learning_config = forecast_module.LearningConfig(preset=preset)
        for f in fields(learning_config):
            if f.name != "preset":
                default_val = getattr(learning_config, f.name)
                if f.name in entries["learning"]:
                    widget = entries["learning"][f.name]
                    if isinstance(widget, ttk.Entry):
                        widget.delete(0, tk.END)
                        widget.insert(0, str(default_val))
                    elif isinstance(widget, tk.BooleanVar):
                        widget.set(default_val)
    except Exception as e:
        logging.error(f"Error updating fields for preset: {e}")
        messagebox.showwarning("Configuration Error", f"Error updating preset fields: {e}")

def on_preset_change(event, entries):
    try:
        preset = entries["learning"]["preset"].get()
        update_fields_for_preset(entries, preset)
    except Exception as e:
        logging.error(f"Error in on_preset_change: {e}")
        messagebox.showwarning("Configuration Error", f"Error updating preset: {e}")

def create_fields_for_config(frame, config_obj):
    entry_dict = {}
    for f in fields(config_obj):
        row = ttk.Frame(frame)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=f.name, width=25).pack(side="left")
        default_val = getattr(config_obj, f.name)
        
        if f.name == "preset":
            widget = ttk.Combobox(row, values=["gpu-high-performance", "high-performance", "high", "medium", "low"], state="readonly")
            widget.set(str(default_val))
            widget.pack(side="left", expand=True, fill="x")
            entry_dict[f.name] = widget
        elif f.type == bool:
            var = tk.BooleanVar(value=default_val)
            widget = ttk.Checkbutton(row, variable=var)
            widget.pack(side="left")
            entry_dict[f.name] = var
        elif isinstance(default_val, str) and f.name.endswith("_method"):
            widget = ttk.Combobox(row, values=["minmax", "log", "sqrt", "cuberoot", "power10"], state="readonly")
            widget.set(default_val)
            widget.pack(side="left", expand=True, fill="x")
            entry_dict[f.name] = widget
        else:
            widget = ttk.Entry(row)
            widget.insert(0, str(default_val))
            widget.pack(side="left", expand=True, fill="x")
            entry_dict[f.name] = widget
    return entry_dict

def update_config_from_gui(entries, app_config):
    try:
        for section, config_obj in {
            "preferences": app_config.learning_pref,
            "prediction": app_config.prediction,
            "plot": app_config.plot,
            "learning": app_config.learning
        }.items():
            for f in fields(config_obj):
                if f.name in entries[section]:
                    value = entries[section][f.name].get()
                    current_val = getattr(config_obj, f.name)
                    if isinstance(current_val, bool):
                        setattr(config_obj, f.name, parse_bool_str(value))
                    elif isinstance(current_val, int):
                        setattr(config_obj, f.name, int(value))
                    elif isinstance(current_val, float):
                        setattr(config_obj, f.name, float(value))
                    else:
                        setattr(config_obj, f.name, value)
        logging.info("Configuration updated. Running on CPU.")
        return True
    except ValueError as e:
        messagebox.showerror("Configuration Error", f"Invalid input: {e}")
        return False
    except Exception as e:
        messagebox.showerror("Configuration Error", f"Error in configuration: {e}")
        return False

def run_forecast_in_subprocess(app_config, output_file, device):
    import os
    import logging
    import sys
    import tensorflow as tf
    import pickle
    from stock_predictor import forecast_module

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    with tf.device(device):
        try:
            plot_data = forecast_module.run_program(app_config)
            confidence_intervals = calculate_confidence_intervals({'predictions': plot_data[5]})
            enhanced_plot_data = plot_data + (confidence_intervals,)
            with open(output_file, 'wb') as f:
                pickle.dump(enhanced_plot_data, f)
        except Exception as e:
            logging.error(f"Forecast failed: {e}")
            with open(output_file, 'wb') as f:
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
    status_var.set(f"Running {len(scenarios)} forecast scenarios...")
    
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
    
    btn_run.config(state="normal")
    status_var.set(f"Completed {len(results)} forecasts")
    if results:
        plot_gui.create_parallel_plot_gui(app_config, results)
    return results

def auto_tune_hyperparameters(app_config, param_grid=None):
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'epoch': [50, 100, 200]
        }
    
    best_score = float('-inf')
    best_params = None
    
    param_combinations = [dict(zip(param_grid.keys(), values)) 
                         for values in itertools.product(*param_grid.values())]
    
    for params in param_combinations:
        config_copy = pickle.loads(pickle.dumps(app_config))
        for key, value in params.items():
            setattr(config_copy.learning, key, value)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            process = multiprocessing.Process(
                target=run_forecast_in_subprocess,
                args=(config_copy, temp_file, '/CPU:0')
            )
            process.start()
            process.join()
            
            if process.exitcode == 0:
                with open(temp_file, 'rb') as f:
                    result = pickle.load(f)
                os.unlink(temp_file)
                if not isinstance(result, dict) or 'error' not in result:
                    score = evaluate_forecast(result)
                    if score > best_score:
                        best_score = score
                        best_params = params
    
    if best_params:
        for key, value in best_params.items():
            setattr(app_config.learning, key, value)
    return best_params, best_score

def evaluate_forecast(result):
    smape_scores = result[6]
    return -np.mean(smape_scores)

def create_gui(app_config):
    root = tk.Tk()
    root.title("Dynamic Forecast Configuration (CPU)")
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    entries = {}
    for group_key, config_obj in [
        ("preferences", app_config.learning_pref),
        ("learning",    app_config.learning),
        ("prediction",  app_config.prediction),
        ("plot",        app_config.plot)
    ]:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=group_key.capitalize())
        entries[group_key] = create_fields_for_config(frame, config_obj)

    if "learning" in entries and "preset" in entries["learning"]:
        entries["learning"]["preset"].bind("<<ComboboxSelected>>", lambda e: on_preset_change(e, entries))

    status_var = tk.StringVar(value="Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

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

    def handle_forecast_result(exitcode, temp_file_path):
        btn_run.config(state="normal")
        if exitcode == 0 and os.path.exists(temp_file_path):
            with open(temp_file, 'rb') as f:
                data = pickle.load(f)
            os.unlink(temp_file_path)
            if isinstance(data, dict) and 'error' in data:
                messagebox.showerror("Error", f"Forecast simulation failed: {data['error']}")
                status_var.set("Forecast failed")
            else:
                status_var.set("Forecast complete - Displaying results")
                plot_gui.create_plot_gui_with_data(app_config, data[:-1])
        else:
            messagebox.showerror("Error", "Forecast simulation failed.")
            status_var.set("Forecast failed")

    def run_parallel_scenarios():
        scenarios = [
            {'learning_rate': 0.001, 'epoch': 100},
            {'learning_rate': 0.01, 'epoch': 150},
            {'learning_rate': 0.1, 'epoch': 200}
        ]
        if update_config_from_gui(entries, app_config):
            processes, temp_files = run_parallel_forecasts(app_config, scenarios, status_var, btn_run)
            root.after(100, lambda: handle_parallel_results(processes, temp_files, app_config, status_var, btn_run))

    def run_auto_tune():
        if update_config_from_gui(entries, app_config):
            status_var.set("Running hyperparameter optimization...")
            btn_run.config(state="disabled")
            def tune_callback():
                best_params, score = auto_tune_hyperparameters(app_config)
                status_var.set(f"Optimization complete. Best score: {score:.4f}")
                btn_run.config(state="normal")
                update_fields_for_preset(entries, app_config.learning.preset)
                messagebox.showinfo("Tuning Complete", f"Best parameters: {best_params}")
            threading.Thread(target=tune_callback, daemon=True).start()

    btn_run = ttk.Button(root, text="Run Forecast", command=on_run)
    btn_run.pack(pady=5)
    ttk.Button(root, text="Run Parallel Scenarios", command=run_parallel_scenarios).pack(pady=5)
    ttk.Button(root, text="Auto-tune Parameters", command=run_auto_tune).pack(pady=5)

    def on_closing():
        root.quit()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    app_config = forecast_module.AppConfig()
    create_gui(app_config)