import tkinter as tk
from tkinter import ttk
from dataclasses import fields
import logging

def create_tab(parent_frame, config_obj, app_config=None):
    """
    Create the prediction tab contents
    
    Args:
        parent_frame: The parent frame to add content to
        config_obj: The PredictionConfig configuration object
        app_config: The full AppConfig object (optional, for synchronization)
    
    Returns:
        Dictionary containing the GUI elements
    """
    # Use app_config if provided, otherwise create a default (adjust based on your implementation)
    if app_config is None:
        from stock_predictor.config import AppConfig
        app_config = AppConfig()
    
    entry_dict = {}
    
    for f in fields(config_obj):
        row = ttk.Frame(parent_frame)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=f.name, width=25).pack(side="left")
        default_val = getattr(config_obj, f.name)
        
        if f.type == bool:
            var = tk.BooleanVar(value=default_val)
            widget = ttk.Checkbutton(row, variable=var)
            widget.pack(side="left")
            entry_dict[f.name] = var
            # For set_initial_data, update initial_data_period state
            if f.name == "set_initial_data":
                def toggle_initial_data_period(*args):
                    state = "disabled" if var.get() else "normal"
                    if 'initial_data_period' in entry_dict:
                        entry_dict['initial_data_period'].config(state=state)
                        if var.get():
                            entry_dict['initial_data_period'].delete(0, tk.END)
                            entry_dict['initial_data_period'].insert(0, str(app_config.learning.timestamp))
                var.trace_add('write', toggle_initial_data_period)
        else:
            widget = ttk.Entry(row)
            if f.name == "initial_data_period":
                widget.insert(0, str(app_config.learning.timestamp))
                widget.config(state="disabled")
            else:
                widget.insert(0, str(default_val))
            widget.pack(side="left", expand=True, fill="x")
            entry_dict[f.name] = widget
    
    # Add help info
    help_frame = ttk.LabelFrame(parent_frame, text="Prediction Settings Info")
    help_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    help_text = """
    predict_days: Number of days to forecast into the future
    
    start_forecast_from_backtest: When enabled, forecasts start from the backtesting date
    
    use_previous_close: Use the previous day's closing price as a feature
    
    set_initial_data: Use historical data as initial state for forecasting
    
    initial_data_period: Number of days of historical data to use as initial state (automatically set to match app_config.learning.timestamp)
    """
    
    ttk.Label(help_frame, text=help_text, wraplength=500, justify="left").pack(padx=10, pady=10)
    
    # Initial toggle of initial_data_period state
    if 'set_initial_data' in entry_dict and 'initial_data_period' in entry_dict:
        state = "disabled" if entry_dict['set_initial_data'].get() else "normal"
        entry_dict['initial_data_period'].config(state=state)
    
    return entry_dict