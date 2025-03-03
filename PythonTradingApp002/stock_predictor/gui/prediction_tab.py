"""
Prediction tab for the stock prediction GUI.
Contains settings for prediction parameters.
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import fields
import logging

def create_tab(parent_frame, config_obj):
    """
    Create the prediction tab contents
    
    Args:
        parent_frame: The parent frame to add content to
        config_obj: The PredictionConfig configuration object
        
    Returns:
        Dictionary containing the GUI elements
    """
    entry_dict = {}
    
    # Create entries for all fields
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
        else:
            widget = ttk.Entry(row)
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
    
    initial_data_period: Number of days of historical data to use as initial state
    """
    
    ttk.Label(help_frame, text=help_text, wraplength=500, justify="left").pack(padx=10, pady=10)
    
    return entry_dict