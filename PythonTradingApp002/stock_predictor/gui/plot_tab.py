"""
Plot tab for the stock prediction GUI.
Contains settings for plot parameters.
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import fields
import logging

def create_tab(parent_frame, config_obj):
    """
    Create the plot tab contents
    
    Args:
        parent_frame: The parent frame to add content to
        config_obj: The PlotConfig configuration object
        
    Returns:
        Dictionary containing the GUI elements
    """
    entry_dict = {}
    
    # Group settings by category
    groups = {
        "General": ["num_predictions_shown", "plot_mean", "plot_confidence_interval", "show_diagnostics"],
        "Appearance": ["line_width_train", "train_color", "forecast_color", "future_color", "mean_color"]
    }
    
    # Create frames for each group
    for group_name, field_names in groups.items():
        group_frame = ttk.LabelFrame(parent_frame, text=group_name)
        group_frame.pack(fill="x", padx=10, pady=5)
        
        for f_name in field_names:
            row = ttk.Frame(group_frame)
            row.pack(fill="x", padx=5, pady=2)
            ttk.Label(row, text=f_name, width=25).pack(side="left")
            
            # Get field and default value
            for f in fields(config_obj):
                if f.name == f_name:
                    default_val = getattr(config_obj, f.name)
                    
                    if f.type == bool:
                        var = tk.BooleanVar(value=default_val)
                        widget = ttk.Checkbutton(row, variable=var)
                        widget.pack(side="left")
                        entry_dict[f.name] = var
                    elif f.name.endswith("_color"):
                        # For colors, use a combobox with common colors
                        colors = ["black", "red", "green", "blue", "yellow", "orange", 
                                 "purple", "gray", "cyan", "magenta"]
                        widget = ttk.Combobox(row, values=colors, state="readonly")
                        widget.set(default_val)
                        widget.pack(side="left", expand=True, fill="x")
                        entry_dict[f.name] = widget
                    else:
                        widget = ttk.Entry(row)
                        widget.insert(0, str(default_val))
                        widget.pack(side="left", expand=True, fill="x")
                        entry_dict[f.name] = widget
                    
                    break
    
    # Add help information
    info_frame = ttk.LabelFrame(parent_frame, text="Plot Settings Help")
    info_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    help_text = """
    num_predictions_shown: Number of top simulations to display on the plot
    
    plot_mean: Show the mean prediction line
    
    plot_confidence_interval: Show confidence intervals around predictions
    
    line_width_train: Width of the training data line
    
    train_color: Color for training data line
    
    forecast_color: Color for forecast line
    
    future_color: Color for actual future data line
    
    mean_color: Color for mean prediction line
    
    show_diagnostics: Enable diagnostics visualization
    """
    
    ttk.Label(info_frame, text=help_text, wraplength=400, justify="left").pack(padx=10, pady=10)
    
    return entry_dict

def update_fields_from_config(entries, config_obj):
    """
    Update the GUI fields from the current config values
    
    Args:
        entries: Dictionary of GUI entries
        config_obj: Configuration object with current values
    """
    for f in fields(config_obj):
        if f.name in entries:
            value = getattr(config_obj, f.name)
            widget = entries[f.name]
            
            if isinstance(widget, tk.BooleanVar):
                widget.set(value)
            elif isinstance(widget, ttk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, str(value))
            elif isinstance(widget, ttk.Combobox):
                widget.set(value)