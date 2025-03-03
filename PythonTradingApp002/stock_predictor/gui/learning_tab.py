"""
Learning tab for the stock prediction GUI.
Contains settings for learning parameters.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import fields
import logging

def create_tab(parent_frame, config_obj):
    """
    Create the learning tab contents
    
    Args:
        parent_frame: The parent frame to add content to
        config_obj: The LearningConfig configuration object
        
    Returns:
        Dictionary containing the GUI elements
    """
    entry_dict = {}
    manual_batch_size_widget = None
    
    for f in fields(config_obj):
        row = ttk.Frame(parent_frame)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=f.name, width=25).pack(side="left")
        default_val = getattr(config_obj, f.name)
        
        if f.name == "preset":
            widget = ttk.Combobox(row, values=["gpu-high-performance", "high-performance", "high", "medium", "low"], state="readonly")
            widget.set(str(default_val))
            widget.pack(side="left", expand=True, fill="x")
            entry_dict[f.name] = widget
        
        elif f.name == "auto_batch_size":
            var = tk.BooleanVar(value=default_val)
            widget = ttk.Checkbutton(row, text="Enable Auto Batch Size", variable=var)
            widget.pack(side="left")
            entry_dict[f.name] = var
        
        elif f.name == "manual_batch_size":
            widget = ttk.Entry(row)
            widget.insert(0, str(default_val))
            widget.pack(side="left", expand=True, fill="x")
            manual_batch_size_widget = widget
            entry_dict[f.name] = widget
        
        elif f.name == "use_features":
            var = tk.BooleanVar(value=default_val)
            widget = ttk.Checkbutton(row, text="Use Learning Features ", variable=var)
            widget.pack(side="left")
            entry_dict[f.name] = var
            
        elif f.type == bool:
            var = tk.BooleanVar(value=default_val)
            widget = ttk.Checkbutton(row, variable=var)
            widget.pack(side="left")
            entry_dict[f.name] = var
        
        else:
            widget = ttk.Entry(row)
            widget.insert(0, str(default_val))
            widget.pack(side="left", expand=True, fill="x")
            entry_dict[f.name] = widget
    
    # Add dynamic state management for manual batch size
    if "auto_batch_size" in entry_dict and manual_batch_size_widget:
        def update_state(*args):
            if entry_dict["auto_batch_size"].get():
                manual_batch_size_widget.config(state='disabled')
            else:
                manual_batch_size_widget.config(state='normal')
        
        entry_dict["auto_batch_size"].trace_add('write', update_state)
        # Initial state
        if entry_dict["auto_batch_size"].get():
            manual_batch_size_widget.config(state='disabled')
    
    return entry_dict

def update_fields_from_config(entries, config_obj):
    """
    Update the GUI fields based on the configuration
    
    Args:
        entries: Dictionary of GUI entries
        config_obj: Configuration object with current values
    """
    for f in fields(config_obj):
        if f.name != "preset" and f.name in entries:
            value = getattr(config_obj, f.name)
            widget = entries[f.name]
            
            if isinstance(widget, tk.BooleanVar):
                widget.set(value)
            elif isinstance(widget, ttk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, str(value))
            elif isinstance(widget, ttk.Combobox):
                widget.set(value)

def update_fields_for_preset(entries, preset):
    """
    Update fields based on the selected preset
    
    Args:
        entries: Dictionary of GUI entries
        preset: Selected preset name
    """
    try:
        # Import at the function level to avoid circular imports
        from stock_predictor import forecast_module
        
        if "preset" not in entries:
            logging.error("Entries dictionary missing 'preset' key")
            messagebox.showwarning("Configuration Error", "Error: 'preset' not found.")
            return
            
        learning_config = forecast_module.LearningConfig(preset=preset)
        for f in fields(learning_config):
            if f.name != "preset":
                default_val = getattr(learning_config, f.name)
                if f.name in entries:
                    widget = entries[f.name]
                    if isinstance(widget, ttk.Entry):
                        widget.delete(0, tk.END)
                        widget.insert(0, str(default_val))
                    elif isinstance(widget, tk.BooleanVar):
                        widget.set(default_val)
    except Exception as e:
        logging.error(f"Error updating fields for preset: {e}")
        messagebox.showwarning("Configuration Error", f"Error updating preset fields: {e}")

def on_preset_change(event, entries):
    """
    Handler for preset combobox selection change
    
    Args:
        event: The event object
        entries: Dictionary of GUI entries
    """
    try:
        preset = entries["preset"].get()
        update_fields_for_preset(entries, preset)
    except Exception as e:
        logging.error(f"Error in on_preset_change: {e}")
        messagebox.showwarning("Configuration Error", f"Error updating preset: {e}")

def setup_events(entries, notebook):
    """
    Set up event handlers for the learning tab
    
    Args:
        entries: Dictionary of GUI entries
        notebook: The main notebook widget
    """
    if "preset" in entries:
        entries["preset"].bind("<<ComboboxSelected>>", lambda e: on_preset_change(e, entries))