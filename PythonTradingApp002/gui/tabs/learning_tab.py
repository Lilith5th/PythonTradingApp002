import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import fields
import logging

def create_tab(parent_frame, config_obj, app_config=None):
    """
    Create the learning tab contents
    
    Args:
        parent_frame: The parent frame to add content to
        config_obj: The LearningConfig configuration object
        app_config: The full AppConfig object (optional, for synchronization)
    
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
        elif f.name == "use_log_transformation":
            var = tk.BooleanVar(value=default_val)
            widget = ttk.Checkbutton(row, text="Use Logarithmic Price Transformation", variable=var)
            widget.pack(side="left")
            entry_dict[f.name] = var
        elif f.name == "timestamp":
            widget = ttk.Entry(row)
            widget.insert(0, str(default_val))
            widget.pack(side="left", expand=True, fill="x")
            entry_dict[f.name] = widget
            # Synchronize with initial_data_period in Prediction tab
            if app_config is not None:
                def update_initial_data_period(*args):
                    try:
                        timestamp_value = int(widget.get())
                        app_config.learning.timestamp = timestamp_value
                        app_config.prediction.initial_data_period = timestamp_value
                        # Update Prediction tab if it exists
                        if hasattr(parent_frame.winfo_toplevel(), 'entries') and 'prediction' in parent_frame.winfo_toplevel().entries:
                            pred_entries = parent_frame.winfo_toplevel().entries['prediction']
                            if 'initial_data_period' in pred_entries:
                                pred_entries['initial_data_period'].delete(0, tk.END)
                                pred_entries['initial_data_period'].insert(0, str(timestamp_value))
                    except ValueError as e:
                        logging.warning(f"Invalid timestamp value: {e}")
                widget.bind("<FocusOut>", update_initial_data_period)
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
        if entry_dict["auto_batch_size"].get():
            manual_batch_size_widget.config(state='disabled')
    
    # Add help info
    help_frame = ttk.LabelFrame(parent_frame, text="Learning Settings Info")
    help_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    help_text = """
    preset: Predefined configuration preset
    
    auto_batch_size: Automatically determine batch size
    
    manual_batch_size: Manually set batch size (if auto_batch_size is off)
    
    use_features: Enable advanced feature engineering
    
    use_log_transformation: Apply logarithmic transformation to price data (helps with non-stationary time series)
    
    timestamp: Sequence length for training and prediction (must match initial_data_period in Prediction tab if set_initial_data is enabled)
    
    Other fields control model architecture and training parameters.
    """
    
    ttk.Label(help_frame, text=help_text, wraplength=500, justify="left").pack(padx=10, pady=10)
    
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