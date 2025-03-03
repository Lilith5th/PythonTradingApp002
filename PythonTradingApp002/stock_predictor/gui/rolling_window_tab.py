"""
Rolling Window Validation tab for the Dynamic Forecast Configuration GUI
"""
import tkinter as tk
from tkinter import ttk

def create_tab(frame, config_obj):
    """
    Create the rolling window validation tab
    
    Args:
        frame: The parent frame
        config_obj: The RollingWindowConfig object
        
    Returns:
        Dictionary of entry widgets
    """
    entry_dict = {}
    
    # Enable/disable toggle
    enable_frame = ttk.Frame(frame)
    enable_frame.pack(fill="x", padx=10, pady=10)
    
    enable_var = tk.BooleanVar(value=config_obj.use_rolling_window)
    enable_widget = ttk.Checkbutton(enable_frame, text="Enable Rolling Window Validation", variable=enable_var)
    enable_widget.pack(side="left", padx=5)
    entry_dict['use_rolling_window'] = enable_var
    
    # Create a frame for the parameters
    params_frame = ttk.LabelFrame(frame, text="Rolling Window Parameters")
    params_frame.pack(fill="x", padx=10, pady=10)
    
    # Create a grid for the parameters
    grid_frame = ttk.Frame(params_frame)
    grid_frame.pack(fill="x", padx=10, pady=10)
    
    # Add parameters in a grid layout
    params = [
        ("Window Size (days):", "window_size"),
        ("Step Size (days):", "step_size"),
        ("Min Training Size (days):", "min_train_size"),
        ("Refit Frequency (days):", "refit_frequency")
    ]
    
    for i, (label_text, field_name) in enumerate(params):
        row = i // 2
        col = (i % 2) * 2
        
        ttk.Label(grid_frame, text=label_text).grid(row=row, column=col, sticky="w", padx=10, pady=5)
        
        entry = ttk.Entry(grid_frame, width=10)
        entry.insert(0, str(getattr(config_obj, field_name)))
        entry.grid(row=row, column=col + 1, sticky="w", padx=5, pady=5)
        
        entry_dict[field_name] = entry
    
    # Add explanatory text
    info_frame = ttk.LabelFrame(frame, text="About Rolling Window Validation")
    info_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    info_text = """
    Rolling window validation tests model performance across different time periods.
    
    Window Size: Length of each training window
    Step Size: How far to move the window forward each time
    Min Train Size: Minimum required training data length
    Refit Frequency: How often to retrain the model (days)
    
    This approach helps identify how stable the model's forecasting 
    performance is across different market conditions.
    """
    
    ttk.Label(info_frame, text=info_text, wraplength=500, justify="left").pack(padx=10, pady=10)
    
    # Function to toggle the state of parameters based on enable checkbox
    def toggle_params(*args):
        state = 'normal' if enable_var.get() else 'disabled'
        for field_name in ['window_size', 'step_size', 'min_train_size', 'refit_frequency']:
            if field_name in entry_dict:
                entry_dict[field_name].config(state=state)
    
    # Connect the toggle function
    enable_var.trace_add('write', toggle_params)
    
    # Initialize the state
    toggle_params()
    
    return entry_dict