"""
Preferences tab for the stock prediction GUI.
Contains settings for learning preferences.
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import fields
import logging

def create_tab(parent_frame, config_obj):
    """
    Create the preferences tab contents
    
    Args:
        parent_frame: The parent frame to add content to
        config_obj: The LearningPreferences configuration object
        
    Returns:
        Dictionary containing the GUI elements
    """
    entry_dict = {}
    
    # For LearningPreferences, we skip the use_indicators field if present
    for f in fields(config_obj):
        # Skip use_indicators field if present (for backwards compatibility)
        if f.name == 'use_indicators':
            continue
            
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