"""
Main Window module for the Dynamic Forecast Configuration GUI
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

class MainWindow:
    """Main application window class that manages the notebook and tabs"""
    
    def __init__(self, app_config, run_forecast_func=None, run_parallel_func=None, auto_tune_func=None):
        """
        Initialize the main window
        
        Args:
            app_config: Application configuration
            run_forecast_func: Function to run the forecast
            run_parallel_func: Function to run parallel scenarios
            auto_tune_func: Function to auto-tune parameters
        """
        self.app_config = app_config
        self.run_forecast_func = run_forecast_func
        self.run_parallel_func = run_parallel_func
        self.auto_tune_func = auto_tune_func
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Dynamic Forecast Configuration (GPU)")
        self.root.geometry("800x600")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")
        
        # Dictionary to store tab entries
        self.entries = {}
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create button frame
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Add the run button
        self.btn_run = ttk.Button(self.button_frame, text="Run Forecast", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        
        # Add other buttons
        ttk.Button(
            self.button_frame, text="Run Parallel Scenarios", 
            command=self.run_parallel_scenarios
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            self.button_frame, text="Auto-tune Parameters", 
            command=self.run_auto_tune
        ).pack(side=tk.LEFT, padx=5)
        
        # Set up close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def add_tab(self, name, tab_module, config_obj):
        """
        Add a tab to the notebook
        
        Args:
            name: The tab name (used as key in self.entries)
            tab_module: The module containing the create_tab function
            config_obj: The configuration object to pass to the tab
        """
        frame = ttk.Frame(self.notebook)
        # Format the tab title with spaces and capital letters
        tab_title = " ".join(word.capitalize() for word in name.split("_"))
        self.notebook.add(frame, text=tab_title)
        
        # Create the tab content
        self.entries[name] = tab_module.create_tab(frame, config_obj)
        
        return frame
    
    def update_config_from_gui(self):
        """
        Update the app_config from GUI entries
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for section, entries_dict in self.entries.items():
                # Get the corresponding config object
                config_obj = getattr(self.app_config, section)
                
                # Skip configuration if we're using feature override
                if section == "features" and self.app_config.learning.use_features:
                    logging.info("Skipping features tab (override enabled)")
                    continue
                
                # Process special sections
                if section == "advanced_prediction" and hasattr(entries_dict, "update_config"):
                    entries_dict.update_config(config_obj)
                    continue
                
                # Update each field
                if hasattr(entries_dict, "update_config"):
                    # If the tab module provides an update_config method
                    entries_dict.update_config(config_obj)
                else:
                    # Default field-by-field update
                    self.update_object_fields(config_obj, entries_dict)
            
            logging.info("Configuration updated from GUI")
            return True
        except Exception as e:
            logging.error(f"Error updating configuration: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Configuration Error", f"Invalid input: {e}")
            return False
    
    def update_object_fields(self, obj, entries_dict):
        """
        Update object fields from entries dictionary
        
        Args:
            obj: Object to update
            entries_dict: Dictionary of entries
        """
        from dataclasses import fields
        
        for f in fields(obj):
            if f.name in entries_dict:
                value = entries_dict[f.name]
                current_val = getattr(obj, f.name)
                
                # Handle different widget types
                if isinstance(value, tk.BooleanVar):
                    setattr(obj, f.name, value.get())
                elif isinstance(value, ttk.Combobox):
                    setattr(obj, f.name, value.get())
                elif isinstance(value, ttk.Entry):
                    # Type conversion based on current value type
                    if isinstance(current_val, bool):
                        setattr(obj, f.name, self.parse_bool_str(value.get()))
                    elif isinstance(current_val, int):
                        setattr(obj, f.name, int(value.get()))
                    elif isinstance(current_val, float):
                        setattr(obj, f.name, float(value.get()))
                    else:
                        setattr(obj, f.name, value.get())
                else:
                    # Try to get the value if possible
                    try:
                        widget_value = value.get()
                        if isinstance(current_val, bool):
                            setattr(obj, f.name, self.parse_bool_str(widget_value))
                        elif isinstance(current_val, int):
                            setattr(obj, f.name, int(widget_value))
                        elif isinstance(current_val, float):
                            setattr(obj, f.name, float(widget_value))
                        else:
                            setattr(obj, f.name, widget_value)
                    except AttributeError:
                        logging.warning(f"Could not get value for {f.name}")
    
    def parse_bool_str(self, value):
        """Parse a string to a boolean value"""
        if isinstance(value, bool):
            return value
        return value.strip().lower() == 'true'
    
    def on_run(self):
        """Run the forecast"""
        if not self.update_config_from_gui():
            return
        
        # Disable the run button
        self.btn_run.config(state="disabled")
        self.status_var.set("Preparing forecast...")
        
        # Run the forecast in a separate thread
        if self.run_forecast_func:
            threading.Thread(
                target=self.run_forecast_thread,
                daemon=True
            ).start()
    
    def run_forecast_thread(self):
        """Run the forecast in a separate thread"""
        # Create a temporary file for the results
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        # Run the forecast
        result = self.run_forecast_func(self.app_config, temp_file_path)
        
        # Process the result
        self.root.after(0, lambda: self.handle_forecast_result(result, temp_file_path))
    
    def handle_forecast_result(self, exitcode, temp_file_path):
        """Handle the forecast result"""
        self.btn_run.config(state="normal")
        if exitcode == 0 and os.path.exists(temp_file_path):
            with open(temp_file_path, 'rb') as f:
                data = pickle.load(f)
            os.unlink(temp_file_path)
            if isinstance(data, dict) and 'error' in data:
                messagebox.showerror("Error", f"Forecast simulation failed: {data['error']}")
                self.status_var.set("Forecast failed")
            else:
                self.status_var.set("Forecast complete - Displaying results")
                # Call to plot_gui, assuming it's imported elsewhere
                from . import plot_results
                plot_results.create_plot_gui_with_data(self.app_config, data[:-1])
        else:
            messagebox.showerror("Error", "Forecast simulation failed.")
            self.status_var.set("Forecast failed")
    
    def run_parallel_scenarios(self):
        """Run parallel scenarios"""
        if not self.update_config_from_gui():
            return
        
        if self.run_parallel_func:
            self.run_parallel_func(self.app_config, self.status_var, self.btn_run)
    
    def run_auto_tune(self):
        """Auto-tune parameters"""
        if not self.update_config_from_gui():
            return
        
        self.status_var.set("Running hyperparameter optimization...")
        self.btn_run.config(state="disabled")
        
        def tune_callback():
            if self.auto_tune_func:
                best_params, score = self.auto_tune_func(self.app_config)
                self.status_var.set(f"Optimization complete. Best score: {score:.4f}")
                self.btn_run.config(state="normal")
                
                # Update the learning tab with the new parameters
                if "learning" in self.entries:
                    self.update_learning_tab(best_params)
                
                messagebox.showinfo("Tuning Complete", f"Best parameters: {best_params}")
        
        threading.Thread(target=tune_callback, daemon=True).start()
    
    def update_learning_tab(self, params):
        """Update the learning tab with the given parameters"""
        if "learning" not in self.entries:
            return
        
        learning_entries = self.entries["learning"]
        for param_name, param_value in params.items():
            if param_name in learning_entries:
                widget = learning_entries[param_name]
                if isinstance(widget, ttk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(param_value))
                elif isinstance(widget, tk.BooleanVar):
                    widget.set(param_value)
                elif isinstance(widget, ttk.Combobox):
                    widget.set(param_value)
    
    def on_closing(self):
        """Handle window closing"""
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run the main loop"""
        self.root.mainloop()