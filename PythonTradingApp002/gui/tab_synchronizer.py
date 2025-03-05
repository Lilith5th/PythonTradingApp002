"""
Tab Synchronization Module for Stock Prediction Application

This module provides functionality to synchronize related parameters between tabs
and maintain consistency throughout the UI.

Synchronized parameters include:
- timestamp in Learning tab and initial_data_period in Prediction tab
- Feature settings between Learning and Features tabs
- ML optimization settings between Strategy and Advanced Prediction tabs
"""

import tkinter as tk
from tkinter import ttk
import logging
import inspect
from typing import Dict, Any, Callable

# Update the import path to use the full module path
from config.config_utils import update_config_from_gui, update_gui_from_config


class TabSynchronizer:
    """
    Class to manage synchronization between tabs
    """
    
    def __init__(self, app_config, entries=None):
        """
        Initialize the tab synchronizer
        
        Args:
            app_config: Application configuration object
            entries: Dictionary of tab entries
        """
        self.app_config = app_config
        self.entries = entries or {}
        self.callbacks = {}
        self.suspending_callbacks = False
    
    def register_entries(self, entries):
        """
        Register tab entries
        
        Args:
            entries: Dictionary of tab entries
        """
        self.entries = entries
    
    def register_callback(self, source_tab, source_field, target_tab, target_field, transform_func=None):
        """
        Register a synchronization callback
        
        Args:
            source_tab: Source tab name
            source_field: Source field name
            target_tab: Target tab name
            target_field: Target field name
            transform_func: Optional function to transform the value
        """
        if source_tab not in self.callbacks:
            self.callbacks[source_tab] = {}
        
        if source_field not in self.callbacks[source_tab]:
            self.callbacks[source_tab][source_field] = []
        
        self.callbacks[source_tab][source_field].append({
            'target_tab': target_tab,
            'target_field': target_field,
            'transform_func': transform_func
        })
    
    def setup_synchronization(self):
        """Set up synchronization between tabs"""
        self._setup_learning_prediction_sync()
        self._setup_learning_features_sync()
        self._setup_strategy_ml_sync()
        self._setup_advanced_prediction_sync()
    
    def _setup_learning_prediction_sync(self):
        """Set up synchronization between Learning and Prediction tabs"""
        # Register callbacks for timestamp and initial_data_period
        self.register_callback('learning', 'timestamp', 'prediction', 'initial_data_period')
        
        # Connect the learning timestamp to prediction initial_data_period
        if 'learning' in self.entries and 'timestamp' in self.entries['learning']:
            timestamp_widget = self.entries['learning']['timestamp']
            if isinstance(timestamp_widget, ttk.Entry):
                timestamp_widget.bind('<FocusOut>', lambda e: self._sync_timestamp_to_initial_period())
                timestamp_widget.bind('<Return>', lambda e: self._sync_timestamp_to_initial_period())
            elif hasattr(timestamp_widget, 'trace_add'):
                timestamp_widget.trace_add('write', lambda *args: self._sync_timestamp_to_initial_period())
    
    def _setup_learning_features_sync(self):
        """Set up synchronization between Learning and Features tabs"""
        # Connect learning.use_features to Features tab state
        if ('learning' in self.entries and 'use_features' in self.entries['learning'] and
            isinstance(self.entries['learning']['use_features'], tk.BooleanVar)):
            use_features_var = self.entries['learning']['use_features']
            use_features_var.trace_add('write', lambda *args: self._toggle_features_tab())
            
            # Initial setup
            self._toggle_features_tab()
    
    def _setup_strategy_ml_sync(self):
        """Set up synchronization between Strategy and ML tabs"""
        # Connect strategy.strategy_type to enable_ml_optimization
        if 'strategy' in self.entries and 'strategy_type' in self.entries['strategy']:
            strategy_type_widget = self.entries['strategy']['strategy_type']
            if hasattr(strategy_type_widget, 'trace_add'):
                strategy_type_widget.trace_add('write', lambda *args: self._sync_strategy_ml())
            
            # Initial setup
            self._sync_strategy_ml()
    
    def _setup_advanced_prediction_sync(self):
        """Set up synchronization for Advanced Prediction tab"""
        # Sync MC dropout parameters between methods
        if ('advanced_prediction' in self.entries and 
            isinstance(self.entries['advanced_prediction'], object) and
            hasattr(self.entries['advanced_prediction'], 'entries')):
            
            adv_entries = self.entries['advanced_prediction'].entries
            
            # Connect uncertainty method to MC dropout params visibility
            if 'uncertainty_method' in adv_entries:
                method_widget = adv_entries['uncertainty_method']
                if hasattr(method_widget, 'trace_add'):
                    method_widget.trace_add('write', lambda *args: self._toggle_uncertainty_params())
                elif isinstance(method_widget, ttk.Combobox):
                    method_widget.bind('<<ComboboxSelected>>', lambda e: self._toggle_uncertainty_params())
    
    def _sync_timestamp_to_initial_period(self):
        """Synchronize timestamp to initial_data_period"""
        if self.suspending_callbacks:
            return
            
        if ('learning' not in self.entries or 'timestamp' not in self.entries['learning'] or
            'prediction' not in self.entries or 'initial_data_period' not in self.entries['prediction'] or
            'set_initial_data' not in self.entries['prediction']):
            return
            
        try:
            # Only sync if set_initial_data is enabled
            if isinstance(self.entries['prediction']['set_initial_data'], tk.BooleanVar):
                if not self.entries['prediction']['set_initial_data'].get():
                    return
            
            # Get timestamp value
            timestamp_widget = self.entries['learning']['timestamp']
            if isinstance(timestamp_widget, ttk.Entry):
                timestamp_value = timestamp_widget.get()
            elif hasattr(timestamp_widget, 'get'):
                timestamp_value = timestamp_widget.get()
            else:
                return
                
            # Update initial_data_period
            initial_period_widget = self.entries['prediction']['initial_data_period']
            if isinstance(initial_period_widget, ttk.Entry):
                initial_period_widget.delete(0, tk.END)
                initial_period_widget.insert(0, timestamp_value)
                
            # Update config
            try:
                self.app_config.learning.timestamp = int(timestamp_value)
                self.app_config.prediction.initial_data_period = int(timestamp_value)
            except (ValueError, AttributeError):
                logging.warning(f"Invalid timestamp value: {timestamp_value}")
                
        except Exception as e:
            logging.error(f"Error synchronizing timestamp to initial_data_period: {e}")
    
    def _toggle_features_tab(self):
        """Toggle the Features tab based on Learning.use_features"""
        if ('learning' not in self.entries or 'use_features' not in self.entries['learning'] or
            not isinstance(self.entries['learning']['use_features'], tk.BooleanVar)):
            return
            
        use_features = self.entries['learning']['use_features'].get()
        
        # Find the Features tab
        notebook = None
        features_idx = -1
        
        for widget in self.entries.get('root_widgets', []):
            if isinstance(widget, ttk.Notebook):
                notebook = widget
                for i, tab_id in enumerate(notebook.tabs()):
                    tab_text = notebook.tab(tab_id, "text")
                    if tab_text == "Features":
                        features_idx = i
                        break
                break
        
        if notebook and features_idx >= 0:
            if use_features:
                notebook.tab(features_idx, state="normal")
            else:
                notebook.tab(features_idx, state="disabled")
    
    def _sync_strategy_ml(self):
        """Synchronize strategy type with ML optimization settings"""
        if 'strategy' not in self.entries or 'strategy_type' not in self.entries['strategy']:
            return
            
        try:
            strategy_type = self.entries['strategy']['strategy_type'].get()
            
            # If ML strategy is selected, enable ML optimization
            if strategy_type == "ml_optimized":
                if 'enable_ml_optimization' in self.entries['strategy']:
                    self.entries['strategy']['enable_ml_optimization'].set(True)
                
                # Update config
                if hasattr(self.app_config, 'strategy'):
                    self.app_config.strategy.enable_ml_optimization = True
                    
                # Enable ML tab
                notebook = None
                ml_idx = -1
                
                for widget in self.entries.get('root_widgets', []):
                    if isinstance(widget, ttk.Notebook):
                        for i, tab_id in enumerate(widget.tabs()):
                            tab_text = widget.tab(tab_id, "text")
                            if "ML" in tab_text:
                                ml_idx = i
                                notebook = widget
                                break
                
                if notebook and ml_idx >= 0:
                    notebook.tab(ml_idx, state="normal")
        except Exception as e:
            logging.error(f"Error synchronizing strategy ML settings: {e}")
    
    def _toggle_uncertainty_params(self):
        """Toggle visibility of uncertainty parameters based on method"""
        if 'advanced_prediction' not in self.entries:
            return
            
        try:
            entries = self.entries['advanced_prediction'].entries
            if 'uncertainty_method' not in entries:
                return
                
            method = entries['uncertainty_method'].get()
            
            # Show/hide MC dropout parameters
            mc_samples_visible = (method == 'mc_dropout')
            for widget_name in ['mc_dropout_samples']:
                if widget_name in entries:
                    widget = entries[widget_name]
                    parent = widget.master if hasattr(widget, 'master') else None
                    
                    if parent:
                        if mc_samples_visible:
                            parent.pack(fill="x", padx=10, pady=5)
                        else:
                            parent.pack_forget()
        except Exception as e:
            logging.error(f"Error toggling uncertainty parameters: {e}")
    
    def with_callbacks_suspended(self, callback):
        """
        Run a callback with synchronization callbacks temporarily suspended
        
        Args:
            callback: Function to run with callbacks suspended
        """
        self.suspending_callbacks = True
        try:
            callback()
        finally:
            self.suspending_callbacks = False


class SynchronizedTabManager:
    """
    Manager for tabs with synchronized fields
    """
    
    def __init__(self, root, app_config):
        """
        Initialize the tab manager
        
        Args:
            root: Root window
            app_config: Application configuration
        """
        self.root = root
        self.app_config = app_config
        self.notebook = None
        self.tabs = {}
        self.entries = {}
        self.synchronizer = TabSynchronizer(app_config)
    
    def create_notebook(self):
        """Create the notebook for tabs"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")
        
        # Add notebook to root widgets list
        self.entries['root_widgets'] = [self.notebook]
        
        return self.notebook
    
    def add_tab(self, name, tab_module, config_obj):
        """
        Add a tab to the notebook
        
        Args:
            name: Tab name
            tab_module: Module containing the create_tab function
            config_obj: Configuration object for the tab
            
        Returns:
            Frame: Tab frame
        """
        frame = ttk.Frame(self.notebook)
        # Format the tab title with spaces and capital letters
        tab_title = " ".join(word.capitalize() for word in name.split("_"))
        self.notebook.add(frame, text=tab_title)
        
        # Check if create_tab accepts 2 or 3+ parameters
        sig = inspect.signature(tab_module.create_tab)
        if len(sig.parameters) == 2:
            tab_entries = tab_module.create_tab(frame, config_obj)
        else:
            tab_entries = tab_module.create_tab(frame, config_obj, self.app_config)
        
        self.entries[name] = tab_entries
        self.tabs[name] = frame
        
        return frame
    
    def setup_synchronization(self):
        """Set up synchronization between tabs"""
        self.synchronizer.register_entries(self.entries)
        self.synchronizer.setup_synchronization()
    
    def update_config_from_gui(self, validate=True):
        """
        Update configuration from GUI with synchronization
        
        Args:
            validate: Whether to validate values
            
        Returns:
            Tuple[bool, List[str]]: Success flag and list of errors
        """
        # Use the imported function from utils.config_utils
        return update_config_from_gui(self.entries, self.app_config, validate)
    
    def update_gui_from_config(self):
        """Update GUI from configuration"""
        # Use the imported function from utils.config_utils
        self.synchronizer.with_callbacks_suspended(
            lambda: update_gui_from_config(self.entries, self.app_config)
        )