"""
Advanced Prediction tab for the Dynamic Forecast Configuration GUI
"""
import tkinter as tk
from tkinter import ttk

def create_tab(frame, config_obj):
    """
    Creates a class-based tab with full functionality for advanced prediction settings
    """
    class AdvancedPredictionTab:
        def __init__(self):
            # Initialize entries dictionary
            self.entries = {}

            # Create a notebook for sub-tabs within the advanced prediction tab
            self.notebook = ttk.Notebook(frame)
            self.notebook.pack(expand=True, fill="both", padx=5, pady=5)

            # Create sub-tabs for different prediction categories
            self.tab_frames = {
                "Ensemble": ttk.Frame(self.notebook),
                "Uncertainty": ttk.Frame(self.notebook),
                "Monte Carlo": ttk.Frame(self.notebook),
                "Rolling Window": ttk.Frame(self.notebook)
            }

            for name, tab_frame in self.tab_frames.items():
                self.notebook.add(tab_frame, text=name)

            # Populate tabs
            self._create_ensemble_tab()
            self._create_uncertainty_tab()
            self._create_monte_carlo_tab()
            self._create_rolling_window_tab()

        def _create_ensemble_tab(self):
            """Create the ensemble methods tab"""
            ensemble_frame = self.tab_frames["Ensemble"]
            ttk.Label(ensemble_frame, text="Ensemble Methods", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)

            # Enable Ensemble Methods
            enable_frame = ttk.Frame(ensemble_frame)
            enable_frame.pack(fill="x", padx=10, pady=5)

            use_ensemble_var = tk.BooleanVar(value=getattr(config_obj, 'use_ensemble_methods', False))
            use_ensemble_ck = ttk.Checkbutton(enable_frame, text="Enable Ensemble Methods", variable=use_ensemble_var)
            use_ensemble_ck.pack(side="left", padx=5)
            self.entries['use_ensemble_methods'] = use_ensemble_var

            # Ensemble Method
            method_frame = ttk.Frame(ensemble_frame)
            method_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(method_frame, text="Ensemble Method:").pack(side="left", padx=5)
            
            ens_method_var = tk.StringVar(value=str(getattr(config_obj, 'ensemble_method', 'voting')))
            ens_method_combo = ttk.Combobox(method_frame, textvariable=ens_method_var, 
                                            values=["voting","stacking","bagging","boosting"], state="readonly")
            ens_method_combo.pack(side="left", padx=5)
            self.entries['ensemble_method'] = ens_method_combo

            # Model Types
            models_frame = ttk.LabelFrame(ensemble_frame, text="Model Types")
            models_frame.pack(fill="x", padx=10, pady=5)

            model_types = ["lstm", "gru", "bilstm", "transformer", "tcn"]
            model_vars = {}
            for i, model_type in enumerate(model_types):
                var = tk.BooleanVar(value=model_type in getattr(config_obj, 'ensemble_models', []))
                ck = ttk.Checkbutton(models_frame, text=model_type.upper(), variable=var)
                ck.grid(row=i//3, column=i%3, sticky="w", padx=10, pady=5)
                model_vars[model_type] = var
            
            self.entries['model_vars'] = model_vars

            # Weights
            weights_frame = ttk.Frame(ensemble_frame)
            weights_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(weights_frame, text="Model Weights (comma-separated):").pack(side="left", padx=5)
            
            default_weights = getattr(config_obj, 'ensemble_weights', [0.5,0.3,0.2])
            weights_str = ",".join(str(w) for w in default_weights)
            ensemble_weights_var = tk.StringVar(value=weights_str)
            ensemble_weights_entry = ttk.Entry(weights_frame, textvariable=ensemble_weights_var, width=20)
            ensemble_weights_entry.pack(side="left", padx=5)
            self.entries['ensemble_weights'] = ensemble_weights_entry

        def _create_uncertainty_tab(self):
            """Create the uncertainty quantification tab"""
            uncertainty_frame = self.tab_frames["Uncertainty"]
            ttk.Label(uncertainty_frame, text="Uncertainty Quantification", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)

            # Enable Uncertainty
            unc_enable_var = tk.BooleanVar(value=getattr(config_obj, 'enable_uncertainty_quantification', False))
            unc_enable_ck = ttk.Checkbutton(uncertainty_frame, text="Enable Uncertainty Quantification", variable=unc_enable_var)
            unc_enable_ck.pack(anchor="w", padx=10, pady=5)
            self.entries['enable_uncertainty_quantification'] = unc_enable_var

            # Uncertainty Method
            method_frame = ttk.Frame(uncertainty_frame)
            method_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(method_frame, text="Uncertainty Method:").pack(side="left", padx=5)
            
            unc_method_var = tk.StringVar(value=str(getattr(config_obj, 'uncertainty_method', 'mc_dropout')))
            unc_method_combo = ttk.Combobox(method_frame, textvariable=unc_method_var, 
                                            values=["mc_dropout","bootstrap","quantile","evidential"], state="readonly")
            unc_method_combo.pack(side="left", padx=5)
            self.entries['uncertainty_method'] = unc_method_combo

            # MC Dropout Samples
            samples_frame = ttk.Frame(uncertainty_frame)
            samples_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(samples_frame, text="MC Dropout Samples:").pack(side="left", padx=5)
            
            mc_samples_var = tk.StringVar(value=str(getattr(config_obj, 'mc_dropout_samples', 100)))
            mc_samples_entry = ttk.Entry(samples_frame, textvariable=mc_samples_var, width=8)
            mc_samples_entry.pack(side="left", padx=5)
            self.entries['mc_dropout_samples'] = mc_samples_entry

            # Confidence Level
            conf_frame = ttk.Frame(uncertainty_frame)
            conf_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(conf_frame, text="Confidence Level (0-1):").pack(side="left", padx=5)
            
            conf_level_var = tk.StringVar(value=str(getattr(config_obj, 'confidence_level', 0.95)))
            conf_level_entry = ttk.Entry(conf_frame, textvariable=conf_level_var, width=8)
            conf_level_entry.pack(side="left", padx=5)
            self.entries['confidence_level'] = conf_level_entry

        def _create_monte_carlo_tab(self):
            """Create the Monte Carlo simulations tab"""
            mc_frame = self.tab_frames["Monte Carlo"]
            ttk.Label(mc_frame, text="Monte Carlo Simulations", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)

            # Enable Monte Carlo
            mc_enable_var = tk.BooleanVar(value=getattr(config_obj, 'enable_monte_carlo', False))
            mc_enable_ck = ttk.Checkbutton(mc_frame, text="Enable Monte Carlo Simulations", variable=mc_enable_var)
            mc_enable_ck.pack(anchor="w", padx=10, pady=5)
            self.entries['enable_monte_carlo'] = mc_enable_var

            # Number of Simulations
            num_frame = ttk.Frame(mc_frame)
            num_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(num_frame, text="Number of Simulations:").pack(side="left", padx=5)
            
            mc_num_var = tk.StringVar(value=str(getattr(config_obj, 'num_monte_carlo_simulations', 1000)))
            mc_num_entry = ttk.Entry(num_frame, textvariable=mc_num_var, width=8)
            mc_num_entry.pack(side="left", padx=5)
            self.entries['num_monte_carlo_simulations'] = mc_num_entry

            # Random Seed
            seed_frame = ttk.Frame(mc_frame)
            seed_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(seed_frame, text="Random Seed:").pack(side="left", padx=5)
            
            mc_seed_var = tk.StringVar(value=str(getattr(config_obj, 'monte_carlo_seed', 42)))
            mc_seed_entry = ttk.Entry(seed_frame, textvariable=mc_seed_var, width=8)
            mc_seed_entry.pack(side="left", padx=5)
            self.entries['monte_carlo_seed'] = mc_seed_entry

            # Market Scenarios
            scenario_frame = ttk.LabelFrame(mc_frame, text="Market Scenarios")
            scenario_frame.pack(fill="x", padx=10, pady=5)

            scenarios = getattr(config_obj, 'monte_carlo_scenarios', ["baseline","bull","bear","volatile"])
            scenario_vars = {}
            for scenario in ["baseline","bull","bear","volatile","custom"]:
                var = tk.BooleanVar(value=(scenario in scenarios))
                ck = ttk.Checkbutton(scenario_frame, text=scenario.capitalize(), variable=var)
                ck.pack(anchor="w", padx=10, pady=2)
                scenario_vars[scenario] = var
            self.entries['monte_carlo_scenarios'] = scenario_vars

            # Scenario Parameters
            param_frame = ttk.LabelFrame(mc_frame, text="Scenario Parameters (Drift, Volatility)")
            param_frame.pack(fill="x", padx=10, pady=10)

            existing_params = getattr(config_obj, 'scenario_parameters', {
                "baseline": [1.0,1.0],
                "bull": [1.5,0.8],
                "bear": [0.5,1.2],
                "volatile": [1.0,2.0]
            })

            scenario_param_widgets = {}
            row_idx = 0
            for scenario in ["baseline","bull","bear","volatile"]:
                ttk.Label(param_frame, text=f"{scenario.capitalize()}:").grid(
                    row=row_idx, column=0, sticky="w", padx=5, pady=2
                )
                drift, vol = existing_params.get(scenario, [1.0,1.0])

                drift_var = tk.StringVar(value=str(drift))
                vol_var = tk.StringVar(value=str(vol))

                drift_entry = ttk.Entry(param_frame, textvariable=drift_var, width=8)
                drift_entry.grid(row=row_idx, column=1, padx=5, pady=2)

                vol_entry = ttk.Entry(param_frame, textvariable=vol_var, width=8)
                vol_entry.grid(row=row_idx, column=2, padx=5, pady=2)

                scenario_param_widgets[scenario] = (drift_var, vol_var)
                row_idx += 1

            self.entries['scenario_param_widgets'] = scenario_param_widgets

        def _create_rolling_window_tab(self):
            """Create the rolling window tab"""
            rw_frame = self.tab_frames["Rolling Window"]
            ttk.Label(rw_frame, text="Advanced Rolling Window Settings", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)

            # Enable Rolling Window
            rw_enable_var = tk.BooleanVar(value=getattr(config_obj, 'enable_rolling_window', False))
            rw_enable_ck = ttk.Checkbutton(rw_frame, text="Enable Advanced Rolling Window Analysis", variable=rw_enable_var)
            rw_enable_ck.pack(anchor="w", padx=10, pady=5)
            self.entries['enable_rolling_window'] = rw_enable_var

            # Validation Metrics
            metrics_frame = ttk.LabelFrame(rw_frame, text="Validation Metrics")
            metrics_frame.pack(fill="x", padx=10, pady=5)

            metrics = ["rmse", "mae", "mape", "smape", "directional_accuracy"]
            metric_vars = {}
            for i, metric in enumerate(metrics):
                var = tk.BooleanVar(value=metric in getattr(config_obj, 'validation_metrics', []))
                ck = ttk.Checkbutton(metrics_frame, text=metric.upper(), variable=var)
                ck.grid(row=i//3, column=i%3, sticky="w", padx=10, pady=5)
                metric_vars[metric] = var
            
            self.entries['metric_vars'] = metric_vars

            # LSTM Architecture Variations
            arch_frame = ttk.LabelFrame(rw_frame, text="Model Architecture Variations")
            arch_frame.pack(fill="x", padx=10, pady=5)

            # LSTM Layers
            layers_frame = ttk.Frame(arch_frame)
            layers_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(layers_frame, text="LSTM Layers (comma-separated):").pack(side="left", padx=5)
            
            layers_var = tk.StringVar(value=",".join(map(str, getattr(config_obj, 'lstm_layers', [2,3,4]))))
            layers_entry = ttk.Entry(layers_frame, textvariable=layers_var)
            layers_entry.pack(side="left", expand=True, fill="x", padx=5)
            self.entries['lstm_layers'] = layers_var

            # LSTM Units
            units_frame = ttk.Frame(arch_frame)
            units_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(units_frame, text="LSTM Units (comma-separated):").pack(side="left", padx=5)
            
            units_var = tk.StringVar(value=",".join(map(str, getattr(config_obj, 'lstm_units', [64,128,256]))))
            units_entry = ttk.Entry(units_frame, textvariable=units_var)
            units_entry.pack(side="left", expand=True, fill="x", padx=5)
            self.entries['lstm_units'] = units_var

            # Dropout Rates
            dropout_frame = ttk.Frame(arch_frame)
            dropout_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(dropout_frame, text="Dropout Rates (comma-separated):").pack(side="left", padx=5)
            
            dropout_var = tk.StringVar(value=",".join(map(str, getattr(config_obj, 'dropout_rates', [0.2,0.3,0.4]))))
            dropout_entry = ttk.Entry(dropout_frame, textvariable=dropout_var)
            dropout_entry.pack(side="left", expand=True, fill="x", padx=5)
            self.entries['dropout_rates'] = dropout_var

    def update_config(entries, config_obj):
        """
        Update the advanced prediction configuration object from GUI entries.
    
        Args:
            entries: Dictionary containing the GUI entries for the advanced prediction tab
            config_obj: The advanced prediction configuration object to update
        """
        from dataclasses import fields
    
        # Process standard entries (BooleanVar, StringVar, etc.)
        for f in fields(config_obj):
            if f.name in entries:
                value = entries[f.name]
                current_val = getattr(config_obj, f.name)
            
                # Handle different widget types
                if isinstance(value, tk.BooleanVar):
                    setattr(config_obj, f.name, value.get())
                elif hasattr(value, 'get'):
                    # Convert to appropriate type based on current value
                    widget_value = value.get()
                    if isinstance(current_val, bool):
                        setattr(config_obj, f.name, str(widget_value).lower() == 'true')
                    elif isinstance(current_val, int):
                        setattr(config_obj, f.name, int(widget_value))
                    elif isinstance(current_val, float):
                        setattr(config_obj, f.name, float(widget_value))
                    else:
                        setattr(config_obj, f.name, widget_value)
    
        # Handle special case for monte_carlo_scenarios (dictionary of checkboxes)
        if 'monte_carlo_scenarios' in entries:
            scenarios = []
            for scenario, var in entries['monte_carlo_scenarios'].items():
                if var.get():
                    scenarios.append(scenario)
            setattr(config_obj, 'monte_carlo_scenarios', scenarios)
    
        # Handle special case for scenario parameters (tuple of entries)
        if 'scenario_param_widgets' in entries:
            params = {}
            for scenario, (drift_var, vol_var) in entries['scenario_param_widgets'].items():
                drift = float(drift_var.get())
                vol = float(vol_var.get())
                params[scenario] = [drift, vol]
            setattr(config_obj, 'scenario_parameters', params)


    # Return an instance of the class
    return AdvancedPredictionTab()