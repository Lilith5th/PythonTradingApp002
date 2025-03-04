import tkinter as tk
from tkinter import ttk

def create_tab(frame, config_obj):
    class AdvancedPredictionTab:
        def __init__(self):
            self.entries = {}
            self.notebook = ttk.Notebook(frame)
            self.notebook.pack(expand=True, fill="both", padx=5, pady=5)

            self.tab_frames = {
                "Ensemble": ttk.Frame(self.notebook),
                "Uncertainty": ttk.Frame(self.notebook),
                "Monte Carlo": ttk.Frame(self.notebook),
                "Rolling Window": ttk.Frame(self.notebook)
            }

            for name, tab_frame in self.tab_frames.items():
                self.notebook.add(tab_frame, text=name)

            self._create_ensemble_tab()
            self._create_uncertainty_tab()
            self._create_monte_carlo_tab()
            self._create_rolling_window_tab()

        def _create_ensemble_tab(self):
            ensemble_frame = self.tab_frames["Ensemble"]
            ttk.Label(ensemble_frame, text="Ensemble Methods (Not Supported in Current Implementation)", 
                      font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            self.notebook.tab(ensemble_frame, state="disabled")

        def _create_uncertainty_tab(self):
            uncertainty_frame = self.tab_frames["Uncertainty"]
            ttk.Label(uncertainty_frame, text="Uncertainty Quantification (Not Supported in Current Implementation)", 
                      font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            self.notebook.tab(uncertainty_frame, state="disabled")

        def _create_monte_carlo_tab(self):
            mc_frame = self.tab_frames["Monte Carlo"]
            ttk.Label(mc_frame, text="Monte Carlo Simulations", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)

            mc_enable_var = tk.BooleanVar(value=getattr(config_obj, 'enable_monte_carlo', False))
            mc_enable_ck = ttk.Checkbutton(mc_frame, text="Enable Monte Carlo Simulations", variable=mc_enable_var)
            mc_enable_ck.pack(anchor="w", padx=10, pady=5)
            self.entries['enable_monte_carlo'] = mc_enable_var

            num_frame = ttk.Frame(mc_frame)
            num_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(num_frame, text="Number of Simulations:").pack(side="left", padx=5)
            mc_num_var = tk.StringVar(value=str(getattr(config_obj, 'num_monte_carlo_simulations', 1000)))
            mc_num_entry = ttk.Entry(num_frame, textvariable=mc_num_var, width=8)
            mc_num_entry.pack(side="left", padx=5)
            self.entries['num_monte_carlo_simulations'] = mc_num_entry

            seed_frame = ttk.Frame(mc_frame)
            seed_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(seed_frame, text="Random Seed:").pack(side="left", padx=5)
            mc_seed_var = tk.StringVar(value=str(getattr(config_obj, 'monte_carlo_seed', 42)))
            mc_seed_entry = ttk.Entry(seed_frame, textvariable=mc_seed_var, width=8)
            mc_seed_entry.pack(side="left", padx=5)
            self.entries['monte_carlo_seed'] = mc_seed_entry

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
            rw_frame = self.tab_frames["Rolling Window"]
            ttk.Label(rw_frame, text="Advanced Rolling Window Settings", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)

            rw_enable_var = tk.BooleanVar(value=getattr(config_obj, 'enable_rolling_window', False))
            rw_enable_ck = ttk.Checkbutton(rw_frame, text="Enable Advanced Rolling Window Analysis", variable=rw_enable_var)
            rw_enable_ck.pack(anchor="w", padx=10, pady=5)
            self.entries['enable_rolling_window'] = rw_enable_var

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

            arch_frame = ttk.LabelFrame(rw_frame, text="Model Architecture Variations")
            arch_frame.pack(fill="x", padx=10, pady=5)

            layers_frame = ttk.Frame(arch_frame)
            layers_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(layers_frame, text="LSTM Layers (comma-separated):").pack(side="left", padx=5)
            layers_var = tk.StringVar(value=",".join(map(str, getattr(config_obj, 'lstm_layers', [2,3,4]))))
            layers_entry = ttk.Entry(layers_frame, textvariable=layers_var)
            layers_entry.pack(side="left", expand=True, fill="x", padx=5)
            self.entries['lstm_layers'] = layers_var

            units_frame = ttk.Frame(arch_frame)
            units_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(units_frame, text="LSTM Units (comma-separated):").pack(side="left", padx=5)
            units_var = tk.StringVar(value=",".join(map(str, getattr(config_obj, 'lstm_units', [64,128,256]))))
            units_entry = ttk.Entry(units_frame, textvariable=units_var)
            units_entry.pack(side="left", expand=True, fill="x", padx=5)
            self.entries['lstm_units'] = units_var

            dropout_frame = ttk.Frame(arch_frame)
            dropout_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(dropout_frame, text="Dropout Rates (comma-separated):").pack(side="left", padx=5)
            dropout_var = tk.StringVar(value=",".join(map(str, getattr(config_obj, 'dropout_rates', [0.2,0.3,0.4]))))
            dropout_entry = ttk.Entry(dropout_frame, textvariable=dropout_var)
            dropout_entry.pack(side="left", expand=True, fill="x", padx=5)
            self.entries['dropout_rates'] = dropout_var

            def toggle_architecture_fields(*args):
                state = "normal" if rw_enable_var.get() else "disabled"
                layers_entry.config(state=state)
                units_entry.config(state=state)
                dropout_entry.config(state=state)
                for child in metrics_frame.winfo_children():
                    child.config(state=state)

            rw_enable_var.trace_add("write", toggle_architecture_fields)
            toggle_architecture_fields()

        def update_config(self, config_obj):
            from dataclasses import fields
            for f in fields(config_obj):
                if f.name in self.entries:
                    value = self.entries[f.name]
                    current_val = getattr(config_obj, f.name)
                    if isinstance(value, tk.BooleanVar):
                        setattr(config_obj, f.name, value.get())
                    elif hasattr(value, 'get'):
                        widget_value = value.get()
                        if isinstance(current_val, bool):
                            setattr(config_obj, f.name, str(widget_value).lower() == 'true')
                        elif isinstance(current_val, int):
                            setattr(config_obj, f.name, int(widget_value))
                        elif isinstance(current_val, float):
                            setattr(config_obj, f.name, float(widget_value))
                        else:
                            setattr(config_obj, f.name, widget_value)

            if 'monte_carlo_scenarios' in self.entries:
                scenarios = []
                for scenario, var in self.entries['monte_carlo_scenarios'].items():
                    if var.get():
                        scenarios.append(scenario)
                setattr(config_obj, 'monte_carlo_scenarios', scenarios)

            if 'scenario_param_widgets' in self.entries:
                params = {}
                for scenario, (drift_var, vol_var) in self.entries['scenario_param_widgets'].items():
                    drift = float(drift_var.get())
                    vol = float(vol_var.get())
                    params[scenario] = [drift, vol]
                setattr(config_obj, 'scenario_parameters', params)

            if 'metric_vars' in self.entries:
                metrics = []
                for metric, var in self.entries['metric_vars'].items():
                    if var.get():
                        metrics.append(metric)
                setattr(config_obj, 'validation_metrics', metrics)

            if 'lstm_layers' in self.entries:
                setattr(config_obj, 'lstm_layers', [int(x) for x in self.entries['lstm_layers'].get().split(",")])
            if 'lstm_units' in self.entries:
                setattr(config_obj, 'lstm_units', [int(x) for x in self.entries['lstm_units'].get().split(",")])
            if 'dropout_rates' in self.entries:
                setattr(config_obj, 'dropout_rates', [float(x) for x in self.entries['dropout_rates'].get().split(",")])

    return AdvancedPredictionTab()