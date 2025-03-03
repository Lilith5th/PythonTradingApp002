"""
Strategy tab for the stock prediction GUI.
Contains settings for testing different trading strategies and machine learning-based optimization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import List, Dict, Any

# This class definition is for documentation only
# The actual class should be defined in config.py
class StrategyConfig:
    """Configuration for trading strategies."""
    strategy_type: str = "buy_and_hold"  # Default strategy
    enable_ml_optimization: bool = False
    initial_capital: float = 10000.0
    position_size_pct: float = 10.0      # Percentage of capital to invest per trade
    take_profit_pct: float = 5.0         # Take profit percentage
    stop_loss_pct: float = 3.0           # Stop loss percentage
    trailing_stop_pct: float = 2.0       # Trailing stop percentage
    max_positions: int = 5               # Maximum number of open positions
    reinvest_profits: bool = True        # Whether to reinvest profits
    use_saved_ml_model: bool = False     # Whether to use a previously saved ML model
    ml_model_path: str = "strategy_ml_model.pkl"  # Path to saved ML model
    ml_features: List[str] = None
    backtest_period: int = 252           # Number of days to backtest (approximately 1 year)
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize for

def create_tab(parent_frame, config_obj):
    """
    Create the strategy tab contents
    
    Args:
        parent_frame: The parent frame to add content to
        config_obj: The StrategyConfig configuration object
        
    Returns:
        Dictionary containing the GUI elements
    """
    # Make sure we have a valid config object
    if not hasattr(config_obj, 'strategy_type'):
        logging.warning("Invalid strategy config object. Creating default.")
        config_obj = StrategyConfig()

    entry_dict = {}
    
    # Create a notebook for sub-tabs
    notebook = ttk.Notebook(parent_frame)
    notebook.pack(expand=True, fill="both", padx=5, pady=5)
    
    # Create frames for sub-tabs
    basic_frame = ttk.Frame(notebook)
    advanced_frame = ttk.Frame(notebook)
    ml_frame = ttk.Frame(notebook)
    
    notebook.add(basic_frame, text="Basic Settings")
    notebook.add(advanced_frame, text="Advanced Settings")
    notebook.add(ml_frame, text="ML Optimization")
    
    # Basic Settings Tab
    create_basic_settings(basic_frame, config_obj, entry_dict)
    
    # Advanced Settings Tab
    create_advanced_settings(advanced_frame, config_obj, entry_dict)
    
    # ML Optimization Tab
    create_ml_settings(ml_frame, config_obj, entry_dict)
    
    return entry_dict

def create_basic_settings(parent, config_obj, entry_dict):
    """Create the basic strategy settings UI"""
    # Strategy Type Selection
    strategy_frame = ttk.LabelFrame(parent, text="Trading Strategy")
    strategy_frame.pack(fill="x", padx=10, pady=10)
    
    ttk.Label(strategy_frame, text="Strategy Type:").pack(anchor="w", padx=10, pady=5)
    
    # Available strategies
    strategies = [
        ("buy_and_hold", "Buy and Hold"),
        ("moving_average_crossover", "Moving Average Crossover"),
        ("rsi_based", "RSI-Based Strategy"),
        ("macd_based", "MACD-Based Strategy"),
        ("bollinger_bands", "Bollinger Bands Strategy"),
        ("trend_following", "Trend Following"),
        ("mean_reversion", "Mean Reversion"),
        ("breakout", "Breakout Strategy"),
        ("ml_optimized", "ML-Optimized Strategy")
    ]
    
    strategy_var = tk.StringVar(value=config_obj.strategy_type)
    strategy_combobox = ttk.Combobox(strategy_frame, textvariable=strategy_var)
    strategy_combobox['values'] = [s[0] for s in strategies]
    strategy_combobox.pack(fill="x", padx=10, pady=5)
    
    # Add strategy descriptions
    desc_frame = ttk.Frame(strategy_frame)
    desc_frame.pack(fill="x", padx=10, pady=5)
    
    strategy_desc = tk.StringVar()
    desc_label = ttk.Label(desc_frame, textvariable=strategy_desc, wraplength=400, justify="left")
    desc_label.pack(fill="x")
    
    # Dictionary of strategy descriptions
    strategy_descriptions = {
        "buy_and_hold": "Buy and hold strategy simply buys the asset and holds it for the entire period.",
        "moving_average_crossover": "Generates buy/sell signals when short-term moving average crosses above/below long-term moving average.",
        "rsi_based": "Uses Relative Strength Index (RSI) to identify overbought/oversold conditions.",
        "macd_based": "Uses Moving Average Convergence Divergence (MACD) for trend identification and signal generation.",
        "bollinger_bands": "Trades based on price movements relative to Bollinger Bands.",
        "trend_following": "Aims to capture gains by riding the momentum of existing market trends.",
        "mean_reversion": "Assumes that prices will revert to their mean over time.",
        "breakout": "Identifies and trades breakouts from established price patterns or levels.",
        "ml_optimized": "Uses machine learning to optimize trading parameters based on historical data."
    }
    
    # Update description when strategy changes
    def update_description(*args):
        selected = strategy_var.get()
        if selected in strategy_descriptions:
            strategy_desc.set(strategy_descriptions[selected])
            
            # Enable ML tab if ML-optimized strategy is selected
            if selected == "ml_optimized":
                notebook.tab(2, state="normal")
                ml_enable_var.set(True)
            else:
                # Don't disable the tab, but update the ML enable checkbox
                ml_enable_var.set(False)
                
    strategy_var.trace_add("write", update_description)
    
    # Initial capital setting
    capital_frame = ttk.Frame(parent)
    capital_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(capital_frame, text="Initial Capital ($):").pack(side="left", padx=5)
    capital_entry = ttk.Entry(capital_frame)
    capital_entry.insert(0, str(config_obj.initial_capital))
    capital_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Position size setting
    position_frame = ttk.Frame(parent)
    position_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(position_frame, text="Position Size (% of capital):").pack(side="left", padx=5)
    position_entry = ttk.Entry(position_frame)
    position_entry.insert(0, str(config_obj.position_size_pct))
    position_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Backtest period setting
    backtest_frame = ttk.Frame(parent)
    backtest_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(backtest_frame, text="Backtest Period (days):").pack(side="left", padx=5)
    backtest_entry = ttk.Entry(backtest_frame)
    backtest_entry.insert(0, str(config_obj.backtest_period))
    backtest_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Add entries to dictionary
    entry_dict["strategy_type"] = strategy_var
    entry_dict["initial_capital"] = capital_entry
    entry_dict["position_size_pct"] = position_entry
    entry_dict["backtest_period"] = backtest_entry
    
    # Add descriptive text
    info_text = """
    Select a trading strategy to backtest against historical data or to apply to forecasted data.
    Each strategy has different parameters that can be adjusted in the Advanced Settings tab.
    For machine learning optimization, select "ML-Optimized Strategy" and configure the ML settings.
    """
    
    info_frame = ttk.LabelFrame(parent, text="Information")
    info_frame.pack(fill="x", padx=10, pady=10)
    ttk.Label(info_frame, text=info_text, wraplength=400, justify="left").pack(padx=10, pady=10)
    
    # Initial call to set the description
    update_description()

def create_advanced_settings(parent, config_obj, entry_dict):
    """Create the advanced strategy settings UI"""
    # Risk management settings
    risk_frame = ttk.LabelFrame(parent, text="Risk Management")
    risk_frame.pack(fill="x", padx=10, pady=10)
    
    # Take profit setting
    take_profit_frame = ttk.Frame(risk_frame)
    take_profit_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(take_profit_frame, text="Take Profit (%):").pack(side="left", padx=5)
    take_profit_entry = ttk.Entry(take_profit_frame)
    take_profit_entry.insert(0, str(config_obj.take_profit_pct))
    take_profit_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Stop loss setting
    stop_loss_frame = ttk.Frame(risk_frame)
    stop_loss_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(stop_loss_frame, text="Stop Loss (%):").pack(side="left", padx=5)
    stop_loss_entry = ttk.Entry(stop_loss_frame)
    stop_loss_entry.insert(0, str(config_obj.stop_loss_pct))
    stop_loss_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Trailing stop setting
    trailing_stop_frame = ttk.Frame(risk_frame)
    trailing_stop_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(trailing_stop_frame, text="Trailing Stop (%):").pack(side="left", padx=5)
    trailing_stop_entry = ttk.Entry(trailing_stop_frame)
    trailing_stop_entry.insert(0, str(config_obj.trailing_stop_pct))
    trailing_stop_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Portfolio management settings
    portfolio_frame = ttk.LabelFrame(parent, text="Portfolio Management")
    portfolio_frame.pack(fill="x", padx=10, pady=10)
    
    # Max positions setting
    max_pos_frame = ttk.Frame(portfolio_frame)
    max_pos_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(max_pos_frame, text="Maximum Positions:").pack(side="left", padx=5)
    max_pos_entry = ttk.Entry(max_pos_frame)
    max_pos_entry.insert(0, str(config_obj.max_positions))
    max_pos_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Reinvest profits setting
    reinvest_var = tk.BooleanVar(value=config_obj.reinvest_profits)
    reinvest_check = ttk.Checkbutton(portfolio_frame, text="Reinvest Profits", variable=reinvest_var)
    reinvest_check.pack(anchor="w", padx=10, pady=5)
    
    # Strategy-specific parameters (these would change based on the selected strategy)
    strategy_params_frame = ttk.LabelFrame(parent, text="Strategy Parameters")
    strategy_params_frame.pack(fill="x", padx=10, pady=10)
    
    # Moving Average parameters (visible for moving_average_crossover strategy)
    ma_params_frame = ttk.Frame(strategy_params_frame)
    ma_params_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(ma_params_frame, text="Short MA Period:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    short_ma_entry = ttk.Entry(ma_params_frame, width=10)
    short_ma_entry.insert(0, "20")
    short_ma_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
    
    ttk.Label(ma_params_frame, text="Long MA Period:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    long_ma_entry = ttk.Entry(ma_params_frame, width=10)
    long_ma_entry.insert(0, "50")
    long_ma_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
    
    # RSI parameters (visible for rsi_based strategy)
    rsi_params_frame = ttk.Frame(strategy_params_frame)
    rsi_params_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(rsi_params_frame, text="RSI Period:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    rsi_period_entry = ttk.Entry(rsi_params_frame, width=10)
    rsi_period_entry.insert(0, "14")
    rsi_period_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
    
    ttk.Label(rsi_params_frame, text="Overbought Level:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    rsi_overbought_entry = ttk.Entry(rsi_params_frame, width=10)
    rsi_overbought_entry.insert(0, "70")
    rsi_overbought_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
    
    ttk.Label(rsi_params_frame, text="Oversold Level:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    rsi_oversold_entry = ttk.Entry(rsi_params_frame, width=10)
    rsi_oversold_entry.insert(0, "30")
    rsi_oversold_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
    
    # Add optimization metric selector
    optimization_frame = ttk.Frame(parent)
    optimization_frame.pack(fill="x", padx=10, pady=10)
    
    ttk.Label(optimization_frame, text="Optimization Metric:").pack(side="left", padx=5)
    
    metrics = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "total_return",
        "win_rate",
        "profit_factor"
    ]
    
    optimization_var = tk.StringVar(value=config_obj.optimization_metric)
    optimization_combo = ttk.Combobox(optimization_frame, textvariable=optimization_var, values=metrics)
    optimization_combo.pack(side="left", expand=True, fill="x", padx=5)
    
    # Add entries to dictionary
    entry_dict["take_profit_pct"] = take_profit_entry
    entry_dict["stop_loss_pct"] = stop_loss_entry
    entry_dict["trailing_stop_pct"] = trailing_stop_entry
    entry_dict["max_positions"] = max_pos_entry
    entry_dict["reinvest_profits"] = reinvest_var
    entry_dict["optimization_metric"] = optimization_var
    
    # Strategy-specific parameters
    entry_dict["strategy_params"] = {
        "short_ma_period": short_ma_entry,
        "long_ma_period": long_ma_entry,
        "rsi_period": rsi_period_entry,
        "rsi_overbought": rsi_overbought_entry,
        "rsi_oversold": rsi_oversold_entry
    }
    
    # Function to show/hide strategy-specific parameters based on selected strategy
    def update_strategy_params(*args):
        selected = entry_dict["strategy_type"].get()
        
        # Hide all parameter frames
        for widget in strategy_params_frame.winfo_children():
            widget.pack_forget()
        
        # Show relevant parameter frame
        if selected == "moving_average_crossover":
            ma_params_frame.pack(fill="x", padx=10, pady=5)
        elif selected == "rsi_based":
            rsi_params_frame.pack(fill="x", padx=10, pady=5)
        # Add more strategy-specific parameter frames as needed
    
    # Connect the strategy type to parameter visibility
    entry_dict["strategy_type"].trace_add("write", update_strategy_params)
    
    # Initial call to set up the UI
    update_strategy_params()

def create_ml_settings(parent, config_obj, entry_dict):
    """Create the machine learning optimization settings UI"""
    # Enable ML optimization
    global ml_enable_var  # Make it accessible to the strategy selection function
    ml_enable_var = tk.BooleanVar(value=config_obj.enable_ml_optimization)
    ml_enable_check = ttk.Checkbutton(parent, text="Enable Machine Learning Optimization", variable=ml_enable_var)
    ml_enable_check.pack(anchor="w", padx=10, pady=5)
    
    # Model settings
    model_frame = ttk.LabelFrame(parent, text="ML Model Settings")
    model_frame.pack(fill="x", padx=10, pady=10)
    
    # Use saved model option
    saved_model_var = tk.BooleanVar(value=config_obj.use_saved_ml_model)
    saved_model_check = ttk.Checkbutton(
        model_frame, 
        text="Use Saved ML Model", 
        variable=saved_model_var
    )
    saved_model_check.pack(anchor="w", padx=10, pady=5)
    
    # Model path
    model_path_frame = ttk.Frame(model_frame)
    model_path_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(model_path_frame, text="Model Path:").pack(side="left", padx=5)
    model_path_entry = ttk.Entry(model_path_frame)
    model_path_entry.insert(0, config_obj.ml_model_path)
    model_path_entry.pack(side="left", expand=True, fill="x", padx=5)
    
    # Browse button
    def browse_model():
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select ML Model File",
            filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*"))
        )
        if filename:
            model_path_entry.delete(0, tk.END)
            model_path_entry.insert(0, filename)
    
    browse_button = ttk.Button(model_path_frame, text="Browse...", command=browse_model)
    browse_button.pack(side="right", padx=5)
    
    # Features selection
    features_frame = ttk.LabelFrame(parent, text="ML Features")
    features_frame.pack(fill="x", padx=10, pady=10)
    
    # Available features
    all_features = [
        "price_momentum",
        "rsi",
        "macd",
        "bollinger_bands",
        "volume_change",
        "price_volatility",
        "atr",
        "moving_averages",
        "gap_analysis",
        "support_resistance",
        "price_patterns",
        "day_of_week",
        "time_series_decomposition"
    ]
    
    # Create feature checkboxes
    feature_vars = {}
    for i, feature in enumerate(all_features):
        row, col = divmod(i, 2)
        var = tk.BooleanVar(value=feature in (config_obj.ml_features or ["price_momentum", "rsi", "macd", "volume_change"]))
        feature_vars[feature] = var
        ttk.Checkbutton(
            features_frame, 
            text=feature.replace("_", " ").title(),
            variable=var
        ).grid(row=row, column=col, sticky="w", padx=10, pady=2)
    
    # Training settings
    training_frame = ttk.LabelFrame(parent, text="Training Settings")
    training_frame.pack(fill="x", padx=10, pady=10)
    
    # Add common ML training parameters (epochs, batch size, etc.)
    params_frame = ttk.Frame(training_frame)
    params_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(params_frame, text="Training Epochs:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    epochs_entry = ttk.Entry(params_frame, width=10)
    epochs_entry.insert(0, "100")
    epochs_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
    
    ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    batch_size_entry = ttk.Entry(params_frame, width=10)
    batch_size_entry.insert(0, "32")
    batch_size_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
    
    ttk.Label(params_frame, text="Validation Split:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    val_split_entry = ttk.Entry(params_frame, width=10)
    val_split_entry.insert(0, "0.2")
    val_split_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
    
    # Action Buttons
    buttons_frame = ttk.Frame(parent)
    buttons_frame.pack(fill="x", padx=10, pady=10)
    
    ttk.Button(buttons_frame, text="Train New Model", command=lambda: messagebox.showinfo("Training", "Model training would start here.")).pack(side="left", padx=5)
    ttk.Button(buttons_frame, text="Evaluate Model", command=lambda: messagebox.showinfo("Evaluation", "Model evaluation would start here.")).pack(side="left", padx=5)
    
    # Help text
    help_frame = ttk.LabelFrame(parent, text="About ML Optimization")
    help_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    help_text = """
    Machine Learning Optimization allows the system to learn optimal trading parameters 
    based on historical data. The ML model can identify patterns and relationships that 
    may not be obvious in traditional strategies.
    
    To use ML optimization:
    1. Enable ML optimization
    2. Select the features to include in the model
    3. Configure training parameters
    4. Train a new model or use a previously saved model
    5. Evaluate the model's performance
    
    The trained model will learn to identify optimal entry and exit points based on 
    the selected features and historical price action.
    """
    
    help_label = ttk.Label(help_frame, text=help_text, wraplength=400, justify="left")
    help_label.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Add entries to dictionary
    entry_dict["enable_ml_optimization"] = ml_enable_var
    entry_dict["use_saved_ml_model"] = saved_model_var
    entry_dict["ml_model_path"] = model_path_entry
    entry_dict["ml_features"] = feature_vars
    entry_dict["ml_training"] = {
        "epochs": epochs_entry,
        "batch_size": batch_size_entry,
        "validation_split": val_split_entry
    }
    
    # Function to enable/disable ML settings
    def toggle_ml_settings(*args):
        state = "normal" if ml_enable_var.get() else "disabled"
        for child in model_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass
        for child in features_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass
        for child in training_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass
        for child in buttons_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass
    
    # Connect the enable checkbox to settings visibility
    ml_enable_var.trace_add("write", toggle_ml_settings)
    
    # Initialize state
    toggle_ml_settings()
    
    # Make sure this tab is disabled if not using ML optimization
    # (This will be overridden by strategy selection)
    notebook = parent.master
    if not config_obj.enable_ml_optimization:
        notebook.tab(2, state="disabled")

# Global notebook variable for tab state management
notebook = None
ml_enable_var = None