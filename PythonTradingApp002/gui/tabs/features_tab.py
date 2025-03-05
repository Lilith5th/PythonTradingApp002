"""
Features tab for the Dynamic Forecast Configuration GUI
"""
import tkinter as tk
from tkinter import ttk

def create_tab(frame, config_obj):
    """
    Create the features tab
    
    Args:
        frame: The parent frame
        config_obj: The FeatureSelectionConfig object
        
    Returns:
        Dictionary of entry widgets
    """
    entry_dict = {}
    
    # Create a notebook for sub-tabs within the feature selection tab
    notebook = ttk.Notebook(frame)
    notebook.pack(expand=True, fill="both", padx=5, pady=5)
    
    # Create sub-tabs for different feature categories
    tab_frames = {
        "Groups": ttk.Frame(notebook),
        "Indicators": ttk.Frame(notebook),
        "Selection": ttk.Frame(notebook)
    }
    
    for name, tab_frame in tab_frames.items():
        notebook.add(tab_frame, text=name)
    
    # ========== Feature Groups Tab ==========
    create_groups_tab(tab_frames["Groups"], config_obj, entry_dict)
    
    # ========== Technical Indicators Tab ==========
    create_indicators_tab(tab_frames["Indicators"], config_obj, entry_dict)
    
    # ========== Feature Selection Tab ==========
    create_selection_tab(tab_frames["Selection"], config_obj, entry_dict)
    
    return entry_dict

def create_groups_tab(frame, config_obj, entry_dict):
    """Create the feature groups tab"""
    ttk.Label(frame, text="Feature Groups", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
    
    # Create a frame for feature group checkboxes
    groups_frame = ttk.Frame(frame)
    groups_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    # Define the feature group fields
    group_fields = [
        'use_trend_indicators', 
        'use_volatility_indicators', 
        'use_momentum_indicators', 
        'use_volume_indicators', 
        'use_price_patterns', 
        'use_time_features', 
        'use_financial_features', 
        'use_market_regime',
        'use_derivative_features'
    ]
    
    # Add group checkboxes in two columns
    for i, field_name in enumerate(group_fields):
        row = i % 5
        col = i // 5
        
        frame = ttk.Frame(groups_frame)
        frame.grid(row=row, column=col, sticky="w", padx=5, pady=2)
        
        default_val = getattr(config_obj, field_name)
        var = tk.BooleanVar(value=default_val)
        
        # Format field name for display
        display_name = field_name.replace('use_', '').replace('_', ' ').title()
        
        widget = ttk.Checkbutton(frame, text=display_name, variable=var)
        widget.pack(side="left", padx=5)
        entry_dict[field_name] = var

def create_indicators_tab(frame, config_obj, entry_dict):
    """Create the technical indicators tab"""
    ttk.Label(frame, text="Specific Technical Indicators", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
    
    # Create scrollable frame for indicators
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Define indicator fields
    indicator_fields = [
        'use_macd', 'use_rsi', 'use_bollinger_bands', 'use_atr', 'use_stochastic',
        'use_obv', 'use_parabolic_sar', 'use_williams_r', 'use_cci', 'use_adx', 
        'use_ichimoku', 'use_keltner', 'use_donchian', 'use_vpt', 'use_adl', 
        'use_cmf', 'use_mfi', 'use_eom', 'use_linear_regression', 'use_support_resistance',
        'use_day_dummies', 'use_month_dummies', 'use_seasonality', 'use_calendar_effects'
    ]
    
    # Arrange indicators in 3 columns
    num_cols = 3
    rows_per_col = (len(indicator_fields) + num_cols - 1) // num_cols
    
    for i, field_name in enumerate(indicator_fields):
        col = i // rows_per_col
        row = i % rows_per_col
        
        frame = ttk.Frame(scrollable_frame)
        frame.grid(row=row, column=col, sticky="w", padx=5, pady=2)
        
        default_val = getattr(config_obj, field_name)
        var = tk.BooleanVar(value=default_val)
        
        # Format field name for display
        display_name = field_name.replace('use_', '').replace('_', ' ').title()
        
        widget = ttk.Checkbutton(frame, text=display_name, variable=var)
        widget.pack(side="left", padx=5)
        entry_dict[field_name] = var
    
    # Add TA-Lib option at the bottom
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=rows_per_col, column=0, columnspan=3, sticky="w", padx=5, pady=10)
    
    var = tk.BooleanVar(value=getattr(config_obj, 'use_talib'))
    widget = ttk.Checkbutton(frame, text="Use TA-Lib (if available)", variable=var)
    widget.pack(side="left", padx=5)
    entry_dict['use_talib'] = var

def create_selection_tab(frame, config_obj, entry_dict):
    """Create the feature selection tab"""
    ttk.Label(frame, text="Feature Selection Settings", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
    
    # Automatic feature selection
    auto_frame = ttk.Frame(frame)
    auto_frame.pack(fill="x", padx=10, pady=5)
    
    auto_var = tk.BooleanVar(value=getattr(config_obj, 'auto_select_features'))
    auto_widget = ttk.Checkbutton(auto_frame, text="Automatic Feature Selection", variable=auto_var)
    auto_widget.pack(side="left", padx=5)
    entry_dict['auto_select_features'] = auto_var
    
    # Number of features to select
    num_frame = ttk.Frame(frame)
    num_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(num_frame, text="Number of Features to Select:").pack(side="left", padx=5)
    num_widget = ttk.Entry(num_frame, width=10)
    num_widget.insert(0, str(getattr(config_obj, 'num_features_to_select')))
    num_widget.pack(side="left", padx=5)
    entry_dict['num_features_to_select'] = num_widget
    
    # Feature selection method
    method_frame = ttk.Frame(frame)
    method_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(method_frame, text="Selection Method:").pack(side="left", padx=5)
    method_widget = ttk.Combobox(method_frame, values=["importance", "mutual_info"], state="readonly")
    method_widget.set(getattr(config_obj, 'feature_selection_method'))
    method_widget.pack(side="left", padx=5)
    entry_dict['feature_selection_method'] = method_widget
    
    # Add some help text
    help_text = """
    Feature selection determines which indicators are most predictive
    for your specific stock data and includes only the best ones in your model.
    
    Automatic Feature Selection: Enables feature importance analysis to select
    the most predictive features for your model.
    
    Number of Features: Controls how many top features to include.
    
    Selection Method:
    - importance: Uses Random Forest importance (faster)
    - mutual_info: Uses mutual information (more accurate but slower)
    """
    
    help_frame = ttk.LabelFrame(frame, text="Help")
    help_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    help_label = ttk.Label(help_frame, text=help_text, wraplength=350, justify="left")
    help_label.pack(padx=10, pady=10)