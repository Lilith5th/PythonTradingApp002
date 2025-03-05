#config.py

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CSVConfig:
    file_path: str = "aapl_data.csv"
    parse_dates: list = field(default_factory=lambda: ['datetime'])

@dataclass
class TensorFlowConfig:
    split_ratio: float = 0.8

@dataclass
class GUIConfig:
    figure_size: tuple = (15, 5)
    title: str = 'Stock Price Prediction with Top 5 Accuracy Simulations'
    xlabel: str = 'Date'
    ylabel: str = 'Stock Price'
    legend_loc: str = 'upper left'
    bbox_to_anchor: tuple = (1, 1)

@dataclass
class PlotConfig:
    num_predictions_shown: int = 5
    plot_mean: bool = True
    plot_confidence_interval: bool = True
    line_width_train: float = 1.0
    train_color: str = 'black'
    forecast_color: str = 'red'
    future_color: str = 'green'
    mean_color: str = 'blue'
    show_diagnostics: bool = True

# Update LearningConfig in config.py

@dataclass
class LearningConfig:
    preset: str = "medium"
    timestamp: int = 60
    simulation_size: int = 5
    epoch: int = 50
    batch_size: int = 32
    auto_batch_size: bool = True
    manual_batch_size: int = 32
    num_layers: int = 2
    size_layer: int = 10  # Changed from 64 to meet the max requirement
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    l2_reg: float = 0.01
    use_features: bool = False
    use_log_transformation: bool = False  # New field for logarithmic transformation
    
    def __post_init__(self):
        presets = {
            "gpu-high-performance": (16, 4, 0.3, 1024, 0.0005),
            "high-performance": (16, 4, 0.3, 4096, 0.0005),
            "high": (10, 4, 0.3, 256, 0.0005),
            "medium": (5, 2, 0.3, 1024, 0.001),
            "low": (3, 1, 0.2, 2048, 0.002)
        }
        if self.preset not in presets:
            raise ValueError(f"Preset must be one of {list(presets.keys())}")
        (self.simulation_size, self.num_layers, 
         self.dropout_rate, self.batch_size, self.learning_rate) = presets[self.preset]

@dataclass
class RollingWindowConfig:
    """Configuration for rolling window validation."""
    use_rolling_window: bool = False  # Enable rolling window validation
    window_size: int = 252  # Trading days in a year
    step_size: int = 21  # Monthly step
    min_train_size: int = 504  # Minimum 2 years of data for training
    refit_frequency: int = 63  # Refit model every 21 days

@dataclass
class LearningPreferences:
    enable_backtesting: bool = True
    backtesting_start_date: str = "2024-01-03"
    volume_scaling_method: str = "minmax"
    enable_learning_start_date: bool = True
    learning_start_date: str = "2020-01-01"
    use_gpu_if_available: bool = True


@dataclass
class FeatureSelectionConfig:
    # Feature groups
    use_trend_indicators: bool = True
    use_volatility_indicators: bool = True
    use_momentum_indicators: bool = True
    use_volume_indicators: bool = True
    use_price_patterns: bool = False
    use_time_features: bool = False
    use_financial_features: bool = False
    use_market_regime: bool = False
    use_derivative_features: bool = True
    
    # Specific indicators (examples)
    use_macd: bool = True
    use_rsi: bool = True
    use_bollinger_bands: bool = True
    use_atr: bool = False
    use_stochastic: bool = False
    use_obv: bool = False
    use_parabolic_sar: bool = False
    use_williams_r: bool = False
    use_cci: bool = False
    use_adx: bool = True
    use_ichimoku: bool = False
    use_keltner: bool = False
    use_donchian: bool = False
    use_vpt: bool = False
    use_adl: bool = False
    use_cmf: bool = False
    use_mfi: bool = False
    use_eom: bool = False
    use_linear_regression: bool = True
    use_support_resistance: bool = False
    use_day_dummies: bool = False
    use_month_dummies: bool = False
    use_seasonality: bool = False
    use_calendar_effects: bool = False
    
    # Feature selection
    auto_select_features: bool = True
    num_features_to_select: int = 20
    feature_selection_method: str = "importance"  # or "mutual_info"
    use_talib: bool = False  # Use TA-Lib if available


@dataclass
class PredictionConfig:
    predict_days: int = 30
    start_forecast_from_backtest: bool = True
    use_previous_close: bool = True
    set_initial_data: bool = False  # New option to use historical data as initial state
    initial_data_period: int = 20   # Number of days of historical data to use as initial state

@dataclass
class PredictionAdvancedConfig:
    """Advanced prediction configuration with ensemble methods and uncertainty quantification."""
    
    # Ensemble methods
    use_ensemble_methods: bool = False
    ensemble_method: str = "voting"  # Options: "voting", "stacking", "bagging", "boosting"
    ensemble_models: list = field(default_factory=lambda: ["lstm", "gru", "bilstm"])
    ensemble_weights: list = field(default_factory=lambda: [0.5, 0.3, 0.2])  # Weights for voting ensemble
    
    # Uncertainty quantification
    enable_uncertainty_quantification: bool = False
    uncertainty_method: str = "mc_dropout"  # Options: "mc_dropout", "bootstrap", "quantile", "evidential"
    mc_dropout_samples: int = 100
    confidence_level: float = 0.95  # For confidence intervals
    
    # Monte Carlo simulations
    enable_monte_carlo: bool = False
    num_monte_carlo_simulations: int = 1000
    monte_carlo_seed: int = 42
    monte_carlo_scenarios: list = field(default_factory=lambda: ["baseline", "bull", "bear", "volatile"])
    # Scenario adjustments (multipliers for: [drift, volatility])
    scenario_parameters: dict = field(default_factory=lambda: {
        "baseline": [1.0, 1.0],
        "bull": [1.5, 0.8],
        "bear": [0.5, 1.2],
        "volatile": [1.0, 2.0]
    })
    
    # Rolling window validation
    enable_rolling_window: bool = False
    window_size: int = 252  # Trading days in a year
    step_size: int = 21     # Monthly step
    min_train_size: int = 504  # Minimum 2 years of data for training
    refit_frequency: int = 21  # Refit model every 21 days
    
    # Model architecture variations for ensemble
    lstm_layers: list = field(default_factory=lambda: [1, 2])
    lstm_units: list = field(default_factory=lambda: [64, 128])
    dropout_rates: list = field(default_factory=lambda: [0.2, 0.3, 0.4])
    
    # Validation metrics
    validation_metrics: list = field(default_factory=lambda: ["rmse", "mae", "mape", "smape", "directional_accuracy"])



@dataclass
class StrategyConfig:
    """Configuration class for trading strategies."""
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
    ml_features: List[str] = field(default_factory=lambda: ["price_momentum", "rsi", "macd", "volume_change"])
    backtest_period: int = 252           # Number of days to backtest (approximately 1 year)
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize for

@dataclass
class PreferencesConfig:
    """Configuration for user preferences."""
    theme: str = "light"               # Application theme (light, dark)
    auto_save: bool = True             # Auto-save configuration
    save_interval: int = 5             # Auto-save interval in minutes
    chart_style: str = "candle"        # Chart style (line, candle, ohlc)
    decimal_precision: int = 2         # Decimal precision for displayed values
    date_format: str = "%Y-%m-%d"      # Date format for display
    log_level: str = "INFO"            # Logging level
    enable_hardware_acceleration: bool = True  # Enable hardware acceleration if available
    use_cache: bool = True             # Cache data between sessions
    cache_expiry: int = 24             # Cache expiry in hours
    ui_font_size: int = 10             # UI font size
    chart_font_size: int = 10          # Chart font size

@dataclass
class RootWidgetsConfig:
    """Configuration for root-level UI widgets."""
    main_width: int = 1280             # Main window width
    main_height: int = 800             # Main window height
    toolbar_visible: bool = True       # Show toolbar
    statusbar_visible: bool = True     # Show status bar
    sidebar_visible: bool = True       # Show sidebar
    sidebar_width: int = 250           # Sidebar width
    tab_position: str = "top"          # Tab position (top, bottom, left, right)
    enable_animations: bool = True     # Enable UI animations
    show_splash_screen: bool = True    # Show splash screen on startup
    confirm_exit: bool = True          # Confirm before exiting

@dataclass
class FeaturesConfig:
    """Configuration for features tab."""
    selected_features: List[str] = field(default_factory=lambda: ["macd", "rsi"])
    feature_categories: Dict[str, bool] = field(default_factory=lambda: {
        "price": True,
        "volume": True,
        "momentum": True,
        "volatility": True,
        "trend": True
    })
    custom_features: List[Dict[str, Any]] = field(default_factory=list)  # Custom feature definitions
    feature_scaling: str = "standard"  # Feature scaling method (standard, minmax, robust)
    enable_pca: bool = False          # Enable PCA dimensionality reduction
    pca_components: int = 5           # Number of PCA components
    enable_feature_engineering: bool = True  # Enable automatic feature engineering
    lag_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])  # Lag periods for features

@dataclass
class AdvancedPredictionConfig:
    """Advanced prediction settings."""
    # Extending the existing PredictionAdvancedConfig with additional UI-specific settings
    show_ensemble_details: bool = True  # Show detailed ensemble model information
    plot_individual_models: bool = False  # Plot predictions from individual ensemble models
    uncertainty_alpha: float = 0.2  # Alpha value for uncertainty bands
    monte_carlo_alpha: float = 0.1  # Alpha value for monte carlo simulation bands
    color_scheme: str = "default"  # Color scheme for plots
    show_metrics_on_plot: bool = True  # Show performance metrics on plot
    export_predictions: bool = False  # Export predictions to CSV
    export_path: str = "./predictions/"  # Path for exported predictions

@dataclass
class AppConfig:
    csv: CSVConfig = field(default_factory=CSVConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    learning_pref: LearningPreferences = field(default_factory=LearningPreferences)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    prediction_advanced: PredictionAdvancedConfig = field(default_factory=PredictionAdvancedConfig)
    tf_config: TensorFlowConfig = field(default_factory=TensorFlowConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    rolling_window: RollingWindowConfig = field(default_factory=RollingWindowConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    # Adding the missing sections
    preferences: PreferencesConfig = field(default_factory=PreferencesConfig)
    root_widgets: RootWidgetsConfig = field(default_factory=RootWidgetsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    advanced_prediction: AdvancedPredictionConfig = field(default_factory=AdvancedPredictionConfig)