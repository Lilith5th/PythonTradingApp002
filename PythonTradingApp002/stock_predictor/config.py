from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CSVConfig:
    csv_path: str = "stock_data.csv"
    date_column: str = "Date"
    close_column: str = "Close"
    high_column: str = "High"
    low_column: str = "Low"
    open_column: str = "Open"
    volume_column: str = "Volume"

@dataclass
class TensorFlowConfig:
    split_ratio: float = 0.8
    early_stopping_patience: int = 10

@dataclass
class GUIConfig:
    figure_size: tuple = (12, 8)

@dataclass
class PlotConfig:
    num_predictions_shown: int = 5
    plot_mean: bool = True
    plot_confidence_interval: bool = True
    show_diagnostics: bool = True
    line_width_train: float = 2.0
    train_color: str = "blue"
    test_color: str = "green"
    forecast_color: str = "red"
    future_color: str = "orange"
    mean_color: str = "purple"
    confidence_color: str = "gray"
    figure_size: tuple = (12, 8)

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
    size_layer: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    l2_reg: float = 0.01
    use_features: bool = False

@dataclass
class RollingWindowConfig:
    use_rolling_window: bool = False
    window_size: int = 252
    step_size: int = 21
    min_train_size: int = 126
    refit_frequency: int = 21

@dataclass
class LearningPreferences:
    use_indicators: bool = True
    use_features: bool = True
    use_talib: bool = True
    optimize_performance: bool = True

@dataclass
class FeatureSelectionConfig:
    use_trend_indicators: bool = True
    use_volatility_indicators: bool = True
    use_momentum_indicators: bool = True
    use_volume_indicators: bool = True
    use_price_patterns: bool = True
    use_time_features: bool = True
    use_financial_features: bool = True
    use_market_regime: bool = True
    use_derivative_features: bool = True
    use_macd: bool = True
    use_rsi: bool = True
    use_bollinger_bands: bool = True
    use_atr: bool = True
    use_stochastic: bool = True
    use_obv: bool = True
    use_parabolic_sar: bool = True
    use_williams_r: bool = True
    use_cci: bool = True
    use_adx: bool = True
    use_ichimoku: bool = True
    use_keltner: bool = True
    use_donchian: bool = True
    use_vpt: bool = True
    use_adl: bool = True
    use_cmf: bool = True
    use_mfi: bool = True
    use_eom: bool = True
    use_linear_regression: bool = True
    use_support_resistance: bool = True
    use_day_dummies: bool = True
    use_month_dummies: bool = True
    use_seasonality: bool = True
    use_calendar_effects: bool = True
    use_talib: bool = True
    auto_select_features: bool = True
    num_features_to_select: int = 10
    feature_selection_method: str = "importance"

@dataclass
class PredictionConfig:
    predict_days: int = 30
    start_forecast_from_backtest: bool = False
    use_previous_close: bool = True
    set_initial_data: bool = False
    initial_data_period: int = 60  # Updated to match LearningConfig.timestamp

@dataclass
class PredictionAdvancedConfig:
    enable_monte_carlo: bool = False
    num_monte_carlo_simulations: int = 1000
    monte_carlo_seed: int = 42
    monte_carlo_scenarios: List[str] = field(default_factory=lambda: ["baseline", "bull", "bear", "volatile"])
    scenario_parameters: Dict[str, List[float]] = field(default_factory=lambda: {
        "baseline": [1.0, 1.0],
        "bull": [1.5, 0.8],
        "bear": [0.5, 1.2],
        "volatile": [1.0, 2.0]
    })
    use_ensemble_methods: bool = False
    ensemble_method: str = "voting"
    ensemble_models: List[str] = field(default_factory=lambda: ["lstm", "gru", "transformer"])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    enable_uncertainty_quantification: bool = False
    uncertainty_method: str = "mc_dropout"
    mc_dropout_samples: int = 100
    confidence_level: float = 0.95
    enable_rolling_window: bool = False
    validation_metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "mape", "smape", "directional_accuracy"])
    lstm_layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    lstm_units: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4])

@dataclass
class StrategyConfig:
    strategy_type: str = "buy_and_hold"
    enable_ml_optimization: bool = False
    initial_capital: float = 10000.0
    position_size_pct: float = 10.0
    take_profit_pct: float = 5.0
    stop_loss_pct: float = 3.0
    trailing_stop_pct: float = 2.0
    max_positions: int = 5
    reinvest_profits: bool = True
    use_saved_ml_model: bool = False
    ml_model_path: str = "strategy_ml_model.pkl"
    ml_features: List[str] = field(default_factory=lambda: ["price_momentum", "rsi", "macd", "volume_change"])
    backtest_period: int = 252
    optimization_metric: str = "sharpe_ratio"

@dataclass
class BacktestConfig:
    enable_backtesting: bool = False
    test_periods: int = 30
    walk_forward: bool = True

@dataclass
class AppConfig:
    csv: CSVConfig = CSVConfig()
    tf_config: TensorFlowConfig = TensorFlowConfig()
    gui: GUIConfig = GUIConfig()
    plot: PlotConfig = PlotConfig()
    learning: LearningConfig = LearningConfig()
    rolling_window: RollingWindowConfig = RollingWindowConfig()
    learning_pref: LearningPreferences = LearningPreferences()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    prediction: PredictionConfig = PredictionConfig()
    prediction_advanced: PredictionAdvancedConfig = PredictionAdvancedConfig()
    strategy: StrategyConfig = StrategyConfig()
    backtest: BacktestConfig = BacktestConfig()