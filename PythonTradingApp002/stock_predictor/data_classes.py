from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional, Any, Union
import datetime
import logging
import os
import traceback
from pandas.tseries.offsets import BDay


@dataclass
class StockData:
    """Container for stock data and transformations throughout the processing pipeline."""
    # Source data
    csv_file_path: str = None
    symbol: str = "Unknown"
    app_config: Any = None  # App config reference
    
    # Raw data directly from CSV
    csv_data_raw: pd.DataFrame = None
    
    # Trimmed/filtered data based on date ranges
    csv_data_train_period: pd.DataFrame = None  # Training period data
    csv_data_test_period: pd.DataFrame = None   # Testing/backtesting period data
    
    # Feature engineered data
    data_with_features: pd.DataFrame = None
    feature_list: List[str] = field(default_factory=list)
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Scaled data for model input
    data_scaled: pd.DataFrame = None
    
    # Time series information
    date_series_full: pd.Series = None  # Complete date series (train + test + future)
    training_dates: pd.Series = None
    testing_dates: pd.Series = None
    future_dates: pd.Series = None
    
    # Scaling tools
    price_scaler: MinMaxScaler = None  # For price/close values
    volume_scaler: MinMaxScaler = None  # For volume
    feature_scalers: Dict[str, MinMaxScaler] = field(default_factory=dict)  # For other features
    
    # Configuration
    volume_scaling_method: str = "minmax"
    training_start_date: Optional[datetime.datetime] = None
    backtesting_start_date: Optional[datetime.datetime] = None
    lookback_window: int = 100  # Timestamp/sequence length for time series
    predict_days: int = 30  # Number of days to predict

    def __post_init__(self):
        if self.csv_data_raw is not None and not isinstance(self.csv_data_raw, pd.DataFrame):
            raise TypeError("csv_data_raw must be a pandas DataFrame")
    
    @classmethod
    def from_app_config(cls, app_config):
        return cls(
            csv_file_path=app_config.csv.file_path,
            symbol=os.path.basename(app_config.csv.file_path).split('.')[0],
            app_config=app_config,
            volume_scaling_method=app_config.learning_pref.volume_scaling_method,
            training_start_date=pd.to_datetime(app_config.learning_pref.learning_start_date) 
                               if app_config.learning_pref.enable_learning_start_date else None,
            backtesting_start_date=pd.to_datetime(app_config.learning_pref.backtesting_start_date) 
                                 if app_config.learning_pref.enable_backtesting else None,
            lookback_window=app_config.learning.timestamp,
            predict_days=app_config.prediction.predict_days
        )
    
    @property
    def is_ready_for_training(self) -> bool:
        return (
            self.data_scaled is not None and 
            len(self.feature_list) > 0 and
            self.price_scaler is not None
        )
    
    def get_training_array(self) -> np.ndarray:
        if not self.is_ready_for_training:
            raise ValueError("Data not ready for training")
        return self.data_scaled.values
    
    def scale_close_price(self, close_values: np.ndarray) -> np.ndarray:
        if self.price_scaler is None:
            raise ValueError("Price scaler not initialized")
        if len(close_values.shape) == 1:
            close_values = close_values.reshape(-1, 1)
        return self.price_scaler.transform(close_values).flatten()
    
    def unscale_close_price(self, scaled_values: np.ndarray) -> np.ndarray:
        if self.price_scaler is None:
            raise ValueError("Price scaler not initialized")
        if len(scaled_values.shape) == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        return self.price_scaler.inverse_transform(scaled_values).flatten()
    
    def load_from_csv(self, parse_dates=None):
        if parse_dates is None:
            parse_dates = ['datetime']
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, self.csv_file_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        self.csv_data_raw = pd.read_csv(csv_path, parse_dates=parse_dates)
        required_columns = ['datetime', 'open', 'close', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in self.csv_data_raw.columns:
                raise ValueError(f"Required column '{col}' missing in CSV file")
        self.csv_data_raw['datetime'] = pd.to_datetime(self.csv_data_raw['datetime'], errors='coerce')
        self.csv_data_raw = self.csv_data_raw.sort_values(by='datetime')
        return self
    
    def split_by_dates(self):
        if self.csv_data_raw is None:
            raise ValueError("No raw data loaded. Call load_from_csv() first.")
        df = self.csv_data_raw.copy()
        if self.training_start_date is not None:
            df = df[df['datetime'] >= self.training_start_date].copy()
        if self.backtesting_start_date is not None:
            self.csv_data_train_period = df[df['datetime'] < self.backtesting_start_date].copy()
            self.csv_data_test_period = df[df['datetime'] >= self.backtesting_start_date].copy()
        else:
            self.csv_data_train_period = df.copy()
            self.csv_data_test_period = pd.DataFrame()
        return self
    
    def calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def apply_feature_engineering(self):
        """
        Apply feature engineering to the training data with clear decision logic.
        """
        if self.csv_data_train_period is None:
            raise ValueError("No training data. Call split_by_dates() first.")
        
        try:
            logging.info("Starting feature engineering process")
            df = self.csv_data_train_period.copy()
            
            # Always add some basic features
            df['previous_close'] = df['close'].shift(1)
            
            # Apply volume scaling
            method = self.volume_scaling_method
            if method == "minmax":
                df['volume_scaled'] = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())
            elif method == "log":
                df['volume_scaled'] = np.log1p(df['volume'])
            elif method == "sqrt":
                df['volume_scaled'] = np.sqrt(df['volume'])
            else:
                logging.warning(f"Invalid volume_scaling_method '{method}'. Using 'minmax'.")
                df['volume_scaled'] = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())
            
            # Make a clear decision about using advanced features
            use_advanced_features = False
            if hasattr(self.app_config, 'learning') and hasattr(self.app_config.learning, 'use_features'):
                use_advanced_features = self.app_config.learning.use_features
            
            if use_advanced_features:
                try:
                    from .feature_engineering import FeatureEngineer
                    logging.info(f"Applying advanced feature engineering with {len(self.csv_data_train_period)} rows of data")
                    feature_engineer = FeatureEngineer(self.app_config)
                    advanced_df = feature_engineer.create_all_features(df)
                    self.feature_importance_scores = feature_engineer.feature_importance_scores
                    self.data_with_features = advanced_df
                    logging.info("Advanced feature engineering completed successfully")
                except ImportError as e:
                    logging.warning(f"Could not import FeatureEngineer: {str(e)}. Falling back to basic features only")
                    self.data_with_features = df
                    use_advanced_features = False
                except Exception as e:
                    logging.error(f"Error in advanced feature engineering: {str(e)}")
                    logging.error(traceback.format_exc())
                    logging.warning("Falling back to basic features only")
                    self.data_with_features = df
                    use_advanced_features = False
            else:
                logging.info("Advanced feature engineering disabled; using basic CSV data without extra features")
                self.data_with_features = df
            
            # Drop NaN values
            original_len = len(self.data_with_features)
            self.data_with_features = self.data_with_features.dropna()
            dropped = original_len - len(self.data_with_features)
            if dropped > 0:
                logging.info(f"Dropped {dropped} rows due to NaN values")
            
            logging.info(f"Feature engineering completed. Data shape: {self.data_with_features.shape}")
            return self
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Feature engineering failed: {str(e)}")

    def build_feature_list(self):
        """
        Build the feature list for model input.
    
        If app_config.learning.use_features is False (the checkbox is not checked),
        then use all CSV columns (except 'datetime') from the raw data.
        If the flag is True, use the extended features computed during feature engineering.
        """
        if self.data_with_features is None:
            raise ValueError("No feature engineered data. Call apply_feature_engineering() first.")
    
        # If advanced features are disabled, use all CSV columns (except datetime)
        if not self.app_config.learning.use_features:
            basic_features = [col for col in self.data_with_features.columns if col != 'datetime']
            self.feature_list = basic_features
            logging.info(f"Use learning tab features disabled: using CSV columns: {self.feature_list}")
            return self

        # Otherwise, build an extended feature list.
        df = self.data_with_features
        # Start with a basic set of features.
        features = ['open', 'close', 'high', 'low', 'volume_scaled', 'previous_close']
    
        # Gather extra features not in the basic set.
        extra_features = [
            col for col in df.columns 
            if col not in ['datetime', 'open', 'close', 'high', 'low', 'volume', 'volume_scaled', 'previous_close']
        ]
    
        # Optionally, if feature importance scores are available, select top extra features.
        if hasattr(self, 'feature_importance_scores') and self.feature_importance_scores:
            sorted_features = sorted(self.feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, score in sorted_features if feat in extra_features]
            max_features = self.app_config.feature_selection.num_features_to_select if hasattr(self.app_config, 'feature_selection') else 20
            selected_features = selected_features[:min(len(selected_features), max_features)]
            logging.info(f"Selected {len(selected_features)} extra features based on importance scores")
            features.extend(selected_features)
        else:
            features.extend(extra_features)
    
        self.feature_list = features
        logging.info(f"Final feature list contains {len(features)} features: {features}")
        return self
    
    def scale_features(self):
        """Scale features with improved error handling and validation"""
        if self.data_with_features is None or not self.feature_list:
            raise ValueError("No features to scale. Call apply_feature_engineering() and build_feature_list() first.")
        
        try:
            df = self.data_with_features.copy()
            features = self.feature_list
        
            # Log feature scaling information
            logging.info(f"Scaling {len(features)} features: {features[:5]}{'...' if len(features) > 5 else ''}")
            
            # Validate features exist in dataframe
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise ValueError(f"The following features are missing in the dataframe: {missing_features}")
        
            # Replace infinity with NaN and convert to numeric
            for feature in features:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                # Check for infinite values
                inf_count = np.isinf(df[feature]).sum()
                if inf_count > 0:
                    logging.warning(f"Found {inf_count} infinite values in {feature}, replacing with NaN")
                    df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
        
            # Fill NaN values after conversion
            for feature in features:
                na_count = df[feature].isna().sum()
                if na_count > 0:
                    if feature in ['open', 'high', 'low', 'close', 'volume_scaled', 'previous_close']:
                        mean_val = df[feature].mean()
                        if pd.isna(mean_val):
                            logging.warning(f"Mean of {feature} is NaN, using 0 for filling")
                            mean_val = 0
                        df[feature].fillna(mean_val, inplace=True)
                        logging.info(f"Filled {na_count} NaN values in {feature} with mean value {mean_val:.4f}")
                    else:
                        df[feature].fillna(0, inplace=True)
                        logging.info(f"Filled {na_count} NaN values in {feature} with 0")
        
            # Create and apply scalers
            feature_scalers = {}
            for feature in features:
                try:
                    # Create a fresh scaler for each feature
                    scaler = MinMaxScaler()
                    # Reshape for scikit-learn's expected format
                    values = df[[feature]].values
                    scaler.fit(values)
                    feature_scalers[feature] = scaler
                    
                    # Log scaling range
                    min_val, max_val = np.min(values), np.max(values)
                    logging.debug(f"Feature {feature} scaled - Range: [{min_val:.4f}, {max_val:.4f}]")
                    
                except Exception as e:
                    logging.warning(f"Could not scale feature {feature}: {str(e)}. Using dummy scaler.")
                    dummy_scaler = MinMaxScaler()
                    dummy_scaler.fit(np.array([[0], [1]]))
                    feature_scalers[feature] = dummy_scaler
        
            # Store price scaler separately for convenience
            if 'close' in feature_scalers:
                self.price_scaler = feature_scalers['close']
            else:
                logging.error("No 'close' feature found for scaling!")
                raise ValueError("Required 'close' feature not found in feature list")
        
            # Create scaled dataframe
            scaled_data = pd.DataFrame()
            for feature in features:
                try:
                    # Transform using the fitted scaler
                    scaled_values = feature_scalers[feature].transform(df[[feature]]).flatten()
                    scaled_data[feature] = scaled_values
                except Exception as e:
                    logging.warning(f"Error transforming feature {feature}: {str(e)}. Using zeros.")
                    scaled_data[feature] = np.zeros(len(df))
        
            # Store results
            self.data_scaled = scaled_data
            self.feature_scalers = feature_scalers
            
            logging.info(f"Feature scaling completed successfully. Scaled data shape: {scaled_data.shape}")
            return self
            
        except Exception as e:
            logging.error(f"Feature scaling failed: {str(e)}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Feature scaling failed: {str(e)}")

    def prepare_date_series(self):
        """Prepare date series with explicit alignment for training, testing, and future periods"""
        if self.data_with_features is None:
            raise ValueError("No feature engineered data. Call apply_feature_engineering() first.")
        
        # Store training dates
        train_dates = self.data_with_features['datetime']
        self.training_dates = train_dates
        
        # Start with training dates
        date_series = train_dates.copy()
        
        # Add test dates if available
        if not self.csv_data_test_period.empty:
            test_dates = self.csv_data_test_period['datetime']
            self.testing_dates = test_dates
            date_series = pd.concat([date_series, test_dates])
        
        # Generate future dates based on prediction horizon using business day frequency
        last_date = date_series.iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=self.predict_days,
            freq='B'  # Business days to match trading days
        )
        
        self.future_dates = pd.Series(future_dates)
        
        # Store the complete date series
        full_series = pd.concat([date_series, pd.Series(future_dates)])
        self.date_series_full = full_series.reset_index(drop=True)
        
        # Log date series information for debugging
        logging.info(f"Date series created: Training ({len(train_dates)}), " +
                    f"Testing ({len(self.testing_dates) if hasattr(self, 'testing_dates') and self.testing_dates is not None else 0}), " +
                    f"Future ({len(future_dates)}), Total: {len(self.date_series_full)}")
        
        return self
    
    def prepare_all(self, parse_dates=None):
        return (self
                .load_from_csv(parse_dates)
                .split_by_dates()
                .apply_feature_engineering()
                .build_feature_list()
                .scale_features()
                .prepare_date_series())

    def validate_dates_data_consistency(self):
        """
        Ensure dates and data are aligned correctly
        
        Returns:
            bool: True if dates and data are aligned, False otherwise
        """
        if self.date_series_full is None or self.data_with_features is None:
            logging.error("Cannot validate date-data consistency: missing date_series_full or data_with_features")
            return False
            
        try:
            data_length = len(self.data_with_features)
            dates_length = len(self.training_dates)
            
            if data_length != dates_length:
                logging.error(f"Date-data mismatch: {dates_length} dates vs {data_length} data points")
                return False
                
            # Also check test data if available
            if hasattr(self, 'testing_dates') and self.testing_dates is not None and not self.csv_data_test_period.empty:
                test_data_length = len(self.csv_data_test_period)
                test_dates_length = len(self.testing_dates)
                
                if test_data_length != test_dates_length:
                    logging.error(f"Test date-data mismatch: {test_dates_length} dates vs {test_data_length} data points")
                    return False
            
            logging.info("Date-data consistency validation passed")
            return True
        except Exception as e:
            logging.error(f"Date validation error: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def create_sequence_data(self, timestamp_length=None):
        """
        Create sequences for model training with validation
        
        Args:
            timestamp_length (int, optional): Sequence length. Defaults to lookback_window.
            
        Returns:
            numpy.ndarray: Sequences for training
        """
        if timestamp_length is None:
            timestamp_length = self.lookback_window
            
        if self.data_scaled is None:
            raise ValueError("No scaled data available. Call scale_features() first.")
            
        if len(self.data_scaled) < timestamp_length:
            raise ValueError(f"Not enough data points ({len(self.data_scaled)}) for sequence length {timestamp_length}")
            
        # Generate sequences with proper length validation
        sequences = []
        labels = []
        feature_count = len(self.data_scaled.columns)
        close_idx = self.feature_list.index('close') if 'close' in self.feature_list else 0
        
        logging.info(f"Creating sequences with length {timestamp_length} from {len(self.data_scaled)} data points")
        
        for i in range(len(self.data_scaled) - timestamp_length):
            # Sequence as input
            seq = self.data_scaled.iloc[i:i+timestamp_length].values
            sequences.append(seq)
            
            # Target is the next close price
            target = self.data_scaled.iloc[i+timestamp_length][close_idx]
            labels.append(target)
            
        logging.info(f"Created {len(sequences)} sequences with shape {sequences[0].shape if sequences else 'N/A'}")
        
        return np.array(sequences), np.array(labels)

    def ensure_forecast_date_alignment(self, predictions, use_backtest=False):
        """
        Ensure forecast dates and predictions are properly aligned
        
        Args:
            predictions (numpy.ndarray): Predictions to align
            use_backtest (bool): Whether to start from backtest date
            
        Returns:
            tuple: (aligned_dates, aligned_predictions)
        """
        if predictions is None or len(predictions) == 0:
            logging.warning("Cannot align empty predictions")
            return None, None
            
        logging.info(f"Aligning {len(predictions)} predictions with dates")
        
        if use_backtest and self.backtesting_start_date is not None:
            # Get backtest start index
            backtest_date = pd.to_datetime(self.backtesting_start_date)
            backtest_idx = self.training_dates.searchsorted(backtest_date)
            if backtest_idx >= len(self.training_dates):
                logging.warning(f"Backtest date {backtest_date} is beyond available training dates")
                backtest_idx = len(self.training_dates) - 1
            forecast_start_idx = backtest_idx
            logging.info(f"Using backtest starting point at index {backtest_idx} ({backtest_date})")
        else:
            # Start from the end of training data
            forecast_start_idx = len(self.training_dates)
            logging.info(f"Starting forecast from end of training data (index {forecast_start_idx})")
            
        # Get forecast dates
        forecast_end_idx = min(forecast_start_idx + len(predictions), len(self.date_series_full))
        forecast_dates = self.date_series_full[forecast_start_idx:forecast_end_idx]
        
        # Adjust predictions if needed
        if len(forecast_dates) < len(predictions):
            logging.warning(f"Truncating predictions from {len(predictions)} to {len(forecast_dates)} to match available dates")
            predictions = predictions[:len(forecast_dates)]
        elif len(forecast_dates) > len(predictions):
            logging.warning(f"Truncating forecast dates from {len(forecast_dates)} to {len(predictions)} to match predictions")
            forecast_dates = forecast_dates[:len(predictions)]
        
        logging.info(f"Aligned {len(predictions)} predictions with {len(forecast_dates)} dates")
        return forecast_dates, predictions

    def validate_configuration(self):
        """
        Validate critical configuration settings
        
        Returns:
            list: List of validation issues found
        """
        issues = []
        
        # Check that timestamp/lookback window is reasonable
        if self.lookback_window <= 0 or self.lookback_window > 300:
            issues.append(f"Invalid lookback_window: {self.lookback_window}. Should be between 1 and 300.")
            
        # Check that prediction days is reasonable
        if self.predict_days <= 0 or self.predict_days > 365:
            issues.append(f"Invalid predict_days: {self.predict_days}. Should be between 1 and 365.")
            
        # Check that training and backtesting dates make sense
        if self.training_start_date is not None and self.backtesting_start_date is not None:
            if self.training_start_date >= self.backtesting_start_date:
                issues.append(f"Training start date ({self.training_start_date}) must be before backtesting start date ({self.backtesting_start_date})")
        
        # Check that we have enough data for the lookback window
        if self.data_with_features is not None:
            if len(self.data_with_features) < self.lookback_window:
                issues.append(f"Not enough data ({len(self.data_with_features)} points) for lookback window ({self.lookback_window})")
        
        # Check that feature list is not empty
        if hasattr(self, 'feature_list') and not self.feature_list:
            issues.append("Feature list is empty")
        
        # Log validation results
        if issues:
            logging.warning(f"Configuration validation found {len(issues)} issues:")
            for i, issue in enumerate(issues):
                logging.warning(f"  {i+1}. {issue}")
        else:
            logging.info("Configuration validation passed with no issues")
            
        return issues

    def safe_unscale_predictions(self, scaled_predictions):
        """
        Safely unscale predictions with dimension checking
        
        Args:
            scaled_predictions (numpy.ndarray): Scaled predictions
            
        Returns:
            numpy.ndarray: Unscaled predictions
        """
        if scaled_predictions is None:
            logging.warning("Cannot unscale None predictions")
            return None
        
        try:
            # Log the input shape for debugging
            input_shape = scaled_predictions.shape
            logging.debug(f"Unscaling predictions with shape {input_shape}")
            
            if len(input_shape) == 1:
                # Single prediction array
                unscaled = self.unscale_close_price(scaled_predictions)
                logging.debug(f"Unscaled single prediction array, shape: {unscaled.shape}")
                return unscaled
            elif len(input_shape) == 2:
                # Multiple prediction arrays (e.g., ensemble)
                ensemble_unscaled = []
                for i, pred in enumerate(scaled_predictions):
                    unscaled = self.unscale_close_price(pred)
                    ensemble_unscaled.append(unscaled)
                result = np.array(ensemble_unscaled)
                logging.debug(f"Unscaled ensemble predictions, shape: {result.shape}")
                return result
            else:
                raise ValueError(f"Unexpected prediction shape: {input_shape}")
        except Exception as e:
            logging.error(f"Error unscaling predictions: {str(e)}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Failed to unscale predictions: {str(e)}")

@dataclass
class ForecastResults:
    """Container for forecast results, metrics, and model diagnostics."""
    # Individual simulation results
    simulation_predictions: List[np.ndarray] = field(default_factory=list)  # Raw predictions from each simulation
    
    # Ensemble predictions
    ensemble_mean: np.ndarray = None  # Mean predictions across top simulations
    ensemble_std: np.ndarray = None   # Standard deviation of predictions
    
    # Confidence intervals
    confidence_intervals: Dict[str, np.ndarray] = field(default_factory=dict)  # Lower/upper bounds
    
    # Performance metrics
    error_metrics: Dict[str, np.ndarray] = field(default_factory=dict)  # SMAPE, MAE, etc.
    best_simulation_indices: np.ndarray = None  # Indices of simulations with lowest error
    
    # Model training diagnostics
    learning_curves: List[Dict[str, float]] = field(default_factory=list)  # Loss history during training
    final_model_loss: float = None
    
    # Feature analysis
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Performance timing
    model_training_time: float = None
    prediction_generation_time: float = None
    
    # Configuration used
    sequence_length: int = None  # Lookback window used
    forecast_horizon: int = None  # Number of days forecasted
    
    @property
    def has_predictions(self) -> bool:
        return len(self.simulation_predictions) > 0
    
    @property
    def ensemble_size(self) -> int:
        return len(self.simulation_predictions)
    
    @property
    def prediction_length(self) -> int:
        if self.ensemble_mean is not None:
            return len(self.ensemble_mean)
        elif len(self.simulation_predictions) > 0:
            return len(self.simulation_predictions[0])
        return 0
    
    def get_best_predictions(self, n: int = 5) -> List[np.ndarray]:
        if self.best_simulation_indices is None or len(self.best_simulation_indices) < n:
            return self.simulation_predictions[:n]
        return [self.simulation_predictions[i] for i in self.best_simulation_indices[:n]]
    
    def calculate_smape(self, actual, predicted, epsilon=1e-8):
        actual = np.array(actual)
        predicted = np.array(predicted)
        numerator = 2.0 * np.abs(actual - predicted)
        denominator = np.abs(actual) + np.abs(predicted) + epsilon
        return 100.0 * np.mean(numerator / denominator)
    
    def evaluate_predictions(self, stock_data: StockData):
        import time
        start_time = time.time()

        if not self.has_predictions:
            logging.error("No predictions to evaluate!")
            raise ValueError("No predictions to evaluate")

        results_array = np.array(self.simulation_predictions)
        smape_scores = []

        logging.debug(f"Results Array Shape: {results_array.shape if isinstance(results_array, np.ndarray) else 'None'}")

        # Determine the relevant actual data based on forecast horizon
        horizon = self.forecast_horizon  # Example: 30 days
        for result in results_array:
            pred = stock_data.unscale_close_price(result)

            if stock_data.backtesting_start_date is not None and not stock_data.csv_data_test_period.empty:
                actual = stock_data.csv_data_test_period['close'].values[:horizon]
                if len(actual) < horizon:
                    logging.warning(f"Actual backtest data length {len(actual)} is less than forecast horizon {horizon}. Padding with NaN.")
                    actual = np.pad(actual, (0, horizon - len(actual)), 'constant', constant_values=np.nan)
            else:
                actual = stock_data.csv_data_train_period['close'].values[-horizon:]
                if len(actual) < horizon:
                    logging.warning(f"Actual training data length {len(actual)} is less than forecast horizon {horizon}. Padding with NaN.")
                    actual = np.pad(actual, (0, horizon - len(actual)), 'constant', constant_values=np.nan)

            if len(pred) != horizon:
                logging.error(f"Prediction length {len(pred)} does not match forecast horizon {horizon}. This should not happen.")
                pred = pred[:horizon] if len(pred) > horizon else np.pad(pred, (0, horizon - len(pred)), 'constant', constant_values=np.nan)

            smape_scores.append(self.calculate_smape(actual, pred))

        logging.debug(f"sMAPE Scores: {smape_scores if smape_scores else 'EMPTY!'}")

        self.error_metrics['smape'] = np.array(smape_scores)

        if smape_scores:
            num_top = min(5, len(smape_scores))
            self.best_simulation_indices = np.argsort(smape_scores)[:num_top]
        else:
            logging.error("sMAPE Scores are empty! Setting top_indices to an empty list.")
            self.best_simulation_indices = []

        # Compute ensemble mean and standard deviation
        top_predictions = results_array[self.best_simulation_indices] if smape_scores else []
        self.ensemble_mean = np.mean(top_predictions, axis=0) if smape_scores else np.array([])
        self.ensemble_std = np.std(top_predictions, axis=0) if smape_scores else np.array([])

        # Compute confidence intervals
        if smape_scores:
            mean = self.ensemble_mean
            std = self.ensemble_std
            z_score = 1.96  # 95% confidence
            self.confidence_intervals = {
                'lower': mean - z_score * std,
                'upper': mean + z_score * std,
                'mean': mean
            }

        self.prediction_generation_time = time.time() - start_time
        return self

    
    def calculate_confidence_intervals(self, confidence=0.95):
        if self.ensemble_mean is None or self.ensemble_std is None:
            raise ValueError("Ensemble statistics not calculated")
        z_score = 1.96  # for 95% confidence
        mean = self.ensemble_mean
        std = self.ensemble_std
        self.confidence_intervals = {
            'lower': mean - z_score * std,
            'upper': mean + z_score * std,
            'mean': mean
        }
        return self
    
    def to_plot_format(self, stock_data: StockData) -> tuple:
        return (
            stock_data.data_scaled,
            stock_data.date_series_full,
            stock_data.price_scaler,
            self.ensemble_mean,
            self.ensemble_std,
            self.simulation_predictions,  # results_array
            self.error_metrics.get('smape', np.array([])),
            self.best_simulation_indices,  # top_indices
            stock_data.csv_data_train_period,
            stock_data.csv_data_test_period,
            {'epoch_history': self.learning_curves, 'final_loss': self.final_model_loss},
            self.feature_importance_scores
        )