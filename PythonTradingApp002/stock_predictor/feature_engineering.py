"""
feature_engineering.py - Advanced feature engineering for stock price prediction

This module provides a comprehensive set of technical indicators and features
for stock price prediction. It's designed to work with the existing stock
prediction framework and supports flexible configuration.
"""

import numpy as np
import pandas as pd
import logging
import traceback
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
from scipy import stats, signal
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# Try to import TA-Lib, but don't fail if it's not available
try:
    import talib
    TALIB_AVAILABLE = True
    logging.info("TA-Lib is available and will be used for technical indicators")
except ImportError:
    TALIB_AVAILABLE = False
    logging.info("TA-Lib not available. Using custom implementations for technical indicators.")

class FeatureEngineer:
    """Handles the creation and selection of features for stock prediction."""

    def __init__(self, app_config):
        """
        Initialize the feature engineer with application configuration.

        Args:
            app_config: The application configuration object containing feature selection settings
        """
        self.config = app_config
        self.feature_importance_scores = {}
        self.feature_groups = {}
        self.talib_available = TALIB_AVAILABLE

        if not self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
            logging.warning("TA-Lib was requested but is not available. Setting use_talib to False.")
            self.config.feature_selection.use_talib = False

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enabled feature engineering based on configuration.

        Args:
            df: DataFrame with at least datetime, open, high, low, close, volume columns

        Returns:
            DataFrame with additional engineered features
        """
        if df.empty:
            logging.warning("Empty DataFrame provided to feature engineering")
            return df

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Verify required columns exist
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'volume_scaled', 'previous_close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing required columns: {missing_columns}")
            # If critical columns are missing, return the original dataframe
            if any(col in missing_columns for col in ['datetime', 'open', 'high', 'low', 'close', 'volume']):
                logging.error("Critical columns missing. Cannot proceed with feature engineering.")
                return df

        # Store original columns for later reference
        original_columns = df.columns.tolist()

        # Dictionary to track created feature groups
        self.feature_groups = {}

        try:
            # Apply features based on config settings
            if hasattr(self.config.feature_selection, 'use_trend_indicators') and self.config.feature_selection.use_trend_indicators:
                df = self.add_trend_indicators(df)

            if hasattr(self.config.feature_selection, 'use_volatility_indicators') and self.config.feature_selection.use_volatility_indicators:
                df = self.add_volatility_indicators(df)

            if hasattr(self.config.feature_selection, 'use_momentum_indicators') and self.config.feature_selection.use_momentum_indicators:
                df = self.add_momentum_indicators(df)

            if hasattr(self.config.feature_selection, 'use_volume_indicators') and self.config.feature_selection.use_volume_indicators:
                df = self.add_volume_indicators(df)

            if hasattr(self.config.feature_selection, 'use_price_patterns') and self.config.feature_selection.use_price_patterns:
                df = self.add_price_patterns(df)

            if hasattr(self.config.feature_selection, 'use_time_features') and self.config.feature_selection.use_time_features:
                df = self.add_temporal_features(df)

            if hasattr(self.config.feature_selection, 'use_financial_features') and self.config.feature_selection.use_financial_features:
                df = self.add_financial_features(df)

            if hasattr(self.config.feature_selection, 'use_market_regime') and self.config.feature_selection.use_market_regime:
                df = self.add_market_regime_features(df)

            # Calculate derivate features across groups
            if hasattr(self.config.feature_selection, 'use_derivative_features') and self.config.feature_selection.use_derivative_features:
                df = self.add_derivative_features(df, original_columns)

            # Handle NaN values
            df = self.handle_missing_values(df)

            # Log feature creation results
            new_columns = [col for col in df.columns if col not in original_columns]
            logging.info(f"Created {len(new_columns)} new features")
            for group, features in self.feature_groups.items():
                logging.info(f"  {group}: {len(features)} features")

            # Perform feature selection if enabled
            if hasattr(self.config.feature_selection, 'auto_select_features') and self.config.feature_selection.auto_select_features:
                try:
                    selected_columns = self.select_best_features(
                        df,
                        target_col='close',
                        top_n=self.config.feature_selection.num_features_to_select
                    )
                    # Always include essential columns
                    essential_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'volume_scaled', 'previous_close']
                    selected_columns = list(set(essential_columns + selected_columns))
                    df = df[selected_columns]
                    logging.info(f"Selected {len(selected_columns)} features after feature selection")
                except Exception as e:
                    logging.error(f"Error during feature selection: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with all features if selection fails

        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Return the original dataframe with at least basic features

        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added trend indicators
        """
        logging.info("Adding trend indicators")
        trend_features = []

        try:
            # Moving averages
            ma_periods = [5, 10, 20, 50, 100, 200]
            for period in ma_periods:
                if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                    df[f'SMA_{period}'] = talib.SMA(df['close'].values, timeperiod=period)
                    df[f'EMA_{period}'] = talib.EMA(df['close'].values, timeperiod=period)
                else:
                    df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
                    df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                trend_features.extend([f'SMA_{period}', f'EMA_{period}'])

            # MACD (Moving Average Convergence Divergence)
            if hasattr(self.config.feature_selection, 'use_macd') and self.config.feature_selection.use_macd:
                if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
                        df['close'].values,
                        fastperiod=12,
                        slowperiod=26,
                        signalperiod=9
                    )
                else:
                    ema12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = ema12 - ema26
                    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
                trend_features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])

            # Moving average crossovers (binary indicators)
            for fast, slow in [(5, 20), (10, 50), (50, 200)]:
                if f'SMA_{fast}' in df.columns and f'SMA_{slow}' in df.columns:
                    df[f'Cross_{fast}_{slow}'] = ((df[f'SMA_{fast}'] > df[f'SMA_{slow}']) &
                                                (df[f'SMA_{fast}'].shift(1) <= df[f'SMA_{slow}'].shift(1))).astype(int)
                    df[f'Cross_down_{fast}_{slow}'] = ((df[f'SMA_{fast}'] < df[f'SMA_{slow}']) &
                                                    (df[f'SMA_{fast}'].shift(1) >= df[f'SMA_{slow}'].shift(1))).astype(int)
                    trend_features.extend([f'Cross_{fast}_{slow}', f'Cross_down_{fast}_{slow}'])

            # Price relative to moving averages
            for period in [20, 50, 200]:
                if f'SMA_{period}' in df.columns:
                    df[f'Price_to_MA_{period}'] = df['close'] / df[f'SMA_{period}']
                    trend_features.append(f'Price_to_MA_{period}')

            # Parabolic SAR
            if hasattr(self.config.feature_selection, 'use_parabolic_sar') and self.config.feature_selection.use_parabolic_sar:
                if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                    df['SAR'] = talib.SAR(df['high'].values, df['low'].values,
                                          acceleration=0.02, maximum=0.2)
                else:
                    df['SAR'] = self.calculate_parabolic_sar(df)
                df['Above_SAR'] = (df['close'] > df['SAR']).astype(int)
                trend_features.extend(['SAR', 'Above_SAR'])

            # Linear regression slope
            if hasattr(self.config.feature_selection, 'use_linear_regression') and self.config.feature_selection.use_linear_regression:
                for period in [10, 20, 50]:
                    df[f'Linear_Slope_{period}'] = self.calculate_linear_regression_slope(df['close'], period)
                    trend_features.append(f'Linear_Slope_{period}')

            # Ichimoku Cloud (simplified version)
            if hasattr(self.config.feature_selection, 'use_ichimoku') and self.config.feature_selection.use_ichimoku:
                # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
                df['Ichimoku_Conversion'] = (df['high'].rolling(window=9).max() +
                                          df['low'].rolling(window=9).min()) / 2

                # Kijun-sen (Base Line): (26-period high + 26-period low)/2
                df['Ichimoku_Base'] = (df['high'].rolling(window=26).max() +
                                    df['low'].rolling(window=26).min()) / 2

                # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
                df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)

                # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
                df['Ichimoku_SpanB'] = ((df['high'].rolling(window=52).max() +
                                      df['low'].rolling(window=52).min()) / 2).shift(26)

                # Chikou Span (Lagging Span): Current closing price, shifted backwards 26 periods
                df['Ichimoku_Lagging'] = df['close'].shift(-26)

                df['Above_Cloud'] = ((df['close'] > df['Ichimoku_SpanA']) &
                                   (df['close'] > df['Ichimoku_SpanB'])).astype(int)
                df['Below_Cloud'] = ((df['close'] < df['Ichimoku_SpanA']) &
                                   (df['close'] < df['Ichimoku_SpanB'])).astype(int)

                trend_features.extend(['Ichimoku_Conversion', 'Ichimoku_Base', 'Ichimoku_SpanA',
                                      'Ichimoku_SpanB', 'Above_Cloud', 'Below_Cloud'])

            # Store created features for reference
            self.feature_groups['trend'] = trend_features

        except Exception as e:
            logging.error(f"Error adding trend indicators: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added volatility indicators
        """
        logging.info("Adding volatility indicators")
        volatility_features = []

        try:
            # Bollinger Bands
            if hasattr(self.config.feature_selection, 'use_bollinger_bands') and self.config.feature_selection.use_bollinger_bands:
                for period in [5, 20, 50]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'BB_Upper_{period}'], df[f'BB_Middle_{period}'], df[f'BB_Lower_{period}'] = talib.BBANDS(
                            df['close'].values, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0
                        )
                    else:
                        rolling_mean = df['close'].rolling(window=period).mean()
                        rolling_std = df['close'].rolling(window=period).std()
                        df[f'BB_Upper_{period}'] = rolling_mean + (rolling_std * 2)
                        df[f'BB_Middle_{period}'] = rolling_mean
                        df[f'BB_Lower_{period}'] = rolling_mean - (rolling_std * 2)

                    # Bandwidth and %B
                    df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
                    df[f'BB_Pct_{period}'] = (df['close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])

                    volatility_features.extend([f'BB_Upper_{period}', f'BB_Middle_{period}',
                                               f'BB_Lower_{period}', f'BB_Width_{period}', f'BB_Pct_{period}'])

            # Average True Range (ATR)
            if hasattr(self.config.feature_selection, 'use_atr') and self.config.feature_selection.use_atr:
                for period in [7, 14, 21]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'ATR_{period}'] = talib.ATR(df['high'].values, df['low'].values,
                                                       df['close'].values, timeperiod=period)
                    else:
                        df[f'ATR_{period}'] = self.calculate_atr(df, period)

                    # Normalized ATR (ATR %)
                    df[f'ATR_Pct_{period}'] = df[f'ATR_{period}'] / df['close'] * 100
                    volatility_features.extend([f'ATR_{period}', f'ATR_Pct_{period}'])

            # Keltner Channels
            if hasattr(self.config.feature_selection, 'use_keltner') and self.config.feature_selection.use_keltner:
                for period in [20]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        ema = talib.EMA(df['close'].values, timeperiod=period)
                        atr = talib.ATR(df['high'].values, df['low'].values,
                                        df['close'].values, timeperiod=period)
                    else:
                        ema = df['close'].ewm(span=period, adjust=False).mean()
                        atr = self.calculate_atr(df, period)

                    df[f'Keltner_Middle_{period}'] = ema
                    df[f'Keltner_Upper_{period}'] = ema + 2 * atr
                    df[f'Keltner_Lower_{period}'] = ema - 2 * atr

                    volatility_features.extend([f'Keltner_Middle_{period}',
                                               f'Keltner_Upper_{period}',
                                               f'Keltner_Lower_{period}'])

            # Historical Volatility
            for period in [5, 21, 63]:  # ~1 week, 1 month, 3 months
                df[f'Log_Return'] = np.log(df['close'] / df['close'].shift(1))
                df[f'Volatility_{period}'] = df['Log_Return'].rolling(window=period).std() * np.sqrt(252)  # Annualized
                volatility_features.extend([f'Volatility_{period}'])

            # Donchian Channels
            if hasattr(self.config.feature_selection, 'use_donchian') and self.config.feature_selection.use_donchian:
                for period in [20, 50]:
                    df[f'Donchian_High_{period}'] = df['high'].rolling(window=period).max()
                    df[f'Donchian_Low_{period}'] = df['low'].rolling(window=period).min()
                    df[f'Donchian_Mid_{period}'] = (df[f'Donchian_High_{period}'] + df[f'Donchian_Low_{period}']) / 2

                    # Donchian Width
                    df[f'Donchian_Width_{period}'] = (df[f'Donchian_High_{period}'] - df[f'Donchian_Low_{period}']) / df[f'Donchian_Mid_{period}']

                    volatility_features.extend([f'Donchian_High_{period}', f'Donchian_Low_{period}',
                                               f'Donchian_Mid_{period}', f'Donchian_Width_{period}'])

            # Store created features for reference
            self.feature_groups['volatility'] = volatility_features

        except Exception as e:
            logging.error(f"Error adding volatility indicators: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added momentum indicators
        """
        logging.info("Adding momentum indicators")
        momentum_features = []

        try:
            # Relative Strength Index (RSI)
            if hasattr(self.config.feature_selection, 'use_rsi') and self.config.feature_selection.use_rsi:
                for period in [6, 14, 28]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'RSI_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
                    else:
                        df[f'RSI_{period}'] = self.calculate_rsi(df['close'], period)

                    # RSI conditions
                    df[f'RSI_Overbought_{period}'] = (df[f'RSI_{period}'] > 70).astype(int)
                    df[f'RSI_Oversold_{period}'] = (df[f'RSI_{period}'] < 30).astype(int)

                    momentum_features.extend([f'RSI_{period}', f'RSI_Overbought_{period}', f'RSI_Oversold_{period}'])

            # Stochastic Oscillator
            if hasattr(self.config.feature_selection, 'use_stochastic') and self.config.feature_selection.use_stochastic:
                for k_period, d_period in [(14, 3), (21, 7)]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'Stoch_K_{k_period}'], df[f'Stoch_D_{k_period}_{d_period}'] = talib.STOCH(
                            df['high'].values, df['low'].values, df['close'].values,
                            fastk_period=k_period, slowk_period=3, slowk_matype=0,
                            slowd_period=d_period, slowd_matype=0
                        )
                    else:
                        df[f'Stoch_K_{k_period}'] = self.calculate_stochastic_k(df, k_period)
                        df[f'Stoch_D_{k_period}_{d_period}'] = df[f'Stoch_K_{k_period}'].rolling(window=d_period).mean()

                    # Stochastic conditions
                    df[f'Stoch_Overbought_{k_period}'] = (df[f'Stoch_K_{k_period}'] > 80).astype(int)
                    df[f'Stoch_Oversold_{k_period}'] = (df[f'Stoch_K_{k_period}'] < 20).astype(int)
                    df[f'Stoch_Cross_Up_{k_period}'] = ((df[f'Stoch_K_{k_period}'] > df[f'Stoch_D_{k_period}_{d_period}']) &
                                                      (df[f'Stoch_K_{k_period}'].shift(1) <= df[f'Stoch_D_{k_period}_{d_period}'].shift(1))).astype(int)

                    momentum_features.extend([f'Stoch_K_{k_period}', f'Stoch_D_{k_period}_{d_period}',
                                             f'Stoch_Overbought_{k_period}', f'Stoch_Oversold_{k_period}',
                                             f'Stoch_Cross_Up_{k_period}'])

            # Rate of Change (ROC)
            for period in [5, 10, 21, 63]:
                df[f'ROC_{period}'] = df['close'].pct_change(periods=period) * 100
                momentum_features.append(f'ROC_{period}')

            # Average Directional Index (ADX)
            if hasattr(self.config.feature_selection, 'use_adx') and self.config.feature_selection.use_adx:
                for period in [14, 28]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'ADX_{period}'] = talib.ADX(df['high'].values, df['low'].values,
                                                      df['close'].values, timeperiod=period)
                        df[f'DI_Plus_{period}'] = talib.PLUS_DI(df['high'].values, df['low'].values,
                                                              df['close'].values, timeperiod=period)
                        df[f'DI_Minus_{period}'] = talib.MINUS_DI(df['high'].values, df['low'].values,
                                                                df['close'].values, timeperiod=period)
                    else:
                        adx_result = self.calculate_adx(df, period)
                        df[f'ADX_{period}'] = adx_result['ADX']
                        df[f'DI_Plus_{period}'] = adx_result['DI+']
                        df[f'DI_Minus_{period}'] = adx_result['DI-']

                    # ADX conditions
                    df[f'Strong_Trend_{period}'] = (df[f'ADX_{period}'] > 25).astype(int)
                    df[f'DI_Cross_Up_{period}'] = ((df[f'DI_Plus_{period}'] > df[f'DI_Minus_{period}']) &
                                                 (df[f'DI_Plus_{period}'].shift(1) <= df[f'DI_Minus_{period}'].shift(1))).astype(int)

                    momentum_features.extend([f'ADX_{period}', f'DI_Plus_{period}', f'DI_Minus_{period}',
                                             f'Strong_Trend_{period}', f'DI_Cross_Up_{period}'])

            # Williams %R
            if hasattr(self.config.feature_selection, 'use_williams_r') and self.config.feature_selection.use_williams_r:
                for period in [14, 28]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'Williams_R_{period}'] = talib.WILLR(df['high'].values, df['low'].values,
                                                               df['close'].values, timeperiod=period)
                    else:
                        df[f'Williams_R_{period}'] = self.calculate_williams_r(df, period)

                    momentum_features.append(f'Williams_R_{period}')

            # Commodity Channel Index (CCI)
            if hasattr(self.config.feature_selection, 'use_cci') and self.config.feature_selection.use_cci:
                for period in [14, 20]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'CCI_{period}'] = talib.CCI(df['high'].values, df['low'].values,
                                                      df['close'].values, timeperiod=period)
                    else:
                        df[f'CCI_{period}'] = self.calculate_cci(df, period)

                    # CCI conditions
                    df[f'CCI_Overbought_{period}'] = (df[f'CCI_{period}'] > 100).astype(int)
                    df[f'CCI_Oversold_{period}'] = (df[f'CCI_{period}'] < -100).astype(int)

                    momentum_features.extend([f'CCI_{period}', f'CCI_Overbought_{period}', f'CCI_Oversold_{period}'])

            # Store created features for reference
            self.feature_groups['momentum'] = momentum_features

        except Exception as e:
            logging.error(f"Error adding momentum indicators: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added volume indicators
        """
        logging.info("Adding volume indicators")
        volume_features = []

        try:
            # Volume Moving Averages
            for period in [10, 20, 50]:
                df[f'Volume_SMA_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'Volume_Ratio_{period}'] = df['volume'] / df[f'Volume_SMA_{period}']
                volume_features.extend([f'Volume_SMA_{period}', f'Volume_Ratio_{period}'])

            # On-Balance Volume (OBV)
            if hasattr(self.config.feature_selection, 'use_obv') and self.config.feature_selection.use_obv:
                if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                    df['OBV'] = talib.OBV(df['close'].values, df['volume'].values)
                else:
                    df['OBV'] = self.calculate_obv(df)

                # OBV Moving Averages
                df['OBV_SMA_10'] = df['OBV'].rolling(window=10).mean()
                df['OBV_SMA_20'] = df['OBV'].rolling(window=20).mean()

                # OBV Signals
                df['OBV_Cross_Up'] = ((df['OBV'] > df['OBV_SMA_20']) &
                                    (df['OBV'].shift(1) <= df['OBV_SMA_20'].shift(1))).astype(int)

                volume_features.extend(['OBV', 'OBV_SMA_10', 'OBV_SMA_20', 'OBV_Cross_Up'])

            # Volume Price Trend (VPT)
            if hasattr(self.config.feature_selection, 'use_vpt') and self.config.feature_selection.use_vpt:
                df['VPT'] = self.calculate_vpt(df)
                df['VPT_SMA_10'] = df['VPT'].rolling(window=10).mean()
                df['VPT_SMA_20'] = df['VPT'].rolling(window=20).mean()

                volume_features.extend(['VPT', 'VPT_SMA_10', 'VPT_SMA_20'])

            # Accumulation/Distribution Line
            if hasattr(self.config.feature_selection, 'use_adl') and self.config.feature_selection.use_adl:
                if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                    df['ADL'] = talib.AD(df['high'].values, df['low'].values,
                                        df['close'].values, df['volume'].values)
                else:
                    df['ADL'] = self.calculate_adl(df)

                df['ADL_SMA_10'] = df['ADL'].rolling(window=10).mean()
                df['ADL_Slope_5'] = self.calculate_linear_regression_slope(df['ADL'], 5)

                volume_features.extend(['ADL', 'ADL_SMA_10', 'ADL_Slope_5'])

            # Chaikin Money Flow
            if hasattr(self.config.feature_selection, 'use_cmf') and self.config.feature_selection.use_cmf:
                for period in [20, 50]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'CMF_{period}'] = talib.ADOSC(df['high'].values, df['low'].values,
                                                         df['close'].values, df['volume'].values,
                                                         fastperiod=3, slowperiod=period)
                    else:
                        df[f'CMF_{period}'] = self.calculate_cmf(df, period)
                    
                    volume_features.append(f'CMF_{period}')

            # Money Flow Index (MFI)
            if hasattr(self.config.feature_selection, 'use_mfi') and self.config.feature_selection.use_mfi:
                for period in [14, 21]:
                    if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                        df[f'MFI_{period}'] = talib.MFI(df['high'].values, df['low'].values,
                                                      df['close'].values, df['volume'].values,
                                                      timeperiod=period)
                    else:
                        df[f'MFI_{period}'] = self.calculate_mfi(df, period)

                    # MFI conditions
                    df[f'MFI_Overbought_{period}'] = (df[f'MFI_{period}'] > 80).astype(int)
                    df[f'MFI_Oversold_{period}'] = (df[f'MFI_{period}'] < 20).astype(int)

                    volume_features.extend([f'MFI_{period}', f'MFI_Overbought_{period}', f'MFI_Oversold_{period}'])

            # Ease of Movement (EOM)
            if hasattr(self.config.feature_selection, 'use_eom') and self.config.feature_selection.use_eom:
                for period in [14, 21]:
                    df[f'EOM_{period}'] = self.calculate_eom(df, period)
                    volume_features.append(f'EOM_{period}')

            # Volume-price relationship
            df['Close_x_Volume'] = df['close'] * df['volume']
            df['Close_x_Volume_SMA_10'] = df['Close_x_Volume'].rolling(window=10).mean()

            volume_features.extend(['Close_x_Volume', 'Close_x_Volume_SMA_10'])

            # Store created features for reference
            self.feature_groups['volume'] = volume_features

        except Exception as e:
            logging.error(f"Error adding volume indicators: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        return df

    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price pattern and candlestick pattern indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added price pattern indicators
        """
        logging.info("Adding price pattern indicators")
        pattern_features = []

        try:
            # Initialize basic candlestick properties
            df['Body'] = abs(df['close'] - df['open'])
            df['Body_Pct'] = df['Body'] / ((df['high'] - df['low']).replace(0, 0.001)) * 100
            df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['Is_Bullish'] = (df['close'] > df['open']).fillna(False).astype(int)
            df['Is_Bearish'] = (df['close'] < df['open']).fillna(False).astype(int)

            pattern_features.extend(['Body', 'Body_Pct', 'Upper_Shadow', 'Lower_Shadow',
                                    'Is_Bullish', 'Is_Bearish'])

            # Common candlestick patterns
            # Doji
            df['Doji'] = ((df['Body'] / (df['high'] - df['low'] + 0.001)) < 0.1).fillna(False).astype(int)

            # Hammer
            df['Hammer'] = ((df['Lower_Shadow'] > 2 * df['Body']).fillna(False) &
                            (df['Upper_Shadow'] < 0.2 * df['Body']).fillna(False) &
                            (df['Body_Pct'] < 50).fillna(False)).astype(int)

            # Shooting Star
            df['Shooting_Star'] = ((df['Upper_Shadow'] > 2 * df['Body']).fillna(False) &
                                  (df['Lower_Shadow'] < 0.2 * df['Body']).fillna(False) &
                                  (df['Body_Pct'] < 50).fillna(False)).astype(int)

            # Engulfing patterns
            df['Bullish_Engulfing'] = ((df['Is_Bullish'].astype(bool)) &
                                      (df['Is_Bearish'].shift(1).fillna(False).astype(bool)) &
                                      (df['open'] < df['close'].shift(1)).fillna(False) &
                                      (df['close'] > df['open'].shift(1)).fillna(False)).astype(int)

            df['Bearish_Engulfing'] = ((df['Is_Bearish'].astype(bool)) &
                                      (df['Is_Bullish'].shift(1).fillna(False).astype(bool)) &
                                      (df['open'] > df['close'].shift(1)).fillna(False) &
                                      (df['close'] < df['open'].shift(1)).fillna(False)).astype(int)

            pattern_features.extend(['Doji', 'Hammer', 'Shooting_Star',
                                    'Bullish_Engulfing', 'Bearish_Engulfing'])

            # TA-Lib candlestick patterns (if available)
            if self.talib_available and hasattr(self.config.feature_selection, 'use_talib') and self.config.feature_selection.use_talib:
                try:
                    import talib
                    # 2-Day patterns
                    pattern_functions = {
                        'CDL_ENGULFING': talib.CDLENGULFING,
                        'CDL_HARAMI': talib.CDLHARAMI,
                        'CDL_DOJI': talib.CDLDOJI,
                        'CDL_HAMMER': talib.CDLHAMMER,
                        'CDL_SHOOTING_STAR': talib.CDLSHOOTINGSTAR
                    }

                    for pattern_name, pattern_func in pattern_functions.items():
                        df[pattern_name] = pattern_func(df['open'].values, df['high'].values,
                                                       df['low'].values, df['close'].values)
                        pattern_features.append(pattern_name)
                except Exception as e:
                    logging.warning(f"Error calculating TA-Lib candlestick patterns: {str(e)}")

            # Price gaps
            df['Gap_Up'] = ((df['open'] > df['close'].shift(1)).fillna(False) &
                           (df['low'] > df['close'].shift(1)).fillna(False)).astype(int)
            df['Gap_Down'] = ((df['open'] < df['close'].shift(1)).fillna(False) &
                             (df['high'] < df['close'].shift(1)).fillna(False)).astype(int)

            pattern_features.extend(['Gap_Up', 'Gap_Down'])

            # Support and resistance
            if hasattr(self.config.feature_selection, 'use_support_resistance') and self.config.feature_selection.use_support_resistance:
                df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
                df['Support1'] = 2 * df['Pivot'] - df['high']
                df['Resistance1'] = 2 * df['Pivot'] - df['low']

                df['Near_Support'] = ((df['close'] - df['Support1']).abs() < 0.01 * df['close']).fillna(False).astype(int)
                df['Near_Resistance'] = ((df['close'] - df['Resistance1']).abs() < 0.01 * df['close']).fillna(False).astype(int)

                pattern_features.extend(['Pivot', 'Support1', 'Resistance1',
                                        'Near_Support', 'Near_Resistance'])

            # Store created features for reference
            self.feature_groups['pattern'] = pattern_features

        except Exception as e:
            logging.error(f"Error adding price pattern indicators: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the dataframe.

        Args:
            df: DataFrame with datetime column

        Returns:
            DataFrame with added time-based features
        """
        logging.info("Adding temporal features")
        time_features = []

        try:
            # Ensure datetime column exists
            if 'datetime' not in df.columns:
                logging.warning("No datetime column found, skipping temporal features")
                return df

            # Basic datetime decomposition
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['week_of_year'] = df['datetime'].dt.isocalendar().week
            df['month'] = df['datetime'].dt.month
            df['quarter'] = df['datetime'].dt.quarter
            df['year'] = df['datetime'].dt.year
            df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)
            df['is_month_start'] = df['datetime'].dt.is_month_start.astype(int)
            df['is_quarter_end'] = df['datetime'].dt.is_quarter_end.astype(int)
            df['is_quarter_start'] = df['datetime'].dt.is_quarter_start.astype(int)
            df['is_year_end'] = df['datetime'].dt.is_year_end.astype(int)
            df['is_year_start'] = df['datetime'].dt.is_year_start.astype(int)

            time_features.extend(['day_of_week', 'day_of_month', 'week_of_year', 'month',
                                 'quarter', 'year', 'is_month_end', 'is_month_start',
                                 'is_quarter_end', 'is_quarter_start',
                                 'is_year_end', 'is_year_start'])

            # Day of week dummies
            if hasattr(self.config.feature_selection, 'use_day_dummies') and self.config.feature_selection.use_day_dummies:
                for i in range(5):  # Only trading days (0=Monday, 4=Friday)
                    df[f'is_day_{i}'] = (df['day_of_week'] == i).astype(int)
                    time_features.append(f'is_day_{i}')

            # Month dummies
            if hasattr(self.config.feature_selection, 'use_month_dummies') and self.config.feature_selection.use_month_dummies:
                for i in range(1, 13):
                    df[f'is_month_{i}'] = (df['month'] == i).astype(int)
                    time_features.append(f'is_month_{i}')

            # Seasonality - Fourier features
            if hasattr(self.config.feature_selection, 'use_seasonality') and self.config.feature_selection.use_seasonality:
                # Weekly (5-day) seasonality
                for period in [5]:  # Trading days in a week
                    for order in range(1, 3):  # 1st and 2nd order terms
                        df[f'sin_week_{order}'] = np.sin(2 * np.pi * order * df['day_of_week'] / period)
                        df[f'cos_week_{order}'] = np.cos(2 * np.pi * order * df['day_of_week'] / period)
                        time_features.extend([f'sin_week_{order}', f'cos_week_{order}'])

                # Monthly seasonality
                for order in range(1, 3):
                    df[f'sin_month_{order}'] = np.sin(2 * np.pi * order * df['day_of_month'] / 30)
                    df[f'cos_month_{order}'] = np.cos(2 * np.pi * order * df['day_of_month'] / 30)
                    time_features.extend([f'sin_month_{order}', f'cos_month_{order}'])

                # Yearly seasonality
                for order in range(1, 3):
                    df[f'sin_year_{order}'] = np.sin(2 * np.pi * order * (df['month'] - 1) / 12)
                    df[f'cos_year_{order}'] = np.cos(2 * np.pi * order * (df['month'] - 1) / 12)
                    time_features.extend([f'sin_year_{order}', f'cos_year_{order}'])

            # Calendar effects
            if hasattr(self.config.feature_selection, 'use_calendar_effects') and self.config.feature_selection.use_calendar_effects:
                # January effect
                df['january_effect'] = (df['month'] == 1).astype(int)

                # Turn of month effect (last trading day of month and first 3 of next month)
                df['turn_of_month'] = (df['is_month_end'] |
                                      (df['day_of_month'] <= 3)).astype(int)

                # Tax-loss harvesting (December)
                df['tax_loss_month'] = (df['month'] == 12).astype(int)

                time_features.extend(['january_effect', 'turn_of_month', 'tax_loss_month'])

            # Time since specific events (e.g., days since last high)
            if len(df) > 20:  # Only if we have enough data
                # Rolling window max/min for past N days
                for window in [10, 20, 50]:
                    # New high/low indicators
                    df[f'new_{window}d_high'] = df['close'].rolling(window=window).apply(
                        lambda x: x.iloc[-1] == max(x)).fillna(0).astype(int)
                    df[f'new_{window}d_low'] = df['close'].rolling(window=window).apply(
                        lambda x: x.iloc[-1] == min(x)).fillna(0).astype(int)

                    time_features.extend([f'new_{window}d_high', f'new_{window}d_low'])

                # Days since events
                # Initialize counters
                df['days_since_high'] = 0
                df['days_since_low'] = 0

                # Calculate days since high/low using iterative approach
                high_counter = 0
                low_counter = 0

                for i in range(1, len(df)):
                    if df.iloc[i]['close'] >= df.iloc[i-high_counter-1:i+1]['close'].max():
                        high_counter = 0
                    else:
                        high_counter += 1

                    if df.iloc[i]['close'] <= df.iloc[i-low_counter-1:i+1]['close'].min():
                        low_counter = 0
                    else:
                        low_counter += 1

                    df.iloc[i, df.columns.get_loc('days_since_high')] = high_counter
                    df.iloc[i, df.columns.get_loc('days_since_low')] = low_counter

                time_features.extend(['days_since_high', 'days_since_low'])

            # Store created features for reference
            self.feature_groups['temporal'] = time_features

        except Exception as e:
            logging.error(f"Error adding temporal features: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        return df

    def add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add financial-specific indicators and ratios to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added financial features
        """
        logging.info("Adding financial features")
        financial_features = []

        try:
            # Price Returns
            for period in [1, 5, 10, 21, 63, 126, 252]:  # various trading day periods
                df[f'return_{period}d'] = df['close'].pct_change(periods=period)
                financial_features.append(f'return_{period}d')

            # Log Returns
            df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
            financial_features.append('log_return_1d')

            # Realized Volatility
            for period in [10, 21, 63]:
                df[f'realized_vol_{period}d'] = df['log_return_1d'].rolling(window=period).std() * np.sqrt(252)
                financial_features.append(f'realized_vol_{period}d')

            # Price to moving average ratios
            for ma_period in [20, 50, 200]:
                ma_col = f'SMA_{ma_period}'
                if ma_col in df.columns:
                    df[f'price_to_MA_{ma_period}'] = df['close'] / df[ma_col]
                    financial_features.append(f'price_to_MA_{ma_period}')

            # Volume relative to price change (volume delta / price delta)
            df['volume_to_price_delta'] = df['volume'].pct_change() / (df['close'].pct_change().abs() + 0.001)
            financial_features.append('volume_to_price_delta')

            # High-Low Range relative to Close
            df['hl_range_to_close'] = (df['high'] - df['low']) / df['close']
            financial_features.append('hl_range_to_close')

            # Price Velocity and Acceleration
            df['price_velocity'] = df['close'].diff()
            df['price_acceleration'] = df['price_velocity'].diff()
            financial_features.extend(['price_velocity', 'price_acceleration'])

            # Log price velocity and its momentum
            df['log_price_velocity'] = df['log_return_1d']
            df['log_price_momentum'] = df['log_return_1d'].rolling(window=5).mean()
            financial_features.extend(['log_price_velocity', 'log_price_momentum'])

            # Store created features for reference
            self.feature_groups['financial'] = financial_features

        except Exception as e:
            logging.error(f"Error adding financial features: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        return df

    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime indicators to identify trending, mean-reverting,
        or high volatility environments.

        Args:
            df: DataFrame with OHLCV data and technical indicators

        Returns:
            DataFrame with added market regime features
        """
        logging.info("Adding market regime features")
        regime_features = []

        try:
            # Trend strength indicators
            adx_col = 'ADX_14'
            if adx_col in df.columns:
                # Trend strength levels
                df['weak_trend'] = ((df[adx_col] > 15) & (df[adx_col] <= 25)).astype(int)
                df['strong_trend'] = ((df[adx_col] > 25) & (df[adx_col] <= 50)).astype(int)
                df['extreme_trend'] = (df[adx_col] > 50).astype(int)

                regime_features.extend(['weak_trend', 'strong_trend', 'extreme_trend'])
            else:
                # Calculate ADX if not available
                try:
                    adx_data = self.calculate_adx(df, 14)
                    df['ADX_14'] = adx_data['ADX']
                    df['weak_trend'] = ((df['ADX_14'] > 15) & (df['ADX_14'] <= 25)).astype(int)
                    df['strong_trend'] = ((df['ADX_14'] > 25) & (df['ADX_14'] <= 50)).astype(int)
                    df['extreme_trend'] = (df['ADX_14'] > 50).astype(int)

                    regime_features.extend(['ADX_14', 'weak_trend', 'strong_trend', 'extreme_trend'])
                except Exception as e:
                    logging.warning(f"Could not calculate ADX: {str(e)}")

            # Volatility regimes based on Bollinger Band width
            if 'BB_Width_20' in df.columns:
                bb_width_mean = df['BB_Width_20'].rolling(window=100).mean()
                bb_width_std = df['BB_Width_20'].rolling(window=100).std()

                df['low_volatility'] = (df['BB_Width_20'] < (bb_width_mean - 0.5 * bb_width_std)).astype(int)
                df['high_volatility'] = (df['BB_Width_20'] > (bb_width_mean + 0.5 * bb_width_std)).astype(int)
                df['extreme_volatility'] = (df['BB_Width_20'] > (bb_width_mean + 2 * bb_width_std)).astype(int)

                # Expanding Bollinger Bands (volatility increasing)
                df['expanding_volatility'] = (df['BB_Width_20'].pct_change(5) > 0.05).astype(int)

                # Contracting Bollinger Bands (volatility decreasing)
                df['contracting_volatility'] = (df['BB_Width_20'].pct_change(5) < -0.05).astype(int)

                regime_features.extend(['low_volatility', 'high_volatility', 'extreme_volatility',
                                       'expanding_volatility', 'contracting_volatility'])

            # Price relative to moving averages
            ma_cols = ['SMA_20', 'SMA_50', 'SMA_200']
            if all(col in df.columns for col in ma_cols):
                # Bullish/bearish market regime
                df['bullish_trend'] = ((df['close'] > df['SMA_20']) &
                                      (df['SMA_20'] > df['SMA_50']) &
                                      (df['SMA_50'] > df['SMA_200'])).astype(int)

                df['bearish_trend'] = ((df['close'] < df['SMA_20']) &
                                      (df['SMA_20'] < df['SMA_50']) &
                                      (df['SMA_50'] < df['SMA_200'])).astype(int)

                # Mean reversion potential
                df['oversold'] = ((df['close'] < df['SMA_20']) &
                                 (df['RSI_14'] < 30 if 'RSI_14' in df.columns else True)).astype(int)

                df['overbought'] = ((df['close'] > df['SMA_20']) &
                                   (df['RSI_14'] > 70 if 'RSI_14' in df.columns else True)).astype(int)

                regime_features.extend(['bullish_trend', 'bearish_trend', 'oversold', 'overbought'])

            # Momentum regime
            if 'ROC_21' in df.columns:
                momentum_mean = df['ROC_21'].rolling(window=100).mean()
                momentum_std = df['ROC_21'].rolling(window=100).std()

                df['strong_momentum'] = (df['ROC_21'] > (momentum_mean + momentum_std)).astype(int)
                df['weak_momentum'] = (df['ROC_21'] < (momentum_mean - momentum_std)).astype(int)

                regime_features.extend(['strong_momentum', 'weak_momentum'])

            # Range-bound vs Trending detection
            if 'ADX_14' in df.columns and 'high' in df.columns and 'low' in df.columns:
                # Calculate normalized range over past 20 days
                df['norm_range_20d'] = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close']

                # Range-bound if ADX is low and normalized range is small
                df['range_bound'] = ((df['ADX_14'] < 20) &
                                    (df['norm_range_20d'] < df['norm_range_20d'].rolling(window=100).mean())).astype(int)

                regime_features.extend(['norm_range_20d', 'range_bound'])

            # Store created features for reference
            self.feature_groups['regime'] = regime_features

        except Exception as e:
            logging.error(f"Error adding market regime features: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        return df

    def add_derivative_features(self, df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
        """
        Add derivative features based on rate of change and differences of existing features.
        Uses pandas concat to avoid dataframe fragmentation.

        Args:
            df: DataFrame with technical indicators and other features
            exclude_columns: Columns to exclude from derivative calculations

        Returns:
            DataFrame with added derivative features
        """
        logging.info("Adding derivative features")
        derivative_features = []

        try:
            # Only process a subset of the most important features to avoid explosion
            candidate_columns = set(df.columns) - set(exclude_columns)

            # Filter to specific feature types that benefit from derivatives
            derivative_candidates = []
            for col in candidate_columns:
                # Include primary technical indicators but not derivatives
                if any(col.startswith(prefix) for prefix in [
                    'SMA_', 'EMA_', 'RSI_', 'MACD', 'ADX_', 'BB_', 'Stoch_K_', 'OBV'
                ]) and not any(col.startswith(prefix) for prefix in [
                    'slope_', 'pct_change_', 'diff_', 'cross_'
                ]):
                    derivative_candidates.append(col)

            # Limit to a reasonable number (e.g., top 10 indicators)
            derivative_candidates = derivative_candidates[:min(len(derivative_candidates), 10)]

            # Create a dictionary to collect all new features, then add them at once
            new_feature_dict = {}

            # First derivatives for all candidates
            for col in derivative_candidates:
                # Rate of change (percentage change)
                new_feature_dict[f'pct_change_{col}_1d'] = df[col].pct_change(periods=1)
                derivative_features.append(f'pct_change_{col}_1d')

                # First differences
                new_feature_dict[f'diff_{col}_1d'] = df[col].diff(periods=1)
                derivative_features.append(f'diff_{col}_1d')

                # Second derivative (acceleration)
                new_feature_dict[f'accel_{col}'] = df[col].diff(periods=1).diff(periods=1)
                derivative_features.append(f'accel_{col}')

            # Linear regression slopes
            slope_candidates = ['close'] + [c for c in derivative_candidates if 'SMA_' in c or 'EMA_' in c][:2]
            for col in slope_candidates:
                for period in [5, 10]:
                    # Calculate the slope for each feature and period
                    new_feature_dict[f'slope_{col}_{period}d'] = self.calculate_linear_regression_slope(df[col], period)
                    derivative_features.append(f'slope_{col}_{period}d')

            # Convert dictionary to DataFrame and concatenate
            if new_feature_dict:
                new_features_df = pd.DataFrame(new_feature_dict, index=df.index)

                # Using pd.concat is more efficient than adding columns one by one
                result_df = pd.concat([df, new_features_df], axis=1)
            else:
                result_df = df

            # Store created features for reference
            self.feature_groups['derivative'] = derivative_features

        except Exception as e:
            logging.error(f"Error adding derivative features: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Return original DataFrame if there's an error
            result_df = df

        return result_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with handled missing values
        """
        # Check for missing values
        na_count = df.isna().sum().sum()
        if na_count > 0:
            logging.info(f"Handling {na_count} missing values")

            # First, forward fill for time series continuity
            df = df.fillna(method='ffill')

            # If still have NaNs at the beginning, use backward fill
            df = df.fillna(method='bfill')

            # Any remaining NaNs (should be rare) replace with zeros or means
            for col in df.columns:
                if df[col].isna().any():
                    if col in ['datetime', 'open', 'high', 'low', 'close', 'volume']:
                        # These are essential, so we replace remaining NaNs with column mean
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        # For derived features, we can use 0
                        df[col] = df[col].fillna(0)

        return df

    def select_best_features(self, df: pd.DataFrame, target_col: str = 'close',
                           top_n: int = 20, method: str = 'importance') -> List[str]:
        """
        Select best features based on feature importance or mutual information.

        Args:
            df: DataFrame with features
            target_col: Target column name
            top_n: Number of top features to select
            method: Method for feature selection ('importance' or 'mutual_info')

        Returns:
            List of selected feature names
        """
        logging.info(f"Selecting top {top_n} features using {method} method")

        # Define columns to exclude from selection
        exclude_cols = ['datetime', target_col]

        # Prepare feature matrix
        X = df.drop(exclude_cols, axis=1).select_dtypes(include=[np.number])

        # Handle infinite or null values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # Drop any constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            logging.info(f"Dropping {len(constant_cols)} constant columns")
            X = X.drop(constant_cols, axis=1)

        # Define target - next day's return
        y = df[target_col].pct_change().shift(-1).fillna(0)

        # Select features based on method
        if method == 'importance':
            # Use Random Forest for feature importance
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_model.fit(X, y)

            importance = rf_model.feature_importances_
            indices = np.argsort(importance)[::-1]

            selected_features = list(X.columns[indices[:top_n]])

            # Store feature importance for later reference
            self.feature_importance_scores = {
                X.columns[i]: importance[i] for i in indices[:top_n]
            }

        elif method == 'mutual_info':
            # Use mutual information for feature selection
            mi_scores = mutual_info_regression(X, y, discrete_features=False, random_state=42)
            indices = np.argsort(mi_scores)[::-1]

            selected_features = list(X.columns[indices[:top_n]])

            # Store feature importance for later reference
            self.feature_importance_scores = {
                X.columns[i]: mi_scores[i] for i in indices[:top_n]
            }
        else:
            logging.warning(f"Unknown feature selection method: {method}. Using all features.")
            selected_features = list(X.columns)

        logging.info(f"Selected features: {selected_features[:5]}... (total: {len(selected_features)})")
        return selected_features

    # Technical indicator calculation methods
    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff().dropna()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        down = down.abs()

        avg_gain = up.rolling(window=window).mean()
        avg_loss = down.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_stochastic_k(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator %K."""
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()

        # Handle case where high_max equals low_min (avoid division by zero)
        denom = high_max - low_min
        denom = denom.replace(0, np.nan)

        k = 100 * ((df['close'] - low_min) / denom)
        return k

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range calculation
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']

        # Plus Directional Movement (+DM)
        plus_dm = high.diff()
        # Minus Directional Movement (-DM)
        minus_dm = low.diff(-1).abs()

        # Conditions for +DM and -DM
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

        # ATR calculation
        atr = self.calculate_atr(df, period)

        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Directional Index
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

        # Average Directional Index
        adx = dx.rolling(window=period).mean()

        return {
            'ADX': adx,
            'DI+': plus_di,
            'DI-': minus_di
        }

    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        # Handle division by zero
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)

        williams_r = -100 * ((highest_high - df['close']) / denom)
        return williams_r

    def calculate_cci(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = tp.rolling(window=period).mean()
        mean_deviation = (tp - tp_sma).abs().rolling(window=period).mean()

        # Avoid division by zero
        mean_deviation = mean_deviation.replace(0, np.nan)

        cci = (tp - tp_sma) / (0.015 * mean_deviation)
        return cci

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        close_diff = df['close'].diff()
        obv = pd.Series(index=df.index, dtype=float).fillna(0)

        # Iterative calculation of OBV
        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        vpt = pd.Series(index=df.index, dtype=float).fillna(0)

        # Iterative calculation of VPT
        for i in range(1, len(df)):
            price_change_pct = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            vpt.iloc[i] = vpt.iloc[i-1] + df['volume'].iloc[i] * price_change_pct

        return vpt

    def calculate_adl(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)  # Handle division by zero
        mfv = mfm * df['volume']

        adl = mfv.cumsum()
        return adl

    def calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)  # Handle division by zero
        mfv = mfm * df['volume']

        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf

    def calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # Positive and negative money flow
        positive_flow = pd.Series(index=df.index, dtype=float)
        negative_flow = pd.Series(index=df.index, dtype=float)

        # Determine positive and negative money flow
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
                negative_flow.iloc[i] = 0
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.iloc[i] = 0
                negative_flow.iloc[i] = money_flow.iloc[i]
            else:
                positive_flow.iloc[i] = 0
                negative_flow.iloc[i] = 0

        # Calculate money flow ratio
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        # Avoid division by zero
        mfr = positive_sum / negative_sum.replace(0, np.nan)

        # Money Flow Index
        mfi = 100 - (100 / (1 + mfr))
        return mfi

    def calculate_eom(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Ease of Movement."""
        distance_moved = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        box_ratio = (df['volume'] / 1000000) / (df['high'] - df['low'])

        # Handle division by zero or infinity
        box_ratio = box_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Single-period Ease of Movement
        eom_1period = distance_moved / box_ratio

        # Smoothed Ease of Movement
        eom = eom_1period.rolling(window=period).mean()
        return eom

    def calculate_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02,
                              af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR."""
        high = df['high']
        low = df['low']

        # Starting values
        sar = pd.Series(index=df.index)
        sar.iloc[0] = low.iloc[0]  # Start with low (uptrend assumed)

        trend = pd.Series(1, index=df.index)  # 1 for uptrend, -1 for downtrend
        extreme_point = high.iloc[0]  # Highest point in current trend
        af = af_start  # Acceleration factor

        # Calculate SAR values iteratively
        for i in range(1, len(df)):
            # Current SAR value
            sar.iloc[i] = sar.iloc[i-1] + af * (extreme_point - sar.iloc[i-1])

            # Check trend reversal
            if trend.iloc[i-1] == 1:  # Currently in uptrend
                # Ensure SAR is below lows
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

                # Check for trend reversal
                if low.iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1  # Switch to downtrend
                    sar.iloc[i] = extreme_point  # SAR = prior extreme point
                    extreme_point = low.iloc[i]  # New extreme point is low
                    af = af_start  # Reset AF
                else:
                    trend.iloc[i] = 1  # Continue uptrend
                    if high.iloc[i] > extreme_point:
                        extreme_point = high.iloc[i]  # New high
                        af = min(af + af_increment, af_max)  # Increase AF
            else:  # Currently in downtrend
                # Ensure SAR is above highs
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

                # Check for trend reversal
                if high.iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1  # Switch to uptrend
                    sar.iloc[i] = extreme_point  # SAR = prior extreme point
                    extreme_point = high.iloc[i]  # New extreme point is high
                    af = af_start  # Reset AF
                else:
                    trend.iloc[i] = -1  # Continue downtrend
                    if low.iloc[i] < extreme_point:
                        extreme_point = low.iloc[i]  # New low
                        af = min(af + af_increment, af_max)  # Increase AF

        return sar

    def calculate_linear_regression_slope(self, series: pd.Series, period: int = 10) -> pd.Series:
        """Calculate the slope of linear regression for a time series."""
        slopes = pd.Series(index=series.index)

        for i in range(period - 1, len(series)):
            x = np.arange(period)
            y = series.iloc[i - period + 1:i + 1].values

            # Handle NaN values
            if np.isnan(y).any():
                slopes.iloc[i] = np.nan
                continue

            # Simple linear regression
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes.iloc[i] = slope

        return slopes
