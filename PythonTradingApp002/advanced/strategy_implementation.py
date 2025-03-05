"""
Strategy implementation module for the stock prediction application.
Provides various trading strategies and ML optimization capabilities.
"""

import numpy as np
import pandas as pd
import logging
import pickle
from typing import Dict, List, Tuple, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class StrategyBacktester:
    """Backtester for trading strategies"""
    
    def __init__(self, strategy_config, data_handler=None):
        """
        Initialize the strategy backtester
        
        Args:
            strategy_config: Configuration for the trading strategy
            data_handler: DataHandler instance with historical data
        """
        self.config = strategy_config
        self.data_handler = data_handler
        self.ml_model = None
        self.scaler = None
        
        # Initialize performance metrics
        self.metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Trading results
        self.results = pd.DataFrame()
        
    def backtest(self, df=None):
        """
        Run a backtest of the selected strategy
        
        Args:
            df: Optional DataFrame with historical data (if not using data_handler)
            
        Returns:
            Dict: Performance metrics
            pd.DataFrame: Trading results
        """
        # Get data
        if df is None and self.data_handler is not None:
            df = self.data_handler.df_train_raw.copy()
        elif df is None:
            raise ValueError("No data provided for backtesting")
        
        # Create result DataFrame with date and price
        self.results = pd.DataFrame({
            'date': df['datetime'],
            'price': df['close']
        })
        
        # Apply the selected strategy
        strategy_func = self._get_strategy_function()
        signals = strategy_func(df)
        
        # Add signals to results
        self.results['signal'] = signals
        
        # Simulate trading
        self._simulate_trading()
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        return self.metrics, self.results
    
    def _get_strategy_function(self):
        """Get the function for the selected strategy"""
        strategy_map = {
            'buy_and_hold': self._buy_and_hold_strategy,
            'moving_average_crossover': self._ma_crossover_strategy,
            'rsi_based': self._rsi_strategy,
            'macd_based': self._macd_strategy,
            'bollinger_bands': self._bollinger_bands_strategy,
            'trend_following': self._trend_following_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'breakout': self._breakout_strategy,
            'ml_optimized': self._ml_strategy
        }
        
        strategy_func = strategy_map.get(self.config.strategy_type, self._buy_and_hold_strategy)
        logging.info(f"Using {self.config.strategy_type} strategy")
        return strategy_func
    
    def _buy_and_hold_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Simple buy and hold strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals (1=buy, 0=hold, -1=sell)
        """
        signals = np.zeros(len(df))
        signals[0] = 1  # Buy at the beginning
        signals[-1] = -1  # Sell at the end
        return signals
    
    def _ma_crossover_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Moving average crossover strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        # Default parameters if not in config
        short_period = getattr(self.config, 'short_ma_period', 20)
        long_period = getattr(self.config, 'long_ma_period', 50)
        
        # Calculate moving averages
        df['short_ma'] = df['close'].rolling(window=short_period).mean()
        df['long_ma'] = df['close'].rolling(window=long_period).mean()
        
        # Generate signals
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if pd.notna(df['short_ma'].iloc[i]) and pd.notna(df['long_ma'].iloc[i]):
                # Crossover (short MA crosses above long MA)
                if df['short_ma'].iloc[i] > df['long_ma'].iloc[i] and df['short_ma'].iloc[i-1] <= df['long_ma'].iloc[i-1]:
                    signals[i] = 1
                # Crossunder (short MA crosses below long MA)
                elif df['short_ma'].iloc[i] < df['long_ma'].iloc[i] and df['short_ma'].iloc[i-1] >= df['long_ma'].iloc[i-1]:
                    signals[i] = -1
        
        return signals

    def _rsi_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Relative Strength Index (RSI) strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        # Default parameters if not in config
        period = getattr(self.config, 'rsi_period', 14)
        overbought = getattr(self.config, 'rsi_overbought', 70)
        oversold = getattr(self.config, 'rsi_oversold', 30)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if pd.notna(df['rsi'].iloc[i]):
                # Oversold to normal (buy signal)
                if df['rsi'].iloc[i] > oversold and df['rsi'].iloc[i-1] <= oversold:
                    signals[i] = 1
                # Overbought to normal (sell signal)
                elif df['rsi'].iloc[i] < overbought and df['rsi'].iloc[i-1] >= overbought:
                    signals[i] = -1
        
        return signals
    
    def _macd_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Moving Average Convergence Divergence (MACD) strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        # Calculate MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Generate signals
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if pd.notna(df['macd'].iloc[i]) and pd.notna(df['signal_line'].iloc[i]):
                # MACD crosses above signal line (buy)
                if df['macd'].iloc[i] > df['signal_line'].iloc[i] and df['macd'].iloc[i-1] <= df['signal_line'].iloc[i-1]:
                    signals[i] = 1
                # MACD crosses below signal line (sell)
                elif df['macd'].iloc[i] < df['signal_line'].iloc[i] and df['macd'].iloc[i-1] >= df['signal_line'].iloc[i-1]:
                    signals[i] = -1
        
        return signals
    
    def _bollinger_bands_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Bollinger Bands strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        window = 20
        num_std = 2
        
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=window).mean()
        df['std'] = df['close'].rolling(window=window).std()
        df['upper_band'] = df['sma'] + (df['std'] * num_std)
        df['lower_band'] = df['sma'] - (df['std'] * num_std)
        
        # Generate signals
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if pd.notna(df['upper_band'].iloc[i]) and pd.notna(df['lower_band'].iloc[i]):
                # Price crosses below lower band (buy)
                if df['close'].iloc[i] < df['lower_band'].iloc[i] and df['close'].iloc[i-1] >= df['lower_band'].iloc[i-1]:
                    signals[i] = 1
                # Price crosses above upper band (sell)
                elif df['close'].iloc[i] > df['upper_band'].iloc[i] and df['close'].iloc[i-1] <= df['upper_band'].iloc[i-1]:
                    signals[i] = -1
        
        return signals
    
    def _trend_following_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Trend following strategy based on ADX
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        # Calculate ADX (Average Directional Index)
        period = 14
        
        # Calculate +DI and -DI
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = df['low'].diff()
        
        df['plus_dm'] = np.where(
            (df['high_diff'] > 0) & (df['high_diff'] > abs(df['low_diff'])),
            df['high_diff'],
            0
        )
        
        df['minus_dm'] = np.where(
            (df['low_diff'] < 0) & (abs(df['low_diff']) > df['high_diff']),
            abs(df['low_diff']),
            0
        )
        
        # Calculate True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate smoothed values
        df['smoothed_tr'] = df['true_range'].rolling(window=period).sum()
        df['smoothed_plus_dm'] = df['plus_dm'].rolling(window=period).sum()
        df['smoothed_minus_dm'] = df['minus_dm'].rolling(window=period).sum()
        
        # Calculate +DI and -DI
        df['plus_di'] = 100 * df['smoothed_plus_dm'] / df['smoothed_tr']
        df['minus_di'] = 100 * df['smoothed_minus_dm'] / df['smoothed_tr']
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Generate signals based on ADX and DI crossovers
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if pd.notna(df['adx'].iloc[i]) and df['adx'].iloc[i] > 25:  # Strong trend
                if pd.notna(df['plus_di'].iloc[i]) and pd.notna(df['minus_di'].iloc[i]):
                    # +DI crosses above -DI (buy)
                    if df['plus_di'].iloc[i] > df['minus_di'].iloc[i] and df['plus_di'].iloc[i-1] <= df['minus_di'].iloc[i-1]:
                        signals[i] = 1
                    # -DI crosses above +DI (sell)
                    elif df['minus_di'].iloc[i] > df['plus_di'].iloc[i] and df['minus_di'].iloc[i-1] <= df['plus_di'].iloc[i-1]:
                        signals[i] = -1
        
        return signals
    
    def _mean_reversion_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Mean reversion strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        window = 20
        
        # Calculate z-score
        df['sma'] = df['close'].rolling(window=window).mean()
        df['std'] = df['close'].rolling(window=window).std()
        df['z_score'] = (df['close'] - df['sma']) / df['std']
        
        # Generate signals
        signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if pd.notna(df['z_score'].iloc[i]):
                # Oversold (z-score < -2) to less oversold (buy)
                if df['z_score'].iloc[i] > -2 and df['z_score'].iloc[i-1] <= -2:
                    signals[i] = 1
                # Overbought (z-score > 2) to less overbought (sell)
                elif df['z_score'].iloc[i] < 2 and df['z_score'].iloc[i-1] >= 2:
                    signals[i] = -1
        
        return signals
    
    def _breakout_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Breakout strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        window = 20
        
        # Calculate support and resistance
        df['rolling_high'] = df['high'].rolling(window=window).max()
        df['rolling_low'] = df['low'].rolling(window=window).min()
        
        # Generate signals
        signals = np.zeros(len(df))
        for i in range(window, len(df)):
            # Breakout above resistance (buy)
            if df['close'].iloc[i] > df['rolling_high'].iloc[i-1]:
                signals[i] = 1
            # Breakdown below support (sell)
            elif df['close'].iloc[i] < df['rolling_low'].iloc[i-1]:
                signals[i] = -1
        
        return signals
    
    def _ml_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Machine learning optimized strategy
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Array of signals
        """
        if not self.config.enable_ml_optimization:
            logging.warning("ML optimization not enabled. Using buy and hold strategy instead.")
            return self._buy_and_hold_strategy(df)
        
        # If using a saved model, load it
        if self.config.use_saved_ml_model:
            try:
                with open(self.config.ml_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_model = model_data['model']
                    self.scaler = model_data.get('scaler')
                logging.info(f"Loaded ML model from {self.config.ml_model_path}")
            except Exception as e:
                logging.error(f"Failed to load ML model: {e}")
                return self._buy_and_hold_strategy(df)
        
        # If no model loaded, train a new one
        if self.ml_model is None:
            self._train_ml_model(df)
        
        # If still no model, fallback to buy and hold
        if self.ml_model is None:
            logging.warning("Failed to create ML model. Using buy and hold strategy instead.")
            return self._buy_and_hold_strategy(df)
        
        # Generate features for prediction
        X = self._generate_ml_features(df)
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        try:
            predictions = self.ml_model.predict(X)
            
            # Convert predictions to signals
            signals = np.zeros(len(df))
            signals[0] = 0  # No signal for the first day
            
            for i in range(1, len(df)):
                if i < len(predictions):
                    signals[i] = predictions[i]
            
            return signals
        except Exception as e:
            logging.error(f"ML prediction failed: {e}")
            return self._buy_and_hold_strategy(df)
    
    def _train_ml_model(self, df: pd.DataFrame) -> None:
        """
        Train a machine learning model for trading signals
        
        Args:
            df: DataFrame with historical data
        """
        try:
            # Generate features
            X = self._generate_ml_features(df)
            
            # Generate labels (next day return direction)
            y = np.zeros(len(df))
            returns = df['close'].pct_change()
            y[1:] = np.where(returns[1:] > 0, 1, -1)
            
            # Drop rows with NaN
            valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) == 0 or len(y) == 0:
                logging.warning("No valid data for ML training")
                return
            
            # Scale features
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model (Random Forest for demonstration)
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logging.info(f"ML model trained - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, " +
                         f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save model if training was successful and accuracy is reasonable
            if accuracy > 0.5:
                model_data = {'model': self.ml_model, 'scaler': self.scaler}
                with open(self.config.ml_model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logging.info(f"ML model saved to {self.config.ml_model_path}")
        
        except Exception as e:
            logging.error(f"Failed to train ML model: {e}")
            self.ml_model = None
            self.scaler = None
    
    def _generate_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate features for ML model
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            np.ndarray: Feature matrix
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Price momentum
        features_df['price_mom_1d'] = df['close'].pct_change(1)
        features_df['price_mom_5d'] = df['close'].pct_change(5)
        features_df['price_mom_10d'] = df['close'].pct_change(10)
        
        # Moving averages
        features_df['sma_10'] = df['close'].rolling(window=10).mean()
        features_df['sma_20'] = df['close'].rolling(window=20).mean()
        features_df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Moving average ratios
        features_df['ma_ratio_10_20'] = features_df['sma_10'] / features_df['sma_20']
        features_df['ma_ratio_20_50'] = features_df['sma_20'] / features_df['sma_50']
        
        # Volatility
        features_df['volatility_10d'] = df['close'].rolling(window=10).std() / df['close']
        features_df['volatility_20d'] = df['close'].rolling(window=20).std() / df['close']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        features_df['macd'] = ema12 - ema26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_hist'] = features_df['macd'] - features_df['macd_signal']
        
        # Volume metrics
        features_df['volume_change'] = df['volume'].pct_change()
        features_df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Drop rows with NaN
        features_df = features_df.fillna(0)
        
        return features_df.values
    
    def _simulate_trading(self) -> None:
        """Simulate trading based on signals"""
        # Initialize trading variables
        initial_capital = self.config.initial_capital
        position_size_pct = self.config.position_size_pct / 100.0
        take_profit_pct = self.config.take_profit_pct / 100.0
        stop_loss_pct = self.config.stop_loss_pct / 100.0
        max_positions = self.config.max_positions
        
        # Add columns to results
        self.results['position'] = 0
        self.results['capital'] = initial_capital
        self.results['equity'] = initial_capital
        self.results['cash'] = initial_capital
        self.results['shares'] = 0
        self.results['trade_price'] = np.nan
        self.results['trade_type'] = ''
        self.results['pnl'] = 0.0
        
        # Trading logic
        current_position = 0
        entry_price = 0
        open_trades = []
        
        for i in range(len(self.results)):
            # Update current price and signal
            current_price = self.results['price'].iloc[i]
            signal = self.results['signal'].iloc[i]
            
            # Check take profit or stop loss for existing position
            if current_position > 0 and entry_price > 0:
                # Take profit
                if current_price >= entry_price * (1 + take_profit_pct):
                    signal = -1  # Force sell
                    self.results.loc[i, 'trade_type'] = 'take_profit'
                # Stop loss
                elif current_price <= entry_price * (1 - stop_loss_pct):
                    signal = -1  # Force sell
                    self.results.loc[i, 'trade_type'] = 'stop_loss'
            
            # Process signals
            if signal == 1 and current_position < max_positions:  # Buy signal
                # Calculate position size
                position_capital = self.results['cash'].iloc[i-1] * position_size_pct
                shares_to_buy = position_capital / current_price
                
                # Update positions
                self.results.loc[i, 'shares'] = self.results['shares'].iloc[i-1] + shares_to_buy
                self.results.loc[i, 'cash'] = self.results['cash'].iloc[i-1] - (shares_to_buy * current_price)
                self.results.loc[i, 'position'] = current_position + 1
                self.results.loc[i, 'trade_price'] = current_price
                self.results.loc[i, 'trade_type'] = 'buy'
                
                # Update trading state
                current_position += 1
                entry_price = current_price
                open_trades.append({'entry_price': current_price, 'shares': shares_to_buy})
                
            elif signal == -1 and current_position > 0:  # Sell signal
                # Calculate position value
                shares_to_sell = self.results['shares'].iloc[i-1]
                position_value = shares_to_sell * current_price
                
                # Calculate P&L
                if len(open_trades) > 0:
                    trade = open_trades.pop(0)
                    trade_pnl = (current_price - trade['entry_price']) * trade['shares']
                    self.results.loc[i, 'pnl'] = trade_pnl
                
                # Update positions
                self.results.loc[i, 'shares'] = 0
                self.results.loc[i, 'cash'] = self.results['cash'].iloc[i-1] + position_value
                self.results.loc[i, 'position'] = 0
                self.results.loc[i, 'trade_price'] = current_price
                self.results.loc[i, 'trade_type'] = 'sell'
                
                # Update trading state
                current_position = 0
                entry_price = 0
                
            else:  # No change in position
                # Copy previous values
                self.results.loc[i, 'shares'] = self.results['shares'].iloc[i-1]
                self.results.loc[i, 'cash'] = self.results['cash'].iloc[i-1]
                self.results.loc[i, 'position'] = current_position
            
            # Update equity (cash + position value)
            position_value = self.results['shares'].iloc[i] * current_price
            self.results.loc[i, 'equity'] = self.results['cash'].iloc[i] + position_value
            
        # Close any remaining positions at the end
        if self.results['shares'].iloc[-1] > 0:
            final_price = self.results['price'].iloc[-1]
            final_shares = self.results['shares'].iloc[-1]
            final_cash = self.results['cash'].iloc[-1] + (final_shares * final_price)
            self.results.loc[len(self.results)-1, 'equity'] = final_cash
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics"""
        # Basic returns
        initial_equity = self.results['equity'].iloc[0]
        final_equity = self.results['equity'].iloc[-1]
        
        # Total return
        total_return = (final_equity / initial_equity) - 1
        self.metrics['total_return'] = total_return * 100.0  # Convert to percentage
        
        # Annualized return (assuming 252 trading days per year)
        days = len(self.results)
        years = days / 252
        if years > 0:
            self.metrics['annualized_return'] = (((1 + total_return) ** (1 / years)) - 1) * 100.0
        else:
            self.metrics['annualized_return'] = 0.0
        
        # Calculate daily returns
        self.results['daily_return'] = self.results['equity'].pct_change()
        
        # Sharpe Ratio (risk-free rate = 0 for simplicity)
        risk_free_rate = 0.0
        daily_excess_returns = self.results['daily_return'] - (risk_free_rate / 252)
        if len(daily_excess_returns.dropna()) > 0:
            sharpe_ratio = daily_excess_returns.mean() / daily_excess_returns.std() * np.sqrt(252)
            self.metrics['sharpe_ratio'] = sharpe_ratio
        else:
            self.metrics['sharpe_ratio'] = 0.0
        
        # Sortino Ratio (downside risk only)
        negative_returns = self.results['daily_return'][self.results['daily_return'] < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                sortino_ratio = (self.results['daily_return'].mean() * 252) / downside_deviation
                self.metrics['sortino_ratio'] = sortino_ratio
            else:
                self.metrics['sortino_ratio'] = 0.0
        else:
            self.metrics['sortino_ratio'] = 0.0
        
        # Maximum Drawdown
        self.results['cumulative_return'] = (1 + self.results['daily_return']).cumprod()
        self.results['cumulative_max'] = self.results['cumulative_return'].cummax()
        self.results['drawdown'] = (self.results['cumulative_return'] / self.results['cumulative_max']) - 1
        self.metrics['max_drawdown'] = abs(self.results['drawdown'].min()) * 100.0
        
        # Count trades
        buy_signals = self.results[self.results['trade_type'] == 'buy']
        sell_signals = self.results[self.results['trade_type'] == 'sell']
        trades = min(len(buy_signals), len(sell_signals))
        self.metrics['trades'] = trades
        
        # Win rate and profit factor
        winning_trades = len(self.results[self.results['pnl'] > 0])
        losing_trades = len(self.results[self.results['pnl'] < 0])
        self.metrics['winning_trades'] = winning_trades
        self.metrics['losing_trades'] = losing_trades
        
        if trades > 0:
            self.metrics['win_rate'] = (winning_trades / trades) * 100.0
        else:
            self.metrics['win_rate'] = 0.0
        
        # Profit factor
        gross_profits = self.results[self.results['pnl'] > 0]['pnl'].sum()
        gross_losses = abs(self.results[self.results['pnl'] < 0]['pnl'].sum())
        
        if gross_losses > 0:
            self.metrics['profit_factor'] = gross_profits / gross_losses
        else:
            self.metrics['profit_factor'] = 1.0 if gross_profits > 0 else 0.0
            
    def generate_performance_report(self) -> str:
        """
        Generate a text performance report
        
        Returns:
            str: Performance report text
        """
        report = []
        
        # Header
        report.append(f"Strategy: {self.config.strategy_type}")
        report.append(f"Initial Capital: ${self.config.initial_capital:.2f}")
        report.append(f"Final Equity: ${self.results['equity'].iloc[-1]:.2f}")
        report.append("")
        
        # Performance metrics
        report.append("Performance Metrics:")
        report.append(f"Total Return: {self.metrics['total_return']:.2f}%")
        report.append(f"Annualized Return: {self.metrics['annualized_return']:.2f}%")
        report.append(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2f}%")
        report.append("")
        
        # Trading statistics
        report.append("Trading Statistics:")
        report.append(f"Number of Trades: {self.metrics['trades']}")
        report.append(f"Winning Trades: {self.metrics['winning_trades']}")
        report.append(f"Losing Trades: {self.metrics['losing_trades']}")
        report.append(f"Win Rate: {self.metrics['win_rate']:.2f}%")
        report.append(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        
        return "\n".join(report)
        
    def plot_performance(self, plt):
        """
        Create performance visualization plots
        
        Args:
            plt: Matplotlib pyplot instance
            
        Returns:
            fig: Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 12))
        
        # Equity curve
        ax1 = plt.subplot(3, 1, 1)
        self.results['equity'].plot(ax=ax1, color='blue')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True)
        
        # Drawdown
        ax2 = plt.subplot(3, 1, 2)
        (self.results['drawdown'] * 100).plot(ax=ax2, color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # Trade PnL
        ax3 = plt.subplot(3, 1, 3)
        filtered_results = self.results[self.results['pnl'] != 0]
        colors = ['green' if pnl > 0 else 'red' for pnl in filtered_results['pnl']]
        
        if len(filtered_results) > 0:
            ax3.bar(range(len(filtered_results)), filtered_results['pnl'], color=colors)
            ax3.set_title('Trade P&L')
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Profit/Loss ($)')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, "No trades with P&L data available", 
                     horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        return fig
        
    def plot_interactive_performance(self):
        """
        Create interactive performance plots using Plotly
        
        Returns:
            dict: Dictionary of Plotly figure objects
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=3, 
                cols=1,
                subplot_titles=("Equity Curve", "Drawdown", "Trade P&L"),
                vertical_spacing=0.1
            )
            
            # Equity curve
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add buy and sell markers
# Add buy and sell markers
            buys = self.results[self.results['trade_type'] == 'buy']
            sells = self.results[self.results['trade_type'] == 'sell']
            
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys['equity'],
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells['equity'],
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            # Trade P&L
            filtered_results = self.results[self.results['pnl'] != 0]
            colors = ['green' if pnl > 0 else 'red' for pnl in filtered_results['pnl']]
            
            if len(filtered_results) > 0:
                fig.add_trace(
                    go.Bar(
                        x=filtered_results.index,
                        y=filtered_results['pnl'],
                        name='P&L',
                        marker_color=colors
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text=f"Strategy Performance: {self.config.strategy_type}",
                showlegend=True
            )
            
            # Price chart with buy/sell markers
            price_fig = go.Figure()
            
            # Plot price
            price_fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results['price'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add buy and sell markers
            price_fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys['price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                )
            )
            
            price_fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells['price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                )
            )
            
            price_fig.update_layout(
                height=500,
                title_text=f"Price Chart with Buy/Sell Signals: {self.config.strategy_type}",
                showlegend=True
            )
            
            return {'performance': fig, 'price': price_fig}
            
        except ImportError:
            logging.warning("Plotly not available. Install plotly for interactive charts.")
            return None