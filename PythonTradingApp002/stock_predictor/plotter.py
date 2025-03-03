import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

class Plotter:
    def __init__(self, app_config, plt=plt, np=np):
        self.app_config = app_config
        self.plt = plt
        self.np = np

    def plot_forecast_results(self, data_handler, predictions_mean, predictions_std, results_array, smape_scores, top_indices):
        # Create a new figure
        fig = self.plt.figure(figsize=self.app_config.gui.figure_size)

        # Plot training data
        train_dates = data_handler.date_ori[:len(data_handler.df_train_raw)]
        self.plt.plot(
            train_dates,
            data_handler.df_train_raw['close'],
            color=self.app_config.plot.train_color,
            linewidth=self.app_config.plot.line_width_train,
            label='True Trend (Train)'
        )

        # Plot backtesting line if enabled
        if self.app_config.learning_pref.enable_backtesting:
            self.plt.axvline(
                x=data_handler.date_ori[len(data_handler.df_train_raw)],
                color='red',
                linestyle='--',
                label='Backtesting Start'
            )

        # Plot actual future data if available
        if self.app_config.learning_pref.enable_backtesting and not data_handler.df_test_raw.empty:
            test_dates = data_handler.date_ori[len(data_handler.df_train_raw):len(data_handler.df_train_raw) + len(data_handler.df_test_raw)]
            self.plt.plot(
                test_dates,
                data_handler.df_test_raw['close'],
                color=self.app_config.plot.future_color,
                label='Actual Future'
            )

        # Plot forecast if available
        if predictions_mean is not None and len(predictions_mean) > 0:
            # Determine forecast start index
            if self.app_config.prediction.start_forecast_from_backtest:
                idx_bt = len(data_handler.df_train_raw)
            else:
                idx_bt = len(data_handler.df_train_raw) - self.app_config.prediction.predict_days
                idx_bt = max(0, idx_bt)  # Ensure idx_bt is not negative
        
            # Ensure we have enough dates for the forecast
            forecast_end_idx = min(idx_bt + len(predictions_mean), len(data_handler.date_ori))
            forecast_dates = data_handler.date_ori[idx_bt:forecast_end_idx]
        
            # Adjust predictions to match available dates
            pred_length = len(forecast_dates)
            if len(predictions_mean) > pred_length:
                logging.warning(f"Trimming predictions from {len(predictions_mean)} to {pred_length} to match available dates")
                predictions_mean = predictions_mean[:pred_length]
                if predictions_std is not None:
                    predictions_std = predictions_std[:pred_length]
        
            # Inverse transform predictions
            ensemble_forecast = data_handler.close_scaler.inverse_transform(predictions_mean.reshape(-1, 1)).flatten()
        
            self.plt.plot(
                forecast_dates,
                ensemble_forecast,
                color=self.app_config.plot.forecast_color,
                linewidth=2,
                label='Ensemble Forecast (Top 5)'
            )
        
            # Plot confidence intervals if enabled
            if self.app_config.plot.plot_confidence_interval and predictions_std is not None:
                lower_ci = data_handler.close_scaler.inverse_transform((predictions_mean - predictions_std).reshape(-1, 1)).flatten()
                upper_ci = data_handler.close_scaler.inverse_transform((predictions_mean + predictions_std).reshape(-1, 1)).flatten()
            
                self.plt.fill_between(
                    forecast_dates,
                    lower_ci[:len(forecast_dates)],  # Ensure matching lengths
                    upper_ci[:len(forecast_dates)],
                    color=self.app_config.plot.forecast_color,
                    alpha=0.1
                )
        
            # Plot top simulations if top_indices is available
            if top_indices is not None and len(top_indices) > 0:
                logging.info(f"Plotting {len(top_indices)} top simulations with indices: {top_indices}")
                colors = ['green', 'orange', 'purple', 'brown', 'pink']
                for j, sim_index in enumerate(top_indices):
                    if sim_index < len(results_array):
                        sim_forecast = data_handler.close_scaler.inverse_transform(results_array[sim_index].reshape(-1, 1)).flatten()
                        logging.debug(f"Plotting simulation {j+1} at index {sim_index}, sMAPE: {smape_scores[sim_index]:.2f}")
                        self.plt.plot(
                            forecast_dates,
                            sim_forecast,
                            color=colors[j % len(colors)],
                            linestyle='--',
                            label=f'Top Sim {j+1} (sMAPE: {smape_scores[sim_index]:.2f}%)',
                            alpha=0.7
                        )
                    else:
                        logging.warning(f"Invalid simulation index {sim_index} for results_array of length {len(results_array)}")
            else:
                logging.info("No top simulations to plot.")
    
        else:
            logging.warning("No ensemble forecast available to plot.")
    
        # Finalize plot settings
        self.plt.title(self.app_config.gui.title)
        self.plt.xlabel(self.app_config.gui.xlabel)
        self.plt.ylabel(self.app_config.gui.ylabel)
        self.plt.legend(loc=self.app_config.gui.legend_loc, bbox_to_anchor=self.app_config.gui.bbox_to_anchor)
        self.plt.grid(True)
        self.plt.tight_layout()
        return fig

    def plot_forecast_results_candlestick(self, data_handler, predictions_mean, predictions_std, results_array, smape_scores, top_indices):
        """
        Generate a candlestick plot with optional forecast overlay.
        
        Args:
            data_handler: Object containing training data and scaler information.
            predictions_mean: Array of mean forecast predictions (can be None).
            predictions_std: Array of standard deviation for predictions (can be None).
            results_array: Array of results (unused in this function).
            smape_scores: SMAPE scores (unused in this function).
            top_indices: Top indices (unused in this function).
        
        Returns:
            go.Figure: Plotly figure object containing the candlestick plot.
        """
        import plotly.graph_objects as go
        import pandas as pd
        import logging
        import traceback
        
        # Initialize the Plotly figure
        fig = go.Figure()
        
        # Prepare candlestick data from training set
        candle_data = data_handler.df_train_raw[['datetime', 'open', 'high', 'low', 'close']].copy()
        candle_data['datetime'] = pd.to_datetime(candle_data['datetime'])
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=candle_data['datetime'],
            open=candle_data['open'],
            high=candle_data['high'],
            low=candle_data['low'],
            close=candle_data['close'],
            name='Training Data'
        ))
        
        # Add forecast overlay if predictions_mean is available
        if predictions_mean is not None and len(predictions_mean) > 0:
            try:
                # Inverse transform predictions to original price scale
                ensemble_forecast = data_handler.close_scaler.inverse_transform(predictions_mean.reshape(-1, 1)).flatten()
                
                # Determine forecast dates based on configuration
                if self.app_config.prediction.start_forecast_from_backtest:
                    backtest_date = pd.to_datetime(self.app_config.learning_pref.backtesting_start_date)
                    idx_bt = data_handler.date_ori.searchsorted(backtest_date)
                    forecast_dates = data_handler.date_ori.iloc[idx_bt: idx_bt + len(ensemble_forecast)]
                else:
                    forecast_dates = data_handler.date_ori.iloc[-len(ensemble_forecast):]
                
                # Add forecast line trace
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=ensemble_forecast,
                    mode='lines',
                    name='Ensemble Forecast',
                    line=dict(color='red', width=2)
                ))
                
                # Add confidence intervals if enabled and predictions_std is provided
                if self.app_config.plot.plot_confidence_interval and predictions_std is not None:
                    lower_ci = data_handler.close_scaler.inverse_transform((predictions_mean - predictions_std).reshape(-1, 1)).flatten()
                    upper_ci = data_handler.close_scaler.inverse_transform((predictions_mean + predictions_std).reshape(-1, 1)).flatten()
                    
                    # Upper confidence interval trace
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=upper_ci,
                        mode='lines',
                        name='Upper CI',
                        line=dict(color='rgba(255,0,0,0.3)', width=1)
                    ))
                    
                    # Lower confidence interval trace with fill
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=lower_ci,
                        mode='lines',
                        name='Lower CI',
                        line=dict(color='rgba(255,0,0,0.3)', width=1),
                        fill='tonexty'  # Fills area between lower and upper CI
                    ))
            except Exception as e:
                logging.error(f"Error adding forecast to candlestick plot: {e}")
                logging.error(traceback.format_exc())
        else:
            logging.warning("No ensemble forecast available to plot in candlestick chart.")
        
        # Update plot layout
        fig.update_layout(
            title=self.app_config.gui.title + " (Candlestick)",
            xaxis_title=self.app_config.gui.xlabel,
            yaxis_title=self.app_config.gui.ylabel,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def plot_diagnostics(self, diagnostics):
        """
        Plot training and validation loss curves.
        
        Args:
            diagnostics: Dictionary containing training history with epoch_history list
                         Each epoch should have avg_train_loss and avg_val_loss
        
        Returns:
            matplotlib.figure.Figure: The figure object with plotted diagnostics
        """
        fig = self.plt.figure(figsize=self.app_config.gui.figure_size)
        train_loss = [epoch['avg_train_loss'] for epoch in diagnostics['epoch_history'] if epoch.get('avg_train_loss') is not None]
        val_loss = [epoch['avg_val_loss'] for epoch in diagnostics['epoch_history'] if epoch.get('avg_val_loss') is not None]

        if train_loss:
            self.plt.plot(train_loss, label='Training Loss')
        if val_loss:
            self.plt.plot(val_loss, label='Validation Loss')

        self.plt.title('Training and Validation Loss')
        self.plt.xlabel('Epoch')
        self.plt.ylabel('Loss')
        self.plt.legend()
        self.plt.tight_layout()
        return fig

    def plot_feature_importance(self, feature_importance):
        """
        Plot feature importance as a bar chart.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            
        Returns:
            matplotlib.figure.Figure: The figure object with plotted feature importance
        """
        fig = self.plt.figure(figsize=self.app_config.gui.figure_size)
        
        # Sort features by importance for better visualization
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # If there are too many features, show only top N for readability
        max_features_to_show = 20
        if len(sorted_features) > max_features_to_show:
            top_features = dict(list(sorted_features.items())[:max_features_to_show])
            self.plt.bar(top_features.keys(), top_features.values())
            self.plt.title(f'Feature Importance (Top {max_features_to_show})')
        else:
            self.plt.bar(sorted_features.keys(), sorted_features.values())
            self.plt.title('Feature Importance')
            
        self.plt.xlabel('Features')
        self.plt.ylabel('Importance')
        self.plt.xticks(rotation=45, ha='right')
        self.plt.tight_layout()
        return fig
        
    def _inverse_transform_predictions(self, data_handler, predictions):
        """Helper method to inverse transform predictions to original scale.
        
        Args:
            data_handler: Object containing the scaler
            predictions: Array of predictions in scaled format
            
        Returns:
            numpy.ndarray: Predictions in original scale
        """
        return data_handler.close_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()