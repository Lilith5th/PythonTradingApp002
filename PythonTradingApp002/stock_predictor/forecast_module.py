import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import time
from config.config import AppConfig, LearningConfig
from core.data_handler import DataHandler
from core.forecaster import Forecaster
from .plotter import Plotter
from advanced.rolling_window_forecaster import RollingWindowForecaster

LearningConfig = LearningConfig

def run_program(app_config, np=np, pd=pd, plt=plt, tf=tf, multiprocessing=multiprocessing):
    """Main function to run stock prediction with support for rolling window validation"""
    logging.info("Starting program execution")
    
    # Create data handler
    data_handler = DataHandler(app_config)

    if app_config.rolling_window.use_rolling_window:
        logging.info("Using rolling window validation")

        forecaster = RollingWindowForecaster(app_config, data_handler, np=np, tf=tf)

        start_time = time.time()
        results_array, diagnostics_list, feature_importance = forecaster.run_rolling_validation()
        learning_time = time.time() - start_time
        logging.info(f"Rolling window validation completed in {learning_time:.2f} seconds")

        # Get mean, std, and other metrics
        predictions_mean = forecaster.forecast_results.ensemble_mean
        predictions_std = forecaster.forecast_results.ensemble_std

        # Calculate sMAPE scores
        if diagnostics_list and 'window_metrics' in diagnostics_list[0]:
            smape_scores = np.array([m['smape'] for m in diagnostics_list[0]['window_metrics']])
            top_indices = np.argsort(smape_scores)[:min(5, len(smape_scores))]
        else:
            logging.error("No sMAPE scores available! Setting top_indices to empty list.")
            smape_scores = np.array([0.0])
            top_indices = np.array([])

        return forecaster.forecast_results.to_plot_format(data_handler.stock_data)

    elif (app_config.prediction_advanced.use_ensemble_methods or 
          app_config.prediction_advanced.enable_uncertainty_quantification or 
          app_config.prediction_advanced.enable_monte_carlo or 
          app_config.prediction_advanced.enable_rolling_window):

        logging.info("Using advanced prediction methods")

        from stock_predictor.advanced_forecaster import AdvancedForecaster
        forecaster = AdvancedForecaster(app_config,data_handler,np,tf)

        start_time = time.time()
        results = forecaster.run_forecast()
        learning_time = time.time() - start_time
        logging.info(f"Advanced forecast completed in {learning_time:.2f} seconds")

        return forecaster.forecast_results.to_plot_format(data_handler.stock_data)

    else:
        forecaster = Forecaster(app_config, data_handler)

        start_time = time.time()
        results_array, diagnostics_list, feature_importance = forecaster.run_simulations()
        learning_time = time.time() - start_time
        logging.info(f"Learning phase completed in {learning_time:.2f} seconds")

        start_time = time.time()
        predictions_mean, predictions_std, smape_scores, top_indices = forecaster.evaluate(results_array)
        prediction_time = time.time() - start_time
        logging.info(f"Prediction phase completed in {prediction_time:.2f} seconds")

        # Debugging check before returning top_indices
        if top_indices is None:
            logging.error("top_indices is None! Check `evaluate_predictions` for issues.")
            top_indices = []

        return forecaster.forecast_results.to_plot_format(data_handler.stock_data)


def plot_forecast_results(app_config, df_train_scaled, date_ori, close_scaler, predictions_mean, predictions_std, 
                         results_array, smape_scores, top_indices, df_train_raw, df_test_raw, np=np, plt=plt):
    start_time = time.time()
    data_handler = DataHandler(app_config)
    data_handler.df_train_scaled = df_train_scaled
    data_handler.date_ori = date_ori
    data_handler.close_scaler = close_scaler
    data_handler.df_train_raw = df_train_raw
    data_handler.df_test_raw = df_test_raw
    plotter = Plotter(app_config, plt=plt, np=np)
    fig = plotter.plot_forecast_results(data_handler, predictions_mean, predictions_std, results_array, smape_scores, top_indices)
    plot_time = time.time() - start_time
    logging.info(f"Forecast Results plot displayed in {plot_time:.2f} seconds")
    return fig

def plot_forecast_results_candlestick(app_config, df_train_scaled, date_ori, close_scaler, predictions_mean, predictions_std, 
                                     results_array, smape_scores, top_indices, df_train_raw, df_test_raw, np=np, plt=plt):
    start_time = time.time()
    data_handler = DataHandler(app_config)
    data_handler.df_train_scaled = df_train_scaled
    data_handler.date_ori = date_ori
    data_handler.close_scaler = close_scaler
    data_handler.df_train_raw = df_train_raw
    data_handler.df_test_raw = df_test_raw
    plotter = Plotter(app_config, plt=plt, np=np)
    fig = plotter.plot_forecast_results_candlestick(data_handler, predictions_mean, predictions_std, results_array, smape_scores, top_indices)
    plot_time = time.time() - start_time
    logging.info(f"Candlestick plot displayed in {plot_time:.2f} seconds")
    return fig

def plot_diagnostics(app_config, diagnostics, plt=plt, np=np):
    start_time = time.time()
    plotter = Plotter(app_config, plt=plt, np=np)
    fig = plotter.plot_diagnostics(diagnostics)
    plot_time = time.time() - start_time
    logging.info(f"Diagnostics plot displayed in {plot_time:.2f} seconds")
    return fig

def plot_feature_importance(app_config, feature_importance, plt=plt, np=np):
    start_time = time.time()
    plotter = Plotter(app_config, plt=plt, np=np)
    fig = plotter.plot_feature_importance(feature_importance)
    plot_time = time.time() - start_time
    logging.info(f"Feature Importance plot displayed in {plot_time:.2f} seconds")
    return fig

def build_feature_list(self):
    """
    Build the feature list for model input. If the "Use learning tab features" flag
    (stored in app_config.learning.use_features) is not checked, skip using extra features
    and use only a minimal set (for example, only 'close').
    """
    if self.data_with_features is None:
        raise ValueError("No feature engineered data. Call apply_feature_engineering() first.")
    
    # Check the flag directly.
    if not self.app_config.learning.use_features:
        # If false, use only minimal features from the CSV.
        self.feature_list = ['close']
        logging.info("Use learning tab features disabled: using minimal feature list: ['close']")
        return self

    # If the flag is true, use both basic and extra engineered features.
    df = self.data_with_features
    basic_features = ['open', 'close', 'high', 'low', 'volume_scaled', 'previous_close']
    extra_features = [
        col for col in df.columns
        if col not in ['datetime', 'open', 'close', 'high', 'low', 'volume', 'volume_scaled', 'previous_close']
    ]
    
    # Optionally, select top extra features using importance scores if available.
    if hasattr(self, 'feature_importance_scores') and self.feature_importance_scores:
        sorted_feats = sorted(self.feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
        selected_extra = [feat for feat, score in sorted_feats if feat in extra_features]
        max_feats = self.app_config.feature_selection.num_features_to_select if hasattr(self.app_config, 'feature_selection') else 20
        selected_extra = selected_extra[:min(len(selected_extra), max_feats)]
        logging.info(f"Selected {len(selected_extra)} extra features based on importance scores")
        final_features = basic_features + selected_extra
    else:
        final_features = basic_features + extra_features

    self.feature_list = final_features
    logging.info(f"Final feature list contains {len(final_features)} features: {final_features}")
    return self
