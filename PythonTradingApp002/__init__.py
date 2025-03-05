"""Stock Predictor - Machine Learning Stock Price Prediction Application"""

__version__ = '1.0.0'

# Expose key classes for easier imports
from stock_predictor.core.data_classes import StockData, ForecastResults
from stock_predictor.core.forecaster import Forecaster
from stock_predictor.core.advanced_forecaster import AdvancedForecaster
from stock_predictor.core.data_handler import DataHandler