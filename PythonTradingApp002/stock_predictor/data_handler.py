import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay
import logging
from .data_classes import StockData


class DataHandler:
    def __init__(self, app_config):
        self.app_config = app_config
        
        # Create and initialize StockData
        self.stock_data = StockData.from_app_config(app_config)
        self.stock_data.prepare_all(parse_dates=app_config.csv.parse_dates)
        
        # For backward compatibility - these will reference the same data
        self.df_train_raw = self.stock_data.csv_data_train_period
        self.df_test_raw = self.stock_data.csv_data_test_period
        self.df_train_scaled = self.stock_data.data_scaled
        self.features = self.stock_data.feature_list
        self.date_ori = self.stock_data.date_series_full
        self.close_scaler = self.stock_data.price_scaler
    
    # Keep existing methods for backward compatibility
    def load_and_split_data(self):
        """Legacy method for backward compatibility"""
        return self.stock_data.csv_data_train_period, self.stock_data.csv_data_test_period
    
    def prepare_data(self):
        """Legacy method for backward compatibility"""
        return (
            self.stock_data.csv_data_train_period,
            self.stock_data.data_scaled,
            self.stock_data.feature_list,
            self.stock_data.date_series_full,
            self.stock_data.price_scaler
        )
    
    def feature_engineering(self, df):
        """Legacy method for backward compatibility"""
        # This is just a wrapper that calls the new method on a specific DataFrame
        temp_stock_data = StockData(
            use_indicators=self.app_config.learning_pref.use_indicators,
            volume_scaling_method=self.app_config.learning_pref.volume_scaling_method
        )
        temp_stock_data.csv_data_train_period = df
        temp_stock_data.apply_feature_engineering()
        return temp_stock_data.data_with_features
    
    def calculate_rsi(self, series, window=14):
        return self.stock_data.calculate_rsi(series, window)

