"""
Dataset Factory for Stock Prediction Application

This module provides a standardized way to create datasets for training
and prediction, ensuring consistency across the application.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import traceback
from typing import Dict, List, Tuple, Any, Union, Optional


class DatasetFactory:
    """Factory class for creating TensorFlow datasets"""
    
    @staticmethod
    def create_dataset(data, timestamp, batch_size, target_idx=None, auto_batch_size=False, shuffle=True, prefetch=True):
        """
        Create a TensorFlow dataset for time series forecasting
        
        Args:
            data: Input data array (N samples, M features)
            timestamp: Sequence length for time series
            batch_size: Batch size (or base size if auto_batch_size is True)
            target_idx: Index of target column (default is 'close')
            auto_batch_size: Whether to automatically determine batch size
            shuffle: Whether to shuffle the dataset
            prefetch: Whether to prefetch data
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
            
        Raises:
            ValueError: If data is invalid or insufficient
        """
        # Validate input data
        if data is None:
            raise ValueError("Input data is None")
        
        # Convert to numpy array
        try:
            data = np.array(data, dtype=np.float32)
        except (ValueError, TypeError) as e:
            logging.warning(f"Data contains non-numeric values: {e}. Attempting to convert with NaN handling.")
            # Convert non-numeric values to NaN, then to 0
            data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        
        # Check for NaN or infinity
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning("Data contains NaN or infinite values. Replacing with 0.")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure 2D data
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data (samples, features), got {data.shape}")
        
        num_samples, num_features = data.shape
        logging.debug(f"Creating dataset with {num_samples} samples, {num_features} features")
        
        # Determine target index (default to 'close' column)
        if target_idx is None:
            # Use last column as target if no index specified
            target_idx = 0
        
        if target_idx >= num_features:
            raise ValueError(f"Target index {target_idx} out of range (0-{num_features-1})")
        
        # Ensure enough data for at least one sequence
        if num_samples <= timestamp:
            raise ValueError(f"Data length {num_samples} too short for sequence length {timestamp}")
        
        # Auto-adjust batch size if needed
        effective_batch_size = batch_size
        if auto_batch_size:
            num_sequences = num_samples - timestamp
            effective_batch_size = min(max(32, num_sequences // 10), 1024)
            logging.info(f"Auto batch size: {effective_batch_size}")
        
        # Create dataset
        try:
            # Create sequences from time series data
            dataset = tf.keras.utils.timeseries_dataset_from_array(
                data=data[:-1],  # Input data
                targets=data[timestamp:, target_idx],  # Target values (shifted by timestamp)
                sequence_length=timestamp,
                sampling_rate=1,
                batch_size=effective_batch_size,
                shuffle=shuffle
            )
            
            # Apply optimizations
            options = tf.data.Options()
            options.experimental_optimization.map_parallelization = True
            dataset = dataset.with_options(options)
            
            if prefetch:
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Failed to create dataset: {e}")
    
    @staticmethod
    def create_sequence_dataset(data, sequence_length, prediction_length, features=None, target_col='close'):
        """
        Create sequences for forecasting multiple steps ahead
        
        Args:
            data: DataFrame with time series data
            sequence_length: Input sequence length
            prediction_length: Number of steps to predict
            features: List of feature names (default: all columns)
            target_col: Target column name
            
        Returns:
            Tuple: X (sequences), y (targets)
        """
        if features is None:
            features = data.columns.tolist()
            if 'datetime' in features:
                features.remove('datetime')
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(data) - sequence_length - prediction_length + 1):
            # Input sequence
            X.append(data[features].iloc[i:i+sequence_length].values)
            
            # Target sequence (future values of target column)
            y.append(data[target_col].iloc[i+sequence_length:i+sequence_length+prediction_length].values)
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def split_train_val_test(dataset, val_ratio=0.1, test_ratio=0.1, shuffle=True, batch_size=None):
        """
        Split a dataset into training, validation, and test sets
        
        Args:
            dataset: TensorFlow dataset
            val_ratio: Validation ratio
            test_ratio: Test ratio
            shuffle: Whether to shuffle before splitting
            batch_size: Batch size for the split datasets
            
        Returns:
            Tuple: (train_dataset, val_dataset, test_dataset)
        """
        # Count total batches
        total_batches = sum(1 for _ in dataset)
        total_elements = sum(batch[0].shape[0] for batch in dataset)
        
        logging.debug(f"Splitting dataset with {total_batches} batches, {total_elements} elements")
        
        if shuffle:
            # Unbatch, shuffle, and rebatch
            unbatched = dataset.unbatch()
            shuffled = unbatched.shuffle(buffer_size=total_elements)
            
            if batch_size is None:
                # Get batch size from original dataset
                for batch in dataset:
                    batch_size = batch[0].shape[0]
                    break
            
            dataset = shuffled.batch(batch_size)
            total_batches = total_elements // batch_size
        
        # Calculate split points
        val_size = int(total_batches * val_ratio)
        test_size = int(total_batches * test_ratio)
        train_size = total_batches - val_size - test_size
        
        # Split dataset
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_prediction_input(data, sequence_length, features=None):
        """
        Create input for prediction
        
        Args:
            data: DataFrame or array with time series data
            sequence_length: Input sequence length
            features: List of feature names (if data is DataFrame)
            
        Returns:
            np.ndarray: Input sequence for prediction
        """
        # Convert DataFrame to array if needed
        if isinstance(data, pd.DataFrame):
            if features is None:
                features = data.columns.tolist()
                if 'datetime' in features:
                    features.remove('datetime')
            data = data[features].values
        
        # Ensure we have enough data
        if len(data) < sequence_length:
            raise ValueError(f"Not enough data for sequence length {sequence_length}")
        
        # Get the last sequence
        sequence = data[-sequence_length:].copy()
        
        # Reshape for model input
        sequence = sequence.reshape(1, sequence_length, -1)
        
        return sequence
    
    @staticmethod
    def create_sliding_window_datasets(data, window_size, step_size, min_train_size, sequence_length, batch_size, target_idx=None):
        """
        Create datasets for sliding window validation
        
        Args:
            data: Input data array
            window_size: Size of each window
            step_size: Step size between windows
            min_train_size: Minimum size of training data
            sequence_length: Sequence length for each window
            batch_size: Batch size for datasets
            target_idx: Index of target column
            
        Returns:
            List: List of (train_dataset, val_dataset) tuples for each window
        """
        if len(data) < min_train_size + window_size:
            raise ValueError(f"Not enough data for sliding window validation with min_train_size={min_train_size} and window_size={window_size}")
        
        datasets = []
        
        for start_idx in range(min_train_size, len(data) - window_size, step_size):
            end_idx = start_idx + window_size
            
            # Training data up to start_idx
            train_data = data[:start_idx]
            
            # Window data
            window_data = data[start_idx:end_idx]
            
            # Create datasets
            train_dataset = DatasetFactory.create_dataset(
                train_data, sequence_length, batch_size, target_idx, auto_batch_size=True
            )
            
            window_dataset = DatasetFactory.create_dataset(
                window_data, sequence_length, batch_size, target_idx, auto_batch_size=True
            )
            
            datasets.append((train_dataset, window_dataset))
        
        return datasets
    
    @staticmethod
    def create_bootstrap_datasets(data, num_datasets, sample_ratio=0.8, sequence_length=None, batch_size=None, seed=None):
        """
        Create bootstrap datasets by sampling with replacement
        
        Args:
            data: Input data array
            num_datasets: Number of bootstrap datasets to create
            sample_ratio: Ratio of data to sample for each bootstrap
            sequence_length: Sequence length for time series
            batch_size: Batch size for datasets
            seed: Random seed
            
        Returns:
            List: List of bootstrap datasets
        """
        if seed is not None:
            np.random.seed(seed)
        
        datasets = []
        num_samples = int(len(data) * sample_ratio)
        
        for i in range(num_datasets):
            # Sample with replacement
            indices = np.random.choice(len(data), size=num_samples, replace=True)
            bootstrap_data = data[indices]
            
            # Create dataset
            if sequence_length is not None and batch_size is not None:
                dataset = DatasetFactory.create_dataset(
                    bootstrap_data, sequence_length, batch_size, auto_batch_size=True
                )
                datasets.append(dataset)
            else:
                datasets.append(bootstrap_data)
        
        return datasets