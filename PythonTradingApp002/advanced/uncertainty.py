"""
Uncertainty Quantification Module for Stock Prediction

This module implements various uncertainty quantification methods for forecasting:
- Monte Carlo Dropout
- Bootstrap Sampling
- Quantile Regression
- Evidential Regression

These methods help quantify the uncertainty in predictions, providing confidence
intervals and probability distributions rather than single-point forecasts.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Dict, List, Tuple, Any, Union, Optional


class UncertaintyBase:
    """Base class for uncertainty quantification methods"""
    
    def __init__(self, model=None, config=None):
        """
        Initialize the uncertainty quantification method
        
        Args:
            model: Base TensorFlow model (optional)
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.confidence_level = getattr(config, 'confidence_level', 0.95) if config else 0.95
    
    def quantify_uncertainty(self, sequence_data, features):
        """
        Quantify uncertainty in predictions
        
        Args:
            sequence_data: Input data sequence
            features: List of feature names
            
        Returns:
            Dict: Uncertainty quantification results
        """
        raise NotImplementedError("Subclasses must implement quantify_uncertainty method")
    
    def calculate_confidence_intervals(self, predictions, confidence_level=None):
        """
        Calculate confidence intervals for predictions
        
        Args:
            predictions: Array of predictions
            confidence_level: Confidence level (0-1)
            
        Returns:
            Dict: Confidence intervals
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if len(predictions.shape) < 2:
            # Not enough samples to calculate confidence intervals
            return {
                'mean': predictions,
                'lower': predictions,
                'upper': predictions
            }
        
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Calculate z-score based on confidence level
        if confidence_level == 0.95:
            z_score = 1.96
        elif confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645
        else:
            # For other confidence levels, approximate using normal distribution
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return {
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'std': std
        }


class MCDropout(UncertaintyBase):
    """
    Monte Carlo Dropout for uncertainty quantification
    
    This method uses dropout at inference time to generate multiple predictions,
    treating the model as a Bayesian approximation.
    """
    
    def __init__(self, model=None, config=None):
        super().__init__(model, config)
        self.num_samples = getattr(config, 'mc_dropout_samples', 100) if config else 100
    
    def quantify_uncertainty(self, sequence_data, features):
        """
        Generate predictions with uncertainty using MC Dropout
        
        Args:
            sequence_data: Input sequence data
            features: List of feature names
            
        Returns:
            Dict: Prediction results with uncertainty
        """
        if self.model is None:
            raise ValueError("Model must be provided for MC Dropout")
        
        # Ensure we're using a proper dropout model
        if not any(isinstance(layer, tf.keras.layers.Dropout) for layer in self.model.layers):
            logging.warning("Model doesn't contain dropout layers. MC Dropout may not be effective.")
        
        # Generate multiple predictions with dropout enabled
        predictions = []
        input_sequence = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
        
        for _ in range(self.num_samples):
            pred = self.model(input_sequence, training=True).numpy().flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        confidence_intervals = self.calculate_confidence_intervals(predictions)
        
        return {
            'predictions': predictions,
            'mean': confidence_intervals['mean'],
            'lower': confidence_intervals['lower'],
            'upper': confidence_intervals['upper'],
            'std': confidence_intervals['std']
        }
    
    @classmethod
    def create_mc_dropout_model(cls, input_shape, num_features, config=None):
        """
        Create a model with dropout layers for uncertainty estimation
        
        Args:
            input_shape: Shape of input sequences
            num_features: Number of features
            config: Configuration object
            
        Returns:
            tf.keras.Model: Model with dropout layers
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        
        # Extract configuration parameters or use defaults
        if config:
            num_layers = getattr(config, 'num_layers', 2)
            dropout_rate = getattr(config, 'dropout_rate', 0.2)
            l2_reg = getattr(config, 'l2_reg', 0.01)
            learning_rate = getattr(config, 'learning_rate', 0.001)
        else:
            num_layers = 2
            dropout_rate = 0.2
            l2_reg = 0.01
            learning_rate = 0.001
        
        # Create model with dropout that will be active during inference
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = LSTM(
                units=min(64, max(32, num_features * 4)),
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg)
            )(x)
            # This dropout layer will be used during inference
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
        
        x = BatchNormalization()(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mae'
        )
        
        return model


class BootstrapUncertainty(UncertaintyBase):
    """
    Bootstrap sampling for uncertainty quantification
    
    This method trains multiple models on bootstrapped samples of the training data,
    then uses the ensemble of models to generate predictions with uncertainty.
    """
    
    def __init__(self, models=None, config=None):
        super().__init__(None, config)
        self.models = models or []
    
    def quantify_uncertainty(self, sequence_data, features):
        """
        Generate predictions with uncertainty using bootstrap models
        
        Args:
            sequence_data: Input sequence data
            features: List of feature names
            
        Returns:
            Dict: Prediction results with uncertainty
        """
        if not self.models:
            raise ValueError("Bootstrap models must be provided")
        
        # Generate predictions from all bootstrap models
        predictions = []
        input_sequence = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
        
        for model in self.models:
            pred = model.predict(input_sequence).flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        confidence_intervals = self.calculate_confidence_intervals(predictions)
        
        return {
            'predictions': predictions,
            'mean': confidence_intervals['mean'],
            'lower': confidence_intervals['lower'],
            'upper': confidence_intervals['upper'],
            'std': confidence_intervals['std']
        }
    
    @classmethod
    def create_bootstrap_models(cls, model_builder, train_data, num_models=5, sample_ratio=0.8, seed=42):
        """
        Create multiple models trained on bootstrapped samples
        
        Args:
            model_builder: Function to build a model
            train_data: Training data
            num_models: Number of bootstrap models to create
            sample_ratio: Ratio of data to sample for each bootstrap
            seed: Random seed
            
        Returns:
            List: List of trained bootstrap models
        """
        np.random.seed(seed)
        models = []
        
        num_samples = int(len(train_data) * sample_ratio)
        
        for i in range(num_models):
            # Create bootstrap sample
            indices = np.random.choice(len(train_data), size=num_samples, replace=True)
            bootstrap_sample = train_data[indices]
            
            # Build and train model
            model = model_builder()
            model.fit(bootstrap_sample, epochs=50, verbose=0)
            models.append(model)
            
            logging.info(f"Bootstrap model {i+1}/{num_models} trained")
        
        return models


class QuantileRegression(UncertaintyBase):
    """
    Quantile Regression for uncertainty quantification
    
    This method uses quantile regression to predict different quantiles
    of the target distribution, providing a direct estimate of uncertainty.
    """
    
    def __init__(self, model=None, config=None):
        super().__init__(model, config)
        
        # Default quantiles for 95% confidence interval
        self.quantiles = [0.025, 0.5, 0.975]
        
        if config:
            # Adjust quantiles based on confidence level
            confidence = getattr(config, 'confidence_level', 0.95)
            alpha = 1 - confidence
            self.quantiles = [alpha/2, 0.5, 1-alpha/2]
    
    def quantify_uncertainty(self, sequence_data, features):
        """
        Generate predictions with uncertainty using quantile regression
        
        Args:
            sequence_data: Input sequence data
            features: List of feature names
            
        Returns:
            Dict: Prediction results with uncertainty
        """
        if self.model is None:
            raise ValueError("Model must be provided for Quantile Regression")
        
        # Generate predictions for each quantile
        input_sequence = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
        predictions = self.model.predict(input_sequence)
        
        # Handle different model output formats
        if isinstance(predictions, list):
            lower = predictions[0].flatten()
            median = predictions[1].flatten()
            upper = predictions[2].flatten()
        else:
            lower = predictions[:, 0].flatten()
            median = predictions[:, 1].flatten()
            upper = predictions[:, 2].flatten()
        
        # Calculate standard deviation from the confidence interval
        std = (upper - lower) / (2 * 1.96)  # For 95% confidence
        
        return {
            'predictions': np.array([lower, median, upper]),
            'mean': median,
            'lower': lower,
            'upper': upper,
            'std': std
        }
    
    @classmethod
    def create_quantile_model(cls, input_shape, num_features, quantiles=None, config=None):
        """
        Create a quantile regression model
        
        Args:
            input_shape: Shape of input sequences
            num_features: Number of features
            quantiles: List of quantiles to predict
            config: Configuration object
            
        Returns:
            tf.keras.Model: Quantile regression model
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        
        # Default quantiles for 95% confidence interval
        if quantiles is None:
            quantiles = [0.025, 0.5, 0.975]
        
        # Extract configuration parameters or use defaults
        if config:
            num_layers = getattr(config, 'num_layers', 2)
            dropout_rate = getattr(config, 'dropout_rate', 0.2)
            l2_reg = getattr(config, 'l2_reg', 0.01)
            learning_rate = getattr(config, 'learning_rate', 0.001)
        else:
            num_layers = 2
            dropout_rate = 0.2
            l2_reg = 0.01
            learning_rate = 0.001
        
        # Define quantile loss function
        def quantile_loss(q):
            def loss(y_true, y_pred):
                error = y_true - y_pred
                return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            return loss
        
        # Create model
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = LSTM(
                units=min(64, max(32, num_features * 4)),
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
        
        # Output layers for each quantile
        outputs = [Dense(1, name=f'quantile_{i}')(x) for i in range(len(quantiles))]
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=[quantile_loss(q) for q in quantiles]
        )
        
        return model


class EvidentialRegression(UncertaintyBase):
    """
    Evidential Regression for uncertainty quantification
    
    This method uses a model that outputs parameters of a distribution,
    providing both aleatoric and epistemic uncertainty estimates.
    """
    
    def __init__(self, model=None, config=None):
        super().__init__(model, config)
    
    def quantify_uncertainty(self, sequence_data, features):
        """
        Generate predictions with uncertainty using evidential regression
        
        Args:
            sequence_data: Input sequence data
            features: List of feature names
            
        Returns:
            Dict: Prediction results with uncertainty
        """
        if self.model is None:
            raise ValueError("Model must be provided for Evidential Regression")
        
        # Generate predictions
        input_sequence = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
        predictions = self.model.predict(input_sequence)
        
        # Extract distribution parameters
        gamma, v, alpha, beta = predictions[0]
        
        # Calculate statistics
        mean = gamma
        
        # Aleatoric uncertainty (uncertainty in data)
        aleatoric = beta / (v * (alpha - 1))
        
        # Epistemic uncertainty (uncertainty in model)
        epistemic = beta / (v * (alpha - 1)**2)
        
        # Total uncertainty
        total_variance = beta / (v * (alpha - 1))
        std = np.sqrt(total_variance)
        
        # Calculate confidence intervals
        z_score = 1.96  # For 95% confidence
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return {
            'predictions': np.array([mean, mean, mean]),  # Placeholder
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'std': std,
            'aleatoric': aleatoric,
            'epistemic': epistemic
        }
    
    @classmethod
    def create_evidential_model(cls, input_shape, num_features, config=None):
        """
        Create an evidential regression model
        
        Args:
            input_shape: Shape of input sequences
            num_features: Number of features
            config: Configuration object
            
        Returns:
            tf.keras.Model: Evidential regression model
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        
        # Extract configuration parameters or use defaults
        if config:
            num_layers = getattr(config, 'num_layers', 2)
            dropout_rate = getattr(config, 'dropout_rate', 0.2)
            l2_reg = getattr(config, 'l2_reg', 0.01)
            learning_rate = getattr(config, 'learning_rate', 0.001)
        else:
            num_layers = 2
            dropout_rate = 0.2
            l2_reg = 0.01
            learning_rate = 0.001
        
        # Define evidential loss function
        def evidential_loss(y_true, y_pred):
            # Extract distribution parameters
            gamma = y_pred[:, 0:1]  # Mean
            v = y_pred[:, 1:2]      # Degrees of freedom
            alpha = y_pred[:, 2:3]  # Precision
            beta = y_pred[:, 3:4]   # Scale
            
            # NLL loss
            twoBlambda = 2 * beta * (1 + v)
            
            # Compute the negative log likelihood
            nll = 0.5 * tf.math.log(np.pi / v) \
                  - alpha * tf.math.log(twoBlambda) \
                  + (alpha + 0.5) * tf.math.log(v * (y_true - gamma)**2 + twoBlambda) \
                  + tf.math.lgamma(alpha) \
                  - tf.math.lgamma(alpha + 0.5)
            
            return tf.reduce_mean(nll)
        
        # Create model
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = LSTM(
                units=min(64, max(32, num_features * 4)),
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
        
        # Output layer for distribution parameters
        # gamma: mean, v: degrees of freedom, alpha: precision, beta: scale
        evidential_params = Dense(4, activation='softplus')(x)
        
        model = Model(inputs=inputs, outputs=evidential_params)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=evidential_loss
        )
        
        return model


class UncertaintyFactory:
    """Factory class for creating uncertainty quantification objects"""
    
    @staticmethod
    def create(method, model=None, config=None):
        """
        Create an uncertainty quantification object based on method
        
        Args:
            method: Uncertainty quantification method
            model: Model to use
            config: Configuration object
            
        Returns:
            UncertaintyBase: Uncertainty quantification object
        """
        if method == 'mc_dropout':
            return MCDropout(model, config)
        elif method == 'bootstrap':
            return BootstrapUncertainty(model, config)
        elif method == 'quantile':
            return QuantileRegression(model, config)
        elif method == 'evidential':
            return EvidentialRegression(model, config)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    @staticmethod
    def create_model(method, input_shape, num_features, config=None):
        """
        Create a model for uncertainty quantification
        
        Args:
            method: Uncertainty quantification method
            input_shape: Shape of input sequences
            num_features: Number of features
            config: Configuration object
            
        Returns:
            tf.keras.Model: Model for uncertainty quantification
        """
        if method == 'mc_dropout':
            return MCDropout.create_mc_dropout_model(input_shape, num_features, config)
        elif method == 'bootstrap':
            raise ValueError("Bootstrap method requires multiple models, use create_bootstrap_models")
        elif method == 'quantile':
            return QuantileRegression.create_quantile_model(input_shape, num_features, config=config)
        elif method == 'evidential':
            return EvidentialRegression.create_evidential_model(input_shape, num_features, config)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")