"""
Model Factory for Stock Prediction Application

This module provides a standardized way to create different types of models
for stock prediction, ensuring consistency across the application.

Available model types:
- Standard LSTM
- Bidirectional LSTM
- GRU
- Transformer
- TCN (Temporal Convolutional Network)
- Uncertainty models (MC Dropout, Quantile Regression, Evidential)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization,
    Bidirectional, GRU, MultiHeadAttention, Conv1D, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import logging
from typing import Dict, List, Tuple, Any, Union, Optional


class ModelFactory:
    """Factory class for creating different types of models"""
    
    @staticmethod
    def create_model(model_type: str, config: Any, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Create a model based on the specified type and configuration
        
        Args:
            model_type: Type of model to create
            config: Configuration object with model parameters
            input_shape: Shape of input data (sequence_length, num_features)
            
        Returns:
            tf.keras.Model: The created model
            
        Raises:
            ValueError: If the model type is unknown
        """
        # Extract common parameters from config
        num_layers = getattr(config, 'num_layers', 2)
        size_layer = getattr(config, 'size_layer', 64)
        dropout_rate = getattr(config, 'dropout_rate', 0.2)
        l2_reg_val = getattr(config, 'l2_reg', 0.01)
        learning_rate = getattr(config, 'learning_rate', 0.001)
        
        # Create model based on type
        if model_type == 'lstm':
            model = ModelFactory._create_lstm_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        elif model_type == 'bilstm':
            model = ModelFactory._create_bidirectional_lstm_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        elif model_type == 'gru':
            model = ModelFactory._create_gru_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        elif model_type == 'transformer':
            model = ModelFactory._create_transformer_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        elif model_type == 'tcn':
            model = ModelFactory._create_tcn_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        elif model_type == 'mc_dropout':
            model = ModelFactory._create_mc_dropout_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        elif model_type == 'quantile':
            model = ModelFactory._create_quantile_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        elif model_type == 'evidential':
            model = ModelFactory._create_evidential_model(
                input_shape, num_layers, size_layer, dropout_rate, l2_reg_val
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Use appropriate loss function based on model type
        if model_type == 'quantile':
            # Define quantile loss functions
            quantiles = [0.025, 0.5, 0.975]
            
            def quantile_loss(q):
                def loss(y_true, y_pred):
                    error = y_true - y_pred
                    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
                return loss
            
            loss = {f'quantile_{i}': quantile_loss(q) for i, q in enumerate(quantiles)}
        elif model_type == 'evidential':
            # Define evidential loss function
            def evidential_loss(y_true, y_pred):
                gamma = y_pred[:, 0:1]  # Mean
                v = y_pred[:, 1:2]      # Degrees of freedom
                alpha = y_pred[:, 2:3]  # Precision
                beta = y_pred[:, 3:4]   # Scale
                
                twoBlambda = 2 * beta * (1 + v)
                
                nll = 0.5 * tf.math.log(tf.constant(3.14159) / v) \
                    - alpha * tf.math.log(twoBlambda) \
                    + (alpha + 0.5) * tf.math.log(v * (y_true - gamma)**2 + twoBlambda) \
                    + tf.math.lgamma(alpha) \
                    - tf.math.lgamma(alpha + 0.5)
                
                return tf.reduce_mean(nll)
            
            loss = evidential_loss
        else:
            # Default to MAE for all other models
            loss = 'mae'
        
        model.compile(optimizer=optimizer, loss=loss)
        
        return model
    
    @staticmethod
    def create_ensemble_models(model_types: List[str], config: Any, input_shape: Tuple[int, int],
                              weights: Optional[List[float]] = None) -> List[Tuple[str, tf.keras.Model]]:
        """
        Create an ensemble of models
        
        Args:
            model_types: List of model types to include in the ensemble
            config: Configuration object with model parameters
            input_shape: Shape of input data (sequence_length, num_features)
            weights: Optional list of weights for each model
            
        Returns:
            List[Tuple[str, tf.keras.Model]]: List of (model_type, model) tuples
        """
        if weights is not None and len(weights) != len(model_types):
            logging.warning(f"Number of weights ({len(weights)}) doesn't match number of models ({len(model_types)})")
            weights = None
        
        if weights is None:
            # Equal weights if not provided
            weights = [1.0 / len(model_types)] * len(model_types)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        ensemble = []
        for i, model_type in enumerate(model_types):
            try:
                model = ModelFactory.create_model(model_type, config, input_shape)
                ensemble.append((model_type, model))
                logging.info(f"Created {model_type} model with weight {weights[i]:.2f}")
            except Exception as e:
                logging.error(f"Failed to create {model_type} model: {e}")
        
        return ensemble
    
    @staticmethod
    def _create_lstm_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create a standard LSTM model"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = LSTM(
                units=size_layer,
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg_val)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    @staticmethod
    def _create_bidirectional_lstm_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create a bidirectional LSTM model"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = Bidirectional(LSTM(
                units=size_layer // 2,  # Half size for bidirectional to match parameter count
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg_val)
            ))(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    @staticmethod
    def _create_gru_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create a GRU model"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = GRU(
                units=size_layer,
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg_val)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    @staticmethod
    def _create_transformer_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create a transformer model for time series"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add positional encoding
        x = Dense(size_layer, activation='linear')(x)
        
        # Transformer blocks
        for _ in range(num_layers):
            # Self-attention
            attention_output = MultiHeadAttention(
                num_heads=4, key_dim=size_layer // 4
            )(x, x)
            attention_output = Dropout(dropout_rate)(attention_output)
            x = Add()([x, attention_output])
            x = LayerNormalization()(x)
            
            # Feed-forward
            ffn = Dense(size_layer * 4, activation='relu', kernel_regularizer=l2(l2_reg_val))(x)
            ffn = Dropout(dropout_rate)(ffn)
            ffn = Dense(size_layer, kernel_regularizer=l2(l2_reg_val))(ffn)
            ffn = Dropout(dropout_rate)(ffn)
            x = Add()([x, ffn])
            x = LayerNormalization()(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    @staticmethod
    def _create_tcn_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create a Temporal Convolutional Network model"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Initial conv
        x = Conv1D(filters=size_layer, kernel_size=1, padding='same')(x)
        
        # TCN blocks
        for i in range(num_layers):
            dilation_rate = 2 ** i
            
            # Dilated causal convolution
            conv = Conv1D(
                filters=size_layer,
                kernel_size=3,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                kernel_regularizer=l2(l2_reg_val)
            )(x)
            conv = Dropout(dropout_rate)(conv)
            
            # Second conv
            conv = Conv1D(
                filters=size_layer,
                kernel_size=3,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                kernel_regularizer=l2(l2_reg_val)
            )(conv)
            conv = Dropout(dropout_rate)(conv)
            
            # Skip connection
            if x.shape[-1] != size_layer:
                x = Conv1D(filters=size_layer, kernel_size=1, padding='same')(x)
            
            x = Add()([x, conv])
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    @staticmethod
    def _create_mc_dropout_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create a model with dropout layers for uncertainty estimation (MC Dropout)"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = LSTM(
                units=size_layer,
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg_val)
            )(x)
            # Use higher dropout rate for MC Dropout
            x = Dropout(max(dropout_rate, 0.3))(x)
            x = LayerNormalization()(x)
        
        x = BatchNormalization()(x)
        outputs = Dense(1, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    @staticmethod
    def _create_quantile_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create a quantile regression model for uncertainty estimation"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = LSTM(
                units=size_layer,
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg_val)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
        
        # Output layers for each quantile (0.025, 0.5, 0.975 for 95% CI)
        q1 = Dense(1, name='quantile_0')(x)
        q2 = Dense(1, name='quantile_1')(x)
        q3 = Dense(1, name='quantile_2')(x)
        
        return Model(inputs=inputs, outputs=[q1, q2, q3])
    
    @staticmethod
    def _create_evidential_model(input_shape, num_layers, size_layer, dropout_rate, l2_reg_val):
        """Create an evidential regression model for uncertainty estimation"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        for i in range(num_layers):
            x = LSTM(
                units=size_layer,
                return_sequences=(i < num_layers - 1),
                kernel_regularizer=l2(l2_reg_val)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
        
        # Output layer for evidential regression
        # gamma: mean, v: degrees of freedom, alpha: precision, beta: scale
        outputs = Dense(4, activation='softplus')(x)
        
        return Model(inputs=inputs, outputs=outputs)