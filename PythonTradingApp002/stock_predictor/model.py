import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# Set the mixed precision policy
set_global_policy('mixed_float16')

class BidirectionalForecastModel(tf.keras.Model):
    def __init__(self, num_layers, size_layer, output_size, dropout_rate, l2_reg):
        super().__init__()
        self.size_layer = size_layer
        self.num_layers = num_layers
        self.output_size = output_size

        self.bilstm_layers = [
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                units=size_layer,
                return_sequences=i < num_layers - 1,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                recurrent_dropout=0.0
            ))
            for i in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(output_size, activation=None, dtype=tf.float32,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    @tf.function
    def call(self, inputs, training=False):
        if len(inputs.shape) != 3:
            raise ValueError(f"Expected input shape (batch_size, time_steps, features), but got {inputs.shape}")

        x = inputs
        for layer in self.bilstm_layers:
            x = self.dropout(x, training=training)
            x = layer(x, training=training)

        output = self.dense(x)
        return output


import tensorflow as tf
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union, Callable


class BaseEnsembleModel:
    """Base class for ensemble models."""
    
    def __init__(self, models_config, features_dim):
        self.models = []
        self.models_config = models_config
        self.features_dim = features_dim
        self._build_models()
    
    def _build_models(self):
        """Build all models in the ensemble."""
        raise NotImplementedError("Subclasses must implement _build_models method")
    
    def fit(self, train_dataset, **kwargs):
        """Fit all models in the ensemble."""
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X, **kwargs):
        """Generate predictions from all models and combine them."""
        raise NotImplementedError("Subclasses must implement predict method")


class VotingEnsemble(BaseEnsembleModel):
    """Weighted average ensemble of multiple models."""
    
    def __init__(self, models_config, features_dim, weights=None):
        self.weights = weights
        super().__init__(models_config, features_dim)
    
    def _build_models(self):
        """Build all model variants specified in the configuration."""
        from .model import BidirectionalForecastModel
        
        # Extract model types from config
        model_types = self.models_config.get('ensemble_models', ['lstm', 'gru', 'bilstm'])
        logging.info(f"Building ensemble with model types: {model_types}")
        
        for model_type in model_types:
            if model_type == 'lstm':
                model = self._build_lstm_model()
            elif model_type == 'gru':
                model = self._build_gru_model()
            elif model_type == 'bilstm':
                model = self._build_bilstm_model()
            elif model_type == 'transformer':
                model = self._build_transformer_model()
            elif model_type == 'tcn':
                model = self._build_tcn_model()
            else:
                # Default to BidirectionalForecastModel
                model = BidirectionalForecastModel(
                    num_layers=self.models_config.get('num_layers', 2),
                    size_layer=self.models_config.get('size_layer', 64),
                    output_size=1,
                    dropout_rate=self.models_config.get('dropout_rate', 0.3),
                    l2_reg=self.models_config.get('l2_reg', 0.005)
                )
            
            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.models_config.get('learning_rate', 0.001)
                ),
                loss='mae'
            )
            
            self.models.append((model_type, model))
    
    def _build_lstm_model(self):
        """Build a standard LSTM model."""
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=self.models_config.get('size_layer', 64),
                input_shape=(None, self.features_dim),
                return_sequences=False,
                dropout=self.models_config.get('dropout_rate', 0.3),
                recurrent_dropout=0.0,
                kernel_regularizer=tf.keras.regularizers.l2(
                    self.models_config.get('l2_reg', 0.005)
                )
            ),
            tf.keras.layers.Dense(1)
        ])
    
    def _build_gru_model(self):
        """Build a GRU model."""
        return tf.keras.Sequential([
            tf.keras.layers.GRU(
                units=self.models_config.get('size_layer', 64),
                input_shape=(None, self.features_dim),
                return_sequences=False,
                dropout=self.models_config.get('dropout_rate', 0.3),
                recurrent_dropout=0.0,
                kernel_regularizer=tf.keras.regularizers.l2(
                    self.models_config.get('l2_reg', 0.005)
                )
            ),
            tf.keras.layers.Dense(1)
        ])
    
    def _build_bilstm_model(self):
        """Build a Bidirectional LSTM model."""
        from .model import BidirectionalForecastModel
        return BidirectionalForecastModel(
            num_layers=self.models_config.get('num_layers', 2),
            size_layer=self.models_config.get('size_layer', 64),
            output_size=1,
            dropout_rate=self.models_config.get('dropout_rate', 0.3),
            l2_reg=self.models_config.get('l2_reg', 0.005)
        )
    
    def _build_transformer_model(self):
        """Build a Transformer model for time series."""
        # Define transformer parameters
        head_size = 64
        num_heads = 4
        ff_dim = 128
        
        # Define a transformer block
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Attention and Normalization
            x = tf.keras.layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(inputs, inputs)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
            
            # Feed Forward Part
            ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
            ff = tf.keras.layers.Dropout(dropout)(ff)
            ff = tf.keras.layers.Dense(inputs.shape[-1])(ff)
            
            # Add and Normalize
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
            return x
        
        # Build the model
        inputs = tf.keras.layers.Input(shape=(None, self.features_dim))
        x = inputs
        
        # Optional positional encoding
        x = tf.keras.layers.Conv1D(
            filters=self.features_dim, kernel_size=1, activation='linear', padding='same'
        )(x)
        
        # Transformer blocks
        for _ in range(self.models_config.get('num_layers', 2)):
            x = transformer_encoder(
                x, 
                head_size=head_size, 
                num_heads=num_heads,
                ff_dim=ff_dim, 
                dropout=self.models_config.get('dropout_rate', 0.3)
            )
        
        # Final prediction layer
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs, outputs)
    
    def _build_tcn_model(self):
        """Build a Temporal Convolutional Network."""
        # TCN parameters
        filters = 64
        kernel_size = 3
        dilation_rates = [1, 2, 4, 8, 16]
        
        # Define a residual block for the TCN
        def residual_block(x, dilation_rate, filters, kernel_size, dropout_rate):
            # Skip connection
            skip = tf.keras.layers.Conv1D(filters, 1, padding='same')(x)
            
            # Dilated convolutions
            conv1 = tf.keras.layers.Conv1D(
                filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                activation='relu', kernel_regularizer=tf.keras.regularizers.l2(
                    self.models_config.get('l2_reg', 0.005)
                )
            )(x)
            conv1 = tf.keras.layers.Dropout(dropout_rate)(conv1)
            
            conv2 = tf.keras.layers.Conv1D(
                filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                activation='relu', kernel_regularizer=tf.keras.regularizers.l2(
                    self.models_config.get('l2_reg', 0.005)
                )
            )(conv1)
            conv2 = tf.keras.layers.Dropout(dropout_rate)(conv2)
            
            # Add skip connection
            return tf.keras.layers.add([skip, conv2])
        
        # Build the TCN model
        inputs = tf.keras.layers.Input(shape=(None, self.features_dim))
        x = inputs
        
        # Initial convolution to transform feature dimension
        x = tf.keras.layers.Conv1D(filters, 1, padding='same')(x)
        
        # Stack residual blocks with increasing dilation rate
        for dilation_rate in dilation_rates:
            x = residual_block(
                x, 
                dilation_rate, 
                filters, 
                kernel_size, 
                self.models_config.get('dropout_rate', 0.3)
            )
        
        # Output layer
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs, outputs)
    
    def fit(self, train_dataset, validation_data=None, epochs=50, **kwargs):
        """Fit all models in the ensemble."""
        histories = []
        
        for i, (model_type, model) in enumerate(self.models):
            logging.info(f"Training {model_type} model ({i+1}/{len(self.models)})")
            
            # Fit the model
            history = model.fit(
                train_dataset,
                validation_data=validation_data,
                epochs=epochs,
                **kwargs
            )
            
            histories.append((model_type, history))
            logging.info(f"Completed training {model_type} model")
        
        return histories
    
    def predict(self, X, **kwargs):
        """Generate predictions by averaging all models."""
        if not self.models:
            raise ValueError("No models in ensemble. Call fit() first.")
        
        # Get predictions from each model
        all_predictions = []
        
        for model_type, model in self.models:
            pred = model.predict(X, **kwargs)
            all_predictions.append(pred)
            logging.debug(f"{model_type} prediction shape: {pred.shape}")
        
        # Stack predictions for weighted average
        stacked_preds = np.stack(all_predictions, axis=0)
        
        # Apply weights if provided
        if self.weights is None:
            # Equal weighting if no weights provided
            self.weights = np.ones(len(self.models)) / len(self.models)
        elif len(self.weights) != len(self.models):
            logging.warning(f"Number of weights ({len(self.weights)}) does not match number of models ({len(self.models)}). Using equal weights.")
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        # Ensure weights sum to 1
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
        # Apply weights along the models dimension (axis 0)
        weighted_preds = np.sum(stacked_preds * self.weights[:, np.newaxis, np.newaxis], axis=0)
        
        return weighted_preds


# # Example of compiling the model with mixed precision
# model = BidirectionalForecastModel(num_layers=2, size_layer=64, output_size=1, dropout_rate=0.2, l2_reg=1e-4)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# model.compile(optimizer=optimizer, loss='mse')

# Example of using tf.data for efficient data loading
# Example of using tf.data for efficient data loading
def load_dataset(file_path, batch_size=32):
    def parse_csv_line(line):
        # Adjust record_defaults based on actual number of columns in your CSV
        num_columns = 10  # Change this to match your CSV's actual column count
        parsed_line = tf.io.decode_csv(line, record_defaults=[[0.0]] * num_columns)  

        # Assume last column is the label, and rest are features
        features = parsed_line[:-1]  
        label = parsed_line[-1]  
        
        return features, label  

    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# The following lines are commented out to prevent errors when importing this module
# dataset = load_dataset("data.csv")
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# model.fit(dataset, epochs=50, validation_split=0.2, callbacks=[early_stopping])
