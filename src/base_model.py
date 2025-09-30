from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Activation, BatchNormalization, LayerNormalization, Input, 
    Embedding, Dense, Flatten, Dropout, Concatenate, Dot, Add, Multiply
)
import tensorflow as tf

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class BaseModel:
    def __init__(self, config_path):
        try:
            self.config = read_yaml(config_path)
            logger.info('Loaded configuration from config.yaml')
        except Exception as e:
            logger.error('Error in loading config.yaml file :' + str(e))
            raise CustomException('Error in loading config.yaml file', e)
        
    def RecommenderNet(self, n_users, n_animes):
        try:
            logger.info('Initializing enhanced model')

            embedding_size = self.config['model_training']['embedding_size']
            loss = self.config['model_training']['loss']
            learning_rate = self.config['model_training']['learning_rate']
            metrics = self.config['model_training']['metrics']

            # Inputs
            user = Input(name='user', shape=[1])
            anime = Input(name='anime', shape=[1])

            # User embeddings with bias
            user_embedding = Embedding(
                input_dim=n_users,
                output_dim=embedding_size,
                embeddings_regularizer=l2(1e-5),  # Slightly stronger regularization
                embeddings_initializer='glorot_uniform',
                name='user_embedding'
            )(user)
            user_bias = Embedding(
                input_dim=n_users,
                output_dim=1,
                embeddings_initializer='zeros',
                name='user_bias'
            )(user)

            # Anime embeddings with bias
            anime_embedding = Embedding(
                input_dim=n_animes,
                output_dim=embedding_size,
                embeddings_regularizer=l2(1e-5),
                embeddings_initializer='glorot_uniform',
                name='anime_embedding'
            )(anime)
            anime_bias = Embedding(
                input_dim=n_animes,
                output_dim=1,
                embeddings_initializer='zeros',
                name='anime_bias'
            )(anime)

            # Flatten embeddings
            user_vec = Flatten(name='user_flatten')(user_embedding)
            anime_vec = Flatten(name='anime_flatten')(anime_embedding)
            user_bias_vec = Flatten()(user_bias)
            anime_bias_vec = Flatten()(anime_bias)

            # Dot product for direct interaction (Matrix Factorization component)
            dot_product = Dot(axes=1, name='dot_product')([user_vec, anime_vec])
            
            # Element-wise multiplication for feature interaction
            interaction = Multiply(name='element_multiply')([user_vec, anime_vec])

            # Deep learning component with residual connections
            concat = Concatenate(name='concatenate')([user_vec, anime_vec, interaction])

            # First dense block with residual
            x1 = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), name='dense_1')(concat)
            x1 = LayerNormalization(name='layer_norm_1')(x1)
            x1 = Activation('relu', name='relu_1')(x1)
            x1 = Dropout(0.4, name='dropout_1')(x1)

            # Second dense block with residual
            x2 = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), name='dense_2')(x1)
            x2 = LayerNormalization(name='layer_norm_2')(x2)
            x2 = Activation('relu', name='relu_2')(x2)
            x2 = Dropout(0.3, name='dropout_2')(x2)

            # Third dense block
            x3 = Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), name='dense_3')(x2)
            x3 = LayerNormalization(name='layer_norm_3')(x3)
            x3 = Activation('relu', name='relu_3')(x3)
            x3 = Dropout(0.2, name='dropout_3')(x3)

            # Combine deep features with dot product
            x4 = Dense(32, kernel_initializer='he_normal', name='dense_4')(x3)
            x4 = Activation('relu', name='relu_4')(x4)
            
            # Final prediction combining all components
            dense_output = Dense(1, kernel_initializer='glorot_uniform', name='dense_output')(x4)
            
            # Combine: dot product + dense output + biases
            output = Add(name='final_add')([
                dot_product,
                dense_output,
                user_bias_vec,
                anime_bias_vec
            ])
            output = Activation('sigmoid', name='output')(output)

            model = Model(inputs=[user, anime], outputs=output, name='enhanced_recommender')
            
            # Compile with gradient clipping
            optimizer = Adam(
                learning_rate=learning_rate,
                clipnorm=1.0  # Gradient clipping
            )
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=metrics
            )
            
            logger.info(f'Enhanced model initialized with {model.count_params():,} parameters')
            return model
        
        except Exception as e:
            logger.error('Error in model initialization :' + str(e))
            raise CustomException('Error in model initialization', e)