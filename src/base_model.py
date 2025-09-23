
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation,BatchNormalization,Input,Embedding,Dense,Flatten,Dropout,Concatenate

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class BaseModel:
    def __init__(self,config_path):
        try:
            self.config = read_yaml(config_path)
            logger.info('Loaded configuration from config.yaml')
        except Exception as e:
            logger.error('Error in loading config.yaml file :' + str(e))
            raise CustomException('Error in loading config.yaml file ' , e )
        
    def RecommenderNet(self,n_users,n_animes):
        try:
            logger.info('Initializing model started')

            embedding_size = self.config['model_training']['embedding_size']
            loss = self.config['model_training']['loss']
            learning_rate = self.config['model_training']['learning_rate']
            metrics = self.config['model_training']['metrics']

            user = Input(name='user', shape=[1])
            anime = Input(name='anime', shape=[1])

            user_embedding = Embedding(input_dim=n_users,output_dim=embedding_size,embeddings_regularizer=l2(1e-6),name='user_embedding')(user)
            anime_embedding = Embedding(input_dim=n_animes,output_dim=embedding_size,embeddings_regularizer=l2(1e-6),name='anime_embedding')(anime)

            user_vec = Flatten()(user_embedding)
            anime_vec = Flatten()(anime_embedding)

            x = Concatenate(name='concatenate')([user_vec, anime_vec])

            x = Dense(128, kernel_initializer='he_normal', name='dense')(x)
            x = BatchNormalization(name='batch_normalization')(x)
            x = Activation('relu')(x)
            x = Dropout(0.3)(x)

            x = Dense(64, kernel_initializer='he_normal', name='dense_1')(x)
            x = BatchNormalization(name='batch_normalization_1')(x)
            x = Activation('relu')(x)
            x = Dropout(0.3)(x)

            x = Dense(1, activation='sigmoid', name='dense_2')(x)
            model = Model(inputs=[user, anime], outputs=x)
            model.compile(loss=loss,optimizer=Adam(learning_rate=learning_rate),metrics=metrics)
            
            logger.info('Model initialized successfully')
            return model
        
        except Exception as e:
            logger.error('Error in model initialization :' + str(e))
            raise CustomException('Error in model initialization ' , e )
        
