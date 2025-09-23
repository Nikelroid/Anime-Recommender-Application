import joblib
import comet_ml
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard,EarlyStopping
from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException
from src.base_model import BaseModel
from config.paths_config import *



logger = get_logger(__name__)


class ModelTrain:
    def __init__(self, data_path, config_path, processed_dir, train_arr_file, test_arr_file,
                 anime_encoder_file, user_encoder_file, checkpoint_dir, checkpoint_file_name,
                 weights_dir, weights_file_names):
        self.config = read_yaml(config_path)
        self.data_path = data_path
        self.config_path = config_path
        
        # Directory and file paths
        self.processed_dir = processed_dir
        self.train_arr_file = train_arr_file
        self.test_arr_file = test_arr_file
        self.anime_encoder_file = anime_encoder_file
        self.user_encoder_file = user_encoder_file
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file_name = checkpoint_file_name
        self.weights_dir = weights_dir
        self.weights_file_names = weights_file_names  # Expected to be a dict like {'anime': 'filename', 'user': 'filename'}
        
        # Data arrays
        self.x_train_array_user, self.x_train_array_anime = [], []
        self.x_test_array_user, self.x_test_array_anime = [], []
        self.y_train_array, self.y_test_array = [], []
        
        # Encoders and decoders
        self.anime2anime_encoder, self.user2user_encoder = {}, {}
        self.anime2anime_decoder, self.user2user_decoder = {}, {}
        
        # Model and dataset info
        self.model = None
        self.n_anime = 0
        self.n_user = 0
        
        # Initialize CometML experiment
        self.experiment = comet_ml.Experiment(
            api_key='zoBjAHHF2mypXJQfqyw2RHmd9',
            project_name='MLOps-2',
            workspace='nikelroid'
        )
        
        logger.info("CometML initialized")
        logger.info("ModelTrain class initialized")

    def load_data(self):
        try:
            train_array = joblib.load(os.path.join(self.processed_dir, self.train_arr_file))
            test_array = joblib.load(os.path.join(self.processed_dir, self.test_arr_file))
            self.anime2anime_encoder = joblib.load(os.path.join(self.processed_dir, self.anime_encoder_file))
            self.user2user_encoder = joblib.load(os.path.join(self.processed_dir, self.user_encoder_file))
            
            self.n_anime = len(self.anime2anime_encoder)
            self.n_user = len(self.user2user_encoder)

            self.x_train_array_user, self.x_train_array_anime, self.y_train_array = train_array
            self.x_test_array_user, self.x_test_array_anime, self.y_test_array = test_array

            logger.info('Data loaded and splitted successfully')
        except Exception as e:
            logger.error('Error in Loading and splitting X-y data: ' + str(e))
            raise CustomException('Error in Loading and splitting X-y data', e)

    def train_model(self, base_model):
        batch_size = self.config['model_training']['batch_size']
        patience = self.config['model_training']['patience']
        verbose = self.config['model_training']['verbose']
        force_training = self.config['model_training']['force_training']
        epochs = self.config['model_training']['epochs']
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = base_model.RecommenderNet(self.n_user, self.n_anime)

        checkpoint_filepath = os.path.join(self.checkpoint_dir, self.checkpoint_file_name)

        required_training = True

        if not force_training:
            if os.path.exists(checkpoint_filepath):
                try:
                    self.model.load_weights(checkpoint_filepath)
                    required_training = False
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error("Could not load checkpoint: " + str(e))
            else:
                logger.info("Could not find checkpoint")
        
        if required_training:
            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=verbose
            )
            early_stopping = EarlyStopping(
                patience=patience,
                monitor='val_loss',
                mode='min',
                restore_best_weights=True
            )

            history = self.model.fit(
                x=[self.x_train_array_user, self.x_train_array_anime],
                y=self.y_train_array,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([self.x_test_array_user, self.x_test_array_anime], self.y_test_array),
                callbacks=[model_checkpoint, early_stopping]
            )
            
            for epoch in range(len(history.history['loss'])):
                train_loss = {
                    'train_loss': history.history['loss'][epoch],
                    'train_mae_loss': history.history['mae'][epoch],
                    'train_mse_loss': history.history['mse'][epoch]
                }
                val_loss = {
                    'val_loss': history.history['val_loss'][epoch],
                    'val_mae_loss': history.history['val_mae'][epoch],
                    'val_mse_loss': history.history['val_mse'][epoch]
                }

                self.experiment.log_metrics(train_loss, step=epoch)
                self.experiment.log_metrics(val_loss, step=epoch)

        return self.model
    
    def extract_weights(self, name):
        weight_layer = self.model.get_layer(name)
        weights = weight_layer.get_weights()[0]
        weights = weights / (np.linalg.norm(weights, axis=1).reshape((-1, 1)) + 1e-8)
        return weights
    
    def save_weights(self):
        anime_weights = self.extract_weights('anime_embedding')
        user_weights = self.extract_weights('user_embedding')

        os.makedirs(self.weights_dir, exist_ok=True)
        anime_weights_filepath = os.path.join(self.weights_dir, self.weights_file_names['anime'])
        user_weights_filepath = os.path.join(self.weights_dir, self.weights_file_names['user'])

        joblib.dump(anime_weights, anime_weights_filepath)
        joblib.dump(user_weights, user_weights_filepath)

        self.experiment.log_asset(anime_weights_filepath)
        self.experiment.log_asset(user_weights_filepath)
        self.experiment.log_asset(os.path.join(self.checkpoint_dir, self.checkpoint_file_name))
    
    def execute(self):
        self.load_data()
        base_model = BaseModel(config_path=self.config_path)
        self.train_model(base_model)
        self.save_weights()


if __name__ == "__main__":
    model_train = ModelTrain(
        data_path=PROCESSED_DIR,
        config_path=CONFIG_PATH,
        processed_dir=PROCESSED_DIR,
        train_arr_file=TRAIN_ARR,
        test_arr_file=TEST_ARR,
        anime_encoder_file=ANIME_ENCODER_FILE,
        user_encoder_file=USER_ENCODER_FILE,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_file_name=CHECKPOINT_FILE_NAME,
        weights_dir=WEIGHTS_DIR,
        weights_file_names=WEIGHTS_FILE_NAME
    )
    model_train.execute()
            
            
