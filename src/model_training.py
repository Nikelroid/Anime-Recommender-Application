import joblib
import comet_ml
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau)
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
        logger.info("Model Training initializing started ...")
        self.config = read_yaml(config_path)
        self.data_path = data_path
        self.config_path = config_path
        
        self.processed_dir = processed_dir
        self.train_arr_file = train_arr_file
        self.test_arr_file = test_arr_file
        self.anime_encoder_file = anime_encoder_file
        self.user_encoder_file = user_encoder_file
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file_name = checkpoint_file_name
        self.weights_dir = weights_dir
        self.weights_file_names = weights_file_names
        
        self.x_train_array_user, self.x_train_array_anime = [], []
        self.x_test_array_user, self.x_test_array_anime = [], []
        self.y_train_array, self.y_test_array = [], []
        
        self.anime2anime_encoder, self.user2user_encoder = {}, {}
        
        self.model = None
        self.n_anime = 0
        self.n_user = 0
        
        self.experiment = comet_ml.Experiment(
            api_key='zoBjAHHF2mypXJQfqyw2RHmd9',
            project_name='MLOps-2',
            workspace='nikelroid'
        )
        
        logger.info("CometML initialized")
        logger.info("ModelTrain class initialized")

        logger.info("Model Training Initialized successfully")

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

            logger.info(f'Data loaded: {len(self.y_train_array)} train samples, {len(self.y_test_array)} test samples')
            logger.info(f'Users: {self.n_user}, Animes: {self.n_anime}')
        except Exception as e:
            logger.error('Error in Loading data: ' + str(e))
            raise CustomException('Error in Loading data', e)

    def get_callbacks(self, checkpoint_filepath):
        try:
            patience = self.config['model_training']['patience']
            verbose = self.config['model_training']['verbose']
            
            callbacks = []
            
            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=verbose
            )
            callbacks.append(model_checkpoint)
            
            early_stopping = EarlyStopping(
                patience=patience,
                monitor='val_loss',
                mode='min',
                restore_best_weights=True,
                min_delta=1e-5,
                verbose=verbose
            )
            callbacks.append(early_stopping)
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, 
                patience=max(3, patience // 2), 
                min_lr=1e-7,
                mode='min',
                verbose=verbose
            )
            callbacks.append(reduce_lr)
            logger.info(f"Callbacks initialized successfully")
            return callbacks
        except Exception as e:
            logger.error('Error in Initialzing callbacks: ' + str(e))
            raise CustomException('Initialzing callbacks failed,', e)

    def train_model(self, base_model):
        try:
            logger.info("Training Model function started ...")
            batch_size = self.config['model_training']['batch_size']
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
                        logger.info("Model loaded successfully from checkpoint")
                        
                        val_loss = self.model.evaluate(
                            x=[self.x_test_array_user, self.x_test_array_anime],
                            y=self.y_test_array,
                            verbose=0
                        )
                        logger.info(f"Loaded model validation loss: {val_loss}")
                    except Exception as e:
                        logger.error("Could not load checkpoint: " + str(e))
                        raise Exception('Loading model failed ========> ', e)
                else:
                    logger.info("Checkpoint not found, training from scratch")
            
            if required_training:
                callbacks = self.get_callbacks(checkpoint_filepath)

                self.experiment.set_model_graph(self.model)
                
                logger.info("Starting model training...")
                history = self.model.fit(
                    x=[self.x_train_array_user, self.x_train_array_anime],
                    y=self.y_train_array,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([self.x_test_array_user, self.x_test_array_anime], self.y_test_array),
                    callbacks=callbacks,
                    verbose=verbose
                )
                
                for epoch in range(len(history.history['loss'])):
                    metrics_dict = {}
                    
                    for key in history.history.keys():
                        if not key.startswith('val_'):
                            metrics_dict[f'train_{key}'] = history.history[key][epoch]
                        else:
                            metrics_dict[key] = history.history[key][epoch]
                    
                    self.experiment.log_metrics(metrics_dict, step=epoch)
                
                final_metrics = {
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'best_val_loss': min(history.history['val_loss']),
                    'epochs_trained': len(history.history['loss'])
                }
                self.experiment.log_metrics(final_metrics)
                
                logger.info(f"Training completed. Best val_loss: {min(history.history['val_loss']):.4f}")

        except Exception as e:
            logger.error('Error in training model: ' + str(e))
            raise CustomException('Training model failed,', e)


        return self.model
    
    def extract_weights(self, name):
        try:
            logger.info(f"Extracting weights started ...")
            weight_layer = self.model.get_layer(name)
            weights = weight_layer.get_weights()[0]
            weights = weights / (np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8)
            logger.info(f"Weights extracted successfully")
            return weights

        except Exception as e:
            logger.error('Error in extracting weights: ' + str(e))
            raise CustomException('Faild to extract weights,', e)
    
    def save_weights(self):
        
        try:
            logger.info(f"Saving weights started ...")
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
            
            logger.info("Weights saved successfully")
            logger.info(f"Anime weights shape: {anime_weights.shape}")
            logger.info(f"User weights shape: {user_weights.shape}")
        except Exception as e:
            logger.error(f"Error saving weights: {str(e)}")
            raise CustomException("Error saving weights", e)
    
    def execute(self):
        logger.info('=' * 60)
        logger.info('MODEL TRAINING PIPELINE STARTED')
        logger.info('=' * 60)
        try:
            self.load_data()
            base_model = BaseModel(config_path=self.config_path)
            self.train_model(base_model)
            self.save_weights()
            self.experiment.end()
            logger.info('=' * 60)
            logger.info("TRAINING PIPELINE COMPELTED SUCCESSFULLY")
            logger.info('=' * 60)
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            self.experiment.end()
            raise


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