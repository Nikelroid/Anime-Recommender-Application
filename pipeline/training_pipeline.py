from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTrain

if __name__ =='__main__':


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
        