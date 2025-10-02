from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTrain

if __name__ =='__main__':
        
        data_processor = DataProcessor(
                rating_array_train_file=TRAIN_ARR,
                rating_array_test_file = TEST_ARR,
                input_rating_dir= ANIMELIST_CSV,
                input_anime_dir=ANIME_CSV,
                input_synopsis_dir=ANIME_SYN_CSV,
                output_dir=PROCESSED_DIR, 
                anime_file_name=ANIME_DF,
                user_encoder_file = USER_ENCODER_FILE,
                anime_encoder_file = ANIME_ENCODER_FILE,
                config_path=CONFIG_PATH)
        data_processor.execute()

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
        