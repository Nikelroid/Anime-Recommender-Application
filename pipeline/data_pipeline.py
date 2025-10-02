from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor

if __name__ == "__main__":

    
    data_ingestion = DataIngestion(
        config_path=CONFIG_PATH,
        raw_dir=RAW_DIR,
        bucket_name=None,  
        file_names=None,   
        row_limit=None,    
        threshold=None    
    )
    data_ingestion.execute()

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