import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config_path, raw_dir, bucket_name=None, file_names=None, row_limit=None, threshold=None):
        self.config_path = config_path
        self.raw_dir = raw_dir

        self.config = read_yaml(config_path)

        self.bucket_name = bucket_name or self.config["data_ingestion"]["bucket_name"]
        self.file_names = file_names or self.config["data_ingestion"]["bucket_file_names"]
        self.row_limit = row_limit or self.config["data_ingestion"]["row_limit"]
        self.threshold = threshold or self.config["data_ingestion"]["threshold"]

        os.makedirs(self.raw_dir, exist_ok=True)
        
        logger.info(f"Data Ingestion started with {self.bucket_name} and files is {self.file_names}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            
            for file_name in self.file_names:
                file_path = os.path.join(self.raw_dir, file_name)

                if file_name == "animelist.csv":
                    filtered_file_path = os.path.join(self.raw_dir, file_name[:-4] + '_filtered.csv')

                    if os.path.exists(filtered_file_path):
                        try:
                            logger.info("Checking existing animelist_filtered.csv file...")
                            existing_data = pd.read_csv(filtered_file_path)

                            user_rating_counts = existing_data['user_id'].value_counts()
                            
                            min_user_ratings = user_rating_counts.min()
                            all_users_above_limit = min_user_ratings >= self.row_limit
                            has_user_below_threshold = (user_rating_counts < (self.row_limit + self.threshold)).any()
                            
                            logger.info(f"Min user ratings: {min_user_ratings}, Row limit: {self.row_limit}, Threshold: {self.threshold}")
                            logger.info(f"All users above limit: {all_users_above_limit}, Has user below threshold: {has_user_below_threshold}")
                            
                            if all_users_above_limit and has_user_below_threshold:
                                logger.info('Same animelist_filtered.csv file exists, dont need to filter again, go to next file')
                                continue  
                            else:
                                logger.info("Existing filtered file doesn't meet criteria, will reprocess...")
                                
                        except Exception as e:
                            logger.error("Error checking existing filtered file: " + str(e))
                    
                    def anime_list_process():
                        logger.info("Anime rating data filtering started")
                        data = pd.read_csv(file_path)
                        
                        data_cleaned = data.dropna(subset=['rating'])
                        data_cleaned = data_cleaned[data_cleaned['rating'] != 0]
                        n_rating = data_cleaned['user_id'].value_counts()
                        data_filtered = data_cleaned[data_cleaned['user_id'].isin(n_rating[n_rating >= self.row_limit].index)]
                        
                        logger.info(f"Original data: {len(data)} rows")
                        logger.info(f"After cleaning: {len(data_cleaned)} rows")
                        logger.info(f"After user filtering: {len(data_filtered)} rows")
                        logger.info(f"Users with >= {self.row_limit} ratings: {len(n_rating[n_rating >= self.row_limit])}")
                        
                        data_filtered.to_csv(filtered_file_path, index=False)
                        logger.info("Anime rating data filtered successfully")

                    required_download = True
                    
                    if os.path.exists(file_path):
                        try:
                            logger.info("Original animelist.csv found, processing...")
                            anime_list_process()
                            required_download = False
                        except Exception as e:
                            logger.error("Could not process existing animelist.csv: " + str(e))
                    else:
                        logger.info("Could not find animelist.csv, start downloading...")

                    if required_download:
                        blob = bucket.blob(file_name)
                        blob.download_to_filename(file_path)
                        anime_list_process()
                    
                    logger.info(f"CSV file animelist_filtered.csv successfully saved to {self.raw_dir} :)")

                else:
                    if os.path.exists(file_path):
                        logger.info(f"{file_name} found in {self.raw_dir} directory successfully")
                    else:
                        logger.info(f"Could not find {file_name}, start downloading...")
                        blob = bucket.blob(file_name)
                        blob.download_to_filename(file_path)
                        logger.info(f"CSV file {file_name} successfully saved to {self.raw_dir} :)")

        except Exception as e:
            logger.error("Error while downloading CSV file :(")
            raise CustomException("Failed to download csv file", e)
        
    def execute(self):
        logger.info('=' * 60)
        logger.info('DATA INGESTION PIPELINE STARTED')
        logger.info('=' * 60)
        try:
            logger.info("Started data ingestion data process")
            self.download_csv_from_gcp()
            logger.info("Data ingestion done successfully :)")
        
        except CustomException as ce:
            logger.error(f"Custom exception: {str(ce)}")
            raise ce

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise CustomException("Data ingestion failed", e)

        finally:
            logger.info("Data ingestion completed!")

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


