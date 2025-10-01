import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from config.paths_config import *

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, rating_array_train_file, rating_array_test_file, input_rating_dir, 
                 input_anime_dir, input_synopsis_dir, output_dir, anime_file_name,
                 user_encoder_file, anime_encoder_file, config_path):
        logger.info('DataProcessor initializing started')
        
        # Load config
        self.config = read_yaml(config_path)
        
        self.rating_array_train_file = rating_array_train_file
        self.rating_array_test_file = rating_array_test_file

        self.rating_dir = input_rating_dir
        self.anime_dir = input_anime_dir
        self.synopsis_dir = input_synopsis_dir
        self.output_dir = output_dir

        self.anime_file_name = anime_file_name

        self.anime_encoder_file = anime_encoder_file
        self.user_encoder_file = user_encoder_file

        self.rating_df = None 
        self.anime_df = None
        self.rating_array_train, self.rating_array_test = None, None
        self.user2user_encoder, self.anime2anime_encoder = {}, {}
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info('DataProcessor initialized successfully')

    def load_data(self, usecols):
        try:
            logger.info('Data loading started')
            self.rating_df = pd.read_csv(self.rating_dir, low_memory=True, usecols=usecols)
            initial_count = len(self.rating_df)
            logger.info(f'Initial data loaded: {initial_count:,} ratings')
            
            self.rating_df = self.rating_df.drop_duplicates(
                subset=['user_id', 'anime_id'], 
                keep='last'
            )
            logger.info(f'After removing duplicates: {len(self.rating_df):,} ratings')
            
            self.rating_df = self.rating_df[self.rating_df['rating'] > 0]
            logger.info(f'After removing invalid ratings: {len(self.rating_df):,} ratings')
            
            min_anime_ratings = self.config.get('data_processing', {}).get('min_anime_ratings', 0)
            
            if min_anime_ratings > 0:
                logger.info(f'Starting iterative filtering for anime with min_anime_ratings={min_anime_ratings}...')
                prev_count = 0
                iteration = 0
                
                while prev_count != len(self.rating_df) and iteration < 10:
                    prev_count = len(self.rating_df)
                    
                    anime_counts = self.rating_df['anime_id'].value_counts()
                    valid_anime = anime_counts[anime_counts >= min_anime_ratings].index
                    self.rating_df = self.rating_df[self.rating_df['anime_id'].isin(valid_anime)]
                    
                    iteration += 1
                    logger.info(f'Iteration {iteration}: {len(self.rating_df):,} ratings remaining')
                
                logger.info(f'Filtering completed after {iteration} iterations')
            
            n_users = self.rating_df['user_id'].nunique()
            n_anime = self.rating_df['anime_id'].nunique()
            sparsity = 100 * (1 - len(self.rating_df) / (n_users * n_anime))
            
            logger.info(f'Final dataset: {len(self.rating_df):,} ratings')
            logger.info(f'Users: {n_users:,} | Anime: {n_anime:,}')
            logger.info(f'Sparsity: {sparsity:.2f}%')
            logger.info(f'Avg ratings per user: {len(self.rating_df)/n_users:.1f}')
            logger.info(f'Avg ratings per anime: {len(self.rating_df)/n_anime:.1f}')

        except Exception as e:
            logger.error('Error in loading data: ' + str(e))
            raise CustomException('Error in loading data', e)
        
    def scale_data(self):
        try:
            logger.info('Data scaling started')
            logger.info(f'Original rating range: [{self.rating_df["rating"].min()}, {self.rating_df["rating"].max()}]')
            logger.info(f'Original rating mean: {self.rating_df["rating"].mean():.2f}')
            logger.info(f'Original rating std: {self.rating_df["rating"].std():.2f}')
            min_rating = self.rating_df['rating'].min()
            max_rating = self.rating_df['rating'].max()
            self.rating_df['rating'] = (
                (self.rating_df['rating'] - min_rating) / (max_rating - min_rating)
            ).astype('float32')
            
            logger.info(f'Scaled rating range: [{self.rating_df["rating"].min():.3f}, {self.rating_df["rating"].max():.3f}]')
            logger.info(f'Scaled rating mean: {self.rating_df["rating"].mean():.3f}')
            logger.info('Data scaled successfully')
            
        except Exception as e:
            logger.error('Error in scaling data: ' + str(e))
            raise CustomException('Error in scaling data', e)

    def create_encoder_decoder(self, key):
        try:
            logger.info(f'Encoder creation started for: {key}')
            ids = self.rating_df[key].unique().tolist()
            ids.sort()
            encoder = {x: i for i, x in enumerate(ids)}
            maps = self.rating_df[key].map(encoder)
            logger.info(f'Encoder created for {key}: {len(encoder)} unique values')
            return encoder, maps
        except Exception as e:
            logger.error(f'Error in creating encoder for {key}: ' + str(e))
            raise CustomException(f'Error in creating encoder for {key}', e)

    def encode_data(self):
        try:
            self.user2user_encoder, self.rating_df['user'] = self.create_encoder_decoder('user_id')
            self.anime2anime_encoder, self.rating_df['anime'] = self.create_encoder_decoder('anime_id')
            logger.info('Encoders created successfully')
            logger.info(f'User count: {len(self.user2user_encoder)} | Anime count: {len(self.anime2anime_encoder)}')
        except Exception as e:
            logger.error('Error in creating encoders: ' + str(e))
            raise CustomException('Error in creating encoders', e)
        
    def split_data(self, test_size=None, random_state=42, stratify_by_user=None):
        try:
            logger.info('Data splitting started')
            
            # Get values from config if not provided
            if test_size is None:
                test_size = self.config.get('data_processing', {}).get('test_size', 0.1)
            if stratify_by_user is None:
                stratify_by_user = self.config.get('data_processing', {}).get('stratify_by_user', True)
            if test_size >= 1:
                test_size = test_size / len(self.rating_df)
            if stratify_by_user:
                train_list, test_list = [], []
                for user_id in self.rating_df['user'].unique():
                    user_data = self.rating_df[self.rating_df['user'] == user_id]
                    if len(user_data) < 2:
                        train_list.append(user_data)
                    else:
                        user_train, user_test = train_test_split(
                            user_data, 
                            test_size=min(test_size, 0.5), 
                            random_state=random_state
                        )
                        train_list.append(user_train)
                        test_list.append(user_test)
                
                train_df = pd.concat(train_list, ignore_index=True)
                test_df = pd.concat(test_list, ignore_index=True)
                
                logger.info('Used stratified split by user')
            else:
                train_df, test_df = train_test_split(
                    self.rating_df,
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=True
                )
                logger.info('Used random split')
            self.rating_array_train = [
                train_df['user'].values,
                train_df['anime'].values,
                train_df['rating'].values
            ]
            self.rating_array_test = [
                test_df['user'].values,
                test_df['anime'].values,
                test_df['rating'].values
            ]
            logger.info(f'Train size: {len(train_df):,} ({len(train_df)/len(self.rating_df)*100:.1f}%)')
            logger.info(f'Test size: {len(test_df):,} ({len(test_df)/len(self.rating_df)*100:.1f}%)')
            logger.info('Data split successfully')

        except Exception as e:
            logger.error('Error in splitting data: ' + str(e))
            raise CustomException('Error in splitting data', e)
    
    def load_merge_anime_df(self, df, syn_df=None):
        df = df.replace("Unknown", np.nan).rename(columns={
            "MAL_ID": "anime_id",
            "Name": "anime_name",
            "Score": "score",
            "Genres": "genres"
        })
        if syn_df is not None:
            syn_df = syn_df.rename(columns={
                "MAL_ID": "anime_id",
                "sypnopsis": "syn"
            })
            df = df.merge(syn_df[["anime_id", "syn"]], on="anime_id", how="left")
        return df

    def process_anime_data(self):
        try:
            logger.info('Anime data loading and processing started')
            anime_df = pd.read_csv(self.anime_dir)
            syn_df = pd.read_csv(self.synopsis_dir)
            merged_df = self.load_merge_anime_df(anime_df, syn_df)

            mask = (merged_df["English name"] != "Unknown") & (merged_df["English name"].notna())
            merged_df.loc[mask, "anime_name"] = (
                merged_df.loc[mask, "anime_name"].astype(str) + " - " + merged_df.loc[mask, "English name"].astype(str)
            )

            self.anime_df = merged_df[['anime_id', 'anime_name', 'score', 'genres', 'syn']]

            valid_anime_ids = [k for k, v in self.anime2anime_encoder.items()]
            self.anime_df = self.anime_df[self.anime_df['anime_id'].isin(valid_anime_ids)]
            logger.info(f'Filtered anime data: {len(self.anime_df)} anime with ratings')
            self.anime_df.sort_values(by=['score'], inplace=True, ascending=False, na_position='last')
            self.anime_df['genres'] = self.anime_df['genres'].fillna('')
            self.anime_df['genre_list'] = self.anime_df['genres'].str.split(', ').apply(list)
            self.anime_df['syn'] = self.anime_df['syn'].fillna('No synopsis available.')
                        
            logger.info('Anime data processed successfully')

        except Exception as e:
            logger.error('Error in processing anime data: ' + str(e))
            raise CustomException('Error in processing anime data', e)
            
    def save_artifacts(self):
        try:
            logger.info(f'Starting Saving artifacts to {self.output_dir}')
            
            artifacts = { 
                self.user_encoder_file: self.user2user_encoder,
                self.anime_encoder_file: self.anime2anime_encoder
            }
            
            for name, data in artifacts.items():
                filepath = os.path.join(self.output_dir, name)
                joblib.dump(data, filepath)
                logger.info(f'{name} saved ({len(data)} items)')

            joblib.dump(
                self.rating_array_train, 
                os.path.join(self.output_dir, self.rating_array_train_file)
            )
            joblib.dump(
                self.rating_array_test, 
                os.path.join(self.output_dir, self.rating_array_test_file)
            )
            logger.info('Train and test arrays saved')

            self.anime_df.to_csv(
                os.path.join(self.output_dir, self.anime_file_name), 
                index=False
            )
            logger.info('Anime metadata saved')
            logger.info(f'All artifacts saved successfully to {self.output_dir}')

        except Exception as e:
            logger.error('Error in saving data: ' + str(e))
            raise CustomException('Error in saving data', e)
        
    def execute(self):
        try:
            logger.info('=' * 60)
            logger.info('DATA PROCESSING PIPELINE STARTED')
            logger.info('=' * 60)
            
            self.load_data(usecols=['user_id', 'anime_id', 'rating'])
            self.scale_data()
            self.encode_data()
            self.split_data()
            self.process_anime_data()
            self.save_artifacts()
            
            logger.info('=' * 60)
            logger.info('DATA PROCESSING COMPLETED SUCCESSFULLY')
            logger.info(f'Output directory: {self.output_dir}')
            logger.info('=' * 60)

        except Exception as e:
            logger.error('Error in processing pipeline: ' + str(e))
            raise CustomException('Error in processing pipeline', e)
        

if __name__ == "__main__":
    data_processor = DataProcessor(
        rating_array_train_file=TRAIN_ARR,
        rating_array_test_file=TEST_ARR,
        input_rating_dir=ANIMELIST_CSV,
        input_anime_dir=ANIME_CSV,
        input_synopsis_dir=ANIME_SYN_CSV,
        output_dir=PROCESSED_DIR, 
        anime_file_name=ANIME_DF,
        user_encoder_file=USER_ENCODER_FILE,
        anime_encoder_file=ANIME_ENCODER_FILE,
        config_path=CONFIG_PATH
    )
    
    data_processor.execute()