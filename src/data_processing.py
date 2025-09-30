import os
import pandas as pd
import numpy as np
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys

logger = get_logger(__name__)

class DataProcrssor:
    def __init__(self,rating_array_train_file,rating_array_test_file,input_rating_dir,input_anime_dir,
                 input_synopsis_dir,output_dir,anime_file_name,user_encoder_file,anime_encoder_file):
        logger.info('DataProcessor initializing started')
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

        self.rating_array_train , self.rating_array_test = None , None

        self.user2user_encoder,self.anime2anime_encoder  = {},{}

        os.makedirs(self.output_dir,exist_ok=True)
        logger.info('DataProcessor initialized succesfulle')

    def load_data(self,usecols):
        try:
            logger.info ('Data loading started')
            self.rating_df = pd.read_csv(self.rating_dir,low_memory=True,usecols=usecols)
            logger.info ('Data loaded successfully')

        except Exception as e:
            logger.error ('Error in loading data: ' + str(e))
            raise CustomException('Error in loading data',e)
        
    def scale_data(self):
        try:
            logger.info ('Data scaling started')
            min_rating = np.min(self.rating_df['rating'])-1
            max_rating = np.max(self.rating_df['rating'])
            self.rating_df['rating']= self.rating_df['rating'].apply(lambda x:(x-min_rating)/(max_rating-min_rating)).values.astype('float64')
            logger.info ('Data scaled successfully')
        except Exception as e:
            logger.error ('Error in scaling data : ' + str(e))
            raise CustomException('Error in scaling data',e)

    def create_encoder_decoder(self,key):
        try:
            logger.info (f'encoder-encoder creation started for : {key} ')
            ids = user_id = self.rating_df[key].unique().tolist()
            encoder = {x: i for i,x in enumerate(user_id)}
            maps = self.rating_df[key].map(encoder)
            logger.info (f'encoder-encoder created for : {key} successfully! ')
            return encoder,maps
        except Exception as e:
            logger.error (f'Error in creating encoder-encoder for : {key}  : ' + str(e))
            raise CustomException(f'Error in creating encoder-encoder for : {key} | ',e)


    def encode_data(self):
        try:
            self.user2user_encoder,self.rating_df['user'] = self.create_encoder_decoder('user_id')
            self.anime2anime_encoder,self.rating_df['anime'] = self.create_encoder_decoder('anime_id')
            logger.info ('Encoders and Decoders created successfully')
            logger.info (f'User count -> {len(self.user2user_encoder)}   |   Anime count -> {len(self.anime2anime_encoder)} ')
        except Exception as e:
            logger.error ('Error in creating encoder-encoder dictionaries: ' + str(e))
            raise CustomException('Error in creating encoder-encoder dictionaries',e)
        
    def split_data(self,test_size = 100000,random_state = 42):
        try:
            logger.info ('Data splitting started')
            self.rating_df = self.rating_df.sample(frac=1,random_state=43).reset_index(drop=True)
            rating_array = self.rating_df[['user','anime','rating']].values
            train_indices = self.rating_df.shape[0] - test_size

            rating_array_train , rating_array_test = (rating_array[:train_indices],rating_array[train_indices :])
            self.rating_array_train = [rating_array_train[:,0],rating_array_train[:,1],rating_array_train[:,2]]
            self.rating_array_test = [rating_array_test[:,0],rating_array_test[:,1],rating_array_test[:,2]]
            logger.info ('Data splitted to test-train, then converted tp list successfully')

        except Exception as e:
            logger.error ('Error in splitting data to test-train : ' + str(e))
            raise CustomException('Error in splitting data to test-train ',e)
    
    def load_merge_anime_df(self,df,syn_df=None):
        df = df.replace("Unknown", np.nan).rename(columns={"MAL_ID": "anime_id","Name": "anime_name","Score": "score","Genres": "genres"})
        if syn_df is not None:syn_df = syn_df.rename(columns={"MAL_ID": "anime_id","sypnopsis": "syn"})
        return df.merge(syn_df[["anime_id", "syn"]], on="anime_id", how="right")

    def prcess_anime_data(self):
        try:
            logger.info ('Anime data loading and merging with synopsis started')
            anime_df = pd.read_csv(self.anime_dir)
            syn_df = pd.read_csv(self.synopsis_dir)
            self.anime_df = self.load_merge_anime_df(anime_df,syn_df)[['anime_id','anime_name','score','genres','syn']]
            self.anime_df.sort_values(by=['score'],inplace=True,ascending=False,na_position='last')
            self.anime_df['genres'] = self.anime_df['genres'].fillna('')
            self.anime_df['genre_list'] = self.anime_df['genres'].str.split(', ').apply(list)
            logger.info ('Anime data loaded and merged with synopsis successfully ')

        except Exception as e:
            logger.error ('Error in loading or/and mergin anime_list data : ' + str(e))
            raise CustomException('Error in loading or/and mergin anime_list data  ',e)
            
    def save_artifacts(self):
        try:
            logger.info (f'Data is saving in {self.output_dir} directory started')
            artifacts = { 
                self.user_encoder_file : self.user2user_encoder,
                self.anime_encoder_file : self.anime2anime_encoder
            }
            for name,data in artifacts.items():
                joblib.dump(data,os.path.join(self.output_dir,f'{name}'))
                logger.info (f'{name} saved successfully')

            joblib.dump(self.rating_array_train,os.path.join(self.output_dir,self.rating_array_train_file))
            joblib.dump(self.rating_array_test,os.path.join(self.output_dir,self.rating_array_test_file))

            self.anime_df.to_csv(os.path.join(self.output_dir,self.anime_file_name),index=False)

            logger.info (f'All data is saved in {self.output_dir} directory')

        except Exception as e:
            logger.error ('Error in saving data : ' + str(e))
            raise CustomException('Error in saving data ',e)
        
    def execute(self):
        try:
            self.load_data(usecols=['user_id','anime_id','rating'])
            self.scale_data()
            self.encode_data()
            self.split_data(test_size=100000)
            self.prcess_anime_data()
            self.save_artifacts()
            logger.info (f'Data process executed successfully. outputs are in {self.output_dir} directory')

        except Exception as e:
            logger.error ('Error in processing data : ' + str(e))
            raise CustomException('Error in processing data ',e)
        

if __name__ == "__main__":
    data_processor = DataProcrssor(rating_array_train_file=TRAIN_ARR,rating_array_test_file = TEST_ARR,
                                   input_rating_dir= ANIMELIST_CSV,input_anime_dir=ANIME_CSV,input_synopsis_dir=ANIME_SYN_CSV,
                                   output_dir=PROCESSED_DIR, anime_file_name=ANIME_DF,
                                   user_encoder_file = USER_ENCODER_FILE,anime_encoder_file = ANIME_ENCODER_FILE)
    data_processor.execute()
            



        

