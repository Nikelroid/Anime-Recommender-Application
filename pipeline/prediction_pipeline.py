from utils.common_functions import read_yaml
from config.paths_config import *
from src.suggestion import Suggestion

class Recommender:
    def __init__(self,user_ratings={},n=None):
        #For Test!
        user_ratings = {
        'Attack on Titan': 9,
        'Death Note': 8,
        'One Piece': 1,
        'Naruto': 6,
        'Monster': 2}
        self.user_ratings = user_ratings
        self.num = n
        
    def recommend(self):
        suggestion_model = Suggestion(
                None, processed_dir=PROCESSED_DIR, anime_df_filename=ANIME_DF,
                weight_path=WEIGHTS_DIR, weight_names=WEIGHTS_FILE_NAME,
                checkpoint_path=CHECKPOINT_DIR, checkpoint_name=CHECKPOINT_FILE_NAME,
                train_df_name=TRAIN_ARR, test_df_name=TEST_ARR,
                user2user_encoder_name=USER_ENCODER_FILE, 
                anime2anime_encoder_name=ANIME_ENCODER_FILE,
                config=CONFIG_PATH
            )
        return suggestion_model.execute(self.user_ratings,num=self.num)


