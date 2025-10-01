import numpy as np
from collections import defaultdict
import pandas as pd
from src.custom_exception import CustomException

def getAnimeFrame(anime,anime_df):
    try:
        if isinstance(anime,int):return anime_df[anime_df.anime_id == anime]
        if isinstance(anime,str):return anime_df[anime_df.anime_name == anime]
    except Exception as e:
        raise CustomException(f"Failed to get anime frame {anime} ", e)

def getAnimeSyn(anime,anime_df):
    try:
        if isinstance(anime,int):return anime_df[anime_df.anime_id == anime].syn.values[0]
        if isinstance(anime,str):return anime_df[anime_df.anime_name == anime].syn.values[0]
    except Exception as e:
        raise CustomException(f"Failed to get anime synopsis {anime} ", e)

def generate_decoders(user2user_encoder,anime2anime_encoder,logger):
    try:
        logger.info(f"Generating decoders started")
        decoder_user, decoder_anime  = {},{}
        for key,value in user2user_encoder.items(): decoder_user[value] = key
        for key,value in anime2anime_encoder.items(): decoder_anime[value] = key
        logger.info(f"Decoders generated successfully")
        return decoder_user,decoder_anime
    except Exception as e:
        logger.error(f"Error in generation decoders : {str(e)}")
        raise CustomException("Failed to generation decoders ", e)

def create_rating_df(train_arr,test_arr,logger):
    try:
        logger.info(f"Reforming rating_df started")
        x_train_user, x_train_anime, y_train = train_arr
        x_test_user, x_test_anime, y_test = test_arr
        all_user_ids = np.concatenate([x_train_user, x_test_user])
        all_anime_ids = np.concatenate([x_train_anime, x_test_anime])
        all_ratings = np.concatenate([y_train, y_test])
        rating_df = pd.DataFrame({
            'user_id': all_user_ids.astype(int),
            'anime_id': all_anime_ids.astype(int),
            'rating': all_ratings
        })
        rating_df = rating_df.sort_values(by='user_id').reset_index(drop=True)
        logger.info(f"Rating_df reformd successfully")
        return rating_df
    except Exception as e:
        logger.error(f"Error in reforming rating_df : {str(e)}")
        raise CustomException("Failed to reform rating_df ", e)


def create_temp_user_profile(anime_ratings_dict, df,logger):
    try:
        logger.info(f"Building profile for user with {len(anime_ratings_dict)} ratings...")
        temp_profile = []
        for anime_name, rating in anime_ratings_dict.items():
            anime_frame = getAnimeFrame(anime_name, df)
            if not anime_frame.empty:
                anime_id = anime_frame.anime_id.values[0]
                temp_profile.append({
                    'user_id': -1, 
                    'anime_id': anime_id,
                    'anime_name': anime_name,
                    'rating': rating
                })
        logger.info(f"Temp profile created successfully")
        return pd.DataFrame(temp_profile)
    except Exception as e:
        logger.error(f"Error in building temp user profile : {str(e)}")
        raise CustomException("Failed to build temp user profile", e)

def create_result_df(hybrid_results,n_recommendations,df,logger):
    try:
        logger.info(f"Creating final result for {n_recommendations} is starting")
        max_score = hybrid_results[0][1] if hybrid_results else 1
        final_recommendations = []
        for anime_name, score in hybrid_results[:n_recommendations]:
            anime_frame = getAnimeFrame(anime_name, df)
            if not anime_frame.empty:
                probability = min((score / max_score) * 0.8, 0.8)
                final_recommendations.append({
                    'anime_name': anime_name,
                    'genres': anime_frame.genres.values[0],
                    'synopsis': anime_frame.syn.values[0],
                    'hybrid_score': score,
                    'probability': probability
                })
        logger.info(f"Final result for {n_recommendations} created")
        return pd.DataFrame(final_recommendations)
    except Exception as e:
        logger.error(f"Error in handling final results : {str(e)}")
        raise CustomException("Failed to handle final results", e)

