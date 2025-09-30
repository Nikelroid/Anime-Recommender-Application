import numpy as np
from collections import defaultdict
import pandas as pd

def getAnimeFrame(anime,anime_df):
    if isinstance(anime,int):return anime_df[anime_df.anime_id == anime]
    if isinstance(anime,str):return anime_df[anime_df.anime_name == anime]

def getAnimeSyn(anime,anime_df):
    if isinstance(anime,int):return anime_df[anime_df.anime_id == anime].syn.values[0]
    if isinstance(anime,str):return anime_df[anime_df.anime_name == anime].syn.values[0]

def generate_decoders(user2user_encoder,anime2anime_encoder):
    decoder_user, decoder_anime  = {},{}
    for key,value in user2user_encoder.items(): decoder_user[value] = key
    for key,value in anime2anime_encoder.items(): decoder_anime[value] = key
    return decoder_user,decoder_anime

def create_rating_df(train_arr,test_arr):
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
    return rating_df

def getFavGenre(frame):
    cleaned_frame = frame.dropna(inplace=False)
    all_genres = defaultdict(int)
    genres_list = []
    for genres in cleaned_frame["genres"]:
        if isinstance(genres,str):
            for genre in genres.split(','):
                genres_list.append(genre)
                all_genres[genre.strip()] += 1
    return genres_list

def create_temp_user_profile(anime_ratings_dict, df):
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
    return pd.DataFrame(temp_profile)

def create_result_df(hybrid_results,n_recommendations,df):
    max_score = hybrid_results[0][1] if hybrid_results else 1
    
    final_recommendations = []
    for anime_name, score in hybrid_results[:n_recommendations]:
        anime_frame = getAnimeFrame(anime_name, df)
        if not anime_frame.empty:
            probability = min((score / max_score) * 0.8, 0.8)
            final_recommendations.append({
                'anime_name': anime_name,
                'genres': anime_frame.genres.values[0],
                'hybrid_score': score,
                'probability': probability
            })
    return pd.DataFrame(final_recommendations)

