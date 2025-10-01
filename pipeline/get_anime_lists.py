import os
import pandas as pd
from config.paths_config import *
import joblib
from src.logger import get_logger

logger = get_logger(__name__)

def get_anime_list():
    try:
        anime_df = pd.read_csv(os.path.join(PROCESSED_DIR, ANIME_DF), low_memory=True)
        anime_encoder = joblib.load(os.path.join(PROCESSED_DIR, ANIME_ENCODER_FILE))
        valid_ids = set(anime_encoder.keys())
        filtered = anime_df[anime_df["anime_id"].isin(valid_ids)]
        result = filtered["anime_name"].tolist()
        if len(result)==0: 
            logger.error('Anime list is empty')
            return "404" 
        else: 
            return result

    except Exception as e:
        logger.error('Error in finding anime names ' + str(e))
        return "403"
    