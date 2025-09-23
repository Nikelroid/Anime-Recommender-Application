import os

# ========== 1.DATA_INGESTION ===============

RAW_DIR = "artifacts/raw"
CONFIG_PATH = "config/config.yaml"

# ========== 2.DATA_PROCESSING ===============

PROCESSED_DIR = "artifacts/processed"
ANIMELIST_CSV = "artifacts/raw/animelist_filtered.csv"
ANIME_CSV = "artifacts/raw/anime.csv"
ANIME_SYN_CSV = "artifacts/raw/anime_with_synopsis.csv"

TRAIN_ARR = 'train_arr.pkl'
TEST_ARR = 'test_arr.pkl'

ANIME_DF = "anime_data.csv"

USER_ENCODER_FILE = "user2user_encoder.pkl"
ANIME_ENCODER_FILE = "anime2anime_encoder.pkl"

# ============ 3.MODEL_TRAINING =================

CHECKPOINT_DIR = "artifacts/models"
WEIGHTS_DIR = "artifacts/wights"

CHECKPOINT_FILE_NAME = 'best_recommender_model.weights.h5'
WEIGHTS_FILE_NAME = {'anime':'anime_weights.pkl','user':'user_weights.pkl'}
