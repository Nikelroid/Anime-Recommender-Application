import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Concatenate, Activation, Dropout,Dot, Multiply, Add
from tensorflow.keras.models import Model
from collections import Counter
from config.paths_config import *
from utils.common_functions import read_yaml
from utils.suggestion_functions import (getAnimeFrame, getAnimeSyn, generate_decoders, 
                                       create_rating_df, create_temp_user_profile, create_result_df)
from tensorflow import keras
from src.base_model import BaseModel
from src.logger import get_logger
from src.custom_exception import CustomException


logger = get_logger(__name__)

class Suggestion:
    def __init__(self, model, processed_dir, anime_df_filename, weight_path, weight_names, 
                 checkpoint_path, checkpoint_name, train_df_name, test_df_name,
                 user2user_encoder_name, anime2anime_encoder_name, config):
        logger.info('Suggestion Model initilizing started')
        self.config = read_yaml(config)
        self.config_path = config
        
        self.anime_df_route = os.path.join(processed_dir, anime_df_filename)
        self.anime_weight_route = os.path.join(weight_path, weight_names['anime'])
        self.user_weight_route = os.path.join(weight_path, weight_names['user'])
        self.checkpoint_route = os.path.join(checkpoint_path, checkpoint_name)
        self.train_df_route = os.path.join(processed_dir, train_df_name)
        self.test_df_route = os.path.join(processed_dir, test_df_name)
        self.user2user_encoder_route = os.path.join(processed_dir, user2user_encoder_name)
        self.anime2anime_encoder_route = os.path.join(processed_dir, anime2anime_encoder_name)
        
        self.anime_df = None
        self.rating_df = None
        self.anime2anime_encoder = {}
        self.anime2anime_decoder = {}
        self.user2user_encoder = {}
        self.user2user_decoder = {}
        self.anime_weights = None
        self.user_weights = None
        
        self.predictor_model = model
        self.inf_model = None
        self.temp_embedding = None
        self.temp_user_profile = None
        self.similar_users = None
        self.hybrid_results = None

        logger.info('Suggestion Model initilized successfully')

    def load_data(self):
        try:
            logger.info('data loading started')
            self.anime_df = pd.read_csv(self.anime_df_route, low_memory=True)
            self.anime2anime_encoder = joblib.load(self.anime2anime_encoder_route)
            self.user2user_encoder = joblib.load(self.user2user_encoder_route)
            
            self.anime_weights = joblib.load(self.anime_weight_route)
            self.user_weights = joblib.load(self.user_weight_route)
            
            anime_norms = np.linalg.norm(self.anime_weights, axis=1, keepdims=True)
            self.anime_weights = self.anime_weights / (anime_norms + 1e-9)
            
            user_norms = np.linalg.norm(self.user_weights, axis=1, keepdims=True)
            self.user_weights = self.user_weights / (user_norms + 1e-9)
            
            train_arr = joblib.load(self.train_df_route)
            test_arr = joblib.load(self.test_df_route)
            self.rating_df = create_rating_df(train_arr, test_arr,logger=logger)
            
            self.user2user_decoder, self.anime2anime_decoder = generate_decoders(
                self.user2user_encoder, self.anime2anime_encoder,logger=logger
            )
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error('Error in loading data: ' + str(e))
            raise CustomException('Error in loading data', e)

    def initialize_model(self):
        try:
            
            if self.predictor_model is None:
                logger.info("Model is empty: Inference model initializing started")
                self.predictor_model = keras.models.load_model(self.checkpoint_route)

                logger.info("Inference model initialized successfully")
            else:
                logger.info("Model is already loaded, no need to re initialize")
        except Exception as e:
            logger.error('Error in initializing inference model: ' + str(e))
            raise CustomException('Error in initializing inference model', e)


    def find_similar_animes(self, name, n=10, consider_genres=True, genre_weight=0.2):

        try:
            logger.info("Finding similar animes for each anime...")
            index = getAnimeFrame(name, self.anime_df).anime_id.values[0]
            encoded_index = self.anime2anime_encoder.get(index)
            query_vec = self.anime_weights[encoded_index]
            dists = np.dot(self.anime_weights, query_vec)
            sorted_dists = np.argsort(dists)[::-1]
            top_candidates = sorted_dists[:min(200, len(sorted_dists))]
            final_scores = self._calculate_anime_scores(
                top_candidates, dists, index, consider_genres, genre_weight
            )
            final_scores.sort(key=lambda x: x['final_score'], reverse=True)
            logger.info("Final scores found and sorted, creating similar anime df...")
            return self._build_anime_result_df(final_scores, n)
        except Exception as e:
            logger.error('Error in find similar animes: ' + str(e))
            raise CustomException('Error in find similar animes', e)


    def _calculate_anime_scores(self, candidates, dists, query_index, consider_genres, genre_weight):
        try:
            logger.info("Starting final score calculation for each similar anime...")
            query_genres = self._get_anime_genres(query_index) if consider_genres else set()
            final_scores = []
            
            for candidate_idx in candidates:
                decoded_id = self.anime2anime_decoder.get(candidate_idx)
                if decoded_id is None or decoded_id == query_index:
                    continue
                
                cosine_score = dists[candidate_idx]
                genre_overlap = 0
                
                if consider_genres and query_genres:
                    candidate_genres = self._get_anime_genres(decoded_id)
                    if candidate_genres and query_genres:
                        matching = len(query_genres.intersection(candidate_genres))
                        union = len(query_genres.union(candidate_genres))
                        genre_overlap = matching / union if union > 0 else 0
                
                final_score = cosine_score * (1 - genre_weight) + genre_overlap * genre_weight
                
                final_scores.append({
                    'decoded_id': decoded_id,
                    'cosine_similarity': cosine_score,
                    'genre_overlap': genre_overlap,
                    'final_score': final_score
                })
            
            logger.info("Final scores calculated, listing them...")
            return final_scores
        except Exception as e:
            logger.error('Error in calculating scores for each anime: ' + str(e))
            raise CustomException('Error in calculating scores for each anime', e)

    def _get_anime_genres(self, anime_id):
        try:
            frame = self.anime_df[self.anime_df['anime_id'] == anime_id]
            if frame.empty:
                return set()
            genre_list = frame['genre_list'].values[0]
            return set(genre_list) if genre_list and len(genre_list) > 0 else set()
        except Exception as e:
            logger.error(f'Error in getting anime {anime_id} genres: ' + str(e))
            raise CustomException(f'Error in getting anime {anime_id} genres', e)

    def _build_anime_result_df(self, final_scores, n):
        try:
            similarity_arr = []
            for rank, item in enumerate(final_scores[:n * 2], 1):  
                if len(similarity_arr) >= n:
                    break
                decoded_id = item['decoded_id']
                anime_frame = self.anime_df[self.anime_df['anime_id'] == decoded_id]
                if anime_frame.empty or np.isnan(item['final_score']):
                    continue
                
                similarity_arr.append({
                    'rank': len(similarity_arr) + 1,
                    'anime_id': decoded_id,
                    'name': anime_frame['anime_name'].values[0],
                    'similarity': item['cosine_similarity'],
                    'genre_overlap': item['genre_overlap'],
                    'final_score': item['final_score'],
                    'genre': anime_frame['genres'].values[0],
                    'syn': getAnimeSyn(anime=decoded_id, anime_df=self.anime_df)
                })
            logger.info("Similar animes data frame created successfully")
            return pd.DataFrame(similarity_arr).head(n)
        except Exception as e:
            logger.error(f'Error in crating data frame for similar anime results ' + str(e))
            raise CustomException(f'Error in crating data frame for similar anime results ', e)

    def find_similar_users(self, user_id, n=10, negative=False):
        encoded_index = self.user2user_encoder.get(int(user_id))
        dists = np.dot(self.user_weights, self.user_weights[encoded_index])
        
        nan_mask = np.isnan(dists)
        valid_indices = np.where(~nan_mask)[0]
        valid_sorted = valid_indices[np.argsort(dists[valid_indices])[::-1 if not negative else 1]]
        
        similarity_data = [
            {'similar_user': self.user2user_decoder.get(idx), 'similarity': dists[idx]}
            for idx in valid_sorted[:n + 1]
            if idx != encoded_index
        ]
        
        return pd.DataFrame(similarity_data).head(n)

    def get_user_preferences(self, user_id):
        user_ratings = self.rating_df[
            self.rating_df.user_id == self.user2user_encoder[user_id]
        ].dropna(subset=['rating'])
        
        if user_ratings.empty:
            return pd.DataFrame(columns=["anime_name", "genres"])
        
        # Get top 25th percentile
        threshold = np.percentile(user_ratings.rating, 75)
        top_ratings = user_ratings[user_ratings.rating >= threshold]
        top_anime_ids = top_ratings.sort_values('rating', ascending=False).anime_id.values
        
        return self.anime_df[self.anime_df["anime_id"].isin(top_anime_ids)][
            ["anime_name", "genres"]
        ]

    def get_user_recommendations(self, similar_users, user_pref, n=10):
        """Get recommendations based on similar users' preferences with similarity weighting"""
        anime_scores = {}
        
        for idx, row in similar_users.iterrows():
            user_id = row['similar_user']
            similarity = row['similarity']
            
            pref_list = self.get_user_preferences(int(user_id))
            pref_list = pref_list[~pref_list.anime_name.isin(user_pref.anime_name.values)]
            
            if not pref_list.empty:
                # Weight by user similarity
                for anime_name in pref_list.anime_name.values:
                    if anime_name not in anime_scores:
                        anime_scores[anime_name] = 0
                    anime_scores[anime_name] += similarity
        
        if not anime_scores:
            return pd.DataFrame()
        
        # Sort by weighted score
        sorted_animes = sorted(anime_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        recommendations = [
            {
                "score": score,
                "anime_name": anime_name,
                "genres": getAnimeFrame(anime_name, self.anime_df).genres.values[0]
            }
            for anime_name, score in sorted_animes
            if isinstance(anime_name, str)
        ]
        
        return pd.DataFrame(recommendations)

    def build_user_inf_model(self, embedding_size=32):
        """Build inference model for temporary user"""
        user_vec = Input(shape=(embedding_size,), name='user_vec')
        anime = Input(shape=(1,), name='anime')
        
        anime_emb = self.predictor_model.get_layer('anime_embedding')(anime)
        anime_bias = self.predictor_model.get_layer('anime_bias')(anime)
        anime_vec = Flatten(name='flatten_anime')(anime_emb)
        anime_bias_vec = Flatten(name='flatten_anime_bias')(anime_bias)
        
        dot_product = Dot(axes=1, name='dot_product')([user_vec, anime_vec])
        
        interaction = Multiply(name='element_multiply')([user_vec, anime_vec])
        
        x = Concatenate(name='concat')([user_vec, anime_vec, interaction])
        x = self.predictor_model.get_layer('dense_1')(x)
        x = self.predictor_model.get_layer('layer_norm_1')(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        x = self.predictor_model.get_layer('dense_2')(x)
        x = self.predictor_model.get_layer('layer_norm_2')(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        x = self.predictor_model.get_layer('dense_3')(x)
        x = self.predictor_model.get_layer('layer_norm_3')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        
        x = self.predictor_model.get_layer('dense_4')(x)
        x = Activation('relu')(x)
        
        dense_output = self.predictor_model.get_layer('dense_output')(x)
        
        output = Add(name='final_add')([dot_product, dense_output, anime_bias_vec])
        output = Activation('sigmoid', name='output')(output)
        
        self.inf_model = Model(inputs=[user_vec, anime], outputs=output)

    def get_temp_user_embedding(self, embedding_size=32, min_rating=None, 
                               max_rating=None, epochs=100, lr=0.01):
        """Optimize embedding for temporary user with improved training"""
        anime_list, rating_list = [], []
        
        for _, row in self.temp_user_profile.iterrows():
            encoded_anime = self.anime2anime_encoder.get(row['anime_id'])
            if encoded_anime is not None:
                anime_list.append(encoded_anime)
                normalized_rating = (
                    (row['rating'] - min_rating) / (max_rating - min_rating + 1)
                    if min_rating is not None and max_rating is not None
                    else row['rating'] / 10.0
                )
                rating_list.append(normalized_rating)
        
        if not anime_list:
            print("Warning: No valid anime found for temp user")
            self.temp_embedding = np.zeros(embedding_size)
            return
        
        print(f"Training temp user embedding with {len(anime_list)} ratings...")
        
        # Prepare tensors
        anime_indices = tf.constant(anime_list, dtype=tf.int32)[:, tf.newaxis]
        targets = tf.constant(rating_list, dtype=tf.float32)
        
        # Initialize user embedding
        user_emb_var = tf.Variable(tf.random.normal([embedding_size], stddev=0.05))
        
        # Learning rate schedule
        initial_lr = lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=epochs // 3,
            decay_rate=0.5
        )
        optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        
        # Use MSE loss for better rating prediction
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                user_vec_input = tf.tile(user_emb_var[tf.newaxis, :], [len(anime_list), 1])
                preds = tf.squeeze(self.inf_model([user_vec_input, anime_indices]))
                loss = loss_fn(targets, preds)
            
            grads = tape.gradient(loss, [user_emb_var])
            optimizer.apply_gradients(zip(grads, [user_emb_var]))
            
            # Early stopping
            current_loss = loss.numpy()
            if current_loss < best_loss - 1e-5:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, loss: {current_loss:.4f}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f}")
        
        # Normalize embedding
        temp_emb = user_emb_var.numpy()
        norm = np.linalg.norm(temp_emb)
        self.temp_embedding = temp_emb / norm if norm > 0 else temp_emb
        print("Temp user embedding optimized!")

    def find_similar_users_for_temp(self, n=30):
        """Find similar users for temporary user embedding (increased n for diversity)"""
        if np.all(self.temp_embedding == 0):
            print("Warning: Temp user embedding is zero; returning empty user list.")
            self.similar_users = pd.DataFrame(columns=['similar_user', 'similarity'])
            return
        
        similarities = np.dot(self.user_weights, self.temp_embedding)
        valid_indices = np.where(~np.isnan(similarities))[0]
        valid_sorted = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        similar_users_data = [
            {
                'similar_user': self.user2user_decoder.get(user_idx),
                'similarity': similarities[user_idx]
            }
            for user_idx in valid_sorted[:n]
        ]
        
        self.similar_users = pd.DataFrame(similar_users_data)
        print(f"Found {len(self.similar_users)} similar users")

    def _apply_diversity_penalty(self, anime_scores, diversity_weight=0.3):
        """Apply diversity penalty to avoid recommending too many anime from same genre"""
        genre_counts = Counter()
        diversified_scores = []
        
        for anime_name, score in anime_scores:
            anime_frame = getAnimeFrame(anime_name, self.anime_df)
            if anime_frame.empty:
                continue
            
            genres = anime_frame['genres'].values[0]
            genre_list = genres.split(', ') if isinstance(genres, str) else []
            
            # Calculate penalty based on already selected genres
            penalty = 0
            for genre in genre_list:
                penalty += genre_counts.get(genre, 0) * diversity_weight
            
            # Apply penalty
            adjusted_score = score - penalty
            diversified_scores.append((anime_name, adjusted_score, genres))
            
            # Update genre counts
            for genre in genre_list:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Re-sort by adjusted scores
        diversified_scores.sort(key=lambda x: x[1], reverse=True)
        return [(name, score) for name, score, _ in diversified_scores]

    def hybrid_recommendation_for_temp_user(self, user_weight=0.5, content_weight=0.5, 
                                           n=10, genre_weight=0.3, diversity_weight=0.2):
        """
        Combine user-based and content-based recommendations with improved scoring
        """
        print("Generating hybrid recommendations...")
        consider_genres = genre_weight > 0.0
        
        # Get user-based recommendations with scores
        user_recommended = self.get_user_recommendations(self.similar_users, self.temp_user_profile, n=n*2)
        
        if user_recommended.empty:
            print("No user-based recommendations found")
            self.hybrid_results = []
            return
        
        # Filter out already watched anime
        watched_anime = set(self.temp_user_profile['anime_name'].values)
        user_recommended = user_recommended[~user_recommended['anime_name'].isin(watched_anime)]
        
        combined_scores = {}
        
        # Add user-based scores with frequency weighting
        print(f"Processing {len(user_recommended)} user-based recommendations...")
        for idx, row in user_recommended.iterrows():
            anime_name = row['anime_name']
            score = row['score']
            combined_scores[anime_name] = score * user_weight
        
        # Add content-based scores with rank decay
        print("Finding content-based recommendations...")
        content_scores = {}
        for anime in user_recommended['anime_name'].head(n).tolist():
            similar_animes = self.find_similar_animes(
                anime, 
                consider_genres=consider_genres, 
                genre_weight=genre_weight, 
                n=n
            )
            
            if similar_animes is not None and not similar_animes.empty:
                for idx, row in similar_animes.iterrows():
                    similar_name = row['name']
                    if similar_name in watched_anime:
                        continue
                    
                    # Rank decay: higher ranked items get more weight
                    rank = row['rank']
                    rank_decay = 1.0 / (rank + 1)
                    weighted_score = row['final_score'] * rank_decay * content_weight
                    
                    if similar_name not in content_scores:
                        content_scores[similar_name] = 0
                    content_scores[similar_name] += weighted_score
        
        # Merge content scores into combined scores
        for anime_name, score in content_scores.items():
            if anime_name in combined_scores:
                combined_scores[anime_name] += score
            else:
                combined_scores[anime_name] = score
        
        if not combined_scores:
            print("No valid recommendations generated")
            self.hybrid_results = []
            return
        
        # Sort by score
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply diversity penalty
        diversified_scores = self._apply_diversity_penalty(sorted_scores[:n*3], diversity_weight)
        
        # Get top n
        self.hybrid_results = diversified_scores[:n]
        
        print(f"Generated {len(self.hybrid_results)} hybrid recommendations")
        print("Top 3 scores:", [f"{name}: {score:.3f}" for name, score in self.hybrid_results[:3]])

    def execute(self, user_pred_dict,num):
        try:
            logger.info('=' * 60)
            logger.info('RECOMMENDATION PIPELINE STARTED')
            logger.info('=' * 60)
            
            cfg = self.config['suggestion']
            n = cfg['n'] if num==None else num
            model_cfg = self.config['model_training']
            
            self.load_data()
            self.initialize_model()
            

            self.temp_user_profile = create_temp_user_profile(user_pred_dict,self.anime_df,logger )
            
            if self.temp_user_profile.empty:
                logger.warning('Could not create valid user profile')
                return "403"
            
            self.build_user_inf_model(embedding_size=model_cfg['embedding_size'])
            self.get_temp_user_embedding(
                min_rating=cfg['min_rating'],
                max_rating=cfg['max_rating'],
                embedding_size=model_cfg['embedding_size'],
                epochs=cfg.get('inf_epoch', 100),
                lr=cfg.get('inf_lr_rate', 0.01)
            )
            
            self.find_similar_users_for_temp(n=cfg.get('n_similar_users', 30))
            
            self.hybrid_recommendation_for_temp_user(
                user_weight=cfg['user_weight'],
                content_weight=cfg['content_weight'],
                n=n,
                genre_weight=cfg.get('genres_weight', 0.3),
                diversity_weight=cfg.get('diversity_weight', 0.2)
            )
            
            if not self.hybrid_results:
                logger.warning('No recommendations generated')
                return "404"
            
            result = create_result_df(self.hybrid_results, n_recommendations=n, df=self.anime_df,logger=logger)
            logger.info('Recommendation pipeline completed successfully')
            
            return result
        except Exception as e:
            logger.error(f"Error In prediction pipeline : {str(e)}")
            raise CustomException("Failed to complete prediction pipeline", e)


if __name__ == '__main__':
    suggestion_model = Suggestion(
        None, processed_dir=PROCESSED_DIR, anime_df_filename=ANIME_DF,
        weight_path=WEIGHTS_DIR, weight_names=WEIGHTS_FILE_NAME,
        checkpoint_path=CHECKPOINT_DIR, checkpoint_name=CHECKPOINT_FILE_NAME,
        train_df_name=TRAIN_ARR, test_df_name=TEST_ARR,
        user2user_encoder_name=USER_ENCODER_FILE, 
        anime2anime_encoder_name=ANIME_ENCODER_FILE,
        config=CONFIG_PATH
    )
    
    user_ratings = {
        'Attack on Titan': 9,
        'Death Note': 8,
        'One Piece': 1,
        'Naruto': 6,
        'Monster': 2
    }
    
    result = suggestion_model.execute(user_ratings)
    print("\n=== FINAL RECOMMENDATIONS ===")
    print(result)