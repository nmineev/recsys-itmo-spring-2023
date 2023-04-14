import pickle
import numpy as np
from scipy.special import expit as sigmoid

from .random import Random
from .recommender import Recommender
import random


class CustomSVD(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, catalog, num_top_tracks_for_random_choice=100):
        self.tracks_redis = tracks_redis
        self.fallback = Random(tracks_redis)
        self.catalog = catalog

        with open("./data/embeddings.npz", "rb") as f:
            embeddings = np.load(f)
            self.user_embeddings = embeddings["user_embeddings"]
            self.track_embeddings = embeddings["track_embeddings"]
            # self.artist_embeddings = embeddings["artist_embeddings"]
            self.time_embeddings = embeddings["time_embeddings"]
            self.user_bias = embeddings["user_bias"]
            self.track_bias = embeddings["track_bias"]
            self.mu = embeddings["mu"]

        self.num_top_tracks_for_random_choice = num_top_tracks_for_random_choice

    def _recommend_next(self, user: int, track: int, time: float) -> list:
        time = np.array([round(time * 100)]) #self.le_time.transform([time])
        user_final = self.user_embeddings[user] + self.track_embeddings[track] + self.time_embeddings[time]
        scores = self.track_embeddings.dot(user_final.T).squeeze()
        scores += (self.mu + self.user_bias[user] + self.track_bias).squeeze()
        scores = sigmoid(scores).squeeze()
        recommendations = np.argpartition(
            -scores, self.num_top_tracks_for_random_choice
        )[:self.num_top_tracks_for_random_choice].tolist()
        return recommendations

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        shuffled = self._recommend_next(user, prev_track, prev_track_time)
        random.shuffle(shuffled)
        return shuffled[0]
