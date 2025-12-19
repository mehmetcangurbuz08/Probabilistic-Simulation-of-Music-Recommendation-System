import numpy as np
import pandas as pd
import random
import os
from part1 import apply_binning

# PARAMETERS ==========================================
ART_DIR = "artifacts" # csv files' directory

TRACK_DF = pd.read_csv("tracks.csv").fillna("unknown")
TRACK_DF = apply_binning(TRACK_DF) # binning applied track dataframe

ALPHA = 4.3379 #3.0535
BETA = 6.9353 #5.6081
EXPECTED_P = ALPHA / (ALPHA + BETA)   # ≈ 0.3525

# columns for personalization
PERSONALIZABLE_FEATURES = [
    "primary_artist_name",
    "ab_genre_rosamerica_value",
    "ab_genre_dortmund_value",
    "ab_mood_happy_value",
    "ab_mood_sad_value",
    "ab_mood_party_value",
    "ab_mood_relaxed_value",
    "ab_mood_acoustic_value",
    "ab_mood_aggressive_value",
    "ab_mood_electronic_value",
    "ab_danceability_value",
    "ab_timbre_value",
    "ab_voice_instrumental_value",
    "ab_gender_value",
    "year_bin",
    "popularity_bin",
    "duration_bin",
]

MIN_CONCENTRATION = 0.35

# HELPER FUNCTIONS ==========================================
def load_cond_probs(path): # reading conditional probabilities from CSV
    df = pd.read_csv(path)
    prob_map = {}
    for _, r in df.iterrows():
        feat = str(r["feature"])
        val = str(r["value"])
        prob_map[(feat, val)] = float(r["prob"])
    return prob_map

P_GLOBAL = load_cond_probs(os.path.join(ART_DIR, "cond_probs_global.csv")) # global probs
P_PERSONAL = load_cond_probs(os.path.join(ART_DIR, "cond_probs_personal.csv")) # personal probs

def get_prob(feature, value, default=0.2):
    """Global fallback  P(5★ | feature=value)"""
    return P_GLOBAL.get((str(feature), str(value)), default)

def compute_global_probability(track: pd.Series) -> float:
    """
    Computes a global score for a single track using the statistics
    learned in Part 1.

    For each feature of the track (such as year, popularity, mood, or genre),
    the function looks up how often tracks with that feature received a 5★ rating
    in the dataset, and multiplies these values together..
    """
    score = 1.0
    # explicit
    explicit_val = track["explicit"]
    score *= get_prob("explicit", explicit_val)

    # year_bin, popularity_bin, duration_bin, markets_bin
    score *= get_prob("year_bin", track["year_bin"])
    score *= get_prob("popularity_bin", track["popularity_bin"])
    score *= get_prob("duration_bin", track["duration_bin"])
    score *= get_prob("markets_bin", track["markets_bin"])

    ab_cols = [
        "ab_genre_rosamerica_value",
        "ab_genre_dortmund_value",
        "ab_timbre_value",
        "ab_danceability_value",
        "ab_mood_acoustic_value",      
        "ab_mood_aggressive_value",    
        "ab_mood_electronic_value",    
        "ab_mood_happy_value",
        "ab_mood_party_value",
        "ab_mood_relaxed_value",
        "ab_mood_sad_value",
        "ab_gender_value",
        "ab_voice_instrumental_value",
    ]

    for c in ab_cols:
        score *= get_prob(c, track[c])

    return float(score)


# Model 1 is a hybrid recommender that combines global popularity patterns
# with lightweight personalization.

# First, each track is given a global base score using conditional
# probabilities learned in Part 1. This reflects how likely the track
# is to receive a 5★ rating overall.

# Then, the model looks at the user’s liked tracks and detects which
# features the user is consistent in (for example genre, artist, or mood).
# Tracks matching these preferred feature values receive a multiplicative
# bonus, scaled by how strong the user’s consistency is.

# Finally, all tracks are ranked by their final score and the top-k
# tracks are returned as recommendations.

class Model1:
    def __init__(self):
        self.tracks = TRACK_DF

    def _calculate_feature_importance(self, liked_tracks):
        """
        Computes how consistent the user is for each feature
        based on the tracks they liked.

        If most liked tracks share the same value for a feature,
        that feature is considered important.

        Returns features sorted from most to least important.
        """
        importance_scores = {}
        
        for feat in PERSONALIZABLE_FEATURES:
            if feat not in liked_tracks.columns:
                continue
            
            value_counts = liked_tracks[feat].value_counts(normalize=True)
            
            if len(value_counts) == 0:
                continue
            
            max_ratio = value_counts.iloc[0]
            n_unique = len(value_counts)
            n_total = len(liked_tracks)
            
            if n_unique <= 2 and n_total >= 3:
                max_ratio *= 1.2
            
            importance_scores[feat] = {
                "score": min(max_ratio, 1.0),
                "top_value": value_counts.index[0],
                "top_ratio": value_counts.iloc[0],
                "n_unique": n_unique,
            }
        
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        return sorted_features

    def _get_dynamic_bonuses(self, liked_tracks):
        """
        Creates personalized bonus multipliers based on the user’s liked tracks.

        Features where the user shows strong consistency (high concentration)
        receive a bonus. Tracks matching the user’s preferred values for these
        features get their score boosted.

        If there is too little data, simple default bonuses are used.
        """

        if len(liked_tracks) < 2:
            return {
                "primary_artist_name": {
                    "bonus": 1.5,
                    "values": set(liked_tracks["primary_artist_name"]),
                    "concentration": 1.0
                },
                "ab_genre_rosamerica_value": {
                    "bonus": 1.3,
                    "values": set(liked_tracks["ab_genre_rosamerica_value"]),
                    "concentration": 1.0
                },
            }
        
        sorted_features = self._calculate_feature_importance(liked_tracks)
        bonuses = {}
        
        for feat, info in sorted_features:
            if info["score"] < MIN_CONCENTRATION:
                continue

            liked_values = set(liked_tracks[feat].dropna().unique())

            bonuses[feat] = {
                "bonus": 1.5,                
                "values": liked_values,
                "concentration": info["score"]
            }
        return bonuses

    def query(self, song_ratings, topk=5):
        """
        Generates recommendations for a user.

        If the user has no liked tracks, it falls back to a global ranking
        based only on overall probabilities.

        Otherwise, it:
        - Computes a base score for each track using global statistics
        - Applies personalized bonuses for features the user is consistent in
        - Ranks tracks by the final score and returns the top recommendations
        """
        liked_ids = [s["track_id"] for s in song_ratings if s["rating"] >= 4]
        liked_tracks = self.tracks[self.tracks["track_id"].isin(liked_ids)]

        if liked_tracks.empty:
            df = self.tracks.copy()
            df["prob"] = df.apply(compute_global_probability, axis=1)
            out = df.sort_values("prob", ascending=False).head(topk)
            return list(zip(out["track_id"], out["track_name"]))
        
        dynamic_bonuses = self._get_dynamic_bonuses(liked_tracks)

        df = self.tracks.copy()
        df["score"] = 0.0

        for i, tr in df.iterrows():
            score = compute_global_probability(tr)
            for feat, bonus_info in dynamic_bonuses.items():
                track_value = tr.get(feat)
                if track_value in bonus_info["values"]:
                    score *= bonus_info["bonus"] ** bonus_info["concentration"]

            df.at[i, "score"] = score

        recs = df.sort_values("score", ascending=False).head(topk)
        return list(zip(recs["track_id"], recs["track_name"]))



# Model2: Patience-aware probabilistic recommender.
# Steps:
# 1. Compute a score table using the exact same logic as Model1
#    (global feature-based probabilities + lightweight personalization).
# 2. Keep only the top-N scoring tracks to form a candidate pool.
# 3. Estimate the user’s patience (expected Tu) using the Beta-Geometric
#    model from Part 2.
# 4. Map the estimated patience to a greediness level:
#   - Impatient users → more greedy (focus on top scores)
#   - Patient users   → more exploratory (allow more diversity)
# 5. Convert scores into sampling probabilities using a temperature-
#    controlled softmax.
# 6. Sample k tracks from the candidate pool according to these probabilities.

# This preserves Model1’s ranking quality while introducing controlled
# randomness based on user patience.

class Model2:

    def __init__(self,candidate_size=75):
        self.tracks = TRACK_DF
        self.candidate_size = candidate_size
    
    def _compute_score_table(self, liked_tracks):
        """
        Computes track scores using the exact same logic as Model1.
        """
        # No personalization → pure global
        if liked_tracks.empty:
            df = self.tracks.copy()
            df["score"] = df.apply(compute_global_probability, axis=1)
            return df

        model1 = Model1()
        dynamic_bonuses = model1._get_dynamic_bonuses(liked_tracks)

        df = self.tracks.copy()
        df["score"] = 0.0

        for i, tr in df.iterrows():
            score = compute_global_probability(tr)

            for feat, bonus_info in dynamic_bonuses.items():
                if tr.get(feat) in bonus_info["values"]:
                    score *= bonus_info["bonus"] ** bonus_info["concentration"]

            df.at[i, "score"] = score

        return df

    def _estimate_user_patience(self, song_ratings):
        """
        Bayesian estimation of user patience using Part 2.
        """
        if not song_ratings:
            estimated_p = EXPECTED_P
            expected_Tu = 1.0 / estimated_p
            return estimated_p, expected_Tu

        n_total = len(song_ratings)
        n_fives = sum(1 for s in song_ratings if s["rating"] == 5)

        alpha_post = ALPHA + n_fives
        beta_post = BETA + (n_total - n_fives)

        estimated_p = alpha_post / (alpha_post + beta_post)
        expected_Tu = 1.0 / estimated_p

        return estimated_p, expected_Tu


    def _patience_to_exponent(self, expected_Tu):
        """
        Maps expected Tu to a smooth exponent.
        Lower Tu  -> higher exponent (greedy)
        Higher Tu -> lower exponent (exploratory)
        """
        # normalize around EXPECTED_P (~0.35 => Tu~2.8)
        base_Tu = 1.0 / EXPECTED_P

        ratio = expected_Tu / base_Tu

        # smooth inverse mapping
        exponent = 2.5 / ratio

        return float(np.clip(exponent, 1.2, 4.0))

    def query(self, song_ratings, topk=5):
        """
        Probabilistic recommendation based on patience-aware softmax sampling.
        """
        liked_ids = [s["track_id"] for s in song_ratings if s["rating"] >= 4]
        liked_tracks = self.tracks[self.tracks["track_id"].isin(liked_ids)]

        # same as Model1 score computation
        df = self._compute_score_table(liked_tracks)
        df = df.sort_values("score", ascending=False).head(self.candidate_size)

        # estimate user patience
        _, expected_Tu = self._estimate_user_patience(song_ratings)
        exponent = self._patience_to_exponent(expected_Tu)

        # score normalization
        scores = df["score"].values
        scores = scores / (scores.max() + 1e-12)

        # temperature calculation
        temperature = 1.0 / exponent

        # softmax utility
        utilities = np.exp(scores / temperature)
        probabilities = utilities / utilities.sum()

        # sampling
        sample_size = min(topk, len(df))
        sampled_idx = np.random.choice(
            len(df),
            size=sample_size,
            replace=False,
            p=probabilities
        )

        recs = df.iloc[sampled_idx]
        return list(zip(recs["track_id"], recs["track_name"]))
