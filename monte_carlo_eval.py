import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.special import betaln

# Import from recommender (Part 3)
from recommender import (
    Model1,
    Model2,
    TRACK_DF,
    compute_global_probability,
    ALPHA,
    BETA,
    EXPECTED_P,
)

# 1. CONTINUOUS FEATURES (normalized to 0-1)
CONTINUOUS_PREF_FEATURES = [
    "track_popularity_norm",   # 0-100 → 0-1
    "duration_norm",           # ms → 0-1 (0-10 min range)
    "available_markets_norm",  # 0-200 → 0-1
]

# 2. CATEGORICAL FEATURES (user preferred categories)
# Each user will prefer some of these categories
CATEGORICAL_PREF_FEATURES = {
    # Genre preferences (compatible with Part 1 CSV values!)
    "ab_genre_rosamerica_value": [
        "hip", "rhy", "roc", "pop", "dan", "cla", "jaz"  # Only these exist in CSV
    ],
    # Dortmund genre
    "ab_genre_dortmund_value": [
        "alternative", "blues", "electronic", "folkcountry", "funksoulrnb", 
        "jazz", "pop", "raphiphop", "rock", "schlager"
    ],
    # Mood: Acoustic
    "ab_mood_acoustic_value": ["acoustic", "not_acoustic"],
    # Mood: Aggressive
    "ab_mood_aggressive_value": ["aggressive", "not_aggressive"],
    # Mood: Electronic
    "ab_mood_electronic_value": ["electronic", "not_electronic"],
    # Mood: Happy
    "ab_mood_happy_value": ["happy", "not_happy"],
    # Mood: Party
    "ab_mood_party_value": ["party", "not_party"],
    # Mood: Relaxed
    "ab_mood_relaxed_value": ["relaxed", "not_relaxed"],
    # Mood: Sad
    "ab_mood_sad_value": ["sad", "not_sad"],
    # Danceability
    "ab_danceability_value": ["danceable", "not_danceable"],
    # Voice vs Instrumental (compatible with CSV column name)
    "ab_voice_instrumental_value": ["voice", "instrumental"],
    # Timbre
    "ab_timbre_value": ["bright", "dark"],
    # Gender (vocal)
    "ab_gender_value": ["female", "male"],
}

# 3. YEAR PREFERENCE (year_bin from Part 1)
YEAR_BINS = ["pre_1980", "1980s", "1990s", "2000s", "2010s", "2020s"]
YEAR_BIN_RANGES = {
    "pre_1980": (0, 1979),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2030),
}

# Old list (backward compatibility)
USER_PREF_FEATURES = CONTINUOUS_PREF_FEATURES

RNG_SEED = 343
np.random.seed(RNG_SEED)

N_USERS = 2000        # Monte Carlo sample size
TOPK = 5              # k for Hit@k
WARMUP_SIZE = 20      # Warm-up songs per user (preference learning)

# Global vs Personal weights
GLOBAL_WEIGHT = 0.6   # Global P(5★) weight from Part 1
PERSONAL_WEIGHT = 0.4 # User-track match weight


def precompute_track_p5() -> Dict[str, float]:
    """
    Using compute_global_probability from Part 1,
    compute a raw score for each track.
    Rescale these scores to [0.02, 0.6] range
    to use as P(5★ | track).
    """
    df = TRACK_DF.copy()
    raw_scores = df.apply(compute_global_probability, axis=1)

    # Min-max normalization
    min_s = raw_scores.min()
    max_s = raw_scores.max()
    denom = max_s - min_s + 1e-12

    scaled = (raw_scores - min_s) / denom  # [0, 1]

    # Map to [0.02, 0.6] range
    p5 = 0.02 + 0.58 * scaled

    track_ids = df["track_id"].values
    return {tid: float(p) for tid, p in zip(track_ids, p5)}


TRACK_P5 = precompute_track_p5()


def precompute_track_features() -> Dict[str, Dict]:
    """
    Compute extended feature vector for each track.
    
    Includes:
    1. Continuous features (normalized)
    2. Categorical features (genre, mood, etc.)
    3. Year information (as bin)
    """
    df = TRACK_DF.copy()
    track_features = {}
    
    for _, row in df.iterrows():
        tid = row["track_id"]
        features = {
            "continuous": {},
            "categorical": {},
            "year_bin": None,
        }
        
        # Track popularity: 0-100 → 0-1
        if "track_popularity" in row and pd.notna(row["track_popularity"]):
            features["continuous"]["track_popularity_norm"] = float(
                np.clip(row["track_popularity"] / 100.0, 0, 1)
            )
        else:
            features["continuous"]["track_popularity_norm"] = 0.5
        
        # Duration: 0-600000ms (10min) → 0-1
        if "duration_ms" in row and pd.notna(row["duration_ms"]):
            features["continuous"]["duration_norm"] = float(
                np.clip(row["duration_ms"] / 600000.0, 0, 1)
            )
        else:
            features["continuous"]["duration_norm"] = 0.5
        
        # Available markets: 0-200 → 0-1
        if "available_markets_count" in row and pd.notna(row["available_markets_count"]):
            features["continuous"]["available_markets_norm"] = float(
                np.clip(row["available_markets_count"] / 200.0, 0, 1)
            )
        else:
            features["continuous"]["available_markets_norm"] = 0.5
        
        for cat_feature, possible_values in CATEGORICAL_PREF_FEATURES.items():
            if cat_feature in row and pd.notna(row[cat_feature]):
                features["categorical"][cat_feature] = str(row[cat_feature]).lower()
            else:
                features["categorical"][cat_feature] = None
        
        if "album_release_year" in row and pd.notna(row["album_release_year"]):
            year = int(row["album_release_year"])
            for bin_name, (low, high) in YEAR_BIN_RANGES.items():
                if low <= year <= high:
                    features["year_bin"] = bin_name
                    break
            if features["year_bin"] is None:
                features["year_bin"] = "2020s"  # default
        else:
            features["year_bin"] = "2010s"  # default
        
        track_features[tid] = features
    
    return track_features


TRACK_FEATURES = precompute_track_features()


def generate_user_preferences() -> Dict:
    """
    Generate extended preference profile for each user.
    
    Includes:
    1. Continuous preferences: popularity, duration, markets (0-1 range)
    2. Categorical preferences: preferred genre, mood sets
    3. Year preference: preferred decades
    
    This gives each user a realistic and specific profile.
    """
    prefs = {
        "continuous": {},
        "categorical": {},
        "year_preferences": {},
    }
    
    # User's preference level for each continuous feature
    for feat in CONTINUOUS_PREF_FEATURES:
        # Beta(2, 2) → mean 0.5, but with variety
        prefs["continuous"][feat] = np.random.beta(2, 2)
    
    # User's preferred values for each categorical feature
    for cat_feature, possible_values in CATEGORICAL_PREF_FEATURES.items():
        # For each categorical feature:
        # - User prefers some values (liked)
        # - Neutral to some (neutral)
        # - Dislikes some (disliked)
        
        n_values = len(possible_values)
        
        if n_values == 2:
            # Binary feature (e.g., acoustic/not_acoustic)
            # 70% chance to prefer one, 30% neutral
            if np.random.rand() < 0.7:
                preferred = np.random.choice(possible_values)
                prefs["categorical"][cat_feature] = {
                    "liked": {preferred},
                    "disliked": set(possible_values) - {preferred},
                    "weight": np.random.uniform(0.6, 1.0),  # Importance of this preference
                }
            else:
                prefs["categorical"][cat_feature] = {
                    "liked": set(),
                    "disliked": set(),
                    "weight": 0.0,  # Doesn't care
                }
        else:
            # Multi-value feature (e.g., genre)
            # Like 1-3, dislike 0-2
            n_liked = np.random.randint(1, min(4, n_values))
            n_disliked = np.random.randint(0, min(3, n_values - n_liked))
            
            shuffled = np.random.permutation(possible_values)
            liked = set(shuffled[:n_liked])
            disliked = set(shuffled[n_liked:n_liked + n_disliked])
            
            prefs["categorical"][cat_feature] = {
                "liked": liked,
                "disliked": disliked,
                "weight": np.random.uniform(0.5, 1.0),
            }
    
    # Each user prefers certain decades
    n_liked_years = np.random.randint(1, 4)
    n_disliked_years = np.random.randint(0, 3)
    
    shuffled_years = np.random.permutation(YEAR_BINS)
    liked_years = set(shuffled_years[:n_liked_years])
    disliked_years = set(shuffled_years[n_liked_years:n_liked_years + n_disliked_years])
    
    prefs["year_preferences"] = {
        "liked": liked_years,
        "disliked": disliked_years,
        "weight": np.random.uniform(0.4, 0.9),  # Importance of year preference
    }
    
    return prefs


def compute_user_track_match(user_prefs: Dict, track_id: str) -> float:
    """
    Compute match score between user preferences and track features.
    
    Extended formula:
    match_score = weighted_average(
        continuous_match,    # Popularity, duration, etc.
        categorical_match,   # Genre, mood, etc.
        year_match,          # Decade preference
    )
    
    Each component is 0-1, weighted average is taken.
    """
    track_feats = TRACK_FEATURES.get(track_id, None)
    if track_feats is None:
        return 0.5  # Default medium match
    
    match_scores = []
    weights = []
    
    continuous_diffs = []
    for feat in CONTINUOUS_PREF_FEATURES:
        user_val = user_prefs["continuous"].get(feat, 0.5)
        track_val = track_feats["continuous"].get(feat, 0.5)
        diff = abs(user_val - track_val)
        continuous_diffs.append(diff)
    
    if continuous_diffs:
        continuous_match = 1.0 - np.mean(continuous_diffs)
        match_scores.append(continuous_match)
        weights.append(0.2)  # Weight for continuous features
    
    categorical_scores = []
    categorical_weights = []
    
    for cat_feature, pref_info in user_prefs["categorical"].items():
        if pref_info["weight"] == 0:
            continue  # User doesn't care about this feature
        
        track_val = track_feats["categorical"].get(cat_feature)
        if track_val is None:
            continue
        
        # Match score: liked=1, neutral=0.5, disliked=0
        if track_val in pref_info["liked"]:
            score = 1.0
        elif track_val in pref_info["disliked"]:
            score = 0.0
        else:
            score = 0.5  # Neutral
        
        categorical_scores.append(score)
        categorical_weights.append(pref_info["weight"])
    
    if categorical_scores:
        # Weighted average
        total_weight = sum(categorical_weights)
        categorical_match = sum(
            s * w for s, w in zip(categorical_scores, categorical_weights)
        ) / total_weight
        match_scores.append(categorical_match)
        weights.append(0.5)  # Weight for categorical features (most important!)
    
    year_pref = user_prefs["year_preferences"]
    track_year = track_feats["year_bin"]
    
    if year_pref["weight"] > 0 and track_year:
        if track_year in year_pref["liked"]:
            year_match = 1.0
        elif track_year in year_pref["disliked"]:
            year_match = 0.0
        else:
            year_match = 0.5
        
        match_scores.append(year_match)
        weights.append(0.3 * year_pref["weight"])  # Weight for year preference
    
    if not match_scores:
        return 0.5
    
    total_weight = sum(weights)
    final_match = sum(s * w for s, w in zip(match_scores, weights)) / total_weight
    
    return float(final_match)


def sample_user_patience(alpha: float, beta: float, max_t: int = 50) -> int:
    """
    Sample user patience (Tu) from Beta-Geometric distribution (Part 2).
    
    Beta-Geometric: User's success probability p ~ Beta(alpha, beta)
    Tu = number of trials until first success (Geometric(p))
    
    Here we interpret this as "patience limit":
    - User leaves session if no 5★ found after Tu trials
    
    P(T=t) = Beta(alpha+1, beta+t-1) / Beta(alpha, beta)
    """
    # Compute CDF and do inverse transform sampling
    u = np.random.rand()
    cumulative = 0.0
    
    for t in range(1, max_t + 1):
        # P(T=t) = exp(betaln(a+1, b+t-1) - betaln(a, b))
        log_pmf = betaln(alpha + 1, beta + t - 1) - betaln(alpha, beta)
        pmf = np.exp(log_pmf)
        cumulative += pmf
        
        if u <= cumulative:
            return t
    
    return max_t


class UserProfile:
    """
    A user's personal profile.
    Learned from 20-song warm-up phase.
    """
    def __init__(self):
        # True (hidden) preferences - for simulation
        self.true_preferences = generate_user_preferences()
        self.true_p_user = np.random.beta(ALPHA, BETA)  # Overall liking level
        
        # Warm-up data
        self.learned_patience = None  # For churn simulation
        self.warmup_ratings = []
        self.warmup_tracks = []
    
    def rate_track(self, track_id: str) -> int:
        """
        Rate a track based on user's true preferences.
        
        P(5★) CALCULATION - GLOBAL + PERSONAL COMBINATION:
        
        1. GLOBAL FACTOR (from Part 1):
           - base_p5: Track's general liking probability
           - Same for all users (track quality)
        
        2. PERSONAL FACTOR:
           - match_score: User-track feature match
           - p_user: User's general criticalness level (Part 2)
        
        FORMULA:
           P(5★) = GLOBAL_WEIGHT * base_p5 + PERSONAL_WEIGHT * (base_p5 * match_score)
           Then modulate with p_user
        
        This way:
        - Quality tracks (high base_p5) are liked by everyone
        - But personal match also plays an important role
        """
        # GLOBAL: Track's P(5★) value from Part 1
        base_p5 = TRACK_P5.get(track_id, 0.1)
        
        # PERSONAL: User-track feature match [0, 1]
        match_score = compute_user_track_match(self.true_preferences, track_id)
        
        # COMBINED SCORE:
        # - Global part: base_p5 (track quality)
        # - Personal part: base_p5 * match_score (quality × match)
        global_component = base_p5
        personal_component = base_p5 * (0.5 + 0.5 * match_score)  # [0.5*base, base]
        
        combined_p5 = GLOBAL_WEIGHT * global_component + PERSONAL_WEIGHT * personal_component
        
        # Modulate with user's general criticalness (Part 2 connection)
        # Higher p_user = user likes more easily
        eff_p5 = combined_p5 * (self.true_p_user / EXPECTED_P)
        eff_p5 = float(np.clip(eff_p5, 0.02, 0.85))
        
        # Rating
        u = np.random.rand()
        if u < eff_p5:
            return 5
        else:
            # Non-5★ ratings: both global and personal affect
            # High match + high quality → closer to 4
            # Low match or low quality → closer to 1-2
            quality_match = (base_p5 + match_score) / 2  # 0-1 range
            
            if quality_match > 0.55:
                return np.random.choice([1, 2, 3, 4], p=[0.05, 0.15, 0.40, 0.40])
            elif quality_match > 0.4:
                return np.random.choice([1, 2, 3, 4], p=[0.10, 0.25, 0.40, 0.25])
            else:
                return np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
    
    def do_warmup(self, warmup_track_ids: List[str]) -> None:
        """
        Perform 20-song warm-up phase.
        Learn user's preferences and patience.
        """
        self.warmup_tracks = warmup_track_ids
        self.warmup_ratings = []
        
        first_five_idx = None
        
        for idx, tid in enumerate(warmup_track_ids):
            rating = self.rate_track(tid)
            self.warmup_ratings.append(rating)
            
            if first_five_idx is None and rating == 5:
                first_five_idx = idx + 1  # 1-indexed
        
        # Patience learning: Bayesian approach (compatible with Part 2)
        # Use all warm-up data instead of single observation
        n_fives_warmup = sum(1 for r in self.warmup_ratings if r == 5)
        n_total_warmup = len(self.warmup_ratings)
        
        # Bayesian posterior: p | data ~ Beta(ALPHA + n_fives, BETA + n_not_fives)
        alpha_post = ALPHA + n_fives_warmup
        beta_post = BETA + (n_total_warmup - n_fives_warmup)
        
        # Expected Tu from posterior mean
        estimated_p = alpha_post / (alpha_post + beta_post)
        expected_Tu = 1.0 / estimated_p
        
        # Set patience limit based on this estimate
        # Integer version of Tu + some tolerance
        self.learned_patience = max(2, int(np.ceil(expected_Tu)))
    
    def get_song_ratings_for_model(self) -> List[dict]:
        """
        Return warm-up ratings in format for model input.
        """
        return [
            {"track_id": tid, "rating": rating}
            for tid, rating in zip(self.warmup_tracks, self.warmup_ratings)
        ]


def get_random_warmup_tracks(n: int = WARMUP_SIZE) -> List[str]:
    """
    Select n random tracks for warm-up.
    """
    all_track_ids = list(TRACK_FEATURES.keys())
    return list(np.random.choice(all_track_ids, size=n, replace=False))


def simulate_personalized_test(
    user: UserProfile,
    track_ids: List[str],
    topk: int = TOPK,
) -> Tuple[List[int], bool, float, bool]:
    """
    Test model recommendations with learned user profile.
    
    Uses user's TRUE preferences (for simulation),
    but patience is LEARNED from warm-up.
    
    Returns:
        ratings: Given ratings
        hit: At least one 5★?
        time_to_5: Position of first 5★
        churned: Did patience run out?
    """
    # Use learned patience limit
    patience_limit = min(user.learned_patience, len(track_ids))
    
    ratings = []
    time_to_5 = np.nan
    hit = False
    churned = False
    
    for idx, tid in enumerate(track_ids, start=1):
        # If patience ran out and no 5★ found -> churn
        if (not hit) and idx > patience_limit:
            churned = True
            break
        
        # Rate based on user's true preferences
        rating = user.rate_track(tid)
        ratings.append(rating)
        
        if (not hit) and rating == 5:
            hit = True
            time_to_5 = float(idx)
    
    return ratings, hit, time_to_5, churned


def run_personalized_monte_carlo(
    model_class,
    n_users: int,
    topk: int,
) -> dict:
    """
    Personalized Monte Carlo evaluation.
    
    For each user:
    1. Learn preferences and patience from 20-song warm-up
    2. Get model recommendations with learned profile
    3. Evaluate with true preferences
    
    This approach:
    - Tests on PERSONAL basis, not global
    - Uses each user's own patience limit
    - Measures model's personalization ability
    """
    hits = []
    avg_ratings = []
    times = []
    churns = []
    learned_patiences = []
    
    for i in range(n_users):
        # 1. Create user profile
        user = UserProfile()
        
        # 2. Warm-up phase
        warmup_tracks = get_random_warmup_tracks(WARMUP_SIZE)
        user.do_warmup(warmup_tracks)
        learned_patiences.append(user.learned_patience)
        
        # 3. Get recommendations from model (with warm-up ratings)
        model = model_class()
        song_ratings = user.get_song_ratings_for_model()
        recs = model.query(song_ratings, topk=topk)
        
        # Remove warm-up tracks from recommendations
        warmup_set = set(warmup_tracks)
        filtered_recs = [(tid, name) for tid, name in recs if tid not in warmup_set]
        
        # If not enough recommendations, use unfiltered
        if len(filtered_recs) < topk:
            track_ids = [tid for tid, name in recs[:topk]]
        else:
            track_ids = [tid for tid, name in filtered_recs[:topk]]
        
        # 4. Test
        ratings, hit, time_to_5, churned = simulate_personalized_test(
            user, track_ids, topk
        )
        
        hits.append(1.0 if hit else 0.0)
        avg_ratings.append(float(np.mean(ratings)) if ratings else 0.0)
        times.append(time_to_5)
        churns.append(1.0 if churned else 0.0)
    
    # Compute metrics
    hits = np.array(hits)
    avg_ratings = np.array(avg_ratings)
    times = np.array(times)
    churns = np.array(churns)
    learned_patiences = np.array(learned_patiences)
    
    churn_rate = churns.mean()
    satisfaction_rate = hits.mean() * (1 - churn_rate)
    
    return {
        "hit_at_k": hits,
        "avg_rating": avg_ratings,
        "time_to_5": times,
        "churns": churns,
        "churn_rate": churn_rate,
        "satisfaction_rate": satisfaction_rate,
        "avg_learned_patience": learned_patiences.mean(),
    }


def mean_and_ci_diff(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    ignore_nan: bool = False,
) -> Tuple[float, float, float]:
    """
    Compute 95% (or 1-alpha) confidence interval for the difference
    between two model metric means using normal approximation.

    diff = mean(a) - mean(b)

    If ignore_nan=True, NaNs are ignored (for time_to_5).
    """
    if ignore_nan:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

    n_a = len(a)
    n_b = len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, 0.0, 0.0
        
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1) if n_a > 1 else 0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0

    diff = mean_a - mean_b
    se = np.sqrt(var_a / n_a + var_b / n_b)

    z = 1.96  # approximately 95% CI
    lower = diff - z * se
    upper = diff + z * se

    return diff, lower, upper


def print_comparison(
    results1: dict,
    results2: dict,
    model1_name: str = "Model1",
    model2_name: str = "Model2",
    k: int = TOPK,
    n_users: int = N_USERS,
):
    """
    Compare Monte Carlo results.
    Metrics: Hit@k, Average Rating, Time-to-5★
    """
    hit1, avg1, t1 = results1["hit_at_k"], results1["avg_rating"], results1["time_to_5"]
    hit2, avg2, t2 = results2["hit_at_k"], results2["avg_rating"], results2["time_to_5"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<20} {model1_name:<15} {model2_name:<15}")
    print("-" * 60)
    print(f"{'Hit@' + str(k):<20} {hit1.mean():<15.3f} {hit2.mean():<15.3f}")
    print(f"{'Average Rating':<20} {avg1.mean():<15.3f} {avg2.mean():<15.3f}")
    print(f"{'Time-to-5★ (mean)':<20} {np.nanmean(t1):<15.3f} {np.nanmean(t2):<15.3f}")

    # Confidence intervals
    print("\n" + "-" * 60)
    print(f"DIFFERENCES ({model1_name} - {model2_name}) + 95% CI")
    print("-" * 60)

    diff_hit, l_hit, u_hit = mean_and_ci_diff(hit1, hit2)
    sig_hit = "*" if l_hit > 0 or u_hit < 0 else ""
    print(f"  Hit@{k}:         {diff_hit:+.4f}  [{l_hit:+.4f}, {u_hit:+.4f}] {sig_hit}")

    diff_avg, l_avg, u_avg = mean_and_ci_diff(avg1, avg2)
    sig_avg = "*" if l_avg > 0 or u_avg < 0 else ""
    print(f"  Avg Rating:     {diff_avg:+.4f}  [{l_avg:+.4f}, {u_avg:+.4f}] {sig_avg}")

    diff_t, l_t, u_t = mean_and_ci_diff(t1, t2, ignore_nan=True)
    sig_t = "*" if l_t > 0 or u_t < 0 else ""
    print(f"  Time-to-5★:     {diff_t:+.4f}  [{l_t:+.4f}, {u_t:+.4f}] {sig_t}")
    
    print("\n  * = Statistically significant (95% CI)")
    print("=" * 60)


def main():
    print("=" * 60)
    print("MONTE CARLO SIMULATION")
    print("=" * 60)
    print(f"Users: {N_USERS}, Warm-up: {WARMUP_SIZE}, Top-k: {TOPK}")

    print("\n>>> Running Model 1...")
    results_m1 = run_personalized_monte_carlo(Model1, N_USERS, TOPK)

    print(">>> Running Model 2...")
    results_m2 = run_personalized_monte_carlo(Model2, N_USERS, TOPK)

    # Comparison
    print_comparison(
        results_m1, results_m2,
        model1_name="Model1",
        model2_name="Model2",
        k=TOPK,
        n_users=N_USERS
    )


if __name__ == "__main__":
    main()