# ================================================================
# recommender.py — CMPE343 Tune Duel Recommender
# Two Models:
#   (1) Conditional Filtering (Deterministic)
#   (2) Beta-Geometric Utility Sampling (Probabilistic)
# ================================================================

import numpy as np
import pandas as pd
import random

# ================================================================
# 1. TRACKS VERİSİNİ YÜKLE + BİNLERİ OLUŞTUR
# ================================================================
TRACK_DF = pd.read_csv("tracks.csv").fillna("unknown")


def to_bool(x):
    """explicit kolonunu düzgün booleana çevirmek için."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in ["true", "1", "yes", "y", "t"]
    return False


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- explicit_bool ----
    if "explicit" in df.columns:
        df["explicit_bool"] = df["explicit"].apply(to_bool)
    else:
        df["explicit_bool"] = False

    # ---- year_bin (album_release_year) ----
    def year_to_bin(y):
        try:
            y = int(y)
        except Exception:
            return "unknown"
        if y < 1980:
            return "pre_1980"
        elif y < 1990:
            return "1980s"
        elif y < 2000:
            return "1990s"
        elif y < 2010:
            return "2000s"
        elif y < 2020:
            return "2010s"
        else:
            return "2020s"

    if "album_release_year" in df.columns:
        df["year_bin"] = df["album_release_year"].apply(year_to_bin)
    else:
        df["year_bin"] = "unknown"

    # ---- popularity_bin (track_popularity 0–100) ----
    def pop_to_bin(p):
        try:
            p = float(p)
        except Exception:
            return "low"
        if p >= 75:
            return "very_high"
        elif p >= 50:
            return "high"
        elif p >= 25:
            return "low"
        else:
            return "very_low"

    if "track_popularity" in df.columns:
        df["popularity_bin"] = df["track_popularity"].apply(pop_to_bin)
    else:
        df["popularity_bin"] = "low"

    # ---- duration_bin (duration_ms) ----
    def dur_to_bin(ms):
        try:
            s = float(ms) / 1000.0
        except Exception:
            return "medium"
        if s < 150:      # < 2.5 dakika
            return "short"
        elif s < 270:    # 2.5–4.5 dakika
            return "medium"
        else:
            return "long"

    if "duration_ms" in df.columns:
        df["duration_bin"] = df["duration_ms"].apply(dur_to_bin)
    else:
        df["duration_bin"] = "medium"

    # ---- markets_bin (available_markets_count) ----
    if "available_markets_count" in df.columns:
        median_m = df["available_markets_count"].median()
        def markets_bin_fun(m):
            try:
                m = float(m)
            except Exception:
                return "many"
            return "few" if m < median_m else "many"
        df["markets_bin"] = df["available_markets_count"].apply(markets_bin_fun)
    else:
        df["markets_bin"] = "many"

    return df


TRACK_DF = add_bins(TRACK_DF)

# ================================================================
# 2. PART 1'DEN OLASILIKLAR
# ================================================================
P_explicit = {False: 0.392155, True: 0.082317}

P_year_bin = {
    "2000s": 0.471910,
    "2010s": 0.365425,
    "1990s": 0.210654,
    "2020s": 0.155361,
    "1980s": 0.068670,
    "pre_1980": 0.039711,
}

P_popularity_bin = {
    "very_high": 0.446358,
    "high": 0.390074,
    "low": 0.320666,
    "very_low": 0.203896,
}

P_duration_bin = {
    "short": 0.220986,
    "medium": 0.356762,
    "long": 0.303730,
}

P_markets_bin = {
    "few": 0.355663,
    "many": 0.320277,
}

P_feature = {
    "danceable": 0.318484,
    "not_danceable": 0.371376,
    "acoustic": 0.365523,
    "not_acoustic": 0.324938,
    "aggressive": 0.291925,
    "not_aggressive": 0.335862,
    "electronic": 0.322233,
    "not_electronic": 0.353442,
    "happy": 0.349250,
    "not_happy": 0.325743,
    "party": 0.300887,
    "not_party": 0.349962,
    "relaxed": 0.337026,
    "not_relaxed": 0.323020,
    "sad": 0.326766,
    "not_sad": 0.334915,
    "female": 0.334846,
    "male": 0.328061,
    "instrumental": 0.354535,
    "voice": 0.321097,
    "bright": 0.351262,
    "dark": 0.312401,
}

P_genre_dortmund = {
    "alternative": 0.477273,
    "electronic": 0.334612,
    "folkcountry": 0.300613,
    "rock": 0.239130,
    "jazz": 0.200000,
    "raphiphop": 0.076923,
    "blues": 0.066667,
}

P_genre_ros = {
    "cla": 0.404580,
    "pop": 0.380163,
    "roc": 0.346552,
    "dan": 0.337972,
    "jaz": 0.320000,
    "rhy": 0.291062,
    "hip": 0.278988,
}

# ================================================================
# 3. PART 2 PARAMETRELERİ (BETA-GEOMETRIC)
# ================================================================
ALPHA = 3.0535
BETA = 5.6081
EXPECTED_P = ALPHA / (ALPHA + BETA)   # ≈ 0.3525


# ================================================================
# 4. Yardımcı: P(5★ | track features) Hesabı
# ================================================================
def compute_global_probability(track: pd.Series) -> float:
    """
    Part 1'deki feature bazlı P(5★) tahminlerini çarpıp
    tek bir global skor üretir.
    """
    score = 1.0

    # explicit
    explicit_val = track.get("explicit_bool", False)
    score *= P_explicit.get(explicit_val, 0.1)

    # year_bin
    yb = track.get("year_bin", "2000s")
    score *= P_year_bin.get(yb, 0.2)

    # popularity_bin
    pb = track.get("popularity_bin", "low")
    score *= P_popularity_bin.get(pb, 0.3)

    # duration_bin
    db = track.get("duration_bin", "medium")
    score *= P_duration_bin.get(db, 0.3)

    # markets_bin
    mb = track.get("markets_bin", "many")
    score *= P_markets_bin.get(mb, 0.3)

    # mood / genre kolonları
    for col in track.index:
        val = track[col]
        if val in P_feature:
            score *= P_feature[val]
        if val in P_genre_dortmund:
            score *= P_genre_dortmund[val]
        if val in P_genre_ros:
            score *= P_genre_ros[val]

    return float(score)


# ================================================================
# 5. MODEL 1 — Conditional Filtering (Deterministic)
# ================================================================
class Model1:
    def __init__(self):
        self.tracks = TRACK_DF

    def query(self, song_ratings, topk=5):
        # kullanıcının beğendikleri (>=4)
        liked_ids = [s["track_id"] for s in song_ratings if s["rating"] >= 4]
        liked_tracks = self.tracks[self.tracks["track_id"].isin(liked_ids)]

        # hiç beğenilen yoksa: global top P(5★)
        if liked_tracks.empty:
            df = self.tracks.copy()
            df["prob"] = df.apply(compute_global_probability, axis=1)
            out = df.sort_values("prob", ascending=False).head(topk)
            return list(zip(out["track_id"], out["track_name"]))

        liked_genres = liked_tracks["ab_genre_rosamerica_value"].value_counts()
        liked_artists = liked_tracks["primary_artist_name"].value_counts()

        df = self.tracks.copy()
        df["score"] = 0.0

        for i, tr in df.iterrows():
            score = compute_global_probability(tr)

            if tr["primary_artist_name"] in liked_artists.index:
                score *= 1.5
            if tr["ab_genre_rosamerica_value"] in liked_genres.index:
                score *= 1.3

            df.at[i, "score"] = score

        recs = df.sort_values("score", ascending=False).head(topk)
        return list(zip(recs["track_id"], recs["track_name"]))


# ================================================================
# 6. MODEL 2 — Utility Sampling (Probabilistic)
# ================================================================
class Model2:
    def __init__(self):
        self.tracks = TRACK_DF

    def query(self, song_ratings, topk=5):
        liked = [s for s in song_ratings if s["rating"] == 5]

        # kaba "sabır" tahmini (Part 2 ile bağlantı)
        patience = 1 + len(song_ratings) - (len(liked) if liked else 1)
        patience = max(1, min(10, patience))

        df = self.tracks.copy()
        df["prob"] = df.apply(compute_global_probability, axis=1)
        df["utility"] = df["prob"].apply(lambda p: p ** (1.0 / patience))

        w = df["utility"].values
        w = w / w.sum()

        # bir miktar örnekle, sonra en yüksek utility olanları al
        sample_size = min(50, len(df))
        sampled_idx = np.random.choice(len(df), size=sample_size, replace=False, p=w)
        sampled = df.iloc[sampled_idx]

        recs = sampled.sort_values("utility", ascending=False).head(topk)
        return list(zip(recs["track_id"], recs["track_name"]))


# ================================================================
# 7. Tune Duel entry point (tek fonksiyonlu arayüz)
# ================================================================
def query(song_ratings, topk=5):
    """
    TuneDuel doğrudan bunu çağırırsa default olarak Model2 kullanılacak.
    """
    model = Model2()
    return model.query(song_ratings, topk)



