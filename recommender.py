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
# 6. MODEL 2 — Utility-Based Sampling with Patience Model (Part 2)
# ================================================================
class Model2:
    """
    Part 2 Odaklı Model: Beta-Geometric Sabır Modeli + Utility Sampling
    
    Part 2'nin Temel Konseptleri:
    1. Tu = Kullanıcının 5★ vermesi için gereken öneri sayısı
    2. Her kullanıcının kendi p'si var: p ~ Beta(α, β)
    3. Sabırlı kullanıcılar (yüksek Tu) → exploration yapılabilir
    4. Sabırsız kullanıcılar (düşük Tu) → exploitation önemli
    
    Strateji:
    - Kullanıcının warm-up verisinden sabır tipini tahmin et
    - Sabırlı kullanıcılara riskli/keşif önerileri sun
    - Sabırsız kullanıcılara güvenli/yüksek P(5★) öneriler sun
    """
    
    def __init__(self):
        self.tracks = TRACK_DF
    
    def _estimate_user_patience_type(self, song_ratings):
        """
        Part 2'deki Beta-Geometric modeline göre kullanıcının sabır tipini tahmin et.
        
        Warm-up verisinden:
        - n_fives / n_total → kullanıcının hit rate'i
        - Bu oran düşükse → seçici/sabırlı kullanıcı (düşük p, yüksek E[Tu])
        - Bu oran yüksekse → kolay beğenen/sabırsız kullanıcı (yüksek p, düşük E[Tu])
        
        Returns:
            estimated_p: Kullanıcının tahmini hit olasılığı
            expected_Tu: Beklenen Time-to-5★
            patience_type: "patient", "normal", "impatient"
        """
        if not song_ratings:
            # Veri yoksa prior kullan
            return EXPECTED_P, 1.0 / EXPECTED_P, "normal"
        
        n_total = len(song_ratings)
        n_fives = sum(1 for s in song_ratings if s["rating"] == 5)
        
        # Bayesian güncelleme: Posterior Beta(α', β')
        # Prior: Beta(ALPHA, BETA) → Part 2'den
        # Likelihood: n_fives başarı, n_total - n_fives başarısızlık
        alpha_post = ALPHA + n_fives
        beta_post = BETA + (n_total - n_fives)
        
        # Posterior ortalama: E[p | data]
        estimated_p = alpha_post / (alpha_post + beta_post)
        
        # Beklenen Tu (Geometric): E[Tu] = 1/p
        expected_Tu = 1.0 / estimated_p
        
        # Sabır tipi sınıflandırma
        if estimated_p < 0.25:
            patience_type = "patient"      # Seçici, sabırlı
        elif estimated_p > 0.45:
            patience_type = "impatient"    # Kolay beğenen, sabırsız
        else:
            patience_type = "normal"
        
        return estimated_p, expected_Tu, patience_type
    
    def _compute_utility(self, track, estimated_p, patience_type):
        """
        Part 2 odaklı utility hesabı.
        
        Utility formülü:
            U(track) = P(5★|track)^γ
        
        γ (gamma) sabır tipine göre ayarlanır:
        - Sabırlı kullanıcı (γ < 1): Düşük P(5★) şarkılar da şans alır (exploration)
        - Sabırsız kullanıcı (γ > 1): Yüksek P(5★) şarkılar tercih edilir (exploitation)
        """
        # Global P(5★) - Part 1'den
        p_five = compute_global_probability(track)
        
        # Gamma ayarı: sabır tipine göre
        if patience_type == "patient":
            # Sabırlı kullanıcı: exploration yapılabilir
            # Düşük P şarkılar da şans alır
            gamma = 0.5  # P^0.5 → düzleştirme, daha uniform
        elif patience_type == "impatient":
            # Sabırsız kullanıcı: hızlı hit lazım
            # Yüksek P şarkılar öne çıkar
            gamma = 2.0  # P^2 → keskinleştirme, top şarkılar öne
        else:
            # Normal kullanıcı
            gamma = 1.0  # P olduğu gibi
        
        utility = p_five ** gamma
        
        return utility, p_five
    
    def query(self, song_ratings, topk=5):
        """
        Utility-Based Sampling: Part 2 entegreli öneri.
        
        Algoritma:
        1. Kullanıcının sabır tipini tahmin et (Beta-Geometric posterior)
        2. Her şarkı için utility hesapla (sabır tipine göre γ ayarlı)
        3. Utility'ye orantılı olasılıkla sample et
        4. Sabırlı kullanıcılara daha çeşitli, sabırsız kullanıcılara daha güvenli öner
        """
        # 1. Sabır tipi tahmini (Part 2)
        estimated_p, expected_Tu, patience_type = self._estimate_user_patience_type(song_ratings)
        
        # 2. Zaten dinlenmiş şarkıları çıkar
        rated_ids = set(s["track_id"] for s in song_ratings)
        df = self.tracks[~self.tracks["track_id"].isin(rated_ids)].copy()
        
        if len(df) == 0:
            df = self.tracks.copy()
        
        # 3. Her şarkı için utility hesapla
        utilities = []
        p_fives = []
        
        for _, track in df.iterrows():
            util, p5 = self._compute_utility(track, estimated_p, patience_type)
            utilities.append(util)
            p_fives.append(p5)
        
        df["utility"] = utilities
        df["p_five"] = p_fives
        
        # 4. Sampling stratejisi: sabır tipine göre
        if patience_type == "patient":
            # Sabırlı kullanıcı: Utility-proportional sampling (exploration)
            # Daha çeşitli öneriler, düşük P şarkılar da şans alır
            weights = df["utility"].values
            weights = weights / weights.sum()
            
            sample_size = min(topk * 4, len(df))
            sampled_idx = np.random.choice(
                len(df), 
                size=sample_size, 
                replace=False, 
                p=weights
            )
            sampled = df.iloc[sampled_idx]
            # Sample içinden en yüksek utility olanları seç
            recs = sampled.nlargest(topk, "utility")
            
        elif patience_type == "impatient":
            # Sabırsız kullanıcı: Top utility (exploitation)
            # En yüksek P(5★) şarkıları doğrudan öner
            recs = df.nlargest(topk, "utility")
            
        else:
            # Normal kullanıcı: Hibrit yaklaşım
            # %60 exploitation + %40 exploration
            n_exploit = max(1, int(topk * 0.6))
            n_explore = topk - n_exploit
            
            # Top şarkılar
            top_recs = df.nlargest(n_exploit, "utility")
            
            # Exploration: kalan şarkılardan utility-weighted sampling
            remaining = df[~df["track_id"].isin(top_recs["track_id"])]
            if len(remaining) > 0 and n_explore > 0:
                weights = remaining["utility"].values
                weights = weights / weights.sum()
                explore_idx = np.random.choice(
                    len(remaining),
                    size=min(n_explore, len(remaining)),
                    replace=False,
                    p=weights
                )
                explore_recs = remaining.iloc[explore_idx]
                recs = pd.concat([top_recs, explore_recs])
            else:
                recs = top_recs.head(topk)
        
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



