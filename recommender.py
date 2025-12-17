# ================================================================
# recommender.py — CMPE343 Tune Duel Recommender
# Two Models:
#   (1) Conditional Filtering (Deterministic)
#   (2) Beta-Geometric Utility Sampling (Probabilistic)
# ================================================================

import numpy as np
import pandas as pd
import random

import os

ART_DIR = "artifacts"  # repo’da bu klasörü koyacaksın

def load_cond_probs(path):
    df = pd.read_csv(path)
    # key: (feature, value_str) -> prob
    prob_map = {}
    for _, r in df.iterrows():
        feat = str(r["feature"])
        val = str(r["value"])
        prob_map[(feat, val)] = float(r["prob"])
    return prob_map

# Global probs
P_GLOBAL = load_cond_probs(os.path.join(ART_DIR, "cond_probs_global.csv"))

# Personal probs (opsiyonel)
P_PERSONAL = None
personal_path = os.path.join(ART_DIR, "cond_probs_personal.csv")
if os.path.exists(personal_path):
    P_PERSONAL = load_cond_probs(personal_path)

def get_prob(feature, value, default=0.2):
    """Global fallback ile P(5★ | feature=value)"""
    return P_GLOBAL.get((str(feature), str(value)), default)

def get_blended_prob(feature, value, w_personal=0.35, default=0.2):
    """
    Personal varsa: P = w*Ppersonal + (1-w)*Pglobal
    (w'yu istersen rating sayısına göre dinamik yaparsın)
    """
    pg = get_prob(feature, value, default=default)
    if P_PERSONAL is None:
        return pg
    pp = P_PERSONAL.get((str(feature), str(value)), pg)
    return w_personal * pp + (1 - w_personal) * pg


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
    # Part 1'de qcut ile 4 eşit parça oluşturuluyor
    if "track_popularity" in df.columns:
        try:
            df["popularity_bin"] = pd.qcut(
                df["track_popularity"], q=4,
                labels=["very_low", "low", "high", "very_high"],
                duplicates="drop"
            )
        except:
            # Fallback: sabit eşikler
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
            df["popularity_bin"] = df["track_popularity"].apply(pop_to_bin)
    else:
        df["popularity_bin"] = "low"

    # ---- duration_bin (duration_ms) ----
    # Part 1 ile uyumlu: 180s (3dk) ve 300s (5dk) eşikleri
    def dur_to_bin(ms):
        try:
            s = float(ms) / 1000.0
        except Exception:
            return "medium"
        if s < 180:      # < 3 dakika
            return "short"
        elif s < 300:    # 3–5 dakika
            return "medium"
        else:
            return "long"

    if "duration_ms" in df.columns:
        df["duration_bin"] = df["duration_ms"].apply(dur_to_bin)
    else:
        df["duration_bin"] = "medium"

    # ---- markets_bin (available_markets_count) ----
    # Part 1'de qcut ile 3 kategori oluşturuluyor: few, medium, many
    if "available_markets_count" in df.columns:
        try:
            df["markets_bin"] = pd.qcut(
                df["available_markets_count"], q=3,
                labels=["few", "medium", "many"],
                duplicates="drop"
            )
        except:
            # Eğer qcut başarısız olursa median-based fallback
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
# 3. PART 2 PARAMETRELERİ (BETA-GEOMETRIC)
# ================================================================
ALPHA = 4.3379 #3.0535
BETA = 6.9353 #5.6081
EXPECTED_P = ALPHA / (ALPHA + BETA)   # ≈ 0.3525


# ================================================================
# 4. Yardımcı: P(5★ | track features) Hesabı
# ================================================================
def compute_global_probability(track: pd.Series) -> float:
    score = 1.0

    # ---- Part1’de export ettiğin feature isimleriyle aynı olmalı ----
    # Eğer Part1 scriptinde explicit kolonunu "explicit" diye export ettiysen:
    # burada feature adını "explicit" kullan.
    # Ben recommender’daki explicit_bool’u kullanıyorum ama alt satırda fallback var.

    # explicit
    explicit_val = track.get("explicit_bool", False)
    # CSV’de feature adı "explicit" ise:
    p_exp = get_prob("explicit", explicit_val, default=0.2)
    # CSV’de feature adı "explicit_bool" ise:
    p_exp2 = get_prob("explicit_bool", explicit_val, default=p_exp)
    score *= p_exp2

    # year_bin, popularity_bin, duration_bin, markets_bin
    score *= get_prob("year_bin", track.get("year_bin", "unknown"), default=0.2)
    score *= get_prob("popularity_bin", track.get("popularity_bin", "low"), default=0.2)
    score *= get_prob("duration_bin", track.get("duration_bin", "medium"), default=0.2)
    score *= get_prob("markets_bin", track.get("markets_bin", "many"), default=0.2)

    # ---- ab_ feature’lar (genre/mood/timbre vs) ----
    # Burada “çok fazla kolon” çarpmak skoru aşırı küçültebilir.
    # O yüzden en etkili gördüklerini seç (senin eski dict’lerinden).
    ab_cols = [
        "ab_genre_rosamerica_value",
        "ab_genre_dortmund_value",
        "ab_timbre_value",
        "ab_danceability_value",
        "ab_acousticness_value",
        "ab_aggressiveness_value",
        "ab_electronic_value",
        "ab_mood_happy_value",
        "ab_mood_party_value",
        "ab_mood_relaxed_value",
        "ab_mood_sad_value",
        "ab_gender_value",
        "ab_voice_value",
    ]

    for c in ab_cols:
        if c in track.index:
            score *= get_prob(c, track.get(c, "unknown"), default=0.2)

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
# 6. MODEL 2 — Advanced Combined Model (Global + Personal + Patience)
# ================================================================
class Model2:
    """
    Gelişmiş Birleşik Model: Part 1 + Part 2 + Kişiselleştirme

    Model 1'in yaptığı her şeyi yapar + Part 2'nin sabır modelini ekler.

    Bileşenler:
    1. GLOBAL: Part 1'den feature-based P(5★) hesaplaması
    2. PERSONAL: Genre/artist bonusları (Model 1 gibi)
    3. PATIENCE: Beta-Geometric sabır modeli ile exploration/exploitation dengesi

    Formül:
    Score = Global_P(5★) × Genre_Bonus × Artist_Bonus × Patience_Factor

    Sabır Modeli (Part 2):
    - Tu = Kullanıcının 5★ şarkı bulması için beklenen öneri sayısı
    - p ~ Beta(α, β) → Kullanıcının 5★ verme olasılığı
    - Sabırlı (düşük p, yüksek Tu): exploration bonus
    - Sabırsız (yüksek p, düşük Tu): exploitation bonus
    """

    def __init__(self):
        self.tracks = TRACK_DF

    def _estimate_user_patience(self, song_ratings):
        """
        Part 2: Beta-Geometric modeline göre kullanıcının p değerini ve Tu'yu tahmin et.

        Bayesian Update:
        Prior: p ~ Beta(ALPHA, BETA)
        Likelihood: X ~ Bernoulli(p) for each 5★ rating
        Posterior: p | data ~ Beta(ALPHA + n_fives, BETA + n_not_fives)

        Returns: (estimated_p, expected_Tu)
        """
        if not song_ratings:
            return EXPECTED_P, 1.0 / EXPECTED_P

        n_total = len(song_ratings)
        n_fives = sum(1 for s in song_ratings if s["rating"] == 5)

        # Bayesian posterior update
        alpha_post = ALPHA + n_fives
        beta_post = BETA + (n_total - n_fives)

        # Posterior mean: E[p | data]
        estimated_p = alpha_post / (alpha_post + beta_post)

        # Expected Tu (Geometric): E[Tu] = 1/p
        expected_Tu = 1.0 / estimated_p

        return estimated_p, expected_Tu

    def _compute_patience_factor(self, base_score, estimated_p, expected_Tu):
        """
        Part 2: Sabır modeline göre exploration/exploitation faktörü.

        Sabır Tipi Belirleme (Tu bazlı):
        - Tu < 2.5 → Sabırsız (çok seçici değil, hızlı beğenir)
        - Tu > 4.0 → Sabırlı (seçici, uzun süre arar)
        - Arada → Normal

        Strateji:
        - Sabırsız kullanıcı: Zaten çabuk beğeniyor, exploration şansı ver
          → Yüksek skorlara hafif penalty, orta skorlara bonus
        - Sabırlı kullanıcı: Seçici, güvenli git
          → Yüksek skorlara bonus (exploit), düşüklere penalty
        """
        if expected_Tu < 2.5:
            # Sabırsız kullanıcı - exploration reward
            # Bu kullanıcı zaten çok şeyi beğeniyor, çeşitlilik kazandır
            # Orta skorlu şarkılara şans ver
            if base_score > 0.6:
                # Çok yüksek skorlar: hafif penalty (çeşitlilik için)
                factor = 0.9 + 0.1 * (1 - base_score)
            else:
                # Orta skorlar: exploration bonus
                factor = 1.0 + 0.15 * (1 - base_score)

        elif expected_Tu > 4.0:
            # Sabırlı kullanıcı - exploitation reward
            # Bu kullanıcı seçici, güvenli/yüksek skorlu şarkılar öner
            # Yüksek skorlara büyük bonus
            if base_score > 0.5:
                factor = 1.0 + 0.3 * base_score  # Yüksek skora büyük bonus
            else:
                factor = 0.8 * base_score  # Düşük skora penalty
        else:
            # Normal kullanıcı - dengeli
            factor = 1.0

        return factor

    def query(self, song_ratings, topk=5):
        """
        Global + Personal + Patience birleşik öneri.

        Model 1'in yaptığı her şeyi yapıp üzerine sabır modelini ekler.
        """
        # ========================================
        # STEP 1: Kişisel Tercih Analizi (Model 1 gibi)
        # ========================================
        liked_genres = set()
        liked_artists = set()

        for s in song_ratings:
            if s["rating"] >= 4:
                row = self.tracks[self.tracks["track_id"] == s["track_id"]]
                if len(row) > 0:
                    liked_genres.add(row.iloc[0]["ab_genre_rosamerica_value"])
                    liked_artists.add(row.iloc[0]["primary_artist_name"])

        # ========================================
        # STEP 2: Sabır Tahmini (Part 2)
        # ========================================
        estimated_p, expected_Tu = self._estimate_user_patience(song_ratings)

        # ========================================
        # STEP 3: Aday Şarkıları Filtrele
        # ========================================
        rated_ids = set(s["track_id"] for s in song_ratings)
        df = self.tracks[~self.tracks["track_id"].isin(rated_ids)].copy()

        if len(df) == 0:
            df = self.tracks.copy()

        # ========================================
        # STEP 4: Her Şarkı için Combined Score Hesapla
        # ========================================
        scores = []

        for _, track in df.iterrows():
            # --- A. Global P(5★) - Part 1 ---
            global_p5 = compute_global_probability(track)

            # --- B. Genre Bonus (Model 1 gibi) ---
            genre = track.get("ab_genre_rosamerica_value", "")
            if liked_genres and genre in liked_genres:
                genre_bonus = 1.4  # Beğenilen genre: %40 bonus
            else:
                genre_bonus = 1.0

            # --- C. Artist Bonus (Model 1 gibi) ---
            artist = track.get("primary_artist_name", "")
            if liked_artists and artist in liked_artists:
                artist_bonus = 1.6  # Beğenilen artist: %60 bonus
            else:
                artist_bonus = 1.0

            # --- D. Base Score (Global × Personal) ---
            base_score = global_p5 * genre_bonus * artist_bonus

            # --- E. Patience Factor (Part 2) ---
            patience_factor = self._compute_patience_factor(base_score, estimated_p, expected_Tu)

            # --- F. Final Combined Score ---
            final_score = base_score * patience_factor

            scores.append(final_score)

        df["score"] = scores

        # ========================================
        # STEP 5: Top-K Seçimi (Sabır tipine göre)
        # ========================================
        if expected_Tu > 4.0:
            # Sabırlı kullanıcı: Direkt en yüksek skorlar (exploit)
            recs = df.nlargest(topk, "score")

        elif expected_Tu < 2.5:
            # Sabırsız kullanıcı: Weighted sampling for diversity (explore)
            top_candidates = df.nlargest(min(25, len(df)), "score")
            weights = top_candidates["score"].values
            weights = np.maximum(weights, 1e-10)  # Sıfır olmaması için
            weights = weights / weights.sum()

            sample_size = min(topk, len(top_candidates))
            sample_idx = np.random.choice(
                len(top_candidates),
                size=sample_size,
                replace=False,
                p=weights
            )
            recs = top_candidates.iloc[sample_idx].nlargest(topk, "score")

        else:
            # Normal kullanıcı: Hybrid (%80 exploit + %20 explore)
            n_exploit = max(1, int(topk * 0.8))
            n_explore = topk - n_exploit

            top_recs = df.nlargest(n_exploit, "score")

            if n_explore > 0:
                remaining = df[~df["track_id"].isin(top_recs["track_id"])]
                if len(remaining) > 0:
                    explore_pool = remaining.nlargest(min(20, len(remaining)), "score")
                    weights = explore_pool["score"].values
                    weights = np.maximum(weights, 1e-10)
                    weights = weights / weights.sum()

                    explore_size = min(n_explore, len(explore_pool))
                    explore_idx = np.random.choice(
                        len(explore_pool),
                        size=explore_size,
                        replace=False,
                        p=weights
                    )
                    explore_recs = explore_pool.iloc[explore_idx]
                    recs = pd.concat([top_recs, explore_recs]).nlargest(topk, "score")
                else:
                    recs = top_recs.head(topk)
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
