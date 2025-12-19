import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.special import betaln

# Part 3'te yazdığın dosyadan import
from recommender import (
    Model1,
    Model2,
    TRACK_DF,
    compute_global_probability,
    ALPHA,
    BETA,
    EXPECTED_P,
)

# ============================================================
# KULLANICI TERCİH ÖZELLİKLERİ - GENİŞLETİLMİŞ VERSİYON
# Part 1'deki tüm binning'ler ve audio özellikleri dahil
# ============================================================

# 1. SÜREKLİ ÖZELLİKLER (0-1 arası normalize edilecek)
CONTINUOUS_PREF_FEATURES = [
    "track_popularity_norm",   # 0-100 → 0-1
    "duration_norm",           # ms → 0-1 (0-10 dk arası)
    "available_markets_norm",  # 0-200 → 0-1
]

# 2. KATEGORİK ÖZELLİKLER (kullanıcının tercih ettiği kategoriler)
# Her kullanıcı bu kategorilerden bazılarını tercih edecek
CATEGORICAL_PREF_FEATURES = {
    # Genre tercihleri (Part 1 CSV'deki değerlerle uyumlu!)
    "ab_genre_rosamerica_value": [
        "hip", "rhy", "roc", "pop", "dan", "cla", "jaz"  # CSV'de sadece bunlar var
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
    # Voice vs Instrumental (CSV'deki isimle uyumlu)
    "ab_voice_instrumental_value": ["voice", "instrumental"],
    # Timbre
    "ab_timbre_value": ["bright", "dark"],
    # Gender (vocal)
    "ab_gender_value": ["female", "male"],
}

# 3. YIL TERCİHİ (Part 1'deki year_bin)
YEAR_BINS = ["pre_1980", "1980s", "1990s", "2000s", "2010s", "2020s"]
YEAR_BIN_RANGES = {
    "pre_1980": (0, 1979),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2030),
}

# Eski liste (backward compatibility için)
USER_PREF_FEATURES = CONTINUOUS_PREF_FEATURES

# ============================================================
# 0. GLOBAL AYARLAR
# ============================================================
RNG_SEED = 343
np.random.seed(RNG_SEED)

N_USERS = 2000        # Monte Carlo deneme sayısı
TOPK = 5              # Hit@k için k
WARMUP_SIZE = 20      # Her kullanıcı için warm-up şarkı sayısı (tercih öğrenme)
TEST_SIZE = 10        # Model test için öneri sayısı

# Global vs Kişisel ağırlıkları
GLOBAL_WEIGHT = 0.6   # Part 1'den gelen global P(5★) ağırlığı
PERSONAL_WEIGHT = 0.4 # Kullanıcı özellik eşleşmesi ağırlığı


# ============================================================
# 1. HER TRACK İÇİN P(5★ | track) TAHMİNİ
#    (Part 1 ile bağlantı)
# ============================================================
def precompute_track_p5() -> Dict[str, float]:
    """
    Part 1'deki compute_global_probability fonksiyonunu kullanarak
    her şarkı için bir "ham" skor hesaplıyoruz.
    Bu skorları [0.02, 0.6] aralığına yeniden ölçekleyip
    P(5★ | track) gibi kullanacağız.
    """
    df = TRACK_DF.copy()
    raw_scores = df.apply(compute_global_probability, axis=1)

    # Min–max normalizasyonu
    min_s = raw_scores.min()
    max_s = raw_scores.max()
    denom = max_s - min_s + 1e-12

    scaled = (raw_scores - min_s) / denom  # [0, 1]

    # Bunu [0.02, 0.6] aralığına map edelim
    p5 = 0.02 + 0.58 * scaled

    track_ids = df["track_id"].values
    return {tid: float(p) for tid, p in zip(track_ids, p5)}


TRACK_P5 = precompute_track_p5()


# ============================================================
# 1b. HER TRACK İÇİN GENİŞLETİLMİŞ ÖZELLİK VEKTÖRLERİ
# ============================================================
def precompute_track_features() -> Dict[str, Dict]:
    """
    Her şarkı için genişletilmiş özellik vektörünü hesaplar.
    
    İçerir:
    1. Sürekli özellikler (normalize edilmiş)
    2. Kategorik özellikler (genre, mood, vb.)
    3. Yıl bilgisi (bin olarak)
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
        
        # ========================================
        # A. SÜREKLİ ÖZELLİKLER
        # ========================================
        
        # Track popularity: 0-100 → 0-1
        if "track_popularity" in row and pd.notna(row["track_popularity"]):
            features["continuous"]["track_popularity_norm"] = float(
                np.clip(row["track_popularity"] / 100.0, 0, 1)
            )
        else:
            features["continuous"]["track_popularity_norm"] = 0.5
        
        # Duration: 0-600000ms (10dk) → 0-1
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
        
        # ========================================
        # B. KATEGORİK ÖZELLİKLER
        # ========================================
        for cat_feature, possible_values in CATEGORICAL_PREF_FEATURES.items():
            if cat_feature in row and pd.notna(row[cat_feature]):
                features["categorical"][cat_feature] = str(row[cat_feature]).lower()
            else:
                features["categorical"][cat_feature] = None
        
        # ========================================
        # C. YIL BİN
        # ========================================
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


# ============================================================
# 2. KULLANICI DAVRANIŞI SİMÜLASYONU
#    (Part 2 — Beta-Geometric ile bağlantı + Genişletilmiş Tercihler)
# ============================================================
def generate_user_preferences() -> Dict:
    """
    Her kullanıcı için genişletilmiş tercih profili oluşturur.
    
    İçerir:
    1. Sürekli tercihler: popularity, duration, markets (0-1 arası)
    2. Kategorik tercihler: tercih edilen genre, mood vb. setleri
    3. Yıl tercihi: tercih edilen dönemler
    
    Bu sayede her kullanıcı gerçekçi ve spesifik bir profile sahip olur.
    """
    prefs = {
        "continuous": {},
        "categorical": {},
        "year_preferences": {},
    }
    
    # ========================================
    # A. SÜREKLİ TERCİHLER
    # ========================================
    # Her sürekli özellik için kullanıcının tercih seviyesi
    for feat in CONTINUOUS_PREF_FEATURES:
        # Beta(2, 2) → ortalama 0.5, ama çeşitlilik var
        prefs["continuous"][feat] = np.random.beta(2, 2)
    
    # ========================================
    # B. KATEGORİK TERCİHLER
    # ========================================
    # Her kategorik özellik için kullanıcının tercih ettiği değerler
    for cat_feature, possible_values in CATEGORICAL_PREF_FEATURES.items():
        # Her kategorik özellik için:
        # - Kullanıcı bazı değerleri tercih eder (liked)
        # - Bazılarına nötr (neutral) 
        # - Bazılarını sevmez (disliked)
        
        n_values = len(possible_values)
        
        if n_values == 2:
            # Binary özellik (acoustic/not_acoustic gibi)
            # %70 ihtimalle birini tercih et, %30 nötr
            if np.random.rand() < 0.7:
                preferred = np.random.choice(possible_values)
                prefs["categorical"][cat_feature] = {
                    "liked": {preferred},
                    "disliked": set(possible_values) - {preferred},
                    "weight": np.random.uniform(0.6, 1.0),  # Bu tercihin önemi
                }
            else:
                prefs["categorical"][cat_feature] = {
                    "liked": set(),
                    "disliked": set(),
                    "weight": 0.0,  # Umursamıyor
                }
        else:
            # Multi-value özellik (genre gibi)
            # 1-3 tane tercih et, 0-2 tane sevme
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
    
    # ========================================
    # C. YIL TERCİHLERİ
    # ========================================
    # Her kullanıcı belirli dönemleri tercih eder
    # 1-3 tercih edilen dönem, 0-2 sevmeyen dönem
    n_liked_years = np.random.randint(1, 4)
    n_disliked_years = np.random.randint(0, 3)
    
    shuffled_years = np.random.permutation(YEAR_BINS)
    liked_years = set(shuffled_years[:n_liked_years])
    disliked_years = set(shuffled_years[n_liked_years:n_liked_years + n_disliked_years])
    
    prefs["year_preferences"] = {
        "liked": liked_years,
        "disliked": disliked_years,
        "weight": np.random.uniform(0.4, 0.9),  # Yıl tercihinin önemi
    }
    
    return prefs


def compute_user_track_match(user_prefs: Dict, track_id: str) -> float:
    """
    Kullanıcı tercihleri ile şarkı özellikleri arasındaki uyumu hesaplar.
    
    Genişletilmiş formül:
    match_score = weighted_average(
        continuous_match,    # Popularity, duration vs.
        categorical_match,   # Genre, mood vs.
        year_match,          # Dönem tercihi
    )
    
    Her bileşen 0-1 arası, ağırlıklı ortalama alınır.
    """
    track_feats = TRACK_FEATURES.get(track_id, None)
    if track_feats is None:
        return 0.5  # Varsayılan orta eşleşme
    
    match_scores = []
    weights = []
    
    # ========================================
    # A. SÜREKLİ ÖZELLİK UYUMU
    # ========================================
    continuous_diffs = []
    for feat in CONTINUOUS_PREF_FEATURES:
        user_val = user_prefs["continuous"].get(feat, 0.5)
        track_val = track_feats["continuous"].get(feat, 0.5)
        diff = abs(user_val - track_val)
        continuous_diffs.append(diff)
    
    if continuous_diffs:
        continuous_match = 1.0 - np.mean(continuous_diffs)
        match_scores.append(continuous_match)
        weights.append(0.2)  # Sürekli özelliklerin ağırlığı
    
    # ========================================
    # B. KATEGORİK ÖZELLİK UYUMU
    # ========================================
    categorical_scores = []
    categorical_weights = []
    
    for cat_feature, pref_info in user_prefs["categorical"].items():
        if pref_info["weight"] == 0:
            continue  # Bu özellik umurunda değil
        
        track_val = track_feats["categorical"].get(cat_feature)
        if track_val is None:
            continue
        
        # Eşleşme skoru: liked=1, neutral=0.5, disliked=0
        if track_val in pref_info["liked"]:
            score = 1.0
        elif track_val in pref_info["disliked"]:
            score = 0.0
        else:
            score = 0.5  # Nötr
        
        categorical_scores.append(score)
        categorical_weights.append(pref_info["weight"])
    
    if categorical_scores:
        # Ağırlıklı ortalama
        total_weight = sum(categorical_weights)
        categorical_match = sum(
            s * w for s, w in zip(categorical_scores, categorical_weights)
        ) / total_weight
        match_scores.append(categorical_match)
        weights.append(0.5)  # Kategorik özelliklerin ağırlığı (en önemli!)
    
    # ========================================
    # C. YIL TERCİHİ UYUMU
    # ========================================
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
        weights.append(0.3 * year_pref["weight"])  # Yıl tercihinin ağırlığı
    
    # ========================================
    # FINAL WEIGHTED AVERAGE
    # ========================================
    if not match_scores:
        return 0.5
    
    total_weight = sum(weights)
    final_match = sum(s * w for s, w in zip(match_scores, weights)) / total_weight
    
    return float(final_match)


def sample_user_patience(alpha: float, beta: float, max_t: int = 50) -> int:
    """
    Part 2'deki Beta-Geometric dağılımından kullanıcının sabrını (Tu) örnekler.
    
    Beta-Geometric: Kullanıcının "başarı (5★ bulma)" olasılığı p ~ Beta(alpha, beta)
    Tu = ilk başarıya kadar gereken deneme sayısı (Geometric(p))
    
    Ama burada bunu "sabrı limite" olarak yorumluyoruz:
    - Kullanıcı Tu deneme sonrası 5★ bulamazsa oturumu terk eder
    
    P(T=t) = Beta(alpha+1, beta+t-1) / Beta(alpha, beta)
    """
    # CDF'yi hesaplayıp inverse transform sampling yap
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


# ============================================================
# 3. KULLANICI PROFİLİ OLUŞTURMA (WARM-UP FAZI)
# ============================================================
class UserProfile:
    """
    Bir kullanıcının kişisel profili.
    20 şarkılık warm-up fazından öğrenilir.
    """
    def __init__(self):
        # Gerçek (gizli) tercihler - simülasyon için
        self.true_preferences = generate_user_preferences()
        self.true_p_user = np.random.beta(ALPHA, BETA)  # Genel beğenme seviyesi
        
        # Warm-up verileri
        self.learned_patience = None  # Churn simülasyonu için
        self.warmup_ratings = []
        self.warmup_tracks = []
    
    def rate_track(self, track_id: str) -> int:
        """
        Kullanıcının gerçek tercihlerine göre bir şarkıyı puanlar.
        
        P(5★) HESABI - GLOBAL + KİŞİSEL KOMBİNASYON:
        
        1. GLOBAL FAKTÖR (Part 1'den):
           - base_p5: Şarkının genel beğenilme olasılığı
           - Tüm kullanıcılar için aynı (şarkı kalitesi)
        
        2. KİŞİSEL FAKTÖR:
           - match_score: Kullanıcı özellikleri ile şarkı özellikleri uyumu
           - p_user: Kullanıcının genel kritiklik seviyesi (Part 2)
        
        FORMÜL:
           P(5★) = GLOBAL_WEIGHT * base_p5 + PERSONAL_WEIGHT * (base_p5 * match_score)
           Sonra p_user ile modüle et
        
        Bu sayede:
        - Kaliteli şarkılar (yüksek base_p5) herkes tarafından beğenilir
        - Ama kişisel uyum da önemli bir rol oynar
        """
        # GLOBAL: Şarkının Part 1'den gelen P(5★) değeri
        base_p5 = TRACK_P5.get(track_id, 0.1)
        
        # KİŞİSEL: Kullanıcı-şarkı özellik uyumu [0, 1]
        match_score = compute_user_track_match(self.true_preferences, track_id)
        
        # KOMBİNE SKOR:
        # - Global kısım: base_p5 (şarkı kalitesi)
        # - Kişisel kısım: base_p5 * match_score (kalite × uyum)
        global_component = base_p5
        personal_component = base_p5 * (0.5 + 0.5 * match_score)  # [0.5*base, base]
        
        combined_p5 = GLOBAL_WEIGHT * global_component + PERSONAL_WEIGHT * personal_component
        
        # Kullanıcının genel kritikliği ile modüle et (Part 2 bağlantısı)
        # p_user yüksekse kullanıcı daha kolay beğenir
        eff_p5 = combined_p5 * (self.true_p_user / EXPECTED_P)
        eff_p5 = float(np.clip(eff_p5, 0.02, 0.85))
        
        # Puanlama
        u = np.random.rand()
        if u < eff_p5:
            return 5
        else:
            # 5★ dışındaki puanlar: hem global hem kişisel etkili
            # Yüksek uyum + yüksek kalite → 4'e yakın
            # Düşük uyum veya düşük kalite → 1-2'ye yakın
            quality_match = (base_p5 + match_score) / 2  # 0-1 arası
            
            if quality_match > 0.55:
                return np.random.choice([1, 2, 3, 4], p=[0.05, 0.15, 0.40, 0.40])
            elif quality_match > 0.4:
                return np.random.choice([1, 2, 3, 4], p=[0.10, 0.25, 0.40, 0.25])
            else:
                return np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
    
    def do_warmup(self, warmup_track_ids: List[str]) -> None:
        """
        20 şarkılık warm-up fazını gerçekleştirir.
        Kullanıcının tercihlerini ve sabrını öğrenir.
        """
        self.warmup_tracks = warmup_track_ids
        self.warmup_ratings = []
        
        first_five_idx = None
        
        for idx, tid in enumerate(warmup_track_ids):
            rating = self.rate_track(tid)
            self.warmup_ratings.append(rating)
            
            if first_five_idx is None and rating == 5:
                first_five_idx = idx + 1  # 1-indexed
        
        # Sabır öğrenme: Bayesian yaklaşım (Part 2 ile uyumlu)
        # Tek bir gözlem yerine, tüm warm-up verisinden tahmin et
        n_fives_warmup = sum(1 for r in self.warmup_ratings if r == 5)
        n_total_warmup = len(self.warmup_ratings)
        
        # Bayesian posterior: p | data ~ Beta(ALPHA + n_fives, BETA + n_not_fives)
        alpha_post = ALPHA + n_fives_warmup
        beta_post = BETA + (n_total_warmup - n_fives_warmup)
        
        # Posterior mean'den beklenen Tu
        estimated_p = alpha_post / (alpha_post + beta_post)
        expected_Tu = 1.0 / estimated_p
        
        # Sabır limitini bu tahmine göre ayarla
        # Tu'nun tam sayı versiyonu + biraz tolerans
        self.learned_patience = max(2, int(np.ceil(expected_Tu)))
    
    def get_song_ratings_for_model(self) -> List[dict]:
        """
        Model'e verilecek formatta warm-up puanlarını döndürür.
        """
        return [
            {"track_id": tid, "rating": rating}
            for tid, rating in zip(self.warmup_tracks, self.warmup_ratings)
        ]


def get_random_warmup_tracks(n: int = WARMUP_SIZE) -> List[str]:
    """
    Warm-up için rastgele n şarkı seçer.
    """
    all_track_ids = list(TRACK_FEATURES.keys())
    return list(np.random.choice(all_track_ids, size=n, replace=False))


# ============================================================
# 4. KİŞİSELLEŞTİRİLMİŞ TEST SİMÜLASYONU
# ============================================================
def simulate_personalized_test(
    user: UserProfile,
    track_ids: List[str],
    topk: int = TOPK,
) -> Tuple[List[int], bool, float, bool]:
    """
    Öğrenilmiş kullanıcı profiliyle model önerilerini test eder.
    
    Kullanıcının GERÇEK tercihlerini kullanır (simülasyon için),
    ama sabrı warm-up'tan ÖĞRENILMIŞ değerdir.
    
    Dönüş:
        ratings: Verilen puanlar
        hit: En az bir 5★ var mı?
        time_to_5: İlk 5★ kaçıncı pozisyonda?
        churned: Sabır bitip terk etti mi?
    """
    # Öğrenilmiş sabır limitini kullan
    patience_limit = min(user.learned_patience, len(track_ids))
    
    ratings = []
    time_to_5 = np.nan
    hit = False
    churned = False
    
    for idx, tid in enumerate(track_ids, start=1):
        # Sabır bittiyse ve 5★ bulamadıysa -> terk
        if (not hit) and idx > patience_limit:
            churned = True
            break
        
        # Kullanıcının gerçek tercihlerine göre puanla
        rating = user.rate_track(tid)
        ratings.append(rating)
        
        if (not hit) and rating == 5:
            hit = True
            time_to_5 = float(idx)
    
    return ratings, hit, time_to_5, churned


# ============================================================
# 5. MONTE CARLO - KİŞİSELLEŞTİRİLMİŞ DEĞERLENDIRME
# ============================================================
def run_personalized_monte_carlo(
    model_class,
    n_users: int,
    topk: int,
) -> dict:
    """
    Kişiselleştirilmiş Monte Carlo değerlendirmesi.
    
    Her kullanıcı için:
    1. 20 şarkılık warm-up ile tercih ve sabır öğren
    2. Öğrenilen profille model önerilerini al
    3. Gerçek tercihlerle değerlendir
    
    Bu yaklaşım:
    - Global değil, KİŞİSEL bazda test eder
    - Her kullanıcının kendi sabır limitini kullanır
    - Modelin kişiselleştirme yeteneğini ölçer
    """
    hits = []
    avg_ratings = []
    times = []
    churns = []
    learned_patiences = []
    
    for i in range(n_users):
        # 1. Kullanıcı profili oluştur
        user = UserProfile()
        
        # 2. Warm-up fazı
        warmup_tracks = get_random_warmup_tracks(WARMUP_SIZE)
        user.do_warmup(warmup_tracks)
        learned_patiences.append(user.learned_patience)
        
        # 3. Model'den öneri al (warm-up puanlarıyla)
        model = model_class()
        song_ratings = user.get_song_ratings_for_model()
        recs = model.query(song_ratings, topk=topk)
        
        # Warm-up şarkılarını önerilerden çıkar
        warmup_set = set(warmup_tracks)
        filtered_recs = [(tid, name) for tid, name in recs if tid not in warmup_set]
        
        # Yeterli öneri yoksa, filtresiz al
        if len(filtered_recs) < topk:
            track_ids = [tid for tid, name in recs[:topk]]
        else:
            track_ids = [tid for tid, name in filtered_recs[:topk]]
        
        # 4. Test et
        ratings, hit, time_to_5, churned = simulate_personalized_test(
            user, track_ids, topk
        )
        
        hits.append(1.0 if hit else 0.0)
        avg_ratings.append(float(np.mean(ratings)) if ratings else 0.0)
        times.append(time_to_5)
        churns.append(1.0 if churned else 0.0)
    
    # Metrikleri hesapla
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


# ============================================================
# 6. İKİ MODELİ KARŞILAŞTIRMA + GÜVEN ARALIĞI
# ============================================================
def mean_and_ci_diff(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    ignore_nan: bool = False,
) -> Tuple[float, float, float]:
    """
    İki modelin metrik ortalamaları arasındaki fark için
    normal yaklaşıkla 95% (veya 1-alpha) güven aralığı hesaplar.

    diff = mean(a) - mean(b)

    Eğer ignore_nan=True ise NaN'ler görmezden gelinir (time_to_5 için).
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

    z = 1.96  # yaklaşık 95% CI için
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
    Monte Carlo sonuçlarını karşılaştırır.
    Metrikler: Hit@k, Average Rating, Time-to-5★
    """
    hit1, avg1, t1 = results1["hit_at_k"], results1["avg_rating"], results1["time_to_5"]
    hit2, avg2, t2 = results2["hit_at_k"], results2["avg_rating"], results2["time_to_5"]

    print("\n" + "=" * 60)
    print("MONTE CARLO DEĞERLENDİRME SONUÇLARI")
    print("=" * 60)
    print(f"  Simüle edilen kullanıcı sayısı : {n_users}")
    print(f"  Warm-up şarkı sayısı           : {WARMUP_SIZE}")
    print(f"  Test (k) değeri                : {k}")

    print(f"\n{'Metrik':<20} {model1_name:<15} {model2_name:<15}")
    print("-" * 60)
    print(f"{'Hit@' + str(k):<20} {hit1.mean():<15.3f} {hit2.mean():<15.3f}")
    print(f"{'Average Rating':<20} {avg1.mean():<15.3f} {avg2.mean():<15.3f}")
    print(f"{'Time-to-5★ (mean)':<20} {np.nanmean(t1):<15.3f} {np.nanmean(t2):<15.3f}")

    # Güven aralıkları
    print("\n" + "-" * 60)
    print(f"FARKLAR ({model1_name} - {model2_name}) + 95% CI")
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
    
    print("\n  * = İstatistiksel olarak anlamlı (95% CI)")
    print("=" * 60)


# ============================================================
# 7. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("MONTE CARLO SİMÜLASYONU")
    print("=" * 60)
    print(f"Kullanıcı: {N_USERS}, Warm-up: {WARMUP_SIZE}, Test: {TOPK}")

    print("\n>>> Model 1 çalıştırılıyor...")
    results_m1 = run_personalized_monte_carlo(Model1, N_USERS, TOPK)

    print(">>> Model 2 çalıştırılıyor...")
    results_m2 = run_personalized_monte_carlo(Model2, N_USERS, TOPK)

    # Karşılaştırma
    print_comparison(
        results_m1, results_m2,
        model1_name="Model1",
        model2_name="Model2",
        k=TOPK,
        n_users=N_USERS
    )


if __name__ == "__main__":
    main()