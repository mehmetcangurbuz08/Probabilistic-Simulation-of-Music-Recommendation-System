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
# KULLANICI TERCİH ÖZELLİKLERİ
# Her kullanıcı bu özellikler için farklı tercihlere sahip olacak
# ============================================================
USER_PREF_FEATURES = [
    "danceability",
    "energy",
    "valence",         # mutluluk/pozitiflik
    "acousticness",
    "instrumentalness",
    "tempo_normalized",  # normalize edilecek
]

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
# 1b. HER TRACK İÇİN ÖZELLİK VEKTÖRLERİ
# ============================================================
def precompute_track_features() -> Dict[str, Dict[str, float]]:
    """
    Her şarkı için normalize edilmiş özellik vektörünü hesaplar.
    Kullanıcı-şarkı eşleşmesi için kullanılacak.
    """
    df = TRACK_DF.copy()
    track_features = {}
    
    for _, row in df.iterrows():
        tid = row["track_id"]
        features = {}
        
        # Doğrudan 0-1 aralığında olan özellikler
        for feat in ["danceability", "energy", "valence", "acousticness", "instrumentalness"]:
            if feat in row and pd.notna(row[feat]):
                features[feat] = float(np.clip(row[feat], 0, 1))
            else:
                features[feat] = 0.5  # varsayılan orta değer
        
        # Tempo: 0-250 BPM aralığını 0-1'e normalize et
        if "tempo" in row and pd.notna(row["tempo"]):
            features["tempo_normalized"] = float(np.clip(row["tempo"] / 200.0, 0, 1))
        else:
            features["tempo_normalized"] = 0.5
        
        track_features[tid] = features
    
    return track_features


TRACK_FEATURES = precompute_track_features()


# ============================================================
# 2. KULLANICI DAVRANIŞI SİMÜLASYONU
#    (Part 2 — Beta-Geometric ile bağlantı + Özellik-Bazlı Tercihler)
# ============================================================
def generate_user_preferences() -> Dict[str, float]:
    """
    Her kullanıcı için özellik tercihlerini simüle eder.
    Her özellik için Beta(2, 2) dağılımından tercih seviyesi çekilir.
    Bu, kullanıcının o özelliği ne kadar sevdiğini belirler.
    
    Örnek:
        - danceability tercihi 0.8 olan kullanıcı dans edilebilir şarkıları sever
        - energy tercihi 0.2 olan kullanıcı sakin şarkıları tercih eder
    """
    prefs = {}
    for feat in USER_PREF_FEATURES:
        # Beta(2, 2) -> ortalama 0.5, ama çeşitlilik var
        prefs[feat] = np.random.beta(2, 2)
    return prefs


def compute_user_track_match(user_prefs: Dict[str, float], track_id: str) -> float:
    """
    Kullanıcı tercihleri ile şarkı özellikleri arasındaki uyumu hesaplar.
    
    Yüksek uyum = kullanıcının tercih ettiği özelliklere sahip şarkı
    
    Formül: 1 - ortalama(|user_pref - track_feature|)
    Yani tercihler ve özellikler ne kadar yakınsa skor o kadar yüksek.
    """
    track_feats = TRACK_FEATURES.get(track_id, {})
    if not track_feats:
        return 0.5  # varsayılan orta eşleşme
    
    total_diff = 0.0
    count = 0
    
    for feat in USER_PREF_FEATURES:
        if feat in track_feats and feat in user_prefs:
            # Kullanıcı bu özelliği ne kadar seviyor vs şarkıda ne kadar var
            diff = abs(user_prefs[feat] - track_feats[feat])
            total_diff += diff
            count += 1
    
    if count == 0:
        return 0.5
    
    # Ortalama fark: 0 = mükemmel eşleşme, 1 = tamamen uyumsuz
    avg_diff = total_diff / count
    
    # Eşleşme skoru: 0 = uyumsuz, 1 = mükemmel
    match_score = 1.0 - avg_diff
    
    return float(match_score)


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
        
        # Öğrenilen tercihler - warm-up'tan hesaplanacak
        self.learned_preferences = {}
        self.learned_patience = None
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
        
        # Sabır öğrenme: İlk 5★'a kadar geçen süre
        if first_five_idx is not None:
            self.learned_patience = first_five_idx
        else:
            # Hiç 5★ vermediyse, tüm warm-up boyunca sabretmiş demek
            self.learned_patience = len(warmup_track_ids)
        
        # Tercih öğrenme: Yüksek puanlanan şarkıların özelliklerinden
        self._learn_preferences_from_warmup()
    
    def _learn_preferences_from_warmup(self) -> None:
        """
        Warm-up verilerinden kullanıcı tercihlerini öğrenir.
        Yüksek puanlı şarkıların (4-5★) özelliklerinin ortalamasını alır.
        """
        liked_features = {feat: [] for feat in USER_PREF_FEATURES}
        
        for tid, rating in zip(self.warmup_tracks, self.warmup_ratings):
            if rating >= 4:  # Beğenilen şarkılar
                track_feats = TRACK_FEATURES.get(tid, {})
                for feat in USER_PREF_FEATURES:
                    if feat in track_feats:
                        liked_features[feat].append(track_feats[feat])
        
        # Ortalama al veya varsayılan kullan
        for feat in USER_PREF_FEATURES:
            if liked_features[feat]:
                self.learned_preferences[feat] = np.mean(liked_features[feat])
            else:
                self.learned_preferences[feat] = 0.5  # Varsayılan orta değer
    
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
    true_patiences = []
    preference_errors = []  # Öğrenilen vs gerçek tercih farkı
    
    for i in range(n_users):
        # 1. Kullanıcı profili oluştur
        user = UserProfile()
        
        # Gerçek sabır (karşılaştırma için)
        true_patience = sample_user_patience(ALPHA, BETA, max_t=50)
        true_patiences.append(true_patience)
        
        # 2. Warm-up fazı
        warmup_tracks = get_random_warmup_tracks(WARMUP_SIZE)
        user.do_warmup(warmup_tracks)
        learned_patiences.append(user.learned_patience)
        
        # Tercih öğrenme hatası
        pref_error = 0.0
        for feat in USER_PREF_FEATURES:
            true_val = user.true_preferences.get(feat, 0.5)
            learned_val = user.learned_preferences.get(feat, 0.5)
            pref_error += abs(true_val - learned_val)
        pref_error /= len(USER_PREF_FEATURES)
        preference_errors.append(pref_error)
        
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
    true_patiences = np.array(true_patiences)
    preference_errors = np.array(preference_errors)
    
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
        "avg_true_patience": true_patiences.mean(),
        "avg_preference_error": preference_errors.mean(),
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


def print_personalized_comparison(
    results1: dict,
    results2: dict,
    model1_name: str = "Model1",
    model2_name: str = "Model2",
    k: int = TOPK,
):
    """
    Kişiselleştirilmiş Monte Carlo sonuçlarını karşılaştırır.
    """
    hit1, avg1, t1 = results1["hit_at_k"], results1["avg_rating"], results1["time_to_5"]
    hit2, avg2, t2 = results2["hit_at_k"], results2["avg_rating"], results2["time_to_5"]
    churn1, churn2 = results1["churns"], results2["churns"]

    print("\n" + "=" * 75)
    print("KİŞİSELLEŞTİRİLMİŞ MONTE CARLO DEĞERLENDİRMESİ")
    print("=" * 75)
    print(f"Her kullanıcı için {WARMUP_SIZE} şarkı warm-up + {k} şarkı test")
    
    # Öğrenme kalitesi
    print("\n" + "-" * 75)
    print("ÖĞRENME KALİTESİ (Warm-up Fazından)")
    print("-" * 75)
    print(f"  Ortalama Öğrenilen Sabır: {results1['avg_learned_patience']:.2f} şarkı")
    print(f"  Ortalama Gerçek Sabır   : {results1['avg_true_patience']:.2f} şarkı")
    print(f"  Tercih Öğrenme Hatası   : {results1['avg_preference_error']:.3f} (0=mükemmel)")

    print("\n" + "-" * 75)
    print(f"MODEL KARŞILAŞTIRMASI")
    print("-" * 75)

    print(f"\n[{model1_name}]")
    print(f"  Hit@{k}              : {hit1.mean():.3f}")
    print(f"  Avg Rating           : {avg1.mean():.3f}")
    print(f"  Time-to-5★ (ortalama): {np.nanmean(t1):.3f}")
    print(f"  Churn Rate           : {results1['churn_rate']:.3f}")
    print(f"  Satisfaction Rate    : {results1['satisfaction_rate']:.3f}")

    print(f"\n[{model2_name}]")
    print(f"  Hit@{k}              : {hit2.mean():.3f}")
    print(f"  Avg Rating           : {avg2.mean():.3f}")
    print(f"  Time-to-5★ (ortalama): {np.nanmean(t2):.3f}")
    print(f"  Churn Rate           : {results2['churn_rate']:.3f}")
    print(f"  Satisfaction Rate    : {results2['satisfaction_rate']:.3f}")

    # Farklar ve güven aralıkları
    print("\n" + "-" * 75)
    print(f"FARKLAR ({model1_name} - {model2_name}) + 95% Güven Aralığı")
    print("-" * 75)

    diff_hit, l_hit, u_hit = mean_and_ci_diff(hit1, hit2)
    sig_hit = "***" if l_hit > 0 or u_hit < 0 else ""
    print(f"  Hit@{k} diff        : {diff_hit:+.4f}  [{l_hit:+.4f}, {u_hit:+.4f}] {sig_hit}")

    diff_avg, l_avg, u_avg = mean_and_ci_diff(avg1, avg2)
    sig_avg = "***" if l_avg > 0 or u_avg < 0 else ""
    print(f"  Avg Rating diff     : {diff_avg:+.4f}  [{l_avg:+.4f}, {u_avg:+.4f}] {sig_avg}")

    diff_t, l_t, u_t = mean_and_ci_diff(t1, t2, ignore_nan=True)
    sig_t = "***" if l_t > 0 or u_t < 0 else ""
    print(f"  Time-to-5★ diff     : {diff_t:+.4f}  [{l_t:+.4f}, {u_t:+.4f}] {sig_t}")
    print("    → Negatif = daha hızlı 5★")

    diff_churn, l_ch, u_ch = mean_and_ci_diff(churn1, churn2)
    sig_ch = "***" if l_ch > 0 or u_ch < 0 else ""
    print(f"  Churn Rate diff     : {diff_churn:+.4f}  [{l_ch:+.4f}, {u_ch:+.4f}] {sig_ch}")
    print("    → Negatif = daha az kullanıcı kaybı")
    
    print("\n  *** = İstatistiksel olarak anlamlı fark (95% CI sıfırı içermiyor)")

    # Özet
    print("\n" + "=" * 75)
    print("ÖZET YORUM")
    print("=" * 75)
    
    winner_hit = model1_name if hit1.mean() > hit2.mean() else model2_name
    winner_churn = model1_name if results1['churn_rate'] < results2['churn_rate'] else model2_name
    winner_sat = model1_name if results1['satisfaction_rate'] > results2['satisfaction_rate'] else model2_name
    
    print(f"  • Hit@{k} lideri: {winner_hit}")
    print(f"  • En düşük churn: {winner_churn}")
    print(f"  • En yüksek memnuniyet: {winner_sat}")
    
    if winner_hit == winner_churn == winner_sat:
        print(f"\n  → {winner_hit} tüm metriklerde daha iyi!")
    else:
        print(f"\n  → Sonuçlar karma; metrik önceliğine göre seçim yapılmalı.")


# ============================================================
# 7. MAIN
# ============================================================
def main():
    print("=" * 75)
    print("MONTE CARLO DEĞERLENDİRMESİ - KİŞİSELLEŞTİRİLMİŞ VERSİYON")
    print("Part 1 (Global P(5★)) + Part 2 (Tu-Based Sabır) + Kişisel Tercihler")
    print("=" * 75)
    
    print(f"\n[AYARLAR]")
    print(f"  Kullanıcı sayısı       : {N_USERS}")
    print(f"  Warm-up şarkı sayısı   : {WARMUP_SIZE}")
    print(f"  Test şarkı sayısı      : {TOPK}")
    print(f"  Beta parametreleri     : α={ALPHA:.2f}, β={BETA:.2f}")
    print(f"  Beklenen E[p]          : {EXPECTED_P:.3f}")
    print(f"\n[P(5★) FORMÜLÜ]")
    print(f"  Global Ağırlık         : {GLOBAL_WEIGHT:.0%}  (Part 1: şarkı kalitesi)")
    print(f"  Kişisel Ağırlık        : {PERSONAL_WEIGHT:.0%}  (kullanıcı-şarkı uyumu)")
    print(f"  → P(5★) = {GLOBAL_WEIGHT}*base_p5 + {PERSONAL_WEIGHT}*(base_p5*match) × (p_user/E[p])")

    print("\n" + "-" * 75)
    print(">>> Model 1 (Conditional Filtering) çalıştırılıyor...")
    results_m1 = run_personalized_monte_carlo(Model1, N_USERS, TOPK)
    print(f"    Tamamlandı. Hit@{TOPK}: {results_m1['hit_at_k'].mean():.3f}")

    print("\n>>> Model 2 (Utility Sampling) çalıştırılıyor...")
    results_m2 = run_personalized_monte_carlo(Model2, N_USERS, TOPK)
    print(f"    Tamamlandı. Hit@{TOPK}: {results_m2['hit_at_k'].mean():.3f}")

    # Karşılaştırma
    print_personalized_comparison(
        results_m1, results_m2,
        model1_name="Model1 (Conditional)",
        model2_name="Model2 (Utility)",
        k=TOPK
    )


if __name__ == "__main__":
    main()
