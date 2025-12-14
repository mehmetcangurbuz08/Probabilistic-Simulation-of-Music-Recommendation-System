import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

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
# 0. GLOBAL AYARLAR
# ============================================================
RNG_SEED = 343
np.random.seed(RNG_SEED)

N_USERS = 2000     # Monte Carlo deneme sayısı (1000–5000 arası önerildi)
TOPK = 5           # Hit@k için k
MAX_TIME = TOPK    # Biz burada tek seferde k öneri aldığımız için Tu ∈ {1,…,k} veya NaN


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
# 2. KULLANICI DAVRANIŞI SİMÜLASYONU
#    (Part 2 — Beta-Geometric ile bağlantı)
# ============================================================
def simulate_ratings_for_tracks(
    track_ids: List[str],
    alpha: float,
    beta: float,
) -> Tuple[List[int], bool, float]:
    """
    Verilen track_id listesi (modelin önerdiği TOPK şarkı) için
    bir kullanıcının puanlarını simüle eder.

    - Önce kullanıcıya özel bir p_user ~ Beta(alpha, beta) çekiyoruz
      (Part 2 sabır modeline uygun).
    - Sonra her şarkı için:
        P(5★) = TRACK_P5[track] * (p_user / EXPECTED_P)
      ile ayarlıyoruz (çok büyük/çok küçük olmaması için [0.01, 0.9] clip).
    - Eğer 5★ gelmezse, kalan 1–4 puanları sabit bir dağılımdan seçiyoruz.

    Dönüş:
        ratings: [r1, ..., rk]
        hit: ilk k içinde en az bir 5★ var mı?
        time_to_5: ilk 5★ kaçıncı pozisyonda? (yoksa np.nan)
    """
    # Kullanıcıya özel beğenme seviyesi
    p_user = np.random.beta(alpha, beta)

    ratings = []
    time_to_5 = np.nan
    hit = False

    for idx, tid in enumerate(track_ids, start=1):
        base_p5 = TRACK_P5.get(tid, 0.1)

        # Kullanıcıya uyarlanmış P(5★)
        eff_p5 = base_p5 * (p_user / EXPECTED_P)
        eff_p5 = float(np.clip(eff_p5, 0.01, 0.9))

        u = np.random.rand()
        if u < eff_p5:
            r = 5
        else:
            # 5★ dışındaki puanları basit bir dağılımla seçelim:
            # 1,2,3,4 için [0.1, 0.2, 0.4, 0.3]
            r = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.4, 0.3])

        ratings.append(int(r))

        if (not hit) and r == 5:
            hit = True
            time_to_5 = float(idx)

    return ratings, hit, time_to_5


# ============================================================
# 3. TEK BİR MONTE CARLO KOŞUSU (TEK MODEL İÇİN)
# ============================================================
def run_monte_carlo_for_model(model_class, n_users: int, topk: int) -> dict:
    """
    Belirli bir model sınıfı (Model1 veya Model2) için
    n_users adet simüle kullanıcı çalıştırır.

    Her kullanıcı için:
        - model.query([], topk) ile öneri al
        - bu önerilere puan simüle et
        - Hit@k, ortalama rating, time-to-5 hesapla

    Çıktı:
        {
          "hit_at_k": np.array([...]),
          "avg_rating": np.array([...]),
          "time_to_5": np.array([...])  # NaN olabilir
        }
    """
    hits = []
    avg_ratings = []
    times = []

    for _ in range(n_users):
        model = model_class()

        # Bu projede, Monte Carlo için sıfır geçmiş ile başlıyoruz:
        # Yani session başında kullanıcı hiç şarkı derecelendirmemiş.
        song_ratings = []
        recs = model.query(song_ratings, topk=topk)  # List[(track_id, track_name)]

        track_ids = [tid for (tid, name) in recs]

        ratings, hit, time_to_5 = simulate_ratings_for_tracks(
            track_ids,
            ALPHA,
            BETA
        )

        hits.append(1.0 if hit else 0.0)
        avg_ratings.append(float(np.mean(ratings)))
        times.append(time_to_5)

    # NumPy array'lere çevir
    hits = np.array(hits)
    avg_ratings = np.array(avg_ratings)
    times = np.array(times)  # NaN içerebilir

    return {
        "hit_at_k": hits,
        "avg_rating": avg_ratings,
        "time_to_5": times,
    }


# ============================================================
# 4. İKİ MODELİ KARŞILAŞTIRMA + GÜVEN ARALIĞI
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
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

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
):
    """
    İki modelin metriklerini ve farkların güven aralıklarını ekrana basar.
    """
    hit1, avg1, t1 = results1["hit_at_k"], results1["avg_rating"], results1["time_to_5"]
    hit2, avg2, t2 = results2["hit_at_k"], results2["avg_rating"], results2["time_to_5"]

    print("\n" + "=" * 60)
    print(f"Monte Carlo Evaluation (k = {k})")
    print("=" * 60)

    print(f"\n[{model1_name}]")
    print(f"  Hit@{k}         : {hit1.mean():.3f}")
    print(f"  Avg Rating      : {avg1.mean():.3f}")
    print(f"  Time-to-5★ (mean, sadece 5 alanlar): {np.nanmean(t1):.3f}")

    print(f"\n[{model2_name}]")
    print(f"  Hit@{k}         : {hit2.mean():.3f}")
    print(f"  Avg Rating      : {avg2.mean():.3f}")
    print(f"  Time-to-5★ (mean, sadece 5 alanlar): {np.nanmean(t2):.3f}")

    # Farklar ve güven aralıkları
    print("\n--- Differences (Model1 - Model2) with ~95% CI ---")

    diff_hit, l_hit, u_hit = mean_and_ci_diff(hit1, hit2)
    print(f"Hit@{k} diff     : {diff_hit:+.3f}  [{l_hit:+.3f}, {u_hit:+.3f}]")

    diff_avg, l_avg, u_avg = mean_and_ci_diff(avg1, avg2)
    print(f"Avg Rating diff  : {diff_avg:+.3f}  [{l_avg:+.3f}, {u_avg:+.3f}]")

    diff_t, l_t, u_t = mean_and_ci_diff(t1, t2, ignore_nan=True)
    print(f"Time-to-5★ diff  : {diff_t:+.3f}  [{l_t:+.3f}, {u_t:+.3f}]")
    print("  (Negatif diff, Model1'in daha hızlı 5★ bulduğunu gösterir.)")


# ============================================================
# 5. MAIN
# ============================================================
def main():
    print(">>> Running Monte Carlo evaluation...")
    print(f"Number of users per model: {N_USERS}")
    print(f"Top-k recommendations: {TOPK}")

    # Model 1: Conditional Filtering
    results_m1 = run_monte_carlo_for_model(Model1, N_USERS, TOPK)

    # Model 2: Utility Sampling
    results_m2 = run_monte_carlo_for_model(Model2, N_USERS, TOPK)

    # Karşılaştırma çıktısı
    print_comparison(results_m1, results_m2, model1_name="Model1", model2_name="Model2", k=TOPK)


if __name__ == "__main__":
    main()
