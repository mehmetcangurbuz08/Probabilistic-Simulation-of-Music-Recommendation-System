import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln

# =============================================================================
# 1. VERİ YÜKLEME VE HAZIRLIK
# =============================================================================
print("\n" + "="*50)
print(" 1. VERİ YÜKLEME VE Tu (SABIR) HESAPLAMA")
print("="*50)

try:
    tracks = pd.read_csv("tracks.csv")
    ratings = pd.read_csv("ratings.csv")
    print(f"Veriler başarıyla yüklendi.")
    print(f"- Ratings: {len(ratings)} satır")
    print(f"- Tracks : {len(tracks)} satır")
except FileNotFoundError:
    print("HATA: 'tracks.csv' veya 'ratings.csv' bulunamadı!")
    exit()

def calculate_user_patience(group):
    """
    Kullanıcının 5 yıldız verdiği ilk anı (kaçıncı şarkıda?) bulur.
    Hiç 5 vermediyse NaN döner.
    """
    # Veriyi oluş sırasına göre diz (round_idx veya timestamp)
    if 'round_idx' in group.columns:
        group = group.sort_values('round_idx')

    # 5 yıldızları bul
    fives = group[group['rating'] == 5]

    if not fives.empty:
        if 'round_idx' in group.columns:
            # round_idx 0'dan başlıyorsa +1 ekle ki "kaçıncı şarkı" olsun
            return fives['round_idx'].iloc[0] + 1
        else:
            # round_idx yoksa index sırasına göre kaçıncı olduğunu bul
            return np.where(group['rating'] == 5)[0][0] + 1
    else:
        return np.nan

# Her kullanıcı için Tu hesapla
print(">>> Kullanıcı sabır süreleri hesaplanıyor...")
tu_raw = ratings.groupby('user_id').apply(calculate_user_patience)

# Hiç 5 vermeyenleri temizle (Discard users who never liked a song)
Tu = tu_raw.dropna()

print(f"Toplam Kullanıcı Sayısı: {len(tu_raw)}")
print(f"Analize Giren (En az bir 5 veren): {len(Tu)}")
print(f"Ortalama Sabır (Tu): {Tu.mean():.2f} şarkı")


# =============================================================================
# 2. MODEL 1: GEOMETRIC (BASİT MODEL)
# =============================================================================
print("\n" + "="*50)
print(" 2. GEOMETRIC MODEL (TEK TİP İNSAN)")
print("="*50)

# E[T] = 1/p  =>  p = 1 / mean(T)
p_geo = 1.0 / Tu.mean()

print(f"Hesaplanan p (Başarı Olasılığı): {p_geo:.4f}")
print(f"Yorum: Ortalama bir kullanıcı %{p_geo*100:.1f} ihtimalle bir şarkıyı beğeniyor.")


# =============================================================================
# 3. MODEL 2: BETA-GEOMETRIC (PROFESYONEL MODEL)
# =============================================================================
print("\n" + "="*50)
print(" 3. BETA-GEOMETRIC MODEL (KİŞİSELLEŞTİRİLMİŞ)")
print("="*50)

# --- A. Log-Likelihood Fonksiyonu ---
def neg_log_likelihood(params, t_values):
    alpha, beta = params

    # Sınır koruması (Optimizer eksiye gitmeye çalışırsa engelle)
    if alpha <= 1e-5 or beta <= 1e-5:
        return 1e10 # Çok büyük ceza puanı döndür

    # Beta-Geometric Formülü (Log space)
    # P(T=t) = Beta(alpha+1, beta+t-1) / Beta(alpha, beta)
    log_probs = betaln(alpha + 1, beta + t_values - 1) - betaln(alpha, beta)

    return -np.sum(log_probs)

# --- B. Akıllı Başlangıç Noktası (Method of Moments) ---
# Rastgele 1.0, 1.0 vermek yerine veriden tahmin ediyoruz.
# p ~ 1/Tu varsayımıyla verideki p'lerin ortalaması ve varyansından alpha/beta çekiyoruz.

estimated_ps = 1.0 / Tu
mean_p = estimated_ps.mean()
var_p = estimated_ps.var()

# Method of Moments Formülleri
if var_p > 1e-10:
    common = (mean_p * (1 - mean_p) / var_p) - 1
    alpha_init = max(0.1, mean_p * common)       # 0.1'den küçük olmasın
    beta_init  = max(0.1, (1 - mean_p) * common) # 0.1'den küçük olmasın
else:
    alpha_init, beta_init = 1.0, 1.0

print(f"Başlangıç Tahmini (Method of Moments): Alpha={alpha_init:.2f}, Beta={beta_init:.2f}")

# --- C. Optimizasyon (L-BFGS-B) ---
bounds = [(1e-3, None), (1e-3, None)] # Alpha ve Beta pozitif olmalı

try:
    result = minimize(
        neg_log_likelihood,
        x0=[alpha_init, beta_init],
        args=(Tu.values,),
        method='L-BFGS-B', # Sınırları tanıyan güvenli yöntem
        bounds=bounds
    )
    alpha_fit, beta_fit = result.x
    success = result.success
except Exception as e:
    print(f"Optimizasyon Hatası: {e}")
    alpha_fit, beta_fit = alpha_init, beta_init
    success = False

print("\n>>> SONUÇLAR:")
print(f"Optimizasyon Başarılı mı?: {success}")
print(f"ALPHA : {alpha_fit:.4f}")
print(f"BETA  : {beta_fit:.4f}")

# Beklenen p değeri (Modelin tahmini)
p_model_mean = alpha_fit / (alpha_fit + beta_fit)
print(f"Model Ortalaması p: {p_model_mean:.4f}")


# =============================================================================
# 4. HİPOTEZ TESTİ (POPÜLER VS NİŞ ZEVKLER)
# =============================================================================
print("\n" + "="*50)
print(" 4. HİPOTEZ TESTİ (POPÜLERLİK ETKİSİ)")
print("="*50)

# Kullanıcıları "Popüler sevenler" ve "Niş sevenler" diye ayıralım
merged = pd.merge(ratings, tracks[['track_id', 'track_popularity']],
                  left_on='song_id', right_on='track_id', how='inner')

# Her kullanıcının dinlediği ortalama popülarite
user_pop = merged.groupby('user_id')['track_popularity'].mean()
median_pop = user_pop.median()

high_pop_users = user_pop[user_pop >= median_pop].index
low_pop_users = user_pop[user_pop < median_pop].index

# Tu değerlerini gruplara göre çek
tu_high = Tu[Tu.index.isin(high_pop_users)]
tu_low = Tu[Tu.index.isin(low_pop_users)]

print(f"Popüler Sevenler (Ort. Tu): {tu_high.mean():.2f}")
print(f"Niş Sevenler     (Ort. Tu): {tu_low.mean():.2f}")

# İstatistiksel Test (Mann-Whitney U)
stat, p_val = stats.mannwhitneyu(tu_high, tu_low, alternative='two-sided')
print(f"p-value: {p_val:.5f}")

if p_val < 0.05:
    print("SONUÇ: İki grup arasında ANLAMLI bir sabır farkı var! (H0 Red)")
else:
    print("SONUÇ: Anlamlı bir fark yok. (H0 Kabul)")


# =============================================================================
# 5. GÖRSELLEŞTİRME VE KAYIT
# =============================================================================
print("\n>>> Grafik Çiziliyor...")

plt.figure(figsize=(10, 6))

# 1. Gerçek Veri Histogramı
max_t = int(Tu.quantile(0.95)) # Çok uç değerleri grafiğe almayalım (x ekseni uzamasın)
t_range = np.arange(1, max_t + 1)
plt.hist(Tu, bins=range(1, max_t + 2), density=True, alpha=0.5, color='gray', label='Gerçek Veri', align='left')

# 2. Geometric Model Çizgisi
y_geo = (1 - p_geo)**(t_range - 1) * p_geo
plt.plot(t_range, y_geo, 'r--', lw=2, label=f'Geometric (p={p_geo:.2f})')

# 3. Beta-Geometric Model Çizgisi
# betaln fonksiyonu logaritma döndürdüğü için exp alıyoruz
y_bg = np.exp(betaln(alpha_fit + 1, beta_fit + t_range - 1) - betaln(alpha_fit, beta_fit))
plt.plot(t_range, y_bg, 'b-', lw=3, label=f'Beta-Geo (α={alpha_fit:.1f}, β={beta_fit:.1f})')

plt.title("Kullanıcı Sabrı Analizi ($T_u$ Dağılımı)")
plt.xlabel("İlk 5★ İçin Gereken Deneme Sayısı")
plt.ylabel("Olasılık")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("part2_final_result.png")
print("Grafik 'part2_final_result.png' olarak kaydedildi.")
print("\nANALİZ TAMAMLANDI. Alpha ve Beta değerlerini not etmeyi unutma!")