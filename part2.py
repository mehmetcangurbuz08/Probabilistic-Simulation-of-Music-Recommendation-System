import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln
import matplotlib.pyplot as plt

# ==========================================
# 1. VERİ YÜKLEME VE Tu HESAPLAMA
# ==========================================
print(">>> Veriler yükleniyor...")
try:
    tracks = pd.read_csv("tracks.csv")
    ratings = pd.read_csv("ratings.csv")
    print(f"Veri yüklendi. Toplam Puanlama: {len(ratings)}")
except FileNotFoundError:
    print("HATA: dosyalar bulunamadı.")
    exit()

# Tu (Time-to-5*) Hesaplama Fonksiyonu
def get_time_to_five(group):
    """
    Bir kullanıcının oturumunda 5 yıldız verdiği İLK sırayı (round_idx) döndürür.
    Hiç 5 vermediyse NaN döner.
    """
    # Önce oturum sırasına göre dizelim
    group = group.sort_values('round_idx')

    # 5 yıldız olanları filtrele
    fives = group[group['rating'] == 5]

    if not fives.empty:
        return fives['round_idx'].iloc[0] # İlk 5'in sırası
    else:
        return np.nan

print(">>> Kullanıcı sabır süreleri (Tu) hesaplanıyor...")
# Her kullanıcı için fonksiyonu çalıştır
Tu_series = ratings.groupby('user_id').apply(get_time_to_five)

# 5 yıldız vermeyen (None/NaN dönen) kullanıcıları analizden çıkarıyoruz
Tu = Tu_series.dropna()

print(f"Toplam Kullanıcı: {len(Tu_series)}")
print(f"En az bir '5' veren (Analize giren): {len(Tu)}")
print(f"Ortalama Bekleme Süresi (Mean Tu): {Tu.mean():.2f} round")


# ==========================================
# 2. GEOMETRIC MODEL (Sabit p)
# ==========================================
# Model: Herkesin beğenme ihtimali (p) aynıdır.
# Geometrik dağılımda E[T] = 1/p  =>  p = 1 / mean(T)

p_geo = 1 / Tu.mean()

print("\n" + "="*40)
print(" MODEL 1: GEOMETRIC (Sabit Sabır)")
print("="*40)
print(f"Tahmini p (Beğenme Olasılığı): {p_geo:.4f}")
print(f"Yorum: Ortalama bir kullanıcı, sunulan herhangi bir şarkıyı %{p_geo*100:.1f} ihtimalle 5 yıldızlar.")


# ==========================================
# 3. BETA-GEOMETRIC MODEL (Değişken p)
# ==========================================
# Model: Her kullanıcının p'si farklıdır ve bu p'ler Beta(alpha, beta) dağılımından gelir.

print("\n" + "="*40)
print(" MODEL 2: BETA-GEOMETRIC (Kişisel Sabır)")
print("="*40)

def neg_log_likelihood(params, t_values):
    alpha, beta = params
    if alpha <= 0 or beta <= 0: return np.inf

    # Beta-Geometric Olasılık Kütle Fonksiyonu (Log scale'de işlem yapıyoruz taşmayı önlemek için)
    # P(T=t) = Beta(alpha+1, beta+t-1) / Beta(alpha, beta)
    # Log P(T=t) = betaln(alpha+1, beta+t-1) - betaln(alpha, beta)

    log_probs = betaln(alpha + 1, beta + t_values - 1) - betaln(alpha, beta)
    return -np.sum(log_probs) # Minimize edeceğimiz için negatifi

# Optimizasyon
initial_guess = [1.0, 1.0]
result = minimize(neg_log_likelihood, initial_guess, args=(Tu.values,), method='Nelder-Mead')
alpha_fit, beta_fit = result.x

print(f"Optimizasyon Sonucu:")
print(f"Alpha: {alpha_fit:.4f}")
print(f"Beta : {beta_fit:.4f}")
print(f"Ortalama p (Alpha / Alpha+Beta): {alpha_fit / (alpha_fit + beta_fit):.4f}")


# ==========================================
# 4. HİPOTEZ TESTİ (Gruplar Arası Fark)
# ==========================================
print("\n" + "="*40)
print(" HİPOTEZ TESTİ: Popüler vs Alternatif")
print("="*40)

# Kullanıcıları ayırmak için şarkı popülaritesini kullanacağız
# Hangi kullanıcı ortalama ne kadar popüler şarkı dinlemiş?
merged_df = pd.merge(ratings, tracks[['track_id', 'track_popularity']], left_on='song_id', right_on='track_id')
user_pop_mean = merged_df.groupby('user_id')['track_popularity'].mean()

# Medyan ile ikiye bölüyoruz
median_pop = user_pop_mean.median()
high_pop_users = user_pop_mean[user_pop_mean >= median_pop].index
low_pop_users = user_pop_mean[user_pop_mean < median_pop].index

# Grupların Tu değerlerini çekelim (Sadece 5 verenler)
Tu_high = Tu[Tu.index.isin(high_pop_users)]
Tu_low = Tu[Tu.index.isin(low_pop_users)]

print(f"Grup 1 (Popüler Sevenler) Ort. Tu: {Tu_high.mean():.2f} (n={len(Tu_high)})")
print(f"Grup 2 (Niş Sevenler)     Ort. Tu: {Tu_low.mean():.2f} (n={len(Tu_low)})")

# Mann-Whitney U Testi (Veri normal dağılmadığı için t-test yerine bunu seçtik)
stat, p_val = stats.mannwhitneyu(Tu_high, Tu_low, alternative='two-sided')

print(f"Mann-Whitney U p-value: {p_val:.5f}")
if p_val < 0.05:
    print("SONUÇ: Fark İSTATİSTİKSEL OLARAK ANLAMLI! (H0 Red)")
else:
    print("SONUÇ: Anlamlı bir fark yok. (H0 Kabul)")

# ==========================================
# 5. GÖRSELLEŞTİRME (RAPOR İÇİN)
# ==========================================
plt.figure(figsize=(10, 6))

# Gerçek Veri Histogramı
plt.hist(Tu, bins=range(1, 30), density=True, alpha=0.6, color='gray', label='Gerçek Veri (Tu)')

# Modellerin Çizimi
t_range = np.arange(1, 30)

# Geometric Model Çizgisi
y_geo = (1 - p_geo)**(t_range - 1) * p_geo
plt.plot(t_range, y_geo, 'r--', linewidth=2, label=f'Geometric (p={p_geo:.2f})')

# Beta-Geometric Model Çizgisi (Formül: P(T=t) = B(a+1, b+t-1)/B(a,b))
# betaln logaritma döndürdüğü için exp ile normale çeviriyoruz
y_bg = np.exp(betaln(alpha_fit + 1, beta_fit + t_range - 1) - betaln(alpha_fit, beta_fit))
plt.plot(t_range, y_bg, 'b-', linewidth=2, label=f'Beta-Geo (a={alpha_fit:.1f}, b={beta_fit:.1f})')

plt.title("Kullanıcı Sabrı Dağılımı: Gerçek vs Modeller")
plt.xlabel("İlk 5* İçin Gereken Deneme Sayısı (Tu)")
plt.ylabel("Olasılık")
plt.legend()
plt.grid(True, alpha=0.3)

# Resmi kaydet
plt.savefig("part2_analysis_plot.png")
print("\nGrafik 'part2_analysis_plot.png' olarak kaydedildi.")