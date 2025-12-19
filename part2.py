import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln


# 1. DATA LOADING AND PATIENCE (Tu) COMPUTATION

print("\n" + "=" * 50)
print(" 1. DATA LOADING AND Tu (PATIENCE) COMPUTATION")
print("=" * 50)

try:
    tracks = pd.read_csv("../tracks.csv")
    ratings = pd.read_csv("ratings.csv")
    print("Data loaded successfully.")
    print(f"- Ratings: {len(ratings)} rows")
    print(f"- Tracks : {len(tracks)} rows")
except FileNotFoundError:
    print("ERROR: 'tracks.csv' or 'ratings.csv' not found!")
    exit()


def calculate_user_patience(group):
    """
    Finds when a user gives their first 5-star rating.
    Returns the song index (Tu). If no 5-star exists, returns NaN.
    """

    # Sort by interaction order if available
    if "round_idx" in group.columns:
        group = group.sort_values("round_idx")

    # Find 5-star ratings
    fives = group[group["rating"] == 5]

    if not fives.empty:
        if "round_idx" in group.columns:
            # +1 to represent "which song number"
            return fives["round_idx"].iloc[0] + 1
        else:
            return np.where(group["rating"] == 5)[0][0] + 1
    else:
        return np.nan


print(">>> Computing user patience values...")
tu_raw = ratings.groupby("user_id").apply(calculate_user_patience)

# Remove users who never gave a 5-star rating
Tu = tu_raw.dropna()

print(f"Total users: {len(tu_raw)}")
print(f"Users with at least one 5★: {len(Tu)}")
print(f"Average patience (Tu): {Tu.mean():.2f} songs")


# 2. MODEL 1: GEOMETRIC MODEL (SIMPLE BASELINE)

print("\n" + "=" * 50)
print(" 2. GEOMETRIC MODEL (HOMOGENEOUS USERS)")
print("=" * 50)

# E[T] = 1 / p  →  p = 1 / mean(T)
p_geo = 1.0 / Tu.mean()

print(f"Estimated p (success probability): {p_geo:.4f}")
print(f"Interpretation: An average user likes a song with probability {p_geo * 100:.1f}%.")


# 3. MODEL 2: BETA-GEOMETRIC MODEL (HETEROGENEOUS USERS)

print("\n" + "=" * 50)
print(" 3. BETA-GEOMETRIC MODEL (PERSONALIZED)")
print("=" * 50)


def neg_log_likelihood(params, t_values):
    """Negative log-likelihood of the Beta-Geometric model."""
    alpha, beta = params

    # Prevent invalid values during optimization
    if alpha <= 1e-5 or beta <= 1e-5:
        return 1e10

    # Beta-Geometric PMF in log-space
    log_probs = betaln(alpha + 1, beta + t_values - 1) - betaln(alpha, beta)

    return -np.sum(log_probs)


# --- Method of Moments initialization ---
estimated_ps = 1.0 / Tu
mean_p = estimated_ps.mean()
var_p = estimated_ps.var()

if var_p > 1e-10:
    common = (mean_p * (1 - mean_p) / var_p) - 1
    alpha_init = max(0.1, mean_p * common)
    beta_init = max(0.1, (1 - mean_p) * common)
else:
    alpha_init, beta_init = 1.0, 1.0

print(f"Initial guess (Method of Moments): Alpha={alpha_init:.2f}, Beta={beta_init:.2f}")

# --- Optimization ---
bounds = [(1e-3, None), (1e-3, None)]

try:
    result = minimize(
        neg_log_likelihood,
        x0=[alpha_init, beta_init],
        args=(Tu.values,),
        method="L-BFGS-B",
        bounds=bounds
    )
    alpha_fit, beta_fit = result.x
    success = result.success
except Exception as e:
    print(f"Optimization error: {e}")
    alpha_fit, beta_fit = alpha_init, beta_init
    success = False

print("\n>>> RESULTS:")
print(f"Optimization successful?: {success}")
print(f"ALPHA: {alpha_fit:.4f}")
print(f"BETA : {beta_fit:.4f}")

p_model_mean = alpha_fit / (alpha_fit + beta_fit)
print(f"Model mean p: {p_model_mean:.4f}")


# 4. HYPOTHESIS TESTING (POPULAR VS NICHE TASTE)

print("\n" + "=" * 50)
print(" 4. HYPOTHESIS TESTING (POPULARITY EFFECT)")
print("=" * 50)

merged = pd.merge(
    ratings,
    tracks[["track_id", "track_popularity"]],
    left_on="song_id",
    right_on="track_id",
    how="inner"
)

user_pop = merged.groupby("user_id")["track_popularity"].mean()
median_pop = user_pop.median()

high_pop_users = user_pop[user_pop >= median_pop].index
low_pop_users = user_pop[user_pop < median_pop].index

tu_high = Tu[Tu.index.isin(high_pop_users)]
tu_low = Tu[Tu.index.isin(low_pop_users)]

print(f"Popularity lovers (mean Tu): {tu_high.mean():.2f}")
print(f"Niche lovers      (mean Tu): {tu_low.mean():.2f}")

stat, p_val = stats.mannwhitneyu(tu_high, tu_low, alternative="two-sided")
print(f"p-value: {p_val:.5f}")

if p_val < 0.05:
    print("RESULT: Significant difference in patience (reject H0).")
else:
    print("RESULT: No significant difference (fail to reject H0).")


# 5. VISUALIZATION AND SAVING RESULTS

print("\n>>> Drawing plot...")

plt.figure(figsize=(10, 6))

max_t = int(Tu.quantile(0.95))
t_range = np.arange(1, max_t + 1)

plt.hist(
    Tu,
    bins=range(1, max_t + 2),
    density=True,
    alpha=0.5,
    color="gray",
    label="Observed data",
    align="left"
)

y_geo = (1 - p_geo) ** (t_range - 1) * p_geo
plt.plot(t_range, y_geo, "r--", lw=2, label=f"Geometric (p={p_geo:.2f})")

y_bg = np.exp(
    betaln(alpha_fit + 1, beta_fit + t_range - 1) - betaln(alpha_fit, beta_fit)
)
plt.plot(
    t_range,
    y_bg,
    "b-",
    lw=3,
    label=f"Beta-Geometric (α={alpha_fit:.1f}, β={beta_fit:.1f})"
)

plt.title("User Patience Analysis ($T_u$ Distribution)")
plt.xlabel("Number of songs until first 5★")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("part2_final_result.png")
print("Plot saved as 'part2_final_result.png'")
print("\nANALYSIS COMPLETE. Do not forget to record Alpha and Beta values.")