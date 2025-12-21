import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


# Lanczos coefficients for better gamma function approximation
_lanczos_g = 7
_lanczos_coef = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
]


def log_gamma(z):
    """Lanczos approximation for log(Gamma(z))"""
    if np.isscalar(z):
        if z <= 0:
            return np.inf
        if z < 0.5:
            # Use reflection formula: Gamma(z)*Gamma(1-z) = pi/sin(pi*z)
            return np.log(np.pi) - np.log(np.sin(np.pi * z)) - log_gamma(1 - z)

        z -= 1
        x = _lanczos_coef[0]
        for i in range(1, _lanczos_g + 2):
            x += _lanczos_coef[i] / (z + i)

        t = z + _lanczos_g + 0.5
        return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)
    else:
        return np.array([log_gamma(zi) for zi in z])


def betaln_custom(a, b):
    """Natural logarithm of the beta function"""
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


def neg_log_likelihood(params, t_values):
    """Negative log-likelihood of the Beta-Geometric model."""
    alpha, beta = params

    # Prevent invalid values during optimization
    if alpha <= 1e-5 or beta <= 1e-5:
        return 1e10

    # Beta-Geometric PMF in log-space
    log_probs = betaln_custom(alpha + 1, beta + t_values - 1) - betaln_custom(alpha, beta)

    return -np.sum(log_probs)


def minimize_bfgs(func, x0, args, bounds, maxiter=15000):
    """
    Simple BFGS with bounds enforcement using projection.
    """
    x = np.array(x0, dtype=float)
    n = len(x)

    def project_bounds(x):
        """Project x onto bounds"""
        x_proj = x.copy()
        for i in range(n):
            if bounds[i][0] is not None:
                x_proj[i] = max(x_proj[i], bounds[i][0])
            if bounds[i][1] is not None:
                x_proj[i] = min(x_proj[i], bounds[i][1])
        return x_proj

    def gradient(x):
        """Numerical gradient"""
        grad = np.zeros(n)
        f0 = func(x, *args)
        eps = 1e-8
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_plus = project_bounds(x_plus)
            grad[i] = (func(x_plus, *args) - f0) / eps
        return grad

    # Initialize
    x = project_bounds(x)
    B = np.eye(n)  # Inverse Hessian approximation

    f_old = func(x, *args)
    g_old = gradient(x)

    for iteration in range(maxiter):
        # Compute search direction
        p = -np.dot(B, g_old)

        # Line search with Armijo condition
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5

        x_new = project_bounds(x + alpha * p)
        f_new = func(x_new, *args)

        # Backtracking
        max_backtracks = 30
        for _ in range(max_backtracks):
            if f_new <= f_old + c1 * alpha * np.dot(g_old, p):
                break
            alpha *= rho
            x_new = project_bounds(x + alpha * p)
            f_new = func(x_new, *args)

        # Update
        s = x_new - x
        x = x_new

        g_new = gradient(x)
        y = g_new - g_old

        # BFGS update
        rho_k = 1.0 / (np.dot(y, s) + 1e-10)

        if rho_k > 0:  # Only update if valid
            I = np.eye(n)
            B = np.dot(I - rho_k * np.outer(s, y), np.dot(B, I - rho_k * np.outer(y, s))) + rho_k * np.outer(s, s)

        # Check convergence
        if np.linalg.norm(g_new) < 1e-5:
            break

        if iteration > 50 and abs(f_new - f_old) < 1e-9:
            break

        g_old = g_new
        f_old = f_new

    class Result:
        def __init__(self, x, fun, success):
            self.x = x
            self.fun = fun
            self.success = success

    return Result(x, f_new, True)


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
    result = minimize_bfgs(
        neg_log_likelihood,
        x0=[alpha_init, beta_init],
        args=(Tu.values,),
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


def erf(x):
    """Error function approximation"""
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def norm_cdf(x):
    """Standard normal CDF"""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def mannwhitneyu_custom(x, y, alternative="two-sided"):
    """Mann-Whitney U test"""
    x = np.asarray(x)
    y = np.asarray(y)
    n1 = len(x)
    n2 = len(y)

    # Rank all observations
    combined = np.concatenate([x, y])
    ranks = np.empty(len(combined), dtype=float)

    # Sort and assign ranks
    sorter = np.argsort(combined)
    sorted_data = combined[sorter]

    # Assign ranks (1-indexed)
    ranks[sorter] = np.arange(1, len(combined) + 1)

    # Handle ties - average ranks
    i = 0
    while i < len(sorted_data):
        j = i
        # Find all equal values
        while j < len(sorted_data) and sorted_data[j] == sorted_data[i]:
            j += 1
        # If there are ties, average the ranks
        if j > i + 1:
            avg_rank = ranks[sorter[i:j]].mean()
            ranks[sorter[i:j]] = avg_rank
        i = j

    # Sum ranks for first sample
    R1 = ranks[:n1].sum()

    # Calculate U statistic
    U1 = R1 - n1 * (n1 + 1) / 2

    # Normal approximation with tie correction
    m_u = n1 * n2 / 2

    # Tie correction
    unique_vals, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts ** 3 - counts) / 12.0

    s_u = np.sqrt(n1 * n2 * ((n1 + n2 + 1) / 12.0 - tie_term / ((n1 + n2) * (n1 + n2 - 1))))

    # Continuity correction
    if U1 > m_u:
        z = (U1 - m_u - 0.5) / s_u
    else:
        z = (U1 - m_u + 0.5) / s_u

    # Two-sided p-value
    p_value = 2.0 * (1.0 - norm_cdf(abs(z)))

    return U1, p_value


stat, p_val = mannwhitneyu_custom(tu_high, tu_low, alternative="two-sided")
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
    betaln_custom(alpha_fit + 1, beta_fit + t_range - 1) - betaln_custom(alpha_fit, beta_fit)
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

plt.savefig("plots/part2_final_result.png")
print("Plot saved as 'part2_final_result.png'")
print("\nANALYSIS COMPLETE. Do not forget to record Alpha and Beta values.")