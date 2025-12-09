import pandas as pd
import numpy as np


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data():
    print(">>> Loading CSV files...")

    # TRACKS
    try:
        tracks = pd.read_csv("tracks.csv")
        print(f"Tracks loaded: {len(tracks)} rows.")
    except:
        print("ERROR: tracks.csv not found!")
        return None, None, None

    # GLOBAL RATINGS
    try:
        global_ratings = pd.read_csv("ratings.csv", sep=";")
        if len(global_ratings.columns) < 3:
            global_ratings = pd.read_csv("ratings.csv", sep=",")
        print(f"ratings.csv loaded: {len(global_ratings)} rows.")
    except Exception as e:
        print("ERROR loading ratings.csv:", e)
        return None, None, None

    # PERSONAL RATINGS
    try:
        my_ratings = pd.read_csv("sarkilar.csv", sep=";", header=None)
        my_ratings = my_ratings.iloc[:, [0, 1, 2, 4]]
        my_ratings.columns = ["user_id", "round_idx", "song_id", "rating"]
        my_ratings["rating"] = pd.to_numeric(my_ratings["rating"], errors="coerce")
        print(f"sarkilar.csv (personal) loaded: {len(my_ratings)} rows.")
    except Exception as e:
        print("ERROR loading sarkilar.csv:", e)
        return None, None, None

    return tracks, global_ratings, my_ratings


# ============================================================
# 2. MERGE AND TARGET VARIABLE
# ============================================================

def merge_and_prep(ratings_df, tracks_df):
    df = pd.merge(ratings_df, tracks_df,
                  left_on="song_id", right_on="track_id",
                  how="left")
    df["is_5_star"] = (df["rating"] == 5).astype(int)
    return df


# ============================================================
# 3. BINNING FOR NUMERIC FEATURES
# ============================================================

def apply_binning(df):

    # Popularity
    if "track_popularity" in df.columns:
        df["popularity_bin"] = pd.qcut(
            df["track_popularity"], q=4,
            labels=["very_low", "low", "high", "very_high"],
            duplicates="drop"
        )

    # Duration
    if "duration_ms" in df.columns:
        df["duration_bin"] = pd.cut(
            df["duration_ms"] / 1000,
            bins=[0, 180, 300, float("inf")],
            labels=["short", "medium", "long"]
        )

    # Markets — flexible binning
    if "available_markets_count" in df.columns:
        try:
            _, bins = pd.qcut(
                df["available_markets_count"], q=3,
                retbins=True, duplicates="drop"
            )
            num_bins = len(bins) - 1
            if num_bins == 3:
                labels = ["few", "medium", "many"]
            elif num_bins == 2:
                labels = ["few", "many"]
            else:
                labels = [f"bin_{i}" for i in range(num_bins)]

            df["markets_bin"] = pd.cut(df["available_markets_count"],
                                       bins=bins, labels=labels)
        except:
            df["markets_bin"] = "unknown"

    # Year binning
    if "album_release_year" in df.columns:
        df["year_bin"] = pd.cut(
            df["album_release_year"],
            bins=[0, 1979, 1989, 1999, 2009, 2019, 2035],
            labels=["pre_1980", "1980s", "1990s", "2000s", "2010s", "2020s"]
        )

    return df


# ============================================================
# 4. LAPLACE CONDITIONAL PROBABILITY
# ============================================================

def calculate_smoothed_prob(df, features, target="is_5_star", alpha=1):
    stats = df.groupby(features)[target].agg(["count", "sum"])
    stats.columns = ["total_count", "success_count"]
    k = 2  # binary: 5★ vs not
    stats["prob"] = (stats["success_count"] + alpha) / \
                    (stats["total_count"] + alpha * k)
    return stats.sort_values("prob", ascending=False)


# ============================================================
# 5. TASK 2 INTERACTION SCORING
# ============================================================

def find_top_interactions(df, candidate_pairs,
                          top_k=5, min_count=25, alpha=1):

    results = []
    global_p = df["is_5_star"].mean()

    for f1, f2 in candidate_pairs:

        if f1 not in df.columns or f2 not in df.columns:
            continue

        stats = calculate_smoothed_prob(df, [f1, f2], alpha=alpha)
        valid = stats[stats["total_count"] >= min_count]
        if len(valid) == 0:
            continue

        # Interaction score = weighted deviation from global mean
        weight = valid["total_count"] / valid["total_count"].sum()
        score = ((valid["prob"] - global_p)**2 * weight).sum()

        results.append({
            "f1": f1,
            "f2": f2,
            "score": score,
            "table": valid.sort_values("prob", ascending=False)
        })

    # sort by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def run_task2(df):

    print("\n===== TASK 2: Top Interaction Effects =====\n")

    mood_cols = [c for c in df.columns if c.startswith("ab_mood_")]
    genre_cols = [c for c in df.columns if c.startswith("ab_genre_")]

    candidate_pairs = []

    # Meaningful interaction families
    if "year_bin" in df.columns:
        for g in genre_cols:
            candidate_pairs.append(("year_bin", g))

    if "popularity_bin" in df.columns:
        for m in mood_cols:
            candidate_pairs.append(("popularity_bin", m))

    if "duration_bin" in df.columns:
        for m in mood_cols:
            candidate_pairs.append(("duration_bin", m))

    if "explicit" in df.columns:
        for m in mood_cols:
            candidate_pairs.append(("explicit", m))

    top_results = find_top_interactions(df, candidate_pairs,
                                        top_k=5,
                                        min_count=25,
                                        alpha=1)

    for inter in top_results:
        print("-" * 60)
        print(f"Interaction: {inter['f1']} × {inter['f2']}")
        print(f"Score = {inter['score']:.4f}")
        print("Top categories:")
        print(inter["table"][["prob", "total_count"]].head(5))
        print()


# ============================================================
# 6. TASK 3 — BAYESIAN ANALYSIS
# ============================================================

def run_task3(df):
    print("\n===== TASK 3: Bayesian Interpretation P(Artist | 5★) =====\n")
    subset = df[df["is_5_star"] == 1]
    dist = subset["primary_artist_name"].value_counts(normalize=True)
    print(dist.head(10))


# ============================================================
# 7. TASK 4 — PERSONAL VS GLOBAL
# ============================================================

def run_task4(df_global, df_personal):

    print("\n===== TASK 4: Personal vs Global Comparison =====")

    feature = "ab_genre_rosamerica_value"

    if feature not in df_global.columns:
        print("Genre feature missing.")
        return

    print("\nYour personal taste for genre:")
    print(calculate_smoothed_prob(df_personal, feature).head(5))

    print("\nGlobal taste for genre:")
    print(calculate_smoothed_prob(df_global, feature).head(5))


# ============================================================
# 8. MAIN
# ============================================================

def main():

    tracks, raw_global, raw_personal = load_data()
    if tracks is None:
        return

    df_global = merge_and_prep(raw_global, tracks)
    df_personal = merge_and_prep(raw_personal, tracks)

    df_global = apply_binning(df_global)
    df_personal = apply_binning(df_personal)

    print("\n==============================================")
    print(" PART 1 ANALYSIS RESULTS ")
    print("==============================================\n")

    # ------------------------------
    # TASK 1
    # ------------------------------
    print(">>> TASK 1: Global Conditional Probabilities\n")

    features = [
        "primary_artist_name",
        "explicit",
        "year_bin",
        "popularity_bin",
        "duration_bin",
        "markets_bin",
    ]
    audio = [c for c in df_global.columns if c.startswith("ab_") and "mbid" not in c]
    features.extend(audio)

    for feat in features:
        print("\n" + "-" * 60)
        print(f"FEATURE: {feat}")
        print("-" * 60)

        stats = calculate_smoothed_prob(df_global, feat)

        if len(stats) > 10:
            print("Top 5:")
            print(stats.head(5)[["prob", "total_count"]])
            print("\nBottom 3:")
            print(stats.tail(3)[["prob", "total_count"]])
        else:
            print(stats[["prob", "total_count"]])

    # ------------------------------
    # TASK 2
    # ------------------------------
    run_task2(df_global)

    # ------------------------------
    # TASK 3
    # ------------------------------
    run_task3(df_global)

    # ------------------------------
    # TASK 4
    # ------------------------------
    run_task4(df_global, df_personal)

    print("\nAnalysis Complete.")


if __name__ == "__main__":
    main()