import os
import re
import pandas as pd
import numpy as np
from itertools import combinations


# 1. DATA LOADING

def _safe_name(s: str) -> str:
    """Make a string safe for file or column names."""
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    return s.strip("_")


def stats_to_long(stats_df, feature_cols, pair_name=None):
    """
    Converts conditional probability tables into long format.

    stats_df: output of calculate_smoothed_prob
    feature_cols: ['col'] or ['col1', 'col2']
    """
    out = stats_df.reset_index().copy()
    out.insert(0, "feature", pair_name if pair_name else "|".join(feature_cols))

    # Rename feature value columns as v1, v2, ...
    for i, c in enumerate(feature_cols, start=1):
        out.rename(columns={c: f"v{i}"}, inplace=True)

    keep = (
            ["feature"]
            + [f"v{i}" for i in range(1, len(feature_cols) + 1)]
            + ["prob", "total_count", "success_count"]
    )
    return out[keep]


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_data():
    print(">>> Loading CSV files...")

    # Tracks
    try:
        tracks = pd.read_csv("../tracks.csv")
        print(f"Tracks loaded: {len(tracks)} rows.")
    except Exception:
        print("ERROR: tracks.csv not found!")
        return None, None, None

    # Global ratings
    try:
        global_ratings = pd.read_csv("ratings.csv", sep=";")
        if len(global_ratings.columns) < 3:
            global_ratings = pd.read_csv("ratings.csv", sep=",")
        print(f"ratings.csv loaded: {len(global_ratings)} rows.")
    except Exception as e:
        print("ERROR loading ratings.csv:", e)
        return None, None, None

    # Personal ratings
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


# 2. MERGE AND TARGET VARIABLE

def merge_and_prep(ratings_df, tracks_df):
    """Merge ratings with track metadata and create target variable."""
    df = pd.merge(
        ratings_df,
        tracks_df,
        left_on="song_id",
        right_on="track_id",
        how="left"
    )
    df["is_5_star"] = (df["rating"] == 5).astype(int)
    return df


# 3. BINNING FOR NUMERIC FEATURES

def apply_binning(df):
    """Discretize numeric features into categorical bins."""

    if "track_popularity" in df.columns:
        df["popularity_bin"] = pd.qcut(
            df["track_popularity"],
            q=4,
            labels=["very_low", "low", "high", "very_high"],
            duplicates="drop"
        )

    if "duration_ms" in df.columns:
        df["duration_bin"] = pd.cut(
            df["duration_ms"] / 1000,
            bins=[0, 180, 300, float("inf")],
            labels=["short", "medium", "long"]
        )

    if "available_markets_count" in df.columns:
        try:
            _, bins = pd.qcut(
                df["available_markets_count"],
                q=3,
                retbins=True,
                duplicates="drop"
            )
            num_bins = len(bins) - 1
            if num_bins == 3:
                labels = ["few", "medium", "many"]
            elif num_bins == 2:
                labels = ["few", "many"]
            else:
                labels = [f"bin_{i}" for i in range(num_bins)]

            df["markets_bin"] = pd.cut(
                df["available_markets_count"],
                bins=bins,
                labels=labels
            )
        except Exception:
            df["markets_bin"] = "unknown"

    if "album_release_year" in df.columns:
        df["year_bin"] = pd.cut(
            df["album_release_year"],
            bins=[0, 1979, 1989, 1999, 2009, 2019, 2035],
            labels=[
                "pre_1980", "1980s", "1990s",
                "2000s", "2010s", "2020s"
            ]
        )

    return df


# 4. LAPLACE CONDITIONAL PROBABILITY

def calculate_smoothed_prob(df, features, target="is_5_star", alpha=1):
    """Compute Laplace-smoothed conditional probabilities."""
    stats = df.groupby(features, observed=False)[target].agg(["count", "sum"])
    stats.columns = ["total_count", "success_count"]

    k = 2  # Binary outcome: 5★ vs not 5★
    stats["prob"] = (
            (stats["success_count"] + alpha)
            / (stats["total_count"] + alpha * k)
    )

    return stats.sort_values("prob", ascending=False)


# 4.5 EXPORT HELPERS (CSV OUTPUT)

def export_single_feature_probs(df, feature_list, out_csv, alpha=1):
    """Export P(5★ | feature=value) tables into one CSV."""
    rows = []

    for feat in feature_list:
        if feat not in df.columns:
            continue

        stats = calculate_smoothed_prob(df, [feat], alpha=alpha).reset_index()
        stats.rename(columns={feat: "value"}, inplace=True)
        stats.insert(0, "feature", feat)

        rows.append(
            stats[["feature", "value", "prob", "total_count", "success_count"]]
        )

    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=["feature", "value", "prob", "total_count", "success_count"]
        )
    )

    out.to_csv(out_csv, index=False)


def export_top_interactions(df, out_csv, top_k=10, min_count=30, alpha=1):
    """Export strongest feature interactions into CSV."""

    blacklist = [
        "is_5_star", "rating", "track_id", "user_id",
        "track_name", "artist_name", "album_name",
        "uri", "url", "mbid", "score", "index"
    ]

    valid_cols = [
        col for col in df.columns
        if not any(b in col for b in blacklist)
           and df[col].nunique() < 50
    ]

    all_pairs = list(combinations(valid_cols, 2))

    top_results = find_top_interactions(
        df,
        candidate_pairs=all_pairs,
        top_k=top_k,
        min_count=min_count,
        alpha=alpha
    )

    rows = []

    for inter in top_results:
        table = inter["table"].reset_index()
        table.rename(
            columns={inter["f1"]: "v1", inter["f2"]: "v2"},
            inplace=True
        )

        table.insert(0, "f1", inter["f1"])
        table.insert(1, "f2", inter["f2"])
        table["score"] = inter["score"]

        rows.append(
            table[
                ["f1", "f2", "v1", "v2",
                 "prob", "total_count", "success_count", "score"]
            ]
        )

    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "f1", "f2", "v1", "v2",
                "prob", "total_count", "success_count", "score"
            ]
        )
    )

    out.to_csv(out_csv, index=False)


def export_posteriors_task3(df, out_csv):
    """Export posterior distributions used in Task 3."""
    subset = df[df["is_5_star"] == 1]
    rows = []

    if "primary_artist_name" in subset.columns:
        dist = subset["primary_artist_name"].value_counts(normalize=True).reset_index()
        dist.columns = ["value", "posterior"]
        dist.insert(0, "feature", "P(artist | 5star)")
        rows.append(dist)

    col_name = "ab_genre_dortmund_value"
    if col_name in subset.columns:
        dist = subset[col_name].value_counts(normalize=True).reset_index()
        dist.columns = ["value", "posterior"]
        dist.insert(0, "feature", f"P({col_name} | 5star)")
        rows.append(dist)

    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["feature", "value", "posterior"])
    )

    out.to_csv(out_csv, index=False)


# 5. TASK 2 INTERACTION SCORING

def find_top_interactions(df, candidate_pairs, top_k=5, min_count=25, alpha=1):
    """Find strongest feature interactions based on deviation from global mean."""
    results = []
    global_p = df["is_5_star"].mean()

    for f1, f2 in candidate_pairs:
        if f1 not in df.columns or f2 not in df.columns:
            continue

        stats = calculate_smoothed_prob(df, [f1, f2], alpha=alpha)
        valid = stats[stats["total_count"] >= min_count]

        if valid.empty:
            continue

        weight = valid["total_count"] / valid["total_count"].sum()
        score = ((valid["prob"] - global_p) ** 2 * weight).sum()

        results.append({
            "f1": f1,
            "f2": f2,
            "score": score,
            "table": valid.sort_values("prob", ascending=False)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# 6. TASK 2 — PRINTED ANALYSIS

def run_task2(df):
    print("\n===== TASK 2: Automatic Interaction Discovery =====\n")

    blacklist = [
        "is_5_star", "rating", "track_id", "user_id",
        "track_name", "artist_name", "album_name",
        "uri", "url", "mbid", "score", "index"
    ]

    valid_cols = [
        col for col in df.columns
        if not any(b in col for b in blacklist)
           and df[col].nunique() < 50
    ]

    print(f"Features used ({len(valid_cols)}):")
    print(valid_cols)
    print("-" * 40)

    all_pairs = list(combinations(valid_cols, 2))
    print(f"Checking {len(all_pairs)} feature pairs...")

    top_results = find_top_interactions(
        df,
        candidate_pairs=all_pairs,
        top_k=10,
        min_count=30,
        alpha=1
    )

    print("\n>>> Strongest interactions:\n")
    for i, inter in enumerate(top_results):
        print("=" * 60)
        print(f"#{i + 1}: {inter['f1']} x {inter['f2']}")
        print(f"Score: {inter['score']:.4f}")
        print(inter["table"][["prob", "total_count", "success_count"]].head(3))
        print()


# ============================================================
# 7. TASK 3 — BAYESIAN ANALYSIS
# ============================================================

def run_task3(df):
    print("\n===== TASK 3: Bayesian Interpretation =====\n")

    subset = df[df["is_5_star"] == 1]

    print(">>> P(Artist | 5★)")
    print(subset["primary_artist_name"].value_counts(normalize=True).head(10))
    print("-" * 40)

    col_name = "ab_genre_dortmund_value"
    if col_name in df.columns:
        print(f"\n>>> P({col_name} | 5★)")
        dist = subset[col_name].value_counts(normalize=True)
        print(dist)

        top_genre = dist.index[0]
        top_prob = dist.iloc[0]
        print(f"\n[Comment]: {top_prob * 100:.1f}% are '{top_genre}'")


# 8. TASK 4 — PERSONAL VS GLOBAL

def run_task4(df_global, df_personal):
    print("\n===== TASK 4: Personal vs Global =====")

    feature = "ab_genre_rosamerica_value"
    if feature not in df_global.columns:
        print("Genre feature missing.")
        return

    print("\nPersonal taste:")
    print(calculate_smoothed_prob(df_personal, feature).head(5))

    print("\nGlobal taste:")
    print(calculate_smoothed_prob(df_global, feature).head(5))


# 9. MAIN

def main():
    tracks, raw_global, raw_personal = load_data()
    if tracks is None:
        return

    df_global = apply_binning(merge_and_prep(raw_global, tracks))
    df_personal = apply_binning(merge_and_prep(raw_personal, tracks))

    print("\n========== PART 1 RESULTS ==========\n")

    features = [
        "primary_artist_name", "explicit", "year_bin",
        "popularity_bin", "duration_bin", "markets_bin"
    ]
    audio = [c for c in df_global.columns if c.startswith("ab_") and "mbid" not in c]
    features.extend(audio)

    print(">>> TASK 1: Global Conditional Probabilities\n")

    for feat in features:
        print("\n" + "-" * 60)
        print(f"FEATURE: {feat}")
        print("-" * 60)

        stats = calculate_smoothed_prob(df_global, feat)
        if len(stats) > 10:
            print(stats.head(5)[["prob", "total_count"]])
            print(stats.tail(3)[["prob", "total_count"]])
        else:
            print(stats[["prob", "total_count"]])

    ensure_dir("artifacts")

    export_single_feature_probs(
        df_global, features,
        out_csv="artifacts/cond_probs_global.csv"
    )

    export_single_feature_probs(
        df_personal, features,
        out_csv="artifacts/cond_probs_personal.csv"
    )

    export_top_interactions(
        df_global,
        out_csv="artifacts/interactions_top.csv"
    )

    export_posteriors_task3(
        df_global,
        out_csv="artifacts/posteriors.csv"
    )

    print("Wrote CSV artifacts to ./artifacts/")

    run_task2(df_global)
    run_task3(df_global)
    run_task4(df_global, df_personal)


if __name__ == "__main__":
    main()