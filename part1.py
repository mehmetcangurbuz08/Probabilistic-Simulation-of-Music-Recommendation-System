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

    # ------------------------------------------------
    # Personal + Friends (NO HEADER, SAFE LOAD)
    # ------------------------------------------------
    def _load_personal(path,seperate):
        df = pd.read_csv(path, sep=seperate, header=None, encoding="latin1")
        df = df.iloc[:, [0, 1, 2, 4]]
        df.columns = ["user_id", "round_idx", "song_id", "rating"]
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        return df

    try:
        me = _load_personal("personal_ratings/sarkilar.csv",seperate=";")
        f1 = _load_personal("personal_ratings/mehmet_ratings.csv",seperate=",")
        f2 = _load_personal("personal_ratings/furkan_ratings.csv",seperate=",")

        print(f"Personal loaded: {len(me)} rows.")
        print(f"Friend 1 loaded: {len(f1)} rows.")
        print(f"Friend 2 loaded: {len(f2)} rows.")

    except Exception as e:
        print("ERROR loading personal rating files:", e)
        return None, None, None, None, None

    return tracks, global_ratings, me, f1, f2



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

import matplotlib.pyplot as plt
import numpy as np

def plot_interaction_heatmap(interaction, title=None):
    """
    Matplotlib-only heatmap for Task 2
    (NO seaborn, fully allowed)
    """

    table = interaction["table"].reset_index()

    f1 = interaction["f1"]
    f2 = interaction["f2"]

    # Pivot to matrix
    pivot = table.pivot(index=f1, columns=f2, values="prob")

    data = pivot.values
    x_labels = pivot.columns.astype(str)
    y_labels = pivot.index.astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data, aspect="auto", cmap="viridis")

    # Ticks
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    # Labels
    ax.set_xlabel(f2)
    ax.set_ylabel(f1)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(5★ | feature pair)")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(
                    j, i,
                    f"{data[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if data[i, j] > np.nanmean(data) else "black",
                    fontsize=8
                )

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Interaction Heatmap: {f1} × {f2}")

    plt.tight_layout()
    plt.savefig("plots/heatmap_task2.png")


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

    # Plot heatmap for the strongest interaction
    best_interaction = top_results[0]

    plot_interaction_heatmap(
    best_interaction,
    title=f"Strongest Interaction: {best_interaction['f1']} × {best_interaction['f2']}"
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

import matplotlib.pyplot as plt

def plot_task3_two_posteriors(df):
    """
    One figure with two subplots:
    - P(Artist | 5★)
    - P(Genre | 5★)
    """

    subset = df[df["is_5_star"] == 1]

    # --- Artist posterior ---
    artist_post = (
        subset["primary_artist_name"]
        .value_counts(normalize=True)
        .head(10)
        .sort_values()
    )

    # --- Genre posterior ---
    genre_post = (
        subset["ab_genre_dortmund_value"]
        .value_counts(normalize=True)
        .head(8)
        .sort_values()
    )

    # -----------------------
    # Plot
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    artist_post.plot(
        kind="barh",
        ax=axes[0]
    )
    axes[0].set_title("Posterior Distribution: P(Artist | 5★)")
    axes[0].set_xlabel("Posterior Probability")

    genre_post.plot(
        kind="barh",
        ax=axes[1]
    )
    axes[1].set_title("Posterior Distribution: P(Genre | 5★)")
    axes[1].set_xlabel("Posterior Probability")

    plt.suptitle("Task 3 — Posterior Distributions Given 5★ Ratings", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

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

    plot_task3_two_posteriors(df)
    plt.savefig("plots/car_chart.png")


# 8. TASK 4 — PERSONAL VS GLOBAL

def plot_genre_pie_ax(ax, stats, title, top_k=5):
    top = stats.head(top_k)

    ax.pie(
        top["prob"],
        labels=top.index.astype(str),
        autopct="%1.1f%%",
        startangle=140
    )
    ax.set_title(title)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def run_task4(df_global, df_personal, df_f1, df_f2):
    """
    PART 4:
    Personal vs Friends vs Global
    Single figure with 4 subplots
    """

    feature = "ab_genre_rosamerica_value"

    if feature not in df_global.columns:
        print("Feature missing.")
        return

    # Probabilities
    personal_probs = calculate_smoothed_prob(df_personal, feature)
    furkan_probs = calculate_smoothed_prob(df_f1, feature)
    mehmet_probs = calculate_smoothed_prob(df_f2, feature)
    global_probs = calculate_smoothed_prob(df_global, feature)

    # Print tables (as before)
    print("\n--- PERSONAL (You) ---")
    print(personal_probs.head(5))

    print("\n--- FRIEND 1 (Furkan) ---")
    print(furkan_probs.head(5))

    print("\n--- FRIEND 2 (Mehmet Can) ---")
    print(mehmet_probs.head(5))

    print("\n--- GLOBAL ---")
    print(global_probs.head(5))

    # ============================
    # PLOTTING
    # ============================
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    plot_genre_pie_ax(
        axes[0, 0],
        personal_probs,
        "Personal Taste (You)"
    )

    plot_genre_pie_ax(
        axes[0, 1],
        furkan_probs,
        "Taste of Furkan"
    )

    plot_genre_pie_ax(
        axes[1, 0],
        mehmet_probs,
        "Taste of Mehmet Can"
    )

    plot_genre_pie_ax(
        axes[1, 1],
        global_probs,
        "Global Taste"
    )

    plt.suptitle(
        "Comparison of Musical Preferences via P(5★ | Genre)",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/piechart_for_each.png')


    # 9. MAIN

def main():
    tracks, raw_global, me, f1, f2 = load_data()
    if tracks is None:
        return

    df_personal = apply_binning(merge_and_prep(me, tracks))
    df_f1 = apply_binning(merge_and_prep(f1, tracks))
    df_f2 = apply_binning(merge_and_prep(f2, tracks))
    df_global = apply_binning(merge_and_prep(raw_global, tracks))
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
    run_task4(df_global, df_personal,df_f1,df_f2)


if __name__ == "__main__":
    main()