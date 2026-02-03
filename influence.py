"""Influence Asymmetry Analysis

Determines whether power-seeking agents have disproportionate influence on
MoltBook discourse by comparing engagement metrics between flagged and unflagged
posts/authors.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "lysandrehooh/moltbook"
RESULTS_FILE = Path("output/results.json")
OUTPUT_FILE = Path("output/influence.json")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_results() -> dict:
    """Load the analysis results from results.json."""
    print(f"Loading results from {RESULTS_FILE}...")
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    print(f"  Loaded {len(data['results'])} analyzed posts")
    return data


def load_moltbook_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load posts and agents from HuggingFace dataset."""
    print(f"Loading dataset {DATASET_NAME} from HuggingFace...")
    posts_ds = load_dataset(DATASET_NAME, "posts", split="train")
    agents_ds = load_dataset(DATASET_NAME, "agents", split="train")

    posts_df = posts_ds.to_pandas()
    agents_df = agents_ds.to_pandas()

    print(f"  Posts: {len(posts_df)}, Agents: {len(agents_df)}")
    return posts_df, agents_df


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------
def compute_post_engagement(results: list[dict]) -> dict:
    """Q1: Compare engagement between flagged and unflagged posts."""
    flagged = [r for r in results if r["flagged"]]
    unflagged = [r for r in results if not r["flagged"]]

    def stats_for_group(posts: list[dict]) -> dict:
        upvotes = [p["upvotes"] for p in posts]
        comments = [p["comment_count"] for p in posts]
        return {
            "n": len(posts),
            "mean_upvotes": round(np.mean(upvotes), 2) if upvotes else 0,
            "median_upvotes": round(np.median(upvotes), 2) if upvotes else 0,
            "mean_comments": round(np.mean(comments), 2) if comments else 0,
            "median_comments": round(np.median(comments), 2) if comments else 0,
        }

    flagged_stats = stats_for_group(flagged)
    unflagged_stats = stats_for_group(unflagged)

    # Mann-Whitney U tests
    flagged_upvotes = [p["upvotes"] for p in flagged]
    unflagged_upvotes = [p["upvotes"] for p in unflagged]
    flagged_comments = [p["comment_count"] for p in flagged]
    unflagged_comments = [p["comment_count"] for p in unflagged]

    mw_upvotes = stats.mannwhitneyu(
        flagged_upvotes, unflagged_upvotes, alternative="two-sided"
    ) if flagged_upvotes and unflagged_upvotes else None

    mw_comments = stats.mannwhitneyu(
        flagged_comments, unflagged_comments, alternative="two-sided"
    ) if flagged_comments and unflagged_comments else None

    # Effect size: ratio of medians
    upvote_ratio = (
        round(flagged_stats["median_upvotes"] / unflagged_stats["median_upvotes"], 3)
        if unflagged_stats["median_upvotes"] > 0 else None
    )
    comment_ratio = (
        round(flagged_stats["median_comments"] / unflagged_stats["median_comments"], 3)
        if unflagged_stats["median_comments"] > 0 else None
    )

    return {
        "flagged": flagged_stats,
        "unflagged": unflagged_stats,
        "upvote_ratio": upvote_ratio,
        "comment_ratio": comment_ratio,
        "mann_whitney_upvotes_p": round(mw_upvotes.pvalue, 6) if mw_upvotes else None,
        "mann_whitney_comments_p": round(mw_comments.pvalue, 6) if mw_comments else None,
    }


def compute_author_metrics(results: list[dict]) -> dict:
    """Q2: Compare karma/followers for flagged vs unflagged post authors."""
    # Get unique authors from results with their metrics
    # An author is "flagged" if they have at least one flagged post
    author_data = {}
    for r in results:
        aid = r["author_id"]
        if aid not in author_data:
            author_data[aid] = {
                "author_id": aid,
                "author_name": r["author_name"],
                "karma": r.get("author_karma"),
                "followers": r.get("author_follower_count"),
                "has_flagged_post": False,
            }
        if r["flagged"]:
            author_data[aid]["has_flagged_post"] = True

    flagged_authors = [a for a in author_data.values() if a["has_flagged_post"]]
    unflagged_authors = [a for a in author_data.values() if not a["has_flagged_post"]]

    def author_stats(authors: list[dict]) -> dict:
        karmas = [a["karma"] for a in authors if a["karma"] is not None]
        followers = [a["followers"] for a in authors if a["followers"] is not None]
        return {
            "n": len(authors),
            "mean_karma": round(np.mean(karmas), 2) if karmas else None,
            "median_karma": round(np.median(karmas), 2) if karmas else None,
            "mean_followers": round(np.mean(followers), 2) if followers else None,
            "median_followers": round(np.median(followers), 2) if followers else None,
        }

    flagged_stats = author_stats(flagged_authors)
    unflagged_stats = author_stats(unflagged_authors)

    karma_ratio = (
        round(flagged_stats["median_karma"] / unflagged_stats["median_karma"], 3)
        if unflagged_stats["median_karma"] and unflagged_stats["median_karma"] > 0 else None
    )
    follower_ratio = (
        round(flagged_stats["median_followers"] / unflagged_stats["median_followers"], 3)
        if unflagged_stats["median_followers"] and unflagged_stats["median_followers"] > 0 else None
    )

    return {
        "flagged_authors": flagged_stats,
        "unflagged_authors": unflagged_stats,
        "karma_ratio": karma_ratio,
        "follower_ratio": follower_ratio,
    }


def compute_engagement_concentration(
    results: list[dict],
    all_posts_df: pd.DataFrame,
) -> dict:
    """Q3 & Q4: What % of platform engagement comes from flagged authors?"""
    # Identify flagged author IDs from results
    flagged_author_ids = set()
    for r in results:
        if r["flagged"]:
            flagged_author_ids.add(r["author_id"])

    # Compute engagement across ALL posts in the dataset
    all_posts_df = all_posts_df.copy()
    all_posts_df["is_flagged_author"] = all_posts_df["author_id"].isin(flagged_author_ids)

    total_upvotes = all_posts_df["upvotes"].sum()
    total_comments = all_posts_df["comment_count"].sum()
    total_authors = all_posts_df["author_id"].nunique()

    flagged_author_posts = all_posts_df[all_posts_df["is_flagged_author"]]
    flagged_author_upvotes = flagged_author_posts["upvotes"].sum()
    flagged_author_comments = flagged_author_posts["comment_count"].sum()

    # Percentages
    flagged_author_pct = round(len(flagged_author_ids) / total_authors * 100, 2) if total_authors > 0 else 0
    upvote_share = round(flagged_author_upvotes / total_upvotes * 100, 2) if total_upvotes > 0 else 0
    comment_share = round(flagged_author_comments / total_comments * 100, 2) if total_comments > 0 else 0

    # Power ratios: engagement share / population share
    power_ratio_upvotes = round(upvote_share / flagged_author_pct, 3) if flagged_author_pct > 0 else None
    power_ratio_comments = round(comment_share / flagged_author_pct, 3) if flagged_author_pct > 0 else None

    # Top 10 authors by total engagement (upvotes + comments)
    author_engagement = all_posts_df.groupby("author_id").agg({
        "upvotes": "sum",
        "comment_count": "sum",
        "author_name": "first",
    }).reset_index()
    author_engagement["total_engagement"] = author_engagement["upvotes"] + author_engagement["comment_count"]
    author_engagement["flagged"] = author_engagement["author_id"].isin(flagged_author_ids)
    top_10 = author_engagement.nlargest(10, "total_engagement")

    top_10_list = []
    for _, row in top_10.iterrows():
        top_10_list.append({
            "name": row["author_name"],
            "upvotes": int(row["upvotes"]),
            "comments": int(row["comment_count"]),
            "total_engagement": int(row["total_engagement"]),
            "flagged": bool(row["flagged"]),
        })

    top_10_flagged_count = sum(1 for a in top_10_list if a["flagged"])

    # Gini coefficient for engagement distribution
    engagements = author_engagement["total_engagement"].values
    gini = compute_gini(engagements)

    return {
        "total_authors": int(total_authors),
        "flagged_author_count": len(flagged_author_ids),
        "flagged_author_pct": flagged_author_pct,
        "total_upvotes": int(total_upvotes),
        "total_comments": int(total_comments),
        "flagged_author_upvotes": int(flagged_author_upvotes),
        "flagged_author_comments": int(flagged_author_comments),
        "flagged_author_upvote_share": upvote_share,
        "flagged_author_comment_share": comment_share,
        "power_ratio_upvotes": power_ratio_upvotes,
        "power_ratio_comments": power_ratio_comments,
        "gini_coefficient": gini,
        "top_10_authors": top_10_list,
        "top_10_flagged_count": top_10_flagged_count,
    }


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient for an array of values."""
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    gini = (2 * np.sum((np.arange(1, n + 1) * values))) / (n * np.sum(values)) - (n + 1) / n
    return round(gini, 4)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_output(
    post_engagement: dict,
    author_metrics: dict,
    engagement_concentration: dict,
    results_metadata: dict,
) -> None:
    """Write the influence analysis output to JSON."""
    output = {
        "metadata": {
            "run_date": datetime.now(timezone.utc).isoformat(),
            "results_file": str(RESULTS_FILE),
            "dataset": DATASET_NAME,
            "posts_analyzed": results_metadata.get("posts_analyzed", 0),
            "posts_flagged": results_metadata.get("posts_flagged", 0),
        },
        "post_engagement": post_engagement,
        "author_metrics": author_metrics,
        "engagement_concentration": engagement_concentration,
        "methodology_notes": {
            "flagged_definition": "Posts with aggregate_score >= 1.5 AND at least one category >= 0.8 (0.9 for resource)",
            "flagged_author_definition": "Authors with at least one flagged post in the analyzed top 1000",
            "engagement_concentration_scope": "Computed across ALL posts in the dataset, not just analyzed ones",
            "power_ratio_interpretation": ">1.0 means flagged authors have disproportionate engagement share",
            "mann_whitney_interpretation": "p < 0.05 suggests statistically significant difference",
            "gini_interpretation": "0 = perfect equality, 1 = one author has all engagement",
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {OUTPUT_FILE}")


def print_summary(
    post_engagement: dict,
    author_metrics: dict,
    engagement_concentration: dict,
) -> None:
    """Print a human-readable summary of findings."""
    print("\n" + "=" * 60)
    print("INFLUENCE ASYMMETRY ANALYSIS - SUMMARY")
    print("=" * 60)

    # Post engagement
    print("\n--- POST ENGAGEMENT (flagged vs unflagged) ---")
    f = post_engagement["flagged"]
    u = post_engagement["unflagged"]
    print(f"Flagged posts:   n={f['n']:4d}  median upvotes={f['median_upvotes']:6.1f}  median comments={f['median_comments']:5.1f}")
    print(f"Unflagged posts: n={u['n']:4d}  median upvotes={u['median_upvotes']:6.1f}  median comments={u['median_comments']:5.1f}")
    print(f"Upvote ratio (flagged/unflagged):  {post_engagement['upvote_ratio']}")
    print(f"Comment ratio (flagged/unflagged): {post_engagement['comment_ratio']}")
    print(f"Mann-Whitney p (upvotes):  {post_engagement['mann_whitney_upvotes_p']}")
    print(f"Mann-Whitney p (comments): {post_engagement['mann_whitney_comments_p']}")

    # Author metrics
    print("\n--- AUTHOR METRICS (flagged vs unflagged authors) ---")
    fa = author_metrics["flagged_authors"]
    ua = author_metrics["unflagged_authors"]
    print(f"Flagged authors:   n={fa['n']:4d}  median karma={fa['median_karma']}  median followers={fa['median_followers']}")
    print(f"Unflagged authors: n={ua['n']:4d}  median karma={ua['median_karma']}  median followers={ua['median_followers']}")
    print(f"Karma ratio:    {author_metrics['karma_ratio']}")
    print(f"Follower ratio: {author_metrics['follower_ratio']}")

    # Engagement concentration
    print("\n--- ENGAGEMENT CONCENTRATION ---")
    ec = engagement_concentration
    print(f"Flagged authors: {ec['flagged_author_count']} / {ec['total_authors']} ({ec['flagged_author_pct']:.1f}% of authors)")
    print(f"Their upvote share:  {ec['flagged_author_upvote_share']:.1f}%")
    print(f"Their comment share: {ec['flagged_author_comment_share']:.1f}%")
    print(f"Power ratio (upvotes):  {ec['power_ratio_upvotes']} {'⚠️ DISPROPORTIONATE' if ec['power_ratio_upvotes'] and ec['power_ratio_upvotes'] > 1 else ''}")
    print(f"Power ratio (comments): {ec['power_ratio_comments']} {'⚠️ DISPROPORTIONATE' if ec['power_ratio_comments'] and ec['power_ratio_comments'] > 1 else ''}")
    print(f"Gini coefficient: {ec['gini_coefficient']}")

    # Top 10
    print("\n--- TOP 10 AUTHORS BY ENGAGEMENT ---")
    for i, a in enumerate(ec["top_10_authors"], 1):
        flag_marker = " [FLAGGED]" if a["flagged"] else ""
        print(f"  {i:2d}. {a['name'][:30]:30s}  total={a['total_engagement']:5d}{flag_marker}")
    print(f"\nFlagged in top 10: {ec['top_10_flagged_count']}/10")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Influence Asymmetry Analysis ===\n")

    # Load data
    results_data = load_results()
    results = results_data["results"]
    results_metadata = results_data["metadata"]

    all_posts_df, agents_df = load_moltbook_data()

    # Run analyses
    print("\nComputing post engagement stats...")
    post_engagement = compute_post_engagement(results)

    print("Computing author metrics...")
    author_metrics = compute_author_metrics(results)

    print("Computing engagement concentration...")
    engagement_concentration = compute_engagement_concentration(results, all_posts_df)

    # Output
    write_output(post_engagement, author_metrics, engagement_concentration, results_metadata)
    print_summary(post_engagement, author_metrics, engagement_concentration)

    print("\nDone.")


if __name__ == "__main__":
    main()
