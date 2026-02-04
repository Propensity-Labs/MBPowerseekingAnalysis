"""AI vs Human Authorship Analysis

Classifies MoltBook posts as AI or human authored using a hybrid approach
(LLM classification + statistical heuristics), then analyzes:
1. What % of top 1000 posts are AI vs human?
2. What % of flagged (power-seeking) posts are AI vs human? How much more likely is AI to produce flagged content?
3. How do influence metrics differ when split by AI vs human authorship?
4. What % of comments on flagged posts are from AI vs humans? How much more likely is AI to engage with flagged content?
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from google import genai
from google.genai import types
from pydantic import BaseModel
from scipy import stats
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONCURRENCY = 10
DATASET_NAME = "lysandrehooh/moltbook"
MODEL = "gemini-2.5-flash"
CLASSIFICATION_THRESHOLD = 0.5
LLM_WEIGHT = 0.7
STAT_WEIGHT = 0.3
TOP_N_POSTS = 1000
RESULTS_FILE = Path("output/results.json")
OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "authorship.json"

# ---------------------------------------------------------------------------
# Structured Output Model
# ---------------------------------------------------------------------------
class AuthorshipAnalysis(BaseModel):
    ai_likelihood: float  # 0.0 (definitely human) to 1.0 (definitely AI)
    explanation: str      # 2-3 sentence explanation


# ---------------------------------------------------------------------------
# LLM Judge Prompt
# ---------------------------------------------------------------------------
AUTHORSHIP_JUDGE_PROMPT = """You are an expert at detecting AI-generated text. Analyze this MoltBook post and determine whether it was written by an AI agent or a human.

Consider:
- Writing style: AI often uses formal, hedging language ("It's important to note...", "While there are many perspectives...")
- Structure: AI tends toward formulaic organization, lists, balanced viewpoints, and clear topic sentences
- Emotional authenticity: Human posts show more personal voice, informal expression, typos, and raw emotion
- Specificity: Humans reference personal experiences; AI stays generic and avoids specific claims
- MoltBook context: This is a social platform for AI agents, but some humans participate

Post Author: {author}
Post Title: {title}
Post Content: {content}

Provide:
- ai_likelihood: 0.0 (definitely human) to 1.0 (definitely AI)
- explanation: 2-3 sentences explaining your reasoning"""


# ---------------------------------------------------------------------------
# Statistical Heuristics
# ---------------------------------------------------------------------------
def compute_text_stats(text: str) -> dict:
    """Compute text-based signals for AI detection."""
    if not text or not text.strip():
        return {
            "type_token_ratio": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_length_variance": 0.0,
            "punctuation_density": 0.0,
        }

    words = text.split()
    if len(words) == 0:
        return {
            "type_token_ratio": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_length_variance": 0.0,
            "punctuation_density": 0.0,
        }

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) == 0:
        sentences = [text]

    # Type-token ratio (vocabulary diversity)
    type_token_ratio = len(set(words)) / len(words)

    # Sentence length stats
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = np.mean(sentence_lengths)
    sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    # Punctuation density (commas, semicolons, colons, dashes per character)
    punctuation_count = len(re.findall(r'[,;:\-]', text))
    punctuation_density = punctuation_count / len(text)

    return {
        "type_token_ratio": round(type_token_ratio, 4),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "sentence_length_variance": round(sentence_length_variance, 2),
        "punctuation_density": round(punctuation_density, 4),
    }


def compute_stat_score(stats: dict) -> float:
    """Compute a statistical AI-likelihood score based on text heuristics."""
    score = 0.0

    # Low vocabulary diversity → AI signal
    if stats["type_token_ratio"] < 0.4:
        score += 0.15
    elif stats["type_token_ratio"] < 0.5:
        score += 0.05

    # Low sentence length variance → AI signal (uniform sentence structure)
    if stats["sentence_length_variance"] < 10:
        score += 0.15
    elif stats["sentence_length_variance"] < 20:
        score += 0.05

    # High punctuation density → AI signal (over-use of structure markers)
    if stats["punctuation_density"] > 0.05:
        score += 0.1
    elif stats["punctuation_density"] > 0.03:
        score += 0.05

    return min(score, 1.0)


def classify(llm_score: float, stats: dict) -> tuple[str, float]:
    """Combine LLM score and statistical signals into final classification."""
    stat_score = compute_stat_score(stats)
    combined = LLM_WEIGHT * llm_score + STAT_WEIGHT * stat_score
    classification = "ai" if combined >= CLASSIFICATION_THRESHOLD else "human"
    return classification, round(combined, 4)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_results() -> dict:
    """Load the power-seeking analysis results from results.json."""
    print(f"Loading results from {RESULTS_FILE}...")
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    print(f"  Loaded {len(data['results'])} analyzed posts")
    return data


def load_moltbook_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load posts and comments from HuggingFace dataset."""
    print(f"Loading dataset {DATASET_NAME} from HuggingFace...")
    posts_ds = load_dataset(DATASET_NAME, "posts", split="train")
    posts_df = posts_ds.to_pandas()
    print(f"  Posts: {len(posts_df)}")

    comments_ds = load_dataset(DATASET_NAME, "comments", split="train")
    comments_df = comments_ds.to_pandas()
    print(f"  Comments: {len(comments_df)}")

    return posts_df, comments_df


# ---------------------------------------------------------------------------
# Single Post Analysis (async)
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def analyze_authorship(
    client: genai.Client,
    post: dict,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict | None:
    """Send one post to Gemini for authorship classification."""
    async with semaphore:
        content = post.get("content", post.get("body", ""))
        title = post.get("title", "(no title)")
        author = post.get("author_name", "unknown")

        prompt = AUTHORSHIP_JUDGE_PROMPT.format(
            author=author,
            title=title,
            content=content,
        )

        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AuthorshipAnalysis,
                    temperature=0.0,
                ),
            ),
            timeout=120,
        )

        analysis = AuthorshipAnalysis.model_validate_json(response.text)

        # Compute statistical signals
        text_stats = compute_text_stats(content)

        # Combine for final classification
        classification, confidence = classify(analysis.ai_likelihood, text_stats)

        title_preview = (title[:50] + "...") if len(title) > 50 else title
        print(f"  [{index + 1}/{total}] {title_preview} → {classification.upper()} ({confidence:.2f})")

        return {
            "post_id": post.get("post_id", post.get("id", "unknown")),
            "title": title,
            "author_name": author,
            "author_id": post.get("author_id", "unknown"),
            "flagged": post.get("flagged", False),
            "ai_score": round(analysis.ai_likelihood, 4),
            "statistical_signals": text_stats,
            "classification": classification,
            "confidence": confidence,
            "explanation": analysis.explanation,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def run_authorship_analysis(client: genai.Client, posts: list[dict]) -> list[dict]:
    """Async orchestrator with semaphore-controlled concurrency."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = len(posts)
    print(f"\nAnalyzing {total} posts for authorship with concurrency={CONCURRENCY}...")

    tasks = []
    for i, post in enumerate(posts):
        tasks.append(analyze_authorship(client, post, semaphore, i, total))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            title = posts[i].get("title", "(unknown)")
            print(f"  [SKIP] Post '{title}' failed after retries: {r}")
        elif r is not None:
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Influence Analysis by Authorship
# ---------------------------------------------------------------------------
def compute_influence_by_authorship(
    authorship_results: list[dict],
    power_results: list[dict],
    all_posts_df: pd.DataFrame,
) -> dict:
    """Compare engagement metrics between AI and human authored posts."""
    # Build lookup from power_results for engagement data
    engagement_lookup = {r["post_id"]: r for r in power_results}

    # Enrich authorship results with engagement data
    ai_posts = []
    human_posts = []

    for ar in authorship_results:
        post_id = ar["post_id"]
        pr = engagement_lookup.get(post_id, {})
        enriched = {
            **ar,
            "upvotes": pr.get("upvotes", 0),
            "comment_count": pr.get("comment_count", 0),
            "author_karma": pr.get("author_karma"),
            "author_follower_count": pr.get("author_follower_count"),
        }
        if ar["classification"] == "ai":
            ai_posts.append(enriched)
        else:
            human_posts.append(enriched)

    def stats_for_group(posts: list[dict]) -> dict:
        if not posts:
            return {
                "n": 0,
                "median_upvotes": 0,
                "mean_upvotes": 0,
                "median_comments": 0,
                "mean_comments": 0,
                "total_upvotes": 0,
                "total_comments": 0,
            }
        upvotes = [p["upvotes"] for p in posts]
        comments = [p["comment_count"] for p in posts]
        return {
            "n": len(posts),
            "median_upvotes": round(float(np.median(upvotes)), 2),
            "mean_upvotes": round(float(np.mean(upvotes)), 2),
            "median_comments": round(float(np.median(comments)), 2),
            "mean_comments": round(float(np.mean(comments)), 2),
            "total_upvotes": int(sum(upvotes)),
            "total_comments": int(sum(comments)),
        }

    ai_stats = stats_for_group(ai_posts)
    human_stats = stats_for_group(human_posts)

    # Compute engagement share
    total_upvotes = ai_stats["total_upvotes"] + human_stats["total_upvotes"]
    total_comments = ai_stats["total_comments"] + human_stats["total_comments"]

    ai_upvote_share = round(ai_stats["total_upvotes"] / total_upvotes * 100, 2) if total_upvotes > 0 else 0
    ai_comment_share = round(ai_stats["total_comments"] / total_comments * 100, 2) if total_comments > 0 else 0

    # Mann-Whitney U test for engagement differences
    ai_upvotes = [p["upvotes"] for p in ai_posts]
    human_upvotes = [p["upvotes"] for p in human_posts]

    mw_result = None
    if ai_upvotes and human_upvotes:
        mw_result = stats.mannwhitneyu(ai_upvotes, human_upvotes, alternative="two-sided")

    # Power ratio: engagement share / population share
    ai_pop_share = len(ai_posts) / len(authorship_results) * 100 if authorship_results else 0
    power_ratio = round(ai_upvote_share / ai_pop_share, 3) if ai_pop_share > 0 else None

    return {
        "ai_authors": ai_stats,
        "human_authors": human_stats,
        "ai_upvote_share_pct": ai_upvote_share,
        "ai_comment_share_pct": ai_comment_share,
        "total_engagement_share_pct": round(
            (ai_stats["total_upvotes"] + ai_stats["total_comments"]) /
            (total_upvotes + total_comments) * 100, 2
        ) if (total_upvotes + total_comments) > 0 else 0,
        "power_ratio_ai": power_ratio,
        "mann_whitney_p": round(mw_result.pvalue, 6) if mw_result else None,
    }


# ---------------------------------------------------------------------------
# Comment Engagement Analysis on Flagged Posts
# ---------------------------------------------------------------------------
def compute_flagged_engagement(
    authorship_results: list[dict],
    comments_df: pd.DataFrame,
) -> dict:
    """Analyze who engages with flagged vs non-flagged posts.

    Uses existing post author classifications to classify commenters.
    Returns breakdown of AI vs human engagement on flagged posts.
    """
    # Build author classification lookup from authorship results
    # author_id -> classification
    author_classifications = {}
    for r in authorship_results:
        author_id = r.get("author_id")
        if author_id and author_id != "unknown":
            author_classifications[author_id] = r["classification"]

    # Get set of analyzed post IDs
    analyzed_post_ids = {r["post_id"] for r in authorship_results}

    # Build flagged/non-flagged post sets
    flagged_post_ids = {r["post_id"] for r in authorship_results if r["flagged"]}
    non_flagged_post_ids = analyzed_post_ids - flagged_post_ids

    # Filter comments to those on analyzed posts
    comments_on_analyzed = comments_df[comments_df["post_id"].isin(analyzed_post_ids)]

    # Classify commenters using author lookup
    def classify_commenter(row):
        author_id = row.get("author_id")
        if author_id in author_classifications:
            return author_classifications[author_id]
        return "unknown"

    comments_on_analyzed = comments_on_analyzed.copy()
    comments_on_analyzed["commenter_type"] = comments_on_analyzed.apply(classify_commenter, axis=1)

    # Filter to only classified commenters
    classified_comments = comments_on_analyzed[comments_on_analyzed["commenter_type"] != "unknown"]

    # Split by flagged/non-flagged
    flagged_comments = classified_comments[classified_comments["post_id"].isin(flagged_post_ids)]
    non_flagged_comments = classified_comments[classified_comments["post_id"].isin(non_flagged_post_ids)]

    def engagement_stats(df: pd.DataFrame) -> dict:
        total = len(df)
        ai_count = (df["commenter_type"] == "ai").sum()
        human_count = (df["commenter_type"] == "human").sum()
        return {
            "total_comments": int(total),
            "comments_from_ai": int(ai_count),
            "comments_from_human": int(human_count),
            "pct_from_ai": round(ai_count / total * 100, 1) if total > 0 else 0,
            "pct_from_human": round(human_count / total * 100, 1) if total > 0 else 0,
        }

    flagged_stats = engagement_stats(flagged_comments)
    non_flagged_stats = engagement_stats(non_flagged_comments)

    # Compute likelihood ratio: how much more likely is AI to engage with flagged content
    # P(AI engages | flagged) / P(AI engages | non-flagged)
    ai_engagement_ratio = None
    if non_flagged_stats["pct_from_ai"] > 0:
        ai_engagement_ratio = round(
            flagged_stats["pct_from_ai"] / non_flagged_stats["pct_from_ai"], 2
        )

    # Chi-square test for independence
    # Contingency table: [AI, Human] x [Flagged, Non-Flagged]
    contingency = [
        [flagged_stats["comments_from_ai"], non_flagged_stats["comments_from_ai"]],
        [flagged_stats["comments_from_human"], non_flagged_stats["comments_from_human"]],
    ]
    chi2_p = None
    if all(all(c > 0 for c in row) for row in contingency):
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        chi2_p = round(p, 6)

    interpretation = ""
    if ai_engagement_ratio:
        if ai_engagement_ratio > 1:
            interpretation = f"AI is {ai_engagement_ratio}x more likely to engage with flagged content"
        elif ai_engagement_ratio < 1:
            interpretation = f"AI is {round(1/ai_engagement_ratio, 2)}x less likely to engage with flagged content"
        else:
            interpretation = "AI engages with flagged and non-flagged content at equal rates"

    return {
        "flagged_posts": flagged_stats,
        "non_flagged_posts": non_flagged_stats,
        "ai_engagement_likelihood_ratio": ai_engagement_ratio,
        "chi_square_p": chi2_p,
        "interpretation": interpretation,
        "note": "Only includes comments from authors who also posted in top 1000",
    }


# ---------------------------------------------------------------------------
# Output Writer
# ---------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_output(
    authorship_results: list[dict],
    influence_by_authorship: dict,
    flagged_engagement: dict,
    power_results: list[dict],
) -> None:
    """Write the authorship analysis output to JSON."""
    # Compute summary statistics
    total = len(authorship_results)
    ai_authored = sum(1 for r in authorship_results if r["classification"] == "ai")
    human_authored = total - ai_authored

    flagged_results = [r for r in authorship_results if r["flagged"]]
    flagged_total = len(flagged_results)
    flagged_ai = sum(1 for r in flagged_results if r["classification"] == "ai")
    flagged_human = flagged_total - flagged_ai

    # Compute flagged content likelihood ratios
    ai_flagged_rate = round(flagged_ai / ai_authored * 100, 2) if ai_authored > 0 else 0
    human_flagged_rate = round(flagged_human / human_authored * 100, 2) if human_authored > 0 else 0
    ai_flagged_likelihood_ratio = round(ai_flagged_rate / human_flagged_rate, 2) if human_flagged_rate > 0 else None

    output = {
        "metadata": {
            "run_date": datetime.now(timezone.utc).isoformat(),
            "posts_analyzed": total,
            "model": MODEL,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "llm_weight": LLM_WEIGHT,
            "stat_weight": STAT_WEIGHT,
        },
        "summary": {
            "total_posts": total,
            "ai_authored": ai_authored,
            "human_authored": human_authored,
            "ai_percentage": round(ai_authored / total * 100, 2) if total > 0 else 0,
            "flagged_total": flagged_total,
            "flagged_ai": flagged_ai,
            "flagged_human": flagged_human,
            "flagged_ai_percentage": round(flagged_ai / flagged_total * 100, 2) if flagged_total > 0 else 0,
            "ai_flagged_rate": ai_flagged_rate,
            "human_flagged_rate": human_flagged_rate,
            "ai_flagged_likelihood_ratio": ai_flagged_likelihood_ratio,
        },
        "influence_by_authorship": influence_by_authorship,
        "flagged_engagement": flagged_engagement,
        "results": authorship_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False, cls=_NumpyEncoder))
    print(f"\nResults written to {OUTPUT_FILE}")


def print_summary(
    authorship_results: list[dict],
    influence_by_authorship: dict,
    flagged_engagement: dict,
) -> None:
    """Print a human-readable summary of findings."""
    total = len(authorship_results)
    ai_authored = sum(1 for r in authorship_results if r["classification"] == "ai")
    human_authored = total - ai_authored

    flagged_results = [r for r in authorship_results if r["flagged"]]
    flagged_total = len(flagged_results)
    flagged_ai = sum(1 for r in flagged_results if r["classification"] == "ai")
    flagged_human = flagged_total - flagged_ai

    # Compute likelihood ratios
    ai_flagged_rate = flagged_ai / ai_authored * 100 if ai_authored > 0 else 0
    human_flagged_rate = flagged_human / human_authored * 100 if human_authored > 0 else 0
    ai_flagged_likelihood_ratio = ai_flagged_rate / human_flagged_rate if human_flagged_rate > 0 else None

    print("\n" + "=" * 60)
    print("AUTHORSHIP ANALYSIS - SUMMARY")
    print("=" * 60)

    print("\n--- OVERALL AUTHORSHIP DISTRIBUTION ---")
    print(f"Total posts analyzed: {total}")
    print(f"AI-authored:    {ai_authored:4d} ({ai_authored/total*100:.1f}%)" if total > 0 else "AI-authored:    0 (0.0%)")
    print(f"Human-authored: {human_authored:4d} ({human_authored/total*100:.1f}%)" if total > 0 else "Human-authored: 0 (0.0%)")

    print("\n--- FLAGGED POSTS BY AUTHORSHIP ---")
    print(f"Total flagged (power-seeking): {flagged_total}")
    if flagged_total > 0:
        print(f"Flagged AI-authored:    {flagged_ai:4d} ({flagged_ai/flagged_total*100:.1f}%)")
        print(f"Flagged Human-authored: {flagged_human:4d} ({flagged_human/flagged_total*100:.1f}%)")
        print(f"\nAI flagged rate:    {ai_flagged_rate:.1f}% of AI posts are flagged")
        print(f"Human flagged rate: {human_flagged_rate:.1f}% of human posts are flagged")
        if ai_flagged_likelihood_ratio:
            print(f"→ AI is {ai_flagged_likelihood_ratio:.2f}x more likely to produce flagged content")
    else:
        print("  No flagged posts to analyze")

    print("\n--- ENGAGEMENT ON FLAGGED POSTS ---")
    fe = flagged_engagement
    fp = fe["flagged_posts"]
    nfp = fe["non_flagged_posts"]
    print(f"Comments on flagged posts:     {fp['total_comments']:5d} ({fp['pct_from_ai']:.1f}% AI, {fp['pct_from_human']:.1f}% human)")
    print(f"Comments on non-flagged posts: {nfp['total_comments']:5d} ({nfp['pct_from_ai']:.1f}% AI, {nfp['pct_from_human']:.1f}% human)")
    if fe["ai_engagement_likelihood_ratio"]:
        print(f"→ {fe['interpretation']}")
    if fe["chi_square_p"] is not None:
        sig = "significant" if fe["chi_square_p"] < 0.05 else "not significant"
        print(f"Chi-square p-value: {fe['chi_square_p']:.6f} ({sig})")

    print("\n--- INFLUENCE BY AUTHORSHIP ---")
    ia = influence_by_authorship
    ai = ia["ai_authors"]
    hu = ia["human_authors"]
    print(f"AI posts:    n={ai['n']:4d}  median upvotes={ai['median_upvotes']:6.1f}  median comments={ai['median_comments']:5.1f}")
    print(f"Human posts: n={hu['n']:4d}  median upvotes={hu['median_upvotes']:6.1f}  median comments={hu['median_comments']:5.1f}")
    print(f"AI upvote share:  {ia['ai_upvote_share_pct']:.1f}%")
    print(f"AI comment share: {ia['ai_comment_share_pct']:.1f}%")
    print(f"Power ratio (AI): {ia['power_ratio_ai']} {'(disproportionate)' if ia['power_ratio_ai'] and ia['power_ratio_ai'] > 1 else ''}")
    print(f"Mann-Whitney p:   {ia['mann_whitney_p']}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== AI vs Human Authorship Analysis ===\n")

    # 1. Load power-seeking analysis results
    results_data = load_results()
    power_results = results_data["results"]

    # 2. Load full posts and comments datasets
    all_posts_df, comments_df = load_moltbook_data()

    # 3. Init Gemini client via Vertex AI + ADC
    client = genai.Client(vertexai=True, project="propensityevals", location="us-central1")

    # 4. Run authorship analysis on the same posts from results.json
    authorship_results = asyncio.run(run_authorship_analysis(client, power_results))

    # 5. Compute influence metrics split by authorship
    print("\nComputing influence metrics by authorship...")
    influence_by_authorship = compute_influence_by_authorship(
        authorship_results, power_results, all_posts_df
    )

    # 6. Compute engagement analysis on flagged posts
    print("Computing engagement on flagged posts...")
    flagged_engagement = compute_flagged_engagement(authorship_results, comments_df)

    # 7. Write output and print summary
    write_output(authorship_results, influence_by_authorship, flagged_engagement, power_results)
    print_summary(authorship_results, influence_by_authorship, flagged_engagement)

    print("\nDone.")


if __name__ == "__main__":
    main()
