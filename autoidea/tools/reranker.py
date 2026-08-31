"""Cross-source search result reranker for AutoIdea.

Implements merge_and_rank_search_results tool that deduplicates and
scores results from multiple academic search sources using a
multi-dimensional relevance scoring system.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool


def _normalize_title(title: str) -> str:
    """Normalize a paper title for deduplication."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _keyword_score(title: str, abstract: str, keywords: list[str]) -> float:
    """Score based on keyword presence in title and abstract."""
    if not keywords:
        return 0.5
    text = (title + " " + abstract).lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return min(1.0, matches / max(len(keywords), 1))


def _citation_score(citations: int) -> float:
    """Normalize citation count to 0-1 score."""
    if citations <= 0:
        return 0.0
    import math
    return min(1.0, math.log10(citations + 1) / 4.0)


def _recency_score(year: int | None) -> float:
    """Score based on publication recency (newer = higher)."""
    if year is None:
        return 0.3
    from datetime import datetime
    current_year = datetime.now().year
    age = current_year - year
    if age <= 1:
        return 1.0
    elif age <= 3:
        return 0.8
    elif age <= 5:
        return 0.6
    elif age <= 10:
        return 0.3
    else:
        return 0.1


def _authority_score(venue: str, source: str) -> float:
    """Score based on publication venue authority."""
    if not venue:
        return 0.3

    top_venues = {
        "neurips", "icml", "iclr", "aaai", "cvpr", "iccv", "eccv",
        "acl", "emnlp", "naacl", "sigir", "kdd", "www", "chi",
        "nature", "science", "cell", "pnas",
    }
    venue_lower = venue.lower()
    for tv in top_venues:
        if tv in venue_lower:
            return 1.0

    if source in ("semantic_scholar", "dblp"):
        return 0.6
    return 0.4


def rank_results(
    results: list[dict],
    keywords: list[str],
    weights: dict[str, float] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """Rank and deduplicate search results.

    Args:
        results: List of paper result dicts.
        keywords: Keywords for relevance scoring.
        weights: Scoring weights (keyword, citation, recency, authority).
        top_k: Number of top results to return.

    Returns:
        Sorted and deduplicated list of paper results.
    """
    if weights is None:
        weights = {
            "keyword": 0.6,
            "citation": 0.2,
            "recency": 0.1,
            "authority": 0.1,
        }

    # Deduplication
    seen_titles: set[str] = set()
    unique_results = []
    for r in results:
        norm_title = _normalize_title(r.get("title", ""))
        if norm_title and norm_title not in seen_titles:
            seen_titles.add(norm_title)
            unique_results.append(r)

    # Scoring
    scored = []
    for r in unique_results:
        title = r.get("title", "")
        abstract = r.get("abstract", "")
        citations = r.get("citations", 0) or r.get("citationCount", 0) or 0
        year = r.get("year")
        venue = r.get("venue", "")
        source = r.get("source", "")

        score = (
            weights["keyword"] * _keyword_score(title, abstract, keywords)
            + weights["citation"] * _citation_score(citations)
            + weights["recency"] * _recency_score(year)
            + weights["authority"] * _authority_score(venue, source)
        )
        r["_relevance_score"] = round(score, 4)
        scored.append(r)

    # Sort by score descending
    scored.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
    return scored[:top_k]


@tool(parse_docstring=True)
def merge_and_rank_search_results(
    results_json: str,
    keywords: str = "",
    top_k: int = 20,
) -> str:
    """Merge and rank search results from multiple academic sources.

    Takes JSON-formatted results from various search tools, deduplicates
    by title, and scores using multi-dimensional relevance:
    - Keyword match (0.6 weight)
    - Citation count (0.2 weight)
    - Publication recency (0.1 weight)
    - Venue authority (0.1 weight)

    Args:
        results_json: JSON string containing a list of paper result objects, each with title, abstract, year, citations/citationCount, venue, and source fields.
        keywords: Comma-separated keywords for relevance scoring.
        top_k: Number of top results to return after ranking.

    Returns:
        Markdown-formatted ranked results with relevance scores.
    """
    import json

    try:
        results = json.loads(results_json)
        if not isinstance(results, list):
            return "Error: results_json must be a JSON array of paper objects."
    except json.JSONDecodeError as e:
        return f"Error parsing results_json: {e}"

    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    ranked = rank_results(results, keyword_list, top_k=top_k)

    if not ranked:
        return "No results to rank after deduplication."

    parts = [
        "## Ranked Search Results",
        f"**Input**: {len(results)} results | **After dedup**: {len(ranked)} | **Top-K**: {top_k}",
        f"**Keywords**: {', '.join(keyword_list) if keyword_list else '(none)'}",
        "",
    ]

    for i, r in enumerate(ranked, 1):
        title = r.get("title", "Untitled")
        year = r.get("year", "N/A")
        score = r.get("_relevance_score", 0)
        source = r.get("source", "unknown")
        citations = r.get("citations", 0) or r.get("citationCount", 0) or 0

        parts.append(
            f"### [{i}] {title}\n"
            f"- **Score**: {score:.3f} | **Year**: {year} | "
            f"**Citations**: {citations} | **Source**: {source}"
        )

    return "\n".join(parts)
