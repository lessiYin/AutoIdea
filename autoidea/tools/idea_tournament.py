"""Elo Tournament Ranking for AutoIdea v3.0.

Implements pairwise Elo-based idea comparison and ranking.
Used in Stage 9.5 to rank generated research ideas using
LLM-as-judge pairwise comparisons with stable K=32 ratings.
"""

from __future__ import annotations

import json
import math
import random

from langchain_core.tools import tool

# ── Elo Constants (defaults, can be overridden via config) ───────────────────

DEFAULT_ELO = 1500
K_FACTOR = 32
MIN_COMPARISONS_PER_IDEA = 3


def _get_elo_config() -> tuple[int, int]:
    """Get Elo configuration from config file.

    Returns:
        Tuple of (initial_score, k_factor).
    """
    try:
        from ..config import load_config
        config = load_config()
        initial = getattr(config, "elo_initial_score", DEFAULT_ELO) or DEFAULT_ELO
        k = getattr(config, "elo_k_factor", K_FACTOR) or K_FACTOR
        return int(initial), int(k)
    except Exception:
        return DEFAULT_ELO, K_FACTOR


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against B."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400))


def _update_elo(
    rating_a: float, rating_b: float, score_a: float
) -> tuple[float, float]:
    """Update Elo ratings after a match.

    Args:
        rating_a: Current rating of idea A.
        rating_b: Current rating of idea B.
        score_a: 1.0 if A wins, 0.0 if B wins, 0.5 for draw.

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    _, k_factor = _get_elo_config()
    expected_a = _expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a

    new_a = rating_a + k_factor * (score_a - expected_a)
    new_b = rating_b + k_factor * ((1.0 - score_a) - expected_b)

    return round(new_a, 1), round(new_b, 1)


def _generate_matchups(
    n_ideas: int, min_per_idea: int = MIN_COMPARISONS_PER_IDEA
) -> list[tuple[int, int]]:
    """Generate a balanced set of pairwise matchups.

    Ensures each idea is compared at least min_per_idea times.

    Returns:
        List of (idx_a, idx_b) tuples.
    """
    if n_ideas < 2:
        return []

    matchups: list[tuple[int, int]] = []
    counts = [0] * n_ideas

    # Round-robin first pass
    for i in range(n_ideas):
        for j in range(i + 1, n_ideas):
            matchups.append((i, j))
            counts[i] += 1
            counts[j] += 1

    # If too many matchups for large sets, sample
    if len(matchups) > n_ideas * min_per_idea:
        random.shuffle(matchups)
        # Greedily select until all ideas meet minimum
        selected = []
        sel_counts = [0] * n_ideas
        for a, b in matchups:
            if sel_counts[a] < min_per_idea or sel_counts[b] < min_per_idea:
                selected.append((a, b))
                sel_counts[a] += 1
                sel_counts[b] += 1
        matchups = selected

    random.shuffle(matchups)
    return matchups


@tool(parse_docstring=True)
def rank_ideas_tournament(
    ideas_json: str,
    comparisons_json: str,
) -> str:
    """Run Elo tournament ranking on research ideas.

    Takes a list of ideas and pairwise comparison results, then computes
    Elo ratings. Each comparison should specify which idea won.

    The tournament uses K=32 stable ratings starting at 1500.
    Each idea should be compared at least 3 times for reliable ranking.

    Args:
        ideas_json: JSON array of idea objects, each with at minimum "id" and "title" fields.
        comparisons_json: JSON array of comparison result objects, each with "idea_a", "idea_b", and "winner" fields (winner is the id of the winning idea, or "draw").

    Returns:
        Markdown-formatted tournament results with Elo rankings.
    """
    try:
        ideas = json.loads(ideas_json)
        comparisons = json.loads(comparisons_json)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"

    if not isinstance(ideas, list) or not isinstance(comparisons, list):
        return "Error: Both ideas_json and comparisons_json must be JSON arrays."

    if len(ideas) < 2:
        return "Error: Need at least 2 ideas for tournament ranking."

    # Build id -> info map
    initial_score, k_factor = _get_elo_config()
    idea_map: dict[str, dict] = {}
    for idea in ideas:
        idea_id = str(idea.get("id", ""))
        if not idea_id:
            continue
        idea_map[idea_id] = {
            "title": idea.get("title", "Untitled"),
            "rating": initial_score,
            "wins": 0,
            "losses": 0,
            "draws": 0,
        }

    if len(idea_map) < 2:
        return "Error: Need at least 2 valid ideas with 'id' fields."

    # Process comparisons
    for comp in comparisons:
        id_a = str(comp.get("idea_a", ""))
        id_b = str(comp.get("idea_b", ""))
        winner = str(comp.get("winner", ""))

        if id_a not in idea_map or id_b not in idea_map:
            continue

        info_a = idea_map[id_a]
        info_b = idea_map[id_b]

        if winner == id_a:
            score_a = 1.0
            info_a["wins"] += 1
            info_b["losses"] += 1
        elif winner == id_b:
            score_a = 0.0
            info_b["wins"] += 1
            info_a["losses"] += 1
        else:
            score_a = 0.5
            info_a["draws"] += 1
            info_b["draws"] += 1

        new_a, new_b = _update_elo(info_a["rating"], info_b["rating"], score_a)
        info_a["rating"] = new_a
        info_b["rating"] = new_b

    # Sort by rating
    sorted_ideas = sorted(
        idea_map.items(), key=lambda x: x[1]["rating"], reverse=True
    )

    # Format results
    parts = [
        "## Elo Tournament Rankings",
        f"**Ideas**: {len(idea_map)} | **Comparisons**: {len(comparisons)}",
        f"**K-Factor**: {k_factor} | **Initial Rating**: {initial_score}",
        "",
    ]

    for rank, (idea_id, info) in enumerate(sorted_ideas, 1):
        total = info["wins"] + info["losses"] + info["draws"]
        win_rate = info["wins"] / total * 100 if total > 0 else 0

        parts.append(
            f"### #{rank} — {info['title']} (ID: {idea_id})\n"
            f"- **Elo Rating**: {info['rating']:.0f}\n"
            f"- **Record**: {info['wins']}W / {info['losses']}L / {info['draws']}D "
            f"({win_rate:.0f}% win rate)\n"
        )

    # Add tier classification
    if len(sorted_ideas) >= 3:
        top = sorted_ideas[0]
        parts.append("\n---\n")
        parts.append(f"**Top Idea**: {top[1]['title']} (Elo: {top[0]})")
        parts.append(
            f"**Rating Spread**: {sorted_ideas[0][1]['rating']:.0f} - "
            f"{sorted_ideas[-1][1]['rating']:.0f} "
            f"(Δ={sorted_ideas[0][1]['rating'] - sorted_ideas[-1][1]['rating']:.0f})"
        )

    return "\n".join(parts)


@tool(parse_docstring=True)
def generate_tournament_matchups(
    ideas_json: str,
    min_per_idea: int = 3,
) -> str:
    """Generate balanced pairwise matchups for Elo tournament.

    Creates a set of matchups ensuring each idea is compared at least
    the specified minimum number of times.

    Args:
        ideas_json: JSON array of idea objects with "id" fields.
        min_per_idea: Minimum comparisons per idea (default 3).

    Returns:
        JSON array of matchup objects with "idea_a" and "idea_b" fields.
    """
    try:
        ideas = json.loads(ideas_json)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"

    ids = [str(idea.get("id", f"idea_{i}")) for i, idea in enumerate(ideas)]
    n = len(ids)

    matchups = _generate_matchups(n, min_per_idea)

    result = []
    for a, b in matchups:
        result.append({
            "idea_a": ids[a],
            "idea_b": ids[b],
            "prompt": (
                f"Compare idea '{ids[a]}' vs '{ids[b]}'. "
                f"Which is more novel, feasible, and impactful? "
                f"Declare a winner or draw."
            ),
        })

    return json.dumps(result, indent=2)
