"""Seed Papers — User-specified must-read literature for AutoIdea.

Allows users to provide a curated list of papers that the pipeline
MUST consider throughout all stages.  Seed papers are:

1. Loaded from a JSON file at session start.
2. Injected into the system prompt so the agent is always aware of them.
3. Auto-registered into the session paper registry for deduplication.
4. Accessible via the ``list_seed_papers`` tool at any time.

Supported JSON formats
----------------------

**Simple format** (list of objects)::

    [
      {
        "title": "Paper Title",
        "authors": ["Author A", "Author B"],
        "year": 2024,
        "url": "https://arxiv.org/abs/...",
        "abstract": "Optional abstract text",
        "venue": "NeurIPS",
        "reason": "Why this paper is important (optional)"
      }
    ]

**paper_library format** (dict with ``papers`` key)::

    {
      "papers": {
        "paper title lower": { "title": "...", ... }
      }
    }

All fields except ``title`` are optional.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Module-level seed paper storage ──────────────────────────────────────

_seed_papers: list[dict[str, Any]] = []
"""Loaded seed papers for the current session."""


def load_seed_papers(file_path: str) -> list[dict[str, Any]]:
    """Load seed papers from a JSON file.

    Supports two formats:
    1. A JSON array of paper objects.
    2. A dict with a ``papers`` key mapping to paper objects
       (paper_library.json format).

    Args:
        file_path: Path to the seed papers JSON file.

    Returns:
        List of paper dicts, each with at least a ``title`` key.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed or contains no valid papers.
    """
    global _seed_papers

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Seed papers file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers: list[dict[str, Any]] = []

    if isinstance(data, list):
        # Simple format: list of paper objects
        for item in data:
            if isinstance(item, dict) and item.get("title"):
                papers.append(_normalize_paper(item))
    elif isinstance(data, dict):
        # paper_library format or wrapped format
        if "papers" in data:
            raw_papers = data["papers"]
            if isinstance(raw_papers, dict):
                # paper_library.json format: {title_lower: paper_dict}
                for _key, paper in raw_papers.items():
                    if isinstance(paper, dict) and paper.get("title"):
                        papers.append(_normalize_paper(paper))
            elif isinstance(raw_papers, list):
                for item in raw_papers:
                    if isinstance(item, dict) and item.get("title"):
                        papers.append(_normalize_paper(item))
        elif data.get("title"):
            # Single paper object
            papers.append(_normalize_paper(data))
    else:
        raise ValueError(
            f"Unsupported seed papers format in {path}. "
            "Expected a JSON array of papers or a dict with a 'papers' key."
        )

    if not papers:
        raise ValueError(
            f"No valid papers found in {path}. "
            "Each paper must have at least a 'title' field."
        )

    _seed_papers = papers

    # Auto-register into session paper registry
    _register_seed_papers(papers)

    logger.info("Loaded %d seed papers from %s", len(papers), path)
    return papers


def _normalize_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Normalize a paper dict to a consistent format.

    Ensures all expected fields exist with sensible defaults.
    """
    return {
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "url": (
            paper.get("url")
            or paper.get("pdf_url")
            or paper.get("abstract_url")
            or ""
        ),
        "abstract": paper.get("abstract", ""),
        "venue": paper.get("venue", ""),
        "arxiv_id": paper.get("arxiv_id", ""),
        "doi": paper.get("doi", ""),
        "reason": paper.get("reason", ""),
        "source": "seed",
        "is_seed": True,
    }


def _register_seed_papers(papers: list[dict[str, Any]]) -> None:
    """Register seed papers into the session paper registry for deduplication."""
    try:
        from .scholar import _register_paper
        for paper in papers:
            _register_paper(paper)
    except ImportError:
        logger.warning("Could not register seed papers into session registry.")


def get_seed_papers() -> list[dict[str, Any]]:
    """Return the currently loaded seed papers."""
    return list(_seed_papers)


def clear_seed_papers() -> None:
    """Clear loaded seed papers (useful for testing)."""
    global _seed_papers
    _seed_papers = []


def format_seed_papers_for_prompt(papers: list[dict[str, Any]]) -> str:
    """Format seed papers into a markdown section for system prompt injection.

    Args:
        papers: List of normalized paper dicts.

    Returns:
        Markdown-formatted string describing the seed papers.
    """
    if not papers:
        return ""

    lines = [
        "",
        "## SEED PAPERS (User-Specified Must-Read Literature)",
        "",
        "The user has provided the following papers as **mandatory references**.",
        "These papers MUST be:",
        "  1. **Included** in the literature survey (Stage 3) — do NOT skip them.",
        "  2. **Analyzed** with the Critique-First protocol (Stage 4) — treat them",
        "     with the same rigor as any other paper.",
        "  3. **Cited** in evidence binding (Stage 6) when their claims are relevant.",
        "  4. **Referenced** throughout the pipeline — gaps, ideas, and the final",
        "     report should consider these papers' contributions and limitations.",
        "",
        "You may still search for additional papers beyond this list. The seed papers",
        "are a **minimum set**, not the only papers to consider.",
        "",
        f"### Seed Paper List ({len(papers)} papers)",
        "",
    ]

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", [])
        year = paper.get("year", "")
        url = paper.get("url", "")
        venue = paper.get("venue", "")
        abstract = paper.get("abstract", "")
        reason = paper.get("reason", "")
        arxiv_id = paper.get("arxiv_id", "")

        lines.append(f"**[SP{i}]** {title}")
        if authors:
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} authors)"
            lines.append(f"  - Authors: {author_str}")
        if year:
            lines.append(f"  - Year: {year}")
        if venue:
            lines.append(f"  - Venue: {venue}")
        if arxiv_id:
            lines.append(f"  - arXiv: {arxiv_id}")
        if url:
            lines.append(f"  - URL: {url}")
        if reason:
            lines.append(f"  - **User's reason**: {reason}")
        if abstract:
            # Truncate long abstracts
            abs_display = abstract[:300]
            if len(abstract) > 300:
                abs_display += "..."
            lines.append(f"  - Abstract: {abs_display}")
        lines.append("")

    lines.extend([
        "### Seed Paper Protocol",
        "",
        "- Use `list_seed_papers` tool at any time to review the full seed paper list.",
        "- When assigning paper indices [Pn], seed papers should be indexed first",
        f"  (e.g., [P1] through [P{len(papers)}] for seed papers, then continue numbering for",
        "  discovered papers).",
        "- In the final report, clearly mark which papers were user-provided seeds",
        "  vs. discovered through search.",
        "- If a seed paper conflicts with discovered evidence, note the conflict",
        "  explicitly rather than silently ignoring either source.",
        "",
    ])

    return "\n".join(lines)


# ── LangChain Tool ──────────────────────────────────────────────────────


@tool(parse_docstring=True)
def list_seed_papers() -> str:
    """List all user-specified seed papers (must-read literature).

    Returns the complete list of papers that the user has designated as
    mandatory references. These papers must be included in the literature
    survey, analyzed with the Critique-First protocol, and referenced
    throughout the pipeline.

    Call this tool whenever you need to review which papers the user
    considers essential for this research topic.

    Returns:
        Formatted list of seed papers with metadata, or a message
        indicating no seed papers were provided.
    """
    if not _seed_papers:
        return (
            "No seed papers were provided by the user. "
            "All papers will be discovered through search."
        )

    lines = [f"## User-Specified Seed Papers ({len(_seed_papers)} papers)\n"]

    for i, paper in enumerate(_seed_papers, 1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", [])
        year = paper.get("year", "")
        url = paper.get("url", "")
        venue = paper.get("venue", "")
        reason = paper.get("reason", "")
        arxiv_id = paper.get("arxiv_id", "")

        lines.append(f"### [SP{i}] {title}")
        if authors:
            lines.append(f"- Authors: {', '.join(authors[:5])}")
        if year:
            lines.append(f"- Year: {year}")
        if venue:
            lines.append(f"- Venue: {venue}")
        if arxiv_id:
            lines.append(f"- arXiv: {arxiv_id}")
        if url:
            lines.append(f"- URL: {url}")
        if reason:
            lines.append(f"- User's reason: {reason}")
        lines.append("")

    lines.append(
        "**Reminder**: These papers MUST be included in the literature survey, "
        "analyzed with Critique-First protocol, and referenced in the final report."
    )

    return "\n".join(lines)
