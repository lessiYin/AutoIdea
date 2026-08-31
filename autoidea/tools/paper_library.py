"""Paper library management for AutoIdea.

Provides a persistent paper library that stores metadata about papers
found during research sessions. Used for cross-session deduplication
and reference tracking.
"""

from __future__ import annotations

import json
from pathlib import Path


def _get_library_path() -> Path:
    """Get the paper library file path."""
    try:
        from autoidea.paths import get_active_workspace
        ws = get_active_workspace()
        if ws:
            return Path(ws) / "paper_library.json"
    except (ImportError, Exception):
        pass
    return Path(__file__).resolve().parent.parent.parent / "paper_library.json"


def load_paper_library() -> list[dict]:
    """Load the paper library from disk."""
    path = _get_library_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_paper_library(papers: list[dict]) -> None:
    """Save the paper library to disk."""
    path = _get_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(papers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _normalize_title(title: str) -> str:
    """Normalize paper title for deduplication.

    Strips punctuation and collapses whitespace so that minor formatting
    differences (e.g. trailing period, extra spaces) don't create duplicates.
    """
    import re
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)  # Remove punctuation
    t = re.sub(r"\s+", " ", t)     # Collapse whitespace
    return t


def add_paper(paper: dict) -> bool:
    """Add a paper to the library if not already present.

    Deduplication is based on normalized title (punctuation-insensitive).

    Args:
        paper: Paper metadata dict with at least 'title' key.

    Returns:
        True if paper was added (new), False if duplicate.
    """
    library = load_paper_library()
    title = (paper.get("title") or "").strip()
    if not title:
        return False

    norm_title = _normalize_title(title)  # Improved: punctuation-insensitive dedup
    for existing in library:
        existing_title = (existing.get("title") or "").strip()
        if _normalize_title(existing_title) == norm_title:
            return False

    library.append(paper)
    save_paper_library(library)
    return True
