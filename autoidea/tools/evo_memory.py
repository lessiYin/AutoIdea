"""Persistent Ideation Memory for AutoIdea v3.0.

Implements cross-run knowledge accumulation — the system remembers
insights, patterns, and lessons from previous research sessions.
This enables progressive improvement across multiple runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


def _get_memory_path() -> Path:
    """Get the path to the global ideation memory file."""
    from autoidea.paths import MEMORY_DIR
    memory_dir = Path(MEMORY_DIR)
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / "ideation_memory.json"


def _load_memory() -> dict[str, Any]:
    """Load the ideation memory from disk."""
    path = _get_memory_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "version": "3.0",
        "entries": [],
        "patterns": [],
        "meta": {"created": datetime.now(timezone.utc).isoformat()},
    }


def _save_memory(memory: dict[str, Any]) -> None:
    """Save the ideation memory to disk."""
    path = _get_memory_path()
    memory["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


@tool(parse_docstring=True)
def recall_ideation_memory(
    query: str = "",
    category: str = "",
    max_entries: int = 10,
) -> str:
    """Recall insights from previous research sessions.

    Searches the persistent ideation memory for relevant entries.
    Memory includes research insights, successful patterns, failed
    approaches, and cross-domain connections from past sessions.

    Args:
        query: Search query to filter relevant memories. If empty,
            returns the most recent entries.
        category: Filter by category (e.g. "insight", "pattern",
            "failure", "connection", "method"). Empty for all.
        max_entries: Maximum number of entries to return.

    Returns:
        Markdown-formatted memory entries.
    """
    memory = _load_memory()
    entries = memory.get("entries", [])

    if not entries:
        return (
            "No ideation memory found. This appears to be the first "
            "research session. Memory will accumulate across runs."
        )

    # Filter by category
    if category:
        entries = [e for e in entries if e.get("category", "") == category]

    # Filter by query (simple keyword matching)
    if query:
        query_lower = query.lower()
        keywords = query_lower.split()

        def _match_score(entry: dict) -> int:
            text = (
                entry.get("content", "") + " " +
                entry.get("domain", "") + " " +
                " ".join(entry.get("tags", []))
            ).lower()
            return sum(1 for kw in keywords if kw in text)

        entries = [(e, _match_score(e)) for e in entries]
        entries = [(e, s) for e, s in entries if s > 0]
        entries.sort(key=lambda x: x[1], reverse=True)
        entries = [e for e, _ in entries[:max_entries]]
    else:
        # Most recent entries
        entries = entries[-max_entries:]

    if not entries:
        return f"No relevant memories found for query: '{query}'"

    # Format output
    parts = [
        "## Ideation Memory",
        f"**Matching entries**: {len(entries)}\n",
    ]

    for i, entry in enumerate(entries, 1):
        cat = entry.get("category", "general")
        content = entry.get("content", "")
        domain = entry.get("domain", "")
        tags = entry.get("tags", [])
        timestamp = entry.get("timestamp", "")
        session = entry.get("session_id", "")

        parts.append(
            f"### [{i}] {cat.upper()}"
            + (f" — {domain}" if domain else "")
            + f"\n*Session: {session[:8] if session else 'unknown'}*"
            + (f" | *{timestamp[:10]}*" if timestamp else "")
            + f"\n\n{content}\n"
        )
        if tags:
            parts.append(f"Tags: {', '.join(tags)}\n")

    # Also include patterns if available
    patterns = memory.get("patterns", [])
    if patterns:
        parts.append("\n---\n### Recurring Patterns\n")
        for p in patterns[:5]:
            parts.append(f"- **{p.get('name', 'unnamed')}**: {p.get('description', '')}")

    return "\n".join(parts)


@tool(parse_docstring=True)
def update_ideation_memory(
    content: str,
    category: str = "insight",
    domain: str = "",
    tags: str = "",
    importance: int = 5,
) -> str:
    """Add a new entry to the persistent ideation memory.

    Records insights, patterns, failures, and connections discovered
    during the current research session for use in future sessions.

    Args:
        content: The insight or knowledge to remember. Should be
            self-contained and understandable in future sessions.
        category: Entry category - one of: "insight" (research finding),
            "pattern" (recurring pattern), "failure" (failed approach),
            "connection" (cross-domain link), "method" (useful method).
        domain: Research domain this memory relates to (e.g. "NLP",
            "computer vision", "reinforcement learning").
        tags: Comma-separated tags for this entry.
        importance: Importance score 1-10 (higher = more important
            to recall in future sessions).

    Returns:
        Confirmation with memory statistics.
    """
    valid_categories = {"insight", "pattern", "failure", "connection", "method"}
    if category not in valid_categories:
        return (
            f"Error: Invalid category '{category}'. "
            f"Valid categories: {', '.join(sorted(valid_categories))}"
        )

    importance = max(1, min(10, importance))

    memory = _load_memory()

    # Get session ID from workspace
    session_id = ""
    try:
        from autoidea.paths import get_active_workspace
        workspace = get_active_workspace()
        session_id = Path(workspace).name
    except Exception:
        session_id = "unknown"

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    entry = {
        "content": content,
        "category": category,
        "domain": domain,
        "tags": tag_list,
        "importance": importance,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    memory["entries"].append(entry)

    # Update patterns if category is "pattern"
    if category == "pattern":
        memory.setdefault("patterns", []).append({
            "name": domain or "general",
            "description": content[:200],
            "session": session_id,
        })

    # Prune old low-importance entries if memory too large (keep 500 max)
    MAX_ENTRIES = 500
    entries = memory["entries"]
    if len(entries) > MAX_ENTRIES:
        entries.sort(key=lambda e: e.get("importance", 5), reverse=True)
        memory["entries"] = entries[:MAX_ENTRIES]

    _save_memory(memory)

    total = len(memory["entries"])
    return (
        f"Memory updated. New **{category}** entry added.\n"
        f"- **Domain**: {domain or '(general)'}\n"
        f"- **Importance**: {importance}/10\n"
        f"- **Total memories**: {total}\n"
        f"- **Tags**: {', '.join(tag_list) if tag_list else '(none)'}"
    )


@tool(parse_docstring=True)
def get_memory_stats() -> str:
    """Get statistics about the persistent ideation memory.

    Shows how many entries exist by category, recent activity,
    and storage information.

    Returns:
        Markdown-formatted memory statistics.
    """
    memory = _load_memory()
    entries = memory.get("entries", [])
    patterns = memory.get("patterns", [])
    meta = memory.get("meta", {})

    if not entries:
        return "Ideation memory is empty. No previous sessions recorded."

    # Count by category
    cat_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for e in entries:
        cat = e.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        domain = e.get("domain", "")
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Unique sessions
    sessions = set(e.get("session_id", "") for e in entries if e.get("session_id"))

    parts = [
        "## Ideation Memory Statistics",
        f"- **Total entries**: {len(entries)}",
        f"- **Unique sessions**: {len(sessions)}",
        f"- **Patterns recorded**: {len(patterns)}",
        f"- **Created**: {meta.get('created', 'unknown')}",
        f"- **Last updated**: {meta.get('last_updated', 'unknown')}",
        "",
        "### By Category",
    ]

    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        parts.append(f"- **{cat}**: {count}")

    if domain_counts:
        parts.append("\n### By Domain")
        for dom, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
            parts.append(f"- **{dom}**: {count}")

    return "\n".join(parts)
