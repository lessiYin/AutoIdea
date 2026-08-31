"""Agent loading and workspace helpers for AutoIdea CLI.

Provides utility functions for path shortening, run-name deduplication,
per-session workspace creation, and lazy agent loading.
"""

from __future__ import annotations

import os


def _shorten_path(path: str) -> str:
    """Shorten an absolute path to a relative path from the current directory.

    If *path* starts with the current working directory it is trimmed to
    show only the trailing portion prefixed by the cwd basename, making
    console output more compact.

    Args:
        path: Filesystem path to shorten.

    Returns:
        Shortened path string, or the original if it cannot be shortened.
    """
    if not path:
        return path
    try:
        cwd = os.getcwd()
        if path.startswith(cwd):
            rel = path[len(cwd):].lstrip(os.sep)
            return (
                os.path.join(os.path.basename(cwd), rel)
                if rel
                else os.path.basename(cwd)
            )
        return path
    except Exception:
        return path


def _load_agent(
    workspace_dir: str | None = None,
    checkpointer=None,
    config=None,
):
    """Load the AutoIdea CLI agent with optional persistent checkpointer.

    Args:
        workspace_dir: Optional per-session workspace directory.
        checkpointer: Optional LangGraph checkpointer (e.g.
            ``AsyncSqliteSaver``).  Falls back to ``InMemorySaver`` when
            ``None``.
        config: Optional pre-loaded ``AutoIdeaConfig``.  Forwarded to
            ``create_cli_agent`` to avoid double config loading.

    Returns:
        A compiled LangGraph agent ready for ``astream`` / ``ainvoke``.
    """
    from autoidea.autoidea import create_cli_agent

    return create_cli_agent(
        workspace_dir=workspace_dir,
        checkpointer=checkpointer,
        config=config,
    )
