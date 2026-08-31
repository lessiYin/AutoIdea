"""CLI package for AutoIdea.

Provides the Typer-based command-line interface for interactive and
single-shot usage of the AutoIdea research agent.
"""

import warnings

from ._app import app
from . import commands  # noqa: F401 — triggers @app.command decorator registration


def main():
    """CLI entry point.

    Called by the ``autoidea`` console script (see pyproject.toml) and
    by ``python -m autoidea``.
    """
    # Suppress noisy warnings from LLM provider SDKs
    warnings.filterwarnings("ignore", message=".*not known to support tools.*")
    warnings.filterwarnings(
        "ignore", message=".*type is unknown and inference may fail.*"
    )

    from .commands import _configure_logging

    _configure_logging()
    app()


__all__ = ["app", "main"]
