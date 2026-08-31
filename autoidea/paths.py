"""Path resolution utilities for AutoIdea runtime directories.

Path configuration priority:
    1. Environment variables (AUTOIDEA_WORKSPACE_DIR, etc.)
    2. config.yaml settings
    3. Default values
"""

from __future__ import annotations

import os
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def _env_path(key: str) -> Path | None:
    """Get path from environment variable."""
    value = os.getenv(key)
    if not value:
        return None
    return _expand(value)


def _config_path(key: str) -> Path | None:
    """Get path from config file (lazy loaded)."""
    try:
        from .config import load_config
        config = load_config()
        value = getattr(config, key, None)
        if value:
            return _expand(value)
    except Exception:
        pass
    return None


def _resolve_path(env_key: str, config_key: str, default: Path) -> Path:
    """Resolve path with priority: env > config > default."""
    # 1. Check environment variable (highest priority)
    env_path = _env_path(env_key)
    if env_path:
        return env_path

    # 2. Check config file
    cfg_path = _config_path(config_key)
    if cfg_path:
        return cfg_path

    # 3. Use default
    return default


# Workspace root: cwd/workspace by default (keeps project dir clean)
# Priority: AUTOIDEA_WORKSPACE_DIR env > config.yaml workspace_dir > cwd/workspace
WORKSPACE_ROOT = _resolve_path(
    "AUTOIDEA_WORKSPACE_DIR",
    "workspace_dir",
    Path.cwd() / "workspace"
)

RUNS_DIR = _resolve_path(
    "AUTOIDEA_RUNS_DIR",
    "runs_dir",
    WORKSPACE_ROOT / "runs"
)

MEMORY_DIR = _resolve_path(
    "AUTOIDEA_MEMORY_DIR",
    "memory_dir",
    WORKSPACE_ROOT / "memory"
)


def set_workspace_root(path: str | Path) -> None:
    """Update workspace root and re-derive dependent directories.

    Directories with an explicit environment-variable or config override
    keep their value; all others are re-derived from the new root.
    Also resets ``_active_workspace`` to the new root as a safe default.
    """
    global WORKSPACE_ROOT, RUNS_DIR, MEMORY_DIR, _active_workspace
    WORKSPACE_ROOT = Path(path).resolve()
    _active_workspace = WORKSPACE_ROOT
    RUNS_DIR = _resolve_path(
        "AUTOIDEA_RUNS_DIR",
        "runs_dir",
        WORKSPACE_ROOT / "runs"
    )
    MEMORY_DIR = _resolve_path(
        "AUTOIDEA_MEMORY_DIR",
        "memory_dir",
        WORKSPACE_ROOT / "memory"
    )


def ensure_dirs() -> None:
    """Create runtime subdirectories (workspace, memory) if they do not exist."""
    for path in (WORKSPACE_ROOT, MEMORY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def default_workspace_dir() -> Path:
    """Default workspace for non-CLI usage."""
    return WORKSPACE_ROOT


# Active workspace (may differ from WORKSPACE_ROOT in per-session modes)
_active_workspace: Path = WORKSPACE_ROOT


def set_active_workspace(path: str | Path) -> None:
    """Update the active workspace root (called on agent creation)."""
    global _active_workspace
    _active_workspace = Path(path).resolve()


def get_active_workspace() -> Path:
    """Return the current active workspace path."""
    return _active_workspace


def resolve_virtual_path(virtual_path: str) -> Path:
    """Resolve a virtual workspace path (e.g. /image.png) to a real filesystem path."""
    vpath = virtual_path if virtual_path.startswith("/") else "/" + virtual_path
    return (_active_workspace / vpath.lstrip("/")).resolve()
