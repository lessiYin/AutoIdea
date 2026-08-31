"""Session persistence for AutoIdea.

Provides SQLite-backed checkpointing for conversation persistence,
thread management, and session metadata.
"""

from __future__ import annotations

import secrets
from pathlib import Path


def _get_db_path() -> Path:
    """Get the path to the sessions database."""
    from autoidea.config.settings import get_state_dir

    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "sessions.db"


def generate_thread_id() -> str:
    """Generate a short human-readable thread ID.

    Returns:
        Random 8-character hexadecimal string.
    """
    return secrets.token_hex(4)


def get_checkpointer():
    """Get an async SQLite checkpointer for LangGraph.

    Returns an async context manager — use as::

        async with get_checkpointer() as saver:
            ...

    ``AsyncSqliteSaver.from_conn_string`` already returns an
    ``@asynccontextmanager``, so this function must NOT be ``async def``
    (wrapping it in a coroutine would strip the context-manager protocol).
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    db_path = str(_get_db_path())
    return AsyncSqliteSaver.from_conn_string(db_path)


async def list_threads(limit: int = 20) -> list[dict]:
    """List recent conversation threads.

    Args:
        limit: Maximum number of threads to return.

    Returns:
        List of thread info dicts with thread_id, created_at, etc.
    """
    import aiosqlite
    db_path = str(_get_db_path())
    threads = []

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT DISTINCT thread_id FROM checkpoints "
                "ORDER BY rowid DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            for row in rows:
                threads.append({"thread_id": row[0]})
    except Exception:
        pass

    return threads


async def thread_exists(thread_id: str) -> bool:
    """Check if a thread exists in the database.

    Args:
        thread_id: Thread identifier to check.

    Returns:
        True if the thread exists.
    """
    import aiosqlite
    db_path = str(_get_db_path())

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1",
                (thread_id,),
            )
            row = await cursor.fetchone()
            return row is not None
    except Exception:
        return False


async def find_similar_threads(partial_id: str) -> list[str]:
    """Find threads matching a partial ID.

    Args:
        partial_id: Partial thread ID to search for.

    Returns:
        List of matching thread IDs.
    """
    import aiosqlite
    db_path = str(_get_db_path())
    matches = []

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT DISTINCT thread_id FROM checkpoints "
                "WHERE thread_id LIKE ?",
                (f"{partial_id}%",),
            )
            rows = await cursor.fetchall()
            matches = [row[0] for row in rows]
    except Exception:
        pass

    return matches


async def delete_thread(thread_id: str) -> bool:
    """Delete a thread and all its checkpoints.

    Args:
        thread_id: Thread identifier to delete.

    Returns:
        True if deletion was successful.
    """
    import aiosqlite
    db_path = str(_get_db_path())

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            await db.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (thread_id,),
            )
            await db.commit()
            return True
    except Exception:
        return False
