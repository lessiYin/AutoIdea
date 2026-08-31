"""Monkey-patches for third-party dependencies.

Applied once at agent-creation time so that fixes ship with the autoidea
package and do not require modifying installed library source files.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def _patched_truncate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
    """Improved truncation that produces a clear placeholder instead of
    content that looks real.

    The original implementation truncates to ``value[:20] + "...(argument
    truncated)"``, which the LLM may mistake for actual file content and
    copy back verbatim when re-writing files -- causing data loss.

    This replacement emits an unambiguous machine-readable placeholder that
    tells the model to re-read the file if needed.
    """
    args = tool_call.get("args", {})

    # Determine file_path for a more informative placeholder
    file_path = args.get("file_path") or args.get("path") or ""

    truncated_args = {}
    modified = False

    for key, value in args.items():
        if isinstance(value, str) and len(value) > self._max_arg_length:
            if file_path:
                truncated_args[key] = (
                    f"[Content for '{file_path}' omitted to save context. "
                    f"Original length: {len(value)} chars. "
                    f"Re-read the file with read_file if you need the content.]"
                )
            else:
                truncated_args[key] = (
                    f"[Large content omitted to save context. "
                    f"Original length: {len(value)} chars.]"
                )
            modified = True
        else:
            truncated_args[key] = value

    if modified:
        return {**tool_call, "args": truncated_args}
    return tool_call


def _patch_tool_message():
    """Patch ToolMessage to ensure tool_call_id is never 'unknown'."""
    try:
        from langchain_core.messages import ToolMessage
        import uuid

        original_init = ToolMessage.__init__

        def patched_init(self, content, *, tool_call_id=None, **kwargs):
            # Ensure every ToolMessage has a concrete tool call id.
            if tool_call_id is None or tool_call_id == "unknown" or tool_call_id == "":
                tool_call_id = f"fallback_{uuid.uuid4().hex[:12]}"
            original_init(self, content, tool_call_id=tool_call_id, **kwargs)

        ToolMessage.__init__ = patched_init
        logger.debug("Patched ToolMessage.__init__ to prevent 'unknown' tool_call_id")
    except Exception as exc:
        logger.debug("Skipping ToolMessage patch: %s", exc)


def apply_patches() -> None:
    """Apply all monkey-patches.  Safe to call multiple times."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    try:
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
        )

        SummarizationMiddleware._truncate_tool_call = _patched_truncate_tool_call  # type: ignore[assignment]
        logger.debug("Patched SummarizationMiddleware._truncate_tool_call")
    except (ImportError, AttributeError) as exc:
        logger.debug("Skipping SummarizationMiddleware patch: %s", exc)

    # Patch ToolMessage to prevent 'unknown' tool_call_id
    _patch_tool_message()

    _PATCHES_APPLIED = True
