"""Tool error handler middleware for AutoIdea.

Catches tool execution exceptions and converts them to error ToolMessages
instead of crashing the agent loop.
"""

from __future__ import annotations

import asyncio
import os
import traceback

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:
    try:
        from deepagents_langgraph.middleware import AgentMiddleware
    except ImportError:
        AgentMiddleware = object

try:
    from langgraph.errors import GraphInterrupt as _GraphInterrupt
except ImportError:
    _GraphInterrupt = None


DEFAULT_TOOL_CALL_TIMEOUT_S = 300.0


def _get_tool_call_timeout() -> float | None:
    """Return the async tool-call timeout in seconds; 0 disables it."""
    raw = os.getenv("AUTOIDEA_TOOL_CALL_TIMEOUT_S", str(DEFAULT_TOOL_CALL_TIMEOUT_S))
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        timeout = DEFAULT_TOOL_CALL_TIMEOUT_S
    if timeout <= 0:
        return None
    return timeout


def _get_tool_call_id(request) -> str:
    """Best-effort extraction of a valid tool call id."""
    # Try each supported request shape and never return "unknown".
    tool_call_id = None

    # First try the direct attribute.
    tool_call_id = getattr(request, "tool_call_id", None)

    # Then handle dict-like request objects.
    if tool_call_id is None and hasattr(request, "get"):
        tool_call_id = request.get("tool_call_id")

    # Fall back to the request id.
    if tool_call_id is None:
        tool_call_id = getattr(request, "id", None)

    # Inspect the nested tool_call object.
    if tool_call_id is None:
        tool_call = getattr(request, "tool_call", None)
        if tool_call:
            tool_call_id = getattr(tool_call, "id", None)
            if tool_call_id is None and isinstance(tool_call, dict):
                tool_call_id = tool_call.get("id")

    # Finally inspect tool calls attached to the message.
    if tool_call_id is None:
        message = getattr(request, "message", None)
        if message:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls and len(tool_calls) > 0:
                tool_call_id = tool_calls[0].get("id") if isinstance(tool_calls[0], dict) else getattr(tool_calls[0], "id", None)

    # Generate a valid UUID-like fallback rather than returning "unknown".
    if tool_call_id is None or tool_call_id == "unknown":
        import uuid
        tool_call_id = f"error_fallback_{uuid.uuid4().hex[:8]}"

    return tool_call_id


def _get_tool_name(request) -> str:
    """Best-effort extraction of the requested tool name."""
    tool_call = getattr(request, "tool_call", None)
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or "")
    return str(getattr(tool_call, "name", "") or "")


def _build_timeout_message(request, timeout_s: float):
    """Build an error ToolMessage for a tool call that exceeded the deadline."""
    from langchain_core.messages import ToolMessage

    error_text = (
        f"Tool execution timed out after {timeout_s:g} seconds.\n\n"
        "This usually means an external API, network request, or provider "
        "tool call stopped producing progress. Mark the current batch/query as "
        "failed or retry with a smaller scope; do not wait indefinitely."
    )
    return ToolMessage(
        content=error_text,
        tool_call_id=_get_tool_call_id(request),
        status="error",
    )


def _build_error_message(request):
    """Build an error ToolMessage from a failed tool request."""
    from langchain_core.messages import ToolMessage

    tb = traceback.format_exc()
    error_text = (
        f"Tool execution failed.\n\n"
        f"```\n{tb[-2000:]}\n```\n\n"
        f"Please try a different approach or tool."
    )

    return ToolMessage(
        content=error_text,
        tool_call_id=_get_tool_call_id(request),
        status="error",
    )


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """Middleware that catches tool execution errors.

    LangGraph's default ToolNode only catches argument-validation
    errors. This middleware wraps every tool call to catch runtime
    exceptions, format the traceback, and return it as a ToolMessage
    with status="error".

    GraphInterrupt is explicitly re-raised — it must propagate
    (used by ask_user's interrupt(), among other things).
    """

    name = "tool_error_handler"

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as exc:
            if _GraphInterrupt is not None and isinstance(exc, _GraphInterrupt):
                raise
            return _build_error_message(request)

    async def awrap_tool_call(self, request, handler):
        try:
            # ``task`` runs an entire sub-agent stage containing many individually
            # bounded tool calls. Applying the single-call deadline to that
            # composite operation aborts healthy literature/reading stages.
            if _get_tool_name(request) == "task":
                return await handler(request)
            timeout_s = _get_tool_call_timeout()
            if timeout_s is None:
                return await handler(request)
            return await asyncio.wait_for(handler(request), timeout=timeout_s)
        except asyncio.TimeoutError:
            return _build_timeout_message(request, _get_tool_call_timeout() or 0)
        except Exception as exc:
            if _GraphInterrupt is not None and isinstance(exc, _GraphInterrupt):
                raise
            return _build_error_message(request)
