"""Stream event generator and chunk processing helpers.

Async generator that streams events from an agent graph,
plus helpers for processing AI message chunks and tool results.

Adapted from EvoScientist's stream.events module for AutoIdea's
research pipeline. Unlike EvoScientist, this module inlines the
emitter, tracker and utility functionality rather than importing
from separate submodules.
"""

import asyncio
import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import AIMessage, AIMessageChunk  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Image media types returned by read_file operations
_IMAGE_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
}

# Status marker constants
SUCCESS_PREFIX = "[OK]"
FAILURE_PREFIX = "[FAILED]"

# Display limits
TOOL_RESULT_MAX = 2000


# ---------------------------------------------------------------------------
# Inline StreamEvent + StreamEventEmitter
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Unified stream event."""

    type: str
    data: Dict[str, Any]


class _Emitter:
    """Creates standardized event dicts (inlined from EvoScientist emitter)."""

    @staticmethod
    def thinking(content: str, thinking_id: int = 0) -> StreamEvent:
        return StreamEvent(
            "thinking", {"type": "thinking", "content": content, "id": thinking_id}
        )

    @staticmethod
    def text(content: str) -> StreamEvent:
        return StreamEvent("text", {"type": "text", "content": content})

    @staticmethod
    def tool_call(name: str, args: Dict[str, Any], tool_id: str = "") -> StreamEvent:
        return StreamEvent(
            "tool_call",
            {"type": "tool_call", "name": name, "args": args, "id": tool_id},
        )

    @staticmethod
    def tool_result(name: str, content: str, success: bool = True) -> StreamEvent:
        return StreamEvent(
            "tool_result",
            {
                "type": "tool_result",
                "name": name,
                "content": content,
                "success": success,
            },
        )

    @staticmethod
    def subagent_start(name: str, description: str) -> StreamEvent:
        return StreamEvent(
            "subagent_start",
            {"type": "subagent_start", "name": name, "description": description},
        )

    @staticmethod
    def subagent_tool_call(
        subagent: str, name: str, args: Dict[str, Any], tool_id: str = ""
    ) -> StreamEvent:
        return StreamEvent(
            "subagent_tool_call",
            {
                "type": "subagent_tool_call",
                "subagent": subagent,
                "name": name,
                "args": args,
                "id": tool_id,
            },
        )

    @staticmethod
    def subagent_tool_result(
        subagent: str, name: str, content: str, success: bool = True
    ) -> StreamEvent:
        return StreamEvent(
            "subagent_tool_result",
            {
                "type": "subagent_tool_result",
                "subagent": subagent,
                "name": name,
                "content": content,
                "success": success,
            },
        )

    @staticmethod
    def subagent_end(name: str) -> StreamEvent:
        return StreamEvent("subagent_end", {"type": "subagent_end", "name": name})

    @staticmethod
    def done(response: str = "") -> StreamEvent:
        return StreamEvent(
            "done", {"type": "done", "content": response, "response": response}
        )

    @staticmethod
    def usage_stats(input_tokens: int, output_tokens: int) -> StreamEvent:
        return StreamEvent(
            "usage_stats",
            {
                "type": "usage_stats",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    @staticmethod
    def interrupt(
        interrupt_id: str,
        action_requests: list,
        review_configs: list | None = None,
    ) -> StreamEvent:
        return StreamEvent(
            "interrupt",
            {
                "type": "interrupt",
                "interrupt_id": interrupt_id,
                "action_requests": action_requests,
                "review_configs": review_configs or [],
            },
        )

    @staticmethod
    def ask_user_interrupt(
        interrupt_id: str,
        questions: list,
        tool_call_id: str = "",
    ) -> StreamEvent:
        return StreamEvent(
            "ask_user",
            {
                "type": "ask_user",
                "interrupt_id": interrupt_id,
                "questions": questions,
                "tool_call_id": tool_call_id,
            },
        )

    @staticmethod
    def summarization(content: str) -> StreamEvent:
        return StreamEvent(
            "summarization", {"type": "summarization", "content": content}
        )

    @staticmethod
    def error(message: str) -> StreamEvent:
        return StreamEvent("error", {"type": "error", "message": message})


# ---------------------------------------------------------------------------
# Inline ToolCallTracker
# ---------------------------------------------------------------------------


@dataclass
class _ToolCallInfo:
    """Tool call information."""

    id: str
    name: str
    args: Dict = field(default_factory=dict)
    emitted: bool = False
    args_complete: bool = False
    _json_buffer: str = ""


class _ToolCallTracker:
    """Tracks incremental JSON parsing for tool parameters.

    Handles tool_use blocks where arguments arrive in fragments via
    input_json_delta.
    """

    def __init__(self):
        self._calls: Dict[str, _ToolCallInfo] = {}
        self._last_tool_id: Optional[str] = None

    def update(
        self,
        tool_id: str,
        name: Optional[str] = None,
        args: Optional[Dict] = None,
        args_complete: bool = False,
    ) -> None:
        if tool_id not in self._calls:
            self._calls[tool_id] = _ToolCallInfo(
                id=tool_id,
                name=name or "",
                args=args or {},
                args_complete=args_complete,
            )
            self._last_tool_id = tool_id
        else:
            info = self._calls[tool_id]
            if name:
                info.name = name
            if args:
                info.args = args
            if args_complete:
                info.args_complete = True

    def append_json_delta(self, partial_json: str, index: int = 0) -> None:
        tool_id = self._last_tool_id
        if tool_id and tool_id in self._calls:
            self._calls[tool_id]._json_buffer += partial_json

    def finalize_all(self) -> None:
        for info in self._calls.values():
            if info._json_buffer:
                try:
                    info.args = json.loads(info._json_buffer)
                except json.JSONDecodeError:
                    pass
                info._json_buffer = ""
            info.args_complete = True

    def is_ready(self, tool_id: str) -> bool:
        """Check if ready to emit -- has name, not yet emitted.

        BUG FIX: Do NOT return True on the first chunk alone.
        Validate that we have accumulated sufficient JSON before
        declaring completion.
        """
        if tool_id not in self._calls:
            return False
        info = self._calls[tool_id]
        if not info.name or info.emitted:
            return False
        # If args were provided directly (not via JSON delta), ready
        if info.args and not info._json_buffer:
            return True
        # If accumulating JSON, only ready once buffer parses successfully
        if info._json_buffer:
            try:
                json.loads(info._json_buffer)
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        # Name present but no args yet -- still ready (e.g. no-arg tools)
        return True

    def mark_emitted(self, tool_id: str) -> None:
        if tool_id in self._calls:
            self._calls[tool_id].emitted = True

    def get(self, tool_id: str) -> Optional[_ToolCallInfo]:
        return self._calls.get(tool_id)

    def get_all(self) -> list[_ToolCallInfo]:
        return list(self._calls.values())

    def get_pending(self) -> list[_ToolCallInfo]:
        return [info for info in self._calls.values() if not info.emitted]

    def emit_all_pending(self) -> list[_ToolCallInfo]:
        pending = self.get_pending()
        for info in pending:
            info.emitted = True
        return pending


# ---------------------------------------------------------------------------
# Inline utility helpers
# ---------------------------------------------------------------------------


def _is_success(content: str) -> bool:
    """Determine if tool output indicates successful execution."""
    content = content.strip()
    if content.startswith(SUCCESS_PREFIX):
        return True
    if content.startswith(FAILURE_PREFIX):
        return False
    head = "\n".join(content.splitlines()[:3])
    error_patterns = [
        "Traceback (most recent call last)",
        "Exception:",
        "Error:",
        "Error invoking tool",
        "Failed ",
    ]
    return not any(pattern in head for pattern in error_patterns)


# ---------------------------------------------------------------------------
# Extract helpers
# ---------------------------------------------------------------------------


def _extract_tool_content(msg) -> tuple[str, bool]:
    """Extract display-safe content from a ToolMessage.

    Returns:
        (content_string, is_image) -- a short summary for images,
        or the raw string content for normal results.
    """
    additional = getattr(msg, "additional_kwargs", None) or {}
    media_type = additional.get("read_file_media_type", "")
    if media_type and media_type in _IMAGE_MEDIA_TYPES:
        file_path = additional.get("read_file_path", "")
        if not file_path:
            file_path = getattr(msg, "name", "image")
        return f"[OK] Image displayed: {file_path} ({media_type})", True

    content = getattr(msg, "content", "")
    # Guard against list-type content (image content blocks without metadata)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image" or "base64" in block:
                    return "[OK] Image displayed", True
        # Non-image list content -- join text blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content), False

    return str(content), False


def _extract_summarization_text(msg: Any) -> str:
    """Extract plain text from a summarization chunk.

    The summarization LLM streams ``AIMessageChunk`` objects whose
    ``content`` may be a plain string **or** a list of content blocks
    depending on the provider.  This helper normalises both forms to
    a plain string.
    """
    if not hasattr(msg, "content"):
        return ""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Main async event generator
# ---------------------------------------------------------------------------


async def stream_agent_events(
    agent: Any,
    message: Any,
    thread_id: str,
    metadata: dict | None = None,
    media: list[str] | None = None,
) -> AsyncIterator[dict]:
    """Stream events from the agent graph using async iteration.

    Uses agent.astream() with subgraphs=True to see sub-agent activity.

    Args:
        agent: Compiled state graph from create_cli_agent()
        message: User message
        thread_id: Thread ID for conversation persistence
        metadata: Optional metadata dict merged into the LangGraph config
            (e.g. agent_name, updated_at for checkpoint persistence).
        media: Optional list of local file paths for attachments.

    Yields:
        Event dicts: thinking, text, tool_call, tool_result,
                     subagent_start, subagent_tool_call, subagent_tool_result,
                     subagent_end, interrupt, ask_user, summarization,
                     usage_stats, done, error
    """
    # Preserve every tool call while executing them one at a time.  This is
    # important for providers with low concurrency limits and avoids relying
    # on the model to repeat calls that a middleware discarded.
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": 1,
    }
    if metadata:
        config["metadata"] = metadata
    emitter = _Emitter()
    main_tracker = _ToolCallTracker()
    full_response = ""

    # Track sub-agent names
    _key_to_name: dict[str, str] = {}
    _announced_names: list[str] = []
    _assigned_names: set[str] = set()
    _announced_task_ids: list[str] = []
    _task_id_to_name: dict[str, str] = {}
    _subagent_trackers: dict[str, _ToolCallTracker] = {}

    def _register_task_tool_call(tc_data: dict) -> str | None:
        """Register or update a task tool call, return subagent name if started."""
        tool_id = tc_data.get("id", "")
        if not tool_id:
            return None
        args = tc_data.get("args", {}) or {}
        desc = str(args.get("description", "")).strip()
        sa_name = str(args.get("subagent_type", "")).strip()
        if not sa_name:
            sa_name = desc.split("\n")[0].strip()
            sa_name = sa_name[:30] + "\u2026" if len(sa_name) > 30 else sa_name
        if not sa_name:
            sa_name = "sub-agent"

        if tool_id not in _announced_task_ids:
            _announced_task_ids.append(tool_id)
            _announced_names.append(sa_name)
            _task_id_to_name[tool_id] = sa_name
            return sa_name

        current = _task_id_to_name.get(tool_id, "sub-agent")
        if sa_name != "sub-agent" and current != sa_name:
            _task_id_to_name[tool_id] = sa_name
            try:
                idx = _announced_task_ids.index(tool_id)
                if idx < len(_announced_names):
                    _announced_names[idx] = sa_name
            except ValueError:
                pass
            return sa_name
        return None

    def _extract_task_id(namespace: tuple) -> tuple[str | None, str | None]:
        for part in namespace:
            part_str = str(part)
            if "task:" in part_str:
                tail = part_str.split("task:", 1)[1]
                task_id = tail.split(":", 1)[0] if tail else ""
                if task_id:
                    return task_id, part_str
        return None, None

    def _find_task_id_from_metadata(meta: dict | None) -> str | None:
        if not meta:
            return None
        candidates = (
            "tool_call_id",
            "task_id",
            "parent_run_id",
            "root_run_id",
            "run_id",
        )
        for key in candidates:
            val = meta.get(key)
            if val and val in _task_id_to_name:
                return val
        return None

    def _get_subagent_key(namespace: tuple, meta: dict | None) -> str | None:
        if not namespace:
            return None
        task_id, task_ns = _extract_task_id(namespace)
        if task_ns:
            return task_ns
        meta_task_id = _find_task_id_from_metadata(meta)
        if meta_task_id:
            return f"task:{meta_task_id}"
        if meta:
            for key in (
                "parent_run_id",
                "root_run_id",
                "run_id",
                "graph_id",
                "node_id",
            ):
                val = meta.get(key)
                if val:
                    return f"{key}:{val}"
        return str(namespace)

    def _get_subagent_name(namespace: tuple, meta: dict | None) -> str | None:
        """Resolve sub-agent name from namespace, or None if main agent."""
        if not namespace:
            return None

        key = _get_subagent_key(namespace, meta) or str(namespace)

        # 0) lc_agent_name from metadata
        if meta:
            lc_name = meta.get("lc_agent_name", "")
            if isinstance(lc_name, str):
                lc_name = lc_name.strip()
            if lc_name and lc_name not in (
                "sub-agent",
                "agent",
                "tools",
                "AutoIdea",
                "LangGraph",
                "",
            ):
                _key_to_name[key] = lc_name
                return lc_name

        # 1) Resolve by task_id
        task_id, _task_ns = _extract_task_id(namespace)
        if task_id and task_id in _task_id_to_name:
            name = _task_id_to_name[task_id]
            if name and name != "sub-agent":
                _assigned_names.add(name)
                _key_to_name[key] = name
                return name

        meta_task_id = _find_task_id_from_metadata(meta)
        if meta_task_id and meta_task_id in _task_id_to_name:
            name = _task_id_to_name[meta_task_id]
            if name and name != "sub-agent":
                _assigned_names.add(name)
                _key_to_name[key] = name
                return name

        # 2) Cached real name
        cached = _key_to_name.get(key)
        if cached and cached != "sub-agent":
            return cached

        # 3) Assign next announced name from queue
        for announced in _announced_names:
            if announced not in _assigned_names and announced != "sub-agent":
                _assigned_names.add(announced)
                _key_to_name[key] = announced
                return announced

        # 4) Fallback without caching
        return "sub-agent"

    # Build input for agent.astream()
    if isinstance(message, str):
        user_content: str | list[dict[str, Any]] = message
        if media:
            _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
            _MAX_INLINE_SIZE = 5 * 1024 * 1024  # 5 MB
            content_blocks: list[dict[str, Any]] = []
            if message:
                content_blocks.append({"type": "text", "text": message})

            def _read_file_b64(path: str) -> str:
                with open(path, "rb") as fh:
                    return base64.b64encode(fh.read()).decode("ascii")

            file_refs: list[str] = []
            for path in media:
                ext = os.path.splitext(path)[1].lower()
                is_image = ext in _IMAGE_EXTS and await asyncio.to_thread(
                    os.path.isfile, path
                )
                if is_image:
                    fsize = await asyncio.to_thread(os.path.getsize, path)
                    if fsize <= _MAX_INLINE_SIZE:
                        mime = mimetypes.guess_type(path)[0] or "image/png"
                        b64 = await asyncio.to_thread(_read_file_b64, path)
                        content_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                },
                            }
                        )
                    else:
                        file_refs.append(path)
                else:
                    file_refs.append(path)
            if file_refs:
                ref_text = "\n".join(
                    f"[attached file: {os.path.basename(p)}] path: {p}"
                    for p in file_refs
                )
                content_blocks.append({"type": "text", "text": ref_text})
            if content_blocks:
                user_content = content_blocks
        astream_input: Any = {"messages": [{"role": "user", "content": user_content}]}
    else:
        # HITL resume: Command object passed directly to agent
        astream_input = message

    _summarization_in_progress = False

    try:
        async for chunk in agent.astream(
            astream_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            # Multi-mode + subgraphs: 3-tuple (namespace, mode, data)
            # Single-mode + subgraphs: 2-tuple (namespace, data) -- fallback
            if not isinstance(chunk, tuple):
                continue

            namespace: tuple = ()
            data: Any
            mode_str: str

            if len(chunk) == 3:
                namespace, mode_str, data = chunk
                if not isinstance(namespace, tuple):
                    namespace = ()
            elif len(chunk) == 2:
                first = chunk[0]
                if isinstance(first, tuple):
                    namespace = first
                    data = chunk[1]
                else:
                    data = chunk
                mode_str = "messages"
            else:
                continue

            # Parse HITL / ask_user interrupts from updates mode
            if mode_str == "updates":
                if isinstance(data, dict) and "__interrupt__" in data:
                    for interrupt_obj in data["__interrupt__"]:
                        if isinstance(interrupt_obj, dict):
                            interrupt_value = interrupt_obj.get("value", {})
                        else:
                            interrupt_value = getattr(interrupt_obj, "value", {})

                        iv_type = (
                            interrupt_value.get("type")
                            if isinstance(interrupt_value, dict)
                            else getattr(interrupt_value, "type", None)
                        )
                        if iv_type == "ask_user":
                            questions = (
                                interrupt_value.get("questions", [])
                                if isinstance(interrupt_value, dict)
                                else getattr(interrupt_value, "questions", [])
                            )
                            tc_id = (
                                interrupt_value.get("tool_call_id", "")
                                if isinstance(interrupt_value, dict)
                                else getattr(interrupt_value, "tool_call_id", "")
                            )
                            ns_parts = (
                                interrupt_obj.get("ns", [""])
                                if isinstance(interrupt_obj, dict)
                                else getattr(interrupt_obj, "ns", [""])
                            )
                            interrupt_id = str(ns_parts[0]) if ns_parts else "default"
                            yield emitter.ask_user_interrupt(
                                interrupt_id, questions, tc_id
                            ).data
                            continue

                        # Standard HITL approval interrupt
                        if isinstance(interrupt_value, dict):
                            action_reqs = interrupt_value.get("action_requests", [])
                            review_cfgs = interrupt_value.get("review_configs", [])
                        else:
                            action_reqs = getattr(
                                interrupt_value, "action_requests", []
                            )
                            review_cfgs = getattr(interrupt_value, "review_configs", [])
                        if action_reqs:
                            ns_parts = (
                                interrupt_obj.get("ns", [""])
                                if isinstance(interrupt_obj, dict)
                                else getattr(interrupt_obj, "ns", [""])
                            )
                            interrupt_id = str(ns_parts[0]) if ns_parts else "default"
                            yield emitter.interrupt(
                                interrupt_id, action_reqs, review_cfgs
                            ).data
                continue
            if mode_str != "messages":
                continue

            # Unpack message + metadata from data
            msg: Any
            chunk_metadata: dict = {}
            if isinstance(data, tuple) and len(data) >= 2:
                msg = data[0]
                chunk_metadata = data[1] or {}
            else:
                msg = data

            # Accumulate summarization middleware chunks
            if (
                isinstance(chunk_metadata, dict)
                and chunk_metadata.get("lc_source") == "summarization"
            ):
                if not _summarization_in_progress:
                    _summarization_in_progress = True
                chunk_text = _extract_summarization_text(msg)
                if chunk_text:
                    yield emitter.summarization(chunk_text).data
                continue

            subagent = _get_subagent_name(namespace, chunk_metadata)
            subagent_tracker = None
            if subagent:
                tracker_key = (
                    _get_subagent_key(namespace, chunk_metadata) or str(namespace)
                )
                subagent_tracker = _subagent_trackers.setdefault(
                    tracker_key, _ToolCallTracker()
                )

            # Extract token usage from main-agent AIMessages
            if isinstance(msg, (AIMessageChunk, AIMessage)) and not subagent:
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    inp = (
                        usage.get("input_tokens", 0)
                        if isinstance(usage, dict)
                        else getattr(usage, "input_tokens", 0)
                    )
                    out = (
                        usage.get("output_tokens", 0)
                        if isinstance(usage, dict)
                        else getattr(usage, "output_tokens", 0)
                    )
                    if inp or out:
                        yield emitter.usage_stats(inp, out).data

            # Process AIMessageChunk / AIMessage
            if isinstance(msg, (AIMessageChunk, AIMessage)):
                if subagent:
                    for ev in _process_chunk_content(msg, emitter, subagent_tracker):
                        if ev.type == "tool_call":
                            yield emitter.subagent_tool_call(
                                subagent,
                                ev.data["name"],
                                ev.data["args"],
                                ev.data.get("id", ""),
                            ).data

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            tool_id = tc.get("id", "")
                            if not name and not tool_id:
                                continue
                            yield emitter.subagent_tool_call(
                                subagent,
                                name,
                                args if isinstance(args, dict) else {},
                                tool_id,
                            ).data
                else:
                    for ev in _process_chunk_content(msg, emitter, main_tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        yield ev.data

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for ev in _process_tool_calls(
                            msg.tool_calls, emitter, main_tracker
                        ):
                            yield ev.data
                            tc_data = ev.data
                            if tc_data.get("name") == "task":
                                started_name = _register_task_tool_call(tc_data)
                                if started_name:
                                    desc = str(
                                        tc_data.get("args", {}).get("description", "")
                                    ).strip()
                                    yield emitter.subagent_start(
                                        started_name, desc
                                    ).data

            # Process ToolMessage (tool execution result)
            elif hasattr(msg, "type") and msg.type == "tool":
                if subagent:
                    if subagent_tracker:
                        subagent_tracker.finalize_all()
                        for info in subagent_tracker.emit_all_pending():
                            yield emitter.subagent_tool_call(
                                subagent,
                                info.name,
                                info.args,
                                info.id,
                            ).data
                    name = getattr(msg, "name", "unknown")
                    raw_content, _is_img = _extract_tool_content(msg)
                    content = raw_content[:TOOL_RESULT_MAX]
                    success = _is_success(content)
                    yield emitter.subagent_tool_result(
                        subagent, name, content, success
                    ).data
                else:
                    for ev in _process_tool_result(msg, emitter, main_tracker):
                        yield ev.data
                        if ev.type == "tool_call" and ev.data.get("name") == "task":
                            started_name = _register_task_tool_call(ev.data)
                            if started_name:
                                desc = str(
                                    ev.data.get("args", {}).get("description", "")
                                ).strip()
                                yield emitter.subagent_start(started_name, desc).data
                    # Check if this is a task result -> sub-agent ended
                    name = getattr(msg, "name", "")
                    if name == "task":
                        tool_call_id = getattr(msg, "tool_call_id", "")
                        sa_name = _task_id_to_name.get(tool_call_id, "sub-agent")
                        yield emitter.subagent_end(sa_name).data

    except Exception as e:
        yield emitter.error(str(e)).data
        raise

    yield emitter.done(full_response).data


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------


def _process_chunk_content(
    chunk, emitter: _Emitter, tracker: _ToolCallTracker
):
    """Process content blocks from an AI message chunk."""
    content = chunk.content

    if isinstance(content, str):
        if content:
            yield emitter.text(content)
            return

    blocks = None
    if hasattr(chunk, "content_blocks"):
        try:
            blocks = chunk.content_blocks
        except Exception:
            blocks = None

    if blocks is None:
        if isinstance(content, dict):
            blocks = [content]
        elif isinstance(content, list):
            blocks = content
        else:
            return

    for raw_block in blocks:
        block = raw_block
        if not isinstance(block, dict):
            if hasattr(block, "model_dump"):
                block = block.model_dump()
            elif hasattr(block, "dict"):
                block = block.dict()
            else:
                continue

        block_type = block.get("type")

        if block_type in ("thinking", "reasoning"):
            thinking_text = block.get("thinking") or block.get("reasoning") or ""
            if thinking_text:
                yield emitter.thinking(thinking_text)

        elif block_type == "text":
            text = block.get("text") or block.get("content") or ""
            if text:
                yield emitter.text(text)

        elif block_type in ("tool_use", "tool_call"):
            tool_id = block.get("id", "")
            name = block.get("name", "")
            args = (
                block.get("input") if block_type == "tool_use" else block.get("args")
            )
            args_payload = args if isinstance(args, dict) else {}

            if tool_id:
                tracker.update(tool_id, name=name, args=args_payload)
                if tracker.is_ready(tool_id):
                    tracker.mark_emitted(tool_id)
                    yield emitter.tool_call(name, args_payload, tool_id)

        elif block_type == "input_json_delta":
            partial_json = block.get("partial_json", "")
            if partial_json:
                tracker.append_json_delta(partial_json, block.get("index", 0))

        elif block_type == "tool_call_chunk":
            tool_id = block.get("id", "")
            name = block.get("name", "")
            if tool_id:
                tracker.update(tool_id, name=name)
            partial_args = block.get("args", "")
            if isinstance(partial_args, str) and partial_args:
                tracker.append_json_delta(partial_args, block.get("index", 0))


def _process_tool_calls(
    tool_calls: list, emitter: _Emitter, tracker: _ToolCallTracker
):
    """Process tool_calls from chunk.tool_calls attribute."""
    for tc in tool_calls:
        tool_id = tc.get("id", "")
        if tool_id:
            name = tc.get("name", "")
            args = tc.get("args", {})
            args_payload = args if isinstance(args, dict) else {}

            tracker.update(tool_id, name=name, args=args_payload)
            if tracker.is_ready(tool_id):
                tracker.mark_emitted(tool_id)
                yield emitter.tool_call(name, args_payload, tool_id)


def _process_tool_result(chunk, emitter: _Emitter, tracker: _ToolCallTracker):
    """Process a ToolMessage result."""
    tracker.finalize_all()

    # Re-emit all tool calls with complete args
    for info in tracker.get_all():
        yield emitter.tool_call(info.name, info.args, info.id)

    name = getattr(chunk, "name", "unknown")
    raw_content, _is_img = _extract_tool_content(chunk)
    content = raw_content[:TOOL_RESULT_MAX]
    if len(raw_content) > TOOL_RESULT_MAX:
        content += "\n... (truncated)"

    success = _is_success(content)
    yield emitter.tool_result(name, content, success)
