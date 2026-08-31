"""
Stream module - streaming event processing for CLI display.

Provides:
- SubAgentState / StreamState: Stream state tracking
- stream_agent_events: Async event generator
- Display functions: Rich rendering for streaming and final output

Adapted from EvoScientist's stream module for AutoIdea's research pipeline.
"""

from .state import SubAgentState, StreamState, _parse_todo_items, _build_todo_stats
from .events import stream_agent_events
from .display import (
    console,
    format_tool_result_compact,
    create_streaming_display,
    display_final_results,
    _astream_to_console,
)

__all__ = [
    # State
    "SubAgentState",
    "StreamState",
    "_parse_todo_items",
    "_build_todo_stats",
    # Events
    "stream_agent_events",
    # Display
    "console",
    "format_tool_result_compact",
    "create_streaming_display",
    "display_final_results",
    "_astream_to_console",
]
