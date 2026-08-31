"""Rich display functions for streaming CLI output.

Contains all rendering logic: tool call lines, sub-agent sections,
todo panels, streaming display layout, and final results display.
Also provides the shared console global and file-logging support.

Adapted from EvoScientist's stream.display module for AutoIdea's
research pipeline. Unlike EvoScientist, utility / formatter code is
inlined here rather than imported from separate submodules.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path, PurePath
from typing import Any, Callable

from rich.console import Console, Group  # type: ignore[import-untyped]
from rich.live import Live  # type: ignore[import-untyped]
from rich.markdown import Markdown  # type: ignore[import-untyped]
from rich.panel import Panel  # type: ignore[import-untyped]
from rich.spinner import Spinner  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]

from ..paths import resolve_virtual_path
from .state import (
    StreamState,
    SubAgentState,
    _build_todo_stats,
    _parse_todo_items,
    _INTERNAL_TOOLS,
)
from .events import stream_agent_events

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared globals
# ---------------------------------------------------------------------------

_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".pdf"}

# Token usage display control.  When False, per-turn token counts are
# suppressed from the streaming display to reduce visual noise.  The
# final summary token count is always shown.
# Can be overridden via environment variable AUTOIDEA_SHOW_TOKEN_DETAILS=1
_SHOW_TOKEN_DETAILS: bool = os.getenv("AUTOIDEA_SHOW_TOKEN_DETAILS", "0") == "1"

console = Console(
    legacy_windows=(sys.platform == "win32"),
    no_color=os.getenv("NO_COLOR") is not None,
)

_log_file_handle = None

# ---------------------------------------------------------------------------
# Inline utility constants and helpers
# ---------------------------------------------------------------------------

SUCCESS_PREFIX = "[OK]"
FAILURE_PREFIX = "[FAILED]"
THINKING_STREAM = 1000
THINKING_FINAL = 2000
TOOL_RESULT_MAX = 2000


def _is_success(content: str) -> bool:
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


def _shorten_path(path: str, max_len: int = 40) -> str:
    if len(path) <= max_len:
        return path
    path_obj = PurePath(path)
    parts = path_obj.parts
    if len(parts) > 2:
        return ".../" + "/".join(parts[-2:])
    return path


def _format_tool_compact(name: str, args: dict | None) -> str:
    """Format as compact tool call string adapted for AutoIdea tools."""
    if not args:
        return f"{name}()"

    nl = name.lower()

    if nl == "execute":
        cmd = args.get("command", "")
        if len(cmd) > 50:
            cmd = cmd[:47] + "\u2026"
        return f"execute({cmd})"

    if nl == "read_file":
        path = args.get("path", args.get("file_path", ""))
        if path.endswith("/MEMORY.md") or path == "/MEMORY.md":
            return "Reading memory"
        return f"read_file({_shorten_path(path)})"

    if nl == "read_workspace_file":
        path = args.get("path", args.get("file_path", ""))
        return f"read_workspace_file({_shorten_path(path)})"

    if nl == "write_workspace_file":
        path = args.get("path", args.get("file_path", ""))
        return f"write_workspace_file({_shorten_path(path)})"

    if nl == "write_file":
        path = args.get("path", "")
        if path.endswith("/MEMORY.md") or path == "/MEMORY.md":
            return "Updating memory"
        return f"write_file({_shorten_path(path)})"

    if nl == "edit_file":
        path = args.get("path", "")
        if path.endswith("/MEMORY.md") or path == "/MEMORY.md":
            return "Updating memory"
        return f"edit_file({_shorten_path(path)})"

    if nl == "glob":
        pattern = args.get("pattern", "")
        if len(pattern) > 40:
            pattern = pattern[:37] + "\u2026"
        return f"glob({pattern})"

    if nl == "grep":
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        if len(pattern) > 30:
            pattern = pattern[:27] + "\u2026"
        return f"grep({pattern}, {path})"

    if nl == "ls":
        path = args.get("path", ".")
        return f"ls({path})"

    if nl == "write_todos":
        todos = args.get("todos", [])
        if isinstance(todos, list):
            return f"write_todos({len(todos)} items)"
        return "write_todos(...)"

    if nl == "read_todos":
        return "read_todos()"

    if nl == "task":
        sa_type = args.get("subagent_type", "").strip()
        task_desc = args.get("description", args.get("task", "")).strip()
        task_desc = task_desc.split("\n")[0].strip() if task_desc else ""
        if sa_type:
            if task_desc:
                if len(task_desc) > 50:
                    task_desc = task_desc[:47] + "\u2026"
                return f"Cooking with {sa_type} \u2014 {task_desc}"
            return f"Cooking with {sa_type}"
        if task_desc:
            if len(task_desc) > 50:
                task_desc = task_desc[:47] + "\u2026"
            return f"Cooking with sub-agent \u2014 {task_desc}"
        return "Cooking with sub-agent"

    if nl in ("tavily_search", "web_search", "internet_search"):
        query = args.get("query", "")
        if not query:
            # Claude native web_search uses different arg structure
            query = args.get("search_query", args.get("q", ""))
        if len(query) > 40:
            query = query[:37] + "\u2026"
        if nl == "web_search" and not query:
            return "🔍 web_search (Claude native)"
        return f"🔍 {name}({query})" if nl == "web_search" else f"{name}({query})"

    if nl in (
        "semantic_scholar_search", "arxiv_search", "openalex_search",
        "dblp_search", "crossref_search", "pubmed_search", "cvf_search",
    ):
        query = args.get("query", "")
        if len(query) > 40:
            query = query[:37] + "\u2026"
        return f"{name}({query})"

    if nl in ("semantic_scholar_get_paper", "arxiv_get_paper"):
        pid = args.get("paper_id", args.get("arxiv_id", ""))
        return f"{name}({pid})"

    if nl == "crossref_resolve_doi":
        doi = args.get("doi", "")
        return f"crossref_resolve_doi({doi})"

    if nl == "paper_lookup":
        query = args.get("query", args.get("title", ""))
        if len(query) > 40:
            query = query[:37] + "\u2026"
        return f"paper_lookup({query})"

    if nl == "list_found_papers":
        return "list_found_papers()"

    if nl == "merge_and_rank_search_results":
        return "merge_and_rank_search_results()"

    if nl == "fetch_paper_content":
        pid = args.get("paper_id", args.get("url", ""))
        if len(pid) > 40:
            pid = pid[:37] + "\u2026"
        return f"fetch_paper_content({pid})"

    if nl == "fetch_paper_section":
        section = args.get("section", "")
        return f"fetch_paper_section({section})"

    if nl == "cite_source":
        pid = args.get("paper_id", "")
        return f"cite_source({pid})"

    if nl == "rank_ideas_tournament":
        ideas = args.get("ideas", args.get("ideas_json", []))
        if isinstance(ideas, str):
            try:
                ideas = json.loads(ideas)
            except json.JSONDecodeError:
                ideas = []
        n = len(ideas) if isinstance(ideas, list) else 0
        return f"rank_ideas_tournament({n} ideas)"

    if nl == "check_stage_gate":
        stage = args.get("stage", "")
        return f"check_stage_gate({stage})"

    if nl == "save_stage_reflection":
        stage = args.get("stage", "")
        return f"save_stage_reflection({stage})"

    if nl in ("recall_ideation_memory", "update_ideation_memory"):
        return f"{name}()"

    if nl in ("think_tool", "think"):
        reflection = args.get("reflection", args.get("thought", ""))
        if len(reflection) > 40:
            reflection = reflection[:37] + "\u2026"
        return f"think({reflection})"

    params = []
    for k, v in list(args.items())[:2]:
        v_str = str(v)
        if len(v_str) > 20:
            v_str = v_str[:17] + "\u2026"
        params.append(f"{k}={v_str}")
    params_str = ", ".join(params)
    if len(params_str) > 50:
        params_str = params_str[:47] + "\u2026"
    return f"{name}({params_str})"


# ---------------------------------------------------------------------------
# Todo formatting
# ---------------------------------------------------------------------------


def _format_single_todo(item: dict) -> Text:
    status = str(item.get("status", "todo")).lower()
    content_text = str(item.get("content", item.get("task", item.get("title", ""))))

    if status in ("done", "completed", "complete"):
        symbol, label, style = "\u2713", "done  ", "green dim"
    elif status in ("active", "in_progress", "in-progress", "working"):
        symbol, label, style = "\u25cf", "active", "yellow"
    else:
        symbol, label, style = "\u25cb", "todo  ", "dim"

    line = Text()
    line.append(f"    {symbol} ", style=style)
    line.append(label, style=style)
    line.append(" ", style="dim")
    if len(content_text) > 60:
        content_text = content_text[:57] + "\u2026"
    line.append(content_text, style=style)
    return line


# ---------------------------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------------------------


def format_tool_result_compact(
    _name: str,
    content: str,
    max_lines: int = 5,
    tool_args: dict | None = None,
) -> list:
    """Format tool result as tree output."""
    elements: list = []

    if not content.strip():
        elements.append(Text("  \u2514 (empty)", style="dim"))
        return elements

    # Claude native web_search results — show search result summaries
    if _name == "web_search" and (
        "web_search_tool_result" in content
        or '"type": "web_search_result"' in content
        or '"url":' in content
    ):
        try:
            import json
            data = json.loads(content) if content.strip().startswith(("{", "[")) else None
            if data and isinstance(data, list):
                result_line = Text("  \u2514 ", style="dim")
                result_line.append(
                    f"🔍 {len(data)} search result(s)", style="cyan"
                )
                elements.append(result_line)
                for item in data[:3]:
                    title = item.get("title", item.get("url", ""))
                    if len(title) > 60:
                        title = title[:57] + "\u2026"
                    elements.append(
                        Text(f"    \u2022 {title}", style="dim cyan")
                    )
                if len(data) > 3:
                    elements.append(
                        Text(
                            f"    ... +{len(data) - 3} more",
                            style="dim italic",
                        )
                    )
                return elements
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Fall through to default rendering

    if _name == "write_todos":
        items = _parse_todo_items(content)
        if items:
            stats = _build_todo_stats(items)
            stats_line = Text()
            stats_line.append("  \u2514 ", style="dim")
            stats_line.append(stats, style="dim")
            elements.append(stats_line)
            elements.append(Text("", style="dim"))
            for item in items[:4]:
                elements.append(_format_single_todo(item))
            remaining = len(items) - 4
            if remaining > 0:
                elements.append(
                    Text(f"    ... {remaining} more", style="dim italic")
                )
            return elements

    lines = content.strip().split("\n")
    total_lines = len(lines)
    display_lines = lines[:max_lines]
    for i, line in enumerate(display_lines):
        prefix = "\u2514" if i == 0 else " "
        if len(line) > 80:
            line = line[:77] + "\u2026"
        style = "dim" if _is_success(content) else "red dim"
        elements.append(Text(f"  {prefix} {line}", style=style))

    remaining = total_lines - max_lines
    if remaining > 0:
        elements.append(Text(f"    ... +{remaining} lines", style="dim italic"))
    return elements


# ---------------------------------------------------------------------------
# Tool call line rendering
# ---------------------------------------------------------------------------


def _render_tool_call_line(tc: dict, tr: dict | None) -> Text:
    is_task = tc.get("name", "").lower() == "task"
    if tr is not None:
        content = tr.get("content", "")
        if _is_success(content):
            style, indicator = "bold green", ("\u2713" if is_task else "\u25cf")
        else:
            style, indicator = "bold red", ("\u2717" if is_task else "\u25cf")
    else:
        if is_task:
            style, indicator = "bold cyan", "\u25b6"
        else:
            style, indicator = "bold yellow", "\u25cf"

    tool_compact = _format_tool_compact(tc["name"], tc.get("args"))
    tool_text = Text()
    tool_text.append(f"{indicator} ", style=style)
    tool_text.append(tool_compact, style=style)
    return tool_text


# ---------------------------------------------------------------------------
# Sub-agent section rendering
# ---------------------------------------------------------------------------


def _render_subagent_section(sa: SubAgentState, compact: bool = False) -> list:
    elements: list = []
    BORDER = "dim cyan" if sa.is_active else "dim"
    valid_calls = [tc for tc in sa.tool_calls if tc.get("name")]

    completed, pending = [], []
    for tc in valid_calls:
        tr = sa.get_result_for(tc)
        if tr is not None:
            completed.append((tc, tr))
        else:
            pending.append(tc)

    display_name = f"Cooking with {sa.name}"
    if sa.description:
        desc = sa.description.split("\n")[0].strip()
        desc = desc[:50] + "\u2026" if len(desc) > 50 else desc
        display_name += f" \u2014 {desc}"

    if compact:
        line = Text()
        if not sa.is_active:
            line.append("\u2713 ", style="green")
            line.append(display_name, style="green dim")
            line.append(f" ({len(valid_calls)} tools)", style="dim")
        else:
            line.append("\u25b6 ", style="cyan")
            line.append(display_name, style="bold cyan")
        elements.append(line)
        return elements

    MAX_SA_VISIBLE, MAX_SA_RUNNING = 3, 2

    header = Text()
    header.append("\u250c ", style=BORDER)
    if sa.is_active:
        header.append(f"\u25b6 {display_name}", style="bold cyan")
    else:
        header.append(f"\u2713 {display_name}", style="bold green")
    elements.append(header)

    slots = max(0, MAX_SA_VISIBLE - len(pending))
    hidden = (
        completed[:-slots]
        if slots and len(completed) > slots
        else (completed if not slots else [])
    )
    visible = completed[-slots:] if slots else []

    if hidden:
        ok = sum(1 for _, tr in hidden if tr.get("success", True))
        fail = len(hidden) - ok
        summary = Text("\u2502 ", style=BORDER)
        summary.append(f"\u2713 {ok} completed", style="dim green")
        if fail > 0:
            summary.append(f" | {fail} failed", style="dim red")
        elements.append(summary)

    for tc, tr in visible:
        tc_line = Text("\u2502 ", style=BORDER)
        tc_name = _format_tool_compact(tc["name"], tc.get("args"))
        if tr.get("success", True):
            tc_line.append(f"\u2713 {tc_name}", style="green")
        else:
            tc_line.append(f"\u2717 {tc_name}", style="red")
            first_line = tr.get("content", "").strip().split("\n")[0][:70]
            if first_line:
                err_line = Text("\u2502   ", style=BORDER)
                err_line.append(f"\u2514 {first_line}", style="red dim")
                elements.append(tc_line)
                elements.append(err_line)
                continue
        elements.append(tc_line)

    hidden_running = len(pending) - MAX_SA_RUNNING
    if hidden_running > 0:
        run_summary = Text("\u2502 ", style=BORDER)
        run_summary.append(
            f"\u25cf {hidden_running} more running...", style="dim yellow"
        )
        elements.append(run_summary)
        pending = pending[-MAX_SA_RUNNING:]

    for tc in pending:
        tc_line = Text("\u2502 ", style=BORDER)
        tc_name = _format_tool_compact(tc["name"], tc.get("args"))
        tc_line.append(f"\u25cf {tc_name}", style="bold yellow")
        elements.append(tc_line)
        spinner_line = Text("\u2502   ", style=BORDER)
        spinner_line.append("\u21bb running...", style="yellow dim")
        elements.append(spinner_line)

    if not sa.is_active:
        footer = Text(f"\u2514 done ({len(valid_calls)} tools)", style="dim green")
        elements.append(footer)
    elif valid_calls:
        footer = Text("\u2514 running...", style="dim cyan")
        elements.append(footer)

    return elements


# ---------------------------------------------------------------------------
# Todo panel
# ---------------------------------------------------------------------------


def _render_todo_panel(todo_items: list[dict]) -> Panel:
    lines = Text()
    for i, item in enumerate(todo_items):
        if i > 0:
            lines.append("\n")
        status = str(item.get("status", "todo")).lower()
        content_text = str(
            item.get("content", item.get("task", item.get("title", "")))
        )
        if status in ("done", "completed", "complete"):
            symbol, style = "\u2713", "green dim"
        elif status in ("active", "in_progress", "in-progress", "working"):
            symbol, style = "\u23f3", "yellow"
        else:
            symbol, style = "\u25a1", "dim"
        lines.append(f"{symbol} ", style=style)
        lines.append(content_text, style=style)

    return Panel(
        lines,
        title="Task List",
        title_align="center",
        border_style="cyan",
        padding=(0, 1),
    )


_PROGRESS_PHASE_LABELS = {
    "defining_requirements": "Defining research requirements",
    "formalizing_problem": "Formalizing the research problem",
    "surveying_literature": "Surveying literature",
    "reading_papers": "Reading selected papers",
    "positioning_papers": "Positioning papers",
    "expanding_literature": "Expanding the literature set",
    "binding_evidence": "Binding claims to sources",
    "synthesizing_gaps": "Synthesizing research gaps",
    "mapping_design_space": "Mapping the design space",
    "generating_ideas": "Generating candidate ideas",
    "ranking_ideas": "Ranking candidate ideas",
    "debating_ideas": "Running adversarial review",
    "assessing_feasibility": "Assessing feasibility",
    "writing_report": "Writing the final report",
    "preparing_batches": "Preparing work batches",
    "processing_batch": "Processing a work batch",
    "checking_batches": "Checking batch results",
    "merging_batches": "Merging batch results",
    "searching_sources": "Searching literature sources",
    "retrieving_full_text": "Retrieving paper full text",
    "validating_stage": "Validating the stage",
    "recording_reflection": "Recording stage reflection",
    "writing_artifact": "Writing a stage artifact",
    "running_subagent": "Running a research sub-agent",
    "integrating_subagent": "Integrating sub-agent results",
    "reasoning": "Analyzing research evidence",
    "runtime_error": "A runtime operation failed",
}

_PROGRESS_UNIT_LABELS = {
    "papers_collected": "papers collected",
    "papers_processed": "papers processed",
    "papers_positioned": "papers positioned",
    "batches": "batches",
    "ideas_generated": "ideas generated",
    "ideas_ranked": "ideas ranked",
    "ideas_reviewed": "ideas reviewed",
    "ideas_assessed": "ideas assessed",
    "stage": "stage complete",
}


def _render_pipeline_progress(progress: dict[str, Any]) -> Panel:
    """Render the same structured stage snapshot exposed by the Web API."""
    number = str(progress.get("number") or "—")
    index = int(progress.get("index") or 0)
    total_stages = int(progress.get("total_stages") or 14)
    title = f"Pipeline · Stage {number} ({index}/{total_stages})"
    body = Text()
    body.append(str(progress.get("name") or progress.get("stage") or "Working"), style="bold")
    phase = str(progress.get("phase") or "")
    phase_label = _PROGRESS_PHASE_LABELS.get(phase, phase.replace("_", " ").capitalize())
    if phase_label:
        body.append(f"\n{phase_label}", style="cyan")
    subject = str(progress.get("subject") or "").strip()
    if subject:
        body.append(f" · {subject}", style="dim")
    current = progress.get("current")
    total = progress.get("total")
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        unit = _PROGRESS_UNIT_LABELS.get(str(progress.get("unit") or ""), "items")
        body.append(f"\n{current}/{total} {unit}", style="green")
    counts = progress.get("counts") if isinstance(progress.get("counts"), dict) else {}
    details: list[str] = []
    if counts.get("batches_total"):
        details.append(f"{counts.get('batches_completed', 0)}/{counts['batches_total']} batches")
    labels = {
        "full_text": "full text",
        "failed": "failed",
        "claims": "claims",
        "gaps": "gaps",
        "evidence_links": "evidence links",
        "axes": "design axes",
        "combinations": "combinations",
        "comparisons": "comparisons",
    }
    for key, label in labels.items():
        if key in counts and (counts.get(key) or key in {"full_text", "failed"}):
            details.append(f"{counts[key]} {label}")
    if counts.get("round_target"):
        details.append(
            f"{counts.get('debate_rounds', 0)}/{counts['round_target']} debate rounds"
        )
    if details:
        body.append(" · " + " · ".join(details), style="dim")
    return Panel(body, title=title, title_align="left", border_style="cyan", padding=(0, 1))


# ---------------------------------------------------------------------------
# Streaming display layout
# ---------------------------------------------------------------------------


def create_streaming_display(
    thinking_text: str = "",
    response_text: str = "",
    latest_text: str = "",
    tool_calls: list | None = None,
    tool_results: list | None = None,
    is_thinking: bool = False,
    is_responding: bool = False,
    is_waiting: bool = False,
    is_processing: bool = False,
    show_thinking: bool = True,
    subagents: list | None = None,
    todo_items: list | None = None,
    is_final: bool = False,
    final_show_thinking: bool = False,
    final_thinking_max_length: int = THINKING_FINAL,
    response_markdown: Any = None,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    summarization_text: str = "",
    pipeline_progress: dict[str, Any] | None = None,
) -> Any:
    """Create Rich display layout for streaming output."""
    elements: list = []
    tool_calls = tool_calls or []
    tool_results = tool_results or []
    subagents = subagents or []

    if pipeline_progress:
        elements.append(_render_pipeline_progress(pipeline_progress))

    if is_waiting and not thinking_text and not response_text and not tool_calls:
        elements.append(Spinner("dots", text=" Thinking...", style="cyan"))
        return Group(*elements)

    _show_thinking = final_show_thinking if is_final else show_thinking
    if _show_thinking and thinking_text:
        thinking_title = "Thinking"
        display_thinking = thinking_text.rstrip()
        if is_final:
            if len(display_thinking) > final_thinking_max_length:
                half = final_thinking_max_length // 2
                display_thinking = (
                    display_thinking[:half]
                    + "\n\n... (truncated) ...\n\n"
                    + display_thinking[-half:]
                )
        else:
            if is_thinking:
                thinking_title += " ..."
            if len(display_thinking) > THINKING_STREAM:
                display_thinking = "..." + display_thinking[-THINKING_STREAM:]
        elements.append(
            Panel(
                Text(display_thinking, style="dim"),
                title=thinking_title,
                border_style="blue",
                padding=(0, 1),
            )
        )

    if summarization_text:
        sd = summarization_text.rstrip()
        n = len(sd)
        char_label = f"{n / 1000:.1f}k chars" if n >= 1000 else f"{n:,} chars"
        if n > 300:
            sd = sd[:300] + " ..."
        elements.append(
            Panel(
                Text(sd, style="dim italic"),
                title=f"Context Summarized ({char_label})",
                border_style="#f59e0b",
                padding=(0, 1),
            )
        )

    MAX_VISIBLE_TOOLS = 4
    MAX_VISIBLE_RUNNING = 3

    if tool_calls:
        completed_regular, task_tools, running_regular = [], [], []
        for i, tc in enumerate(tool_calls):
            has_result = i < len(tool_results)
            tr = tool_results[i] if has_result else None
            if tc.get("name") in _INTERNAL_TOOLS:
                continue
            if tc.get("name") == "task":
                if tc.get("args"):
                    task_tools.append((tc, tr))
            elif has_result:
                completed_regular.append((tc, tr))
            else:
                running_regular.append((tc, None))

        if is_final:
            shown_sa_names: set[str] = set()
            for tc, tr in completed_regular:
                elements.append(_render_tool_call_line(tc, tr))
                content = tr.get("content", "") if tr else ""
                if tr and not _is_success(content):
                    elements.extend(
                        format_tool_result_compact(
                            tr["name"], content,
                            max_lines=10, tool_args=tc.get("args"),
                        )
                    )
            for tc, tr in task_tools:
                elements.append(_render_tool_call_line(tc, tr))
                sa_name = tc.get("args", {}).get("subagent_type", "")
                task_desc = tc.get("args", {}).get("description", "")
                matched_sa = None
                for sa in subagents:
                    if sa.name == sa_name or (
                        task_desc and task_desc in (sa.description or "")
                    ):
                        matched_sa = sa
                        break
                if matched_sa:
                    shown_sa_names.add(matched_sa.name)
                    elements.extend(
                        _render_subagent_section(matched_sa, compact=True)
                    )
            for sa in subagents:
                if sa.name not in shown_sa_names and (sa.tool_calls or sa.is_active):
                    elements.extend(_render_subagent_section(sa, compact=True))
        else:
            slots = max(0, MAX_VISIBLE_TOOLS - len(running_regular))
            hidden = (
                completed_regular[:-slots]
                if slots and len(completed_regular) > slots
                else (completed_regular if not slots else [])
            )
            visible = completed_regular[-slots:] if slots else []

            if hidden:
                ok = sum(
                    1 for _, tr in hidden if _is_success(tr.get("content", ""))
                )
                fail = len(hidden) - ok
                s = Text()
                s.append(f"\u2713 {ok} completed", style="dim green")
                if fail > 0:
                    s.append(f" | {fail} failed", style="dim red")
                elements.append(s)

            for tc, tr in visible:
                elements.append(_render_tool_call_line(tc, tr))
                content = tr.get("content", "") if tr else ""
                if tr and not _is_success(content):
                    elements.extend(
                        format_tool_result_compact(
                            tr["name"], content,
                            max_lines=5, tool_args=tc.get("args"),
                        )
                    )

            hr = len(running_regular) - MAX_VISIBLE_RUNNING
            if hr > 0:
                s = Text()
                s.append(f"\u25cf {hr} more running...", style="dim yellow")
                elements.append(s)
                running_regular = running_regular[-MAX_VISIBLE_RUNNING:]

            for tc, tr in running_regular:
                elements.append(_render_tool_call_line(tc, tr))
                elements.append(Spinner("dots", text=" Running...", style="yellow"))

    _n_visible = 0
    _n_visible_done = 0
    for i, tc in enumerate(tool_calls):
        if tc.get("name") in _INTERNAL_TOOLS:
            continue
        _n_visible += 1
        if i < len(tool_results):
            _n_visible_done += 1
    has_pending = _n_visible > _n_visible_done
    any_active_sa = any(sa.is_active for sa in subagents)
    has_used_tools = _n_visible > 0
    all_done = not has_pending and not any_active_sa and not is_processing

    if is_final:
        todo_items = todo_items or []
        if todo_items:
            elements.append(Text(""))
            elements.append(_render_todo_panel(todo_items))
        if response_text:
            cr = response_text.strip()
            while cr.endswith("\n...") or cr.rstrip() == "...":
                cr = cr.rstrip().removesuffix("...").rstrip()
            if cr:
                elements.append(Text(""))
                elements.append(response_markdown or Markdown(cr))
        if (total_input_tokens or total_output_tokens) and (
            is_final or _SHOW_TOKEN_DETAILS
        ):
            st = Text(justify="right")
            st.append("[", style="dim italic")
            st.append("Usage: ", style="dim italic")
            st.append(f"{total_input_tokens:,}", style="cyan italic")
            st.append(" in \u00b7 ", style="dim italic")
            st.append(f"{total_output_tokens:,}", style="green italic")
            st.append(" out", style="dim italic")
            st.append("]", style="dim italic")
            elements.append(st)
    else:
        if latest_text and has_used_tools and not all_done:
            preview = latest_text.strip()
            if preview:
                last_line = preview.split("\n")[-1].strip()
                if last_line:
                    if len(last_line) > 60:
                        last_line = last_line[:57] + "\u2026"
                    elements.append(Text(f"    {last_line}", style="dim italic"))

        todo_items = todo_items or []
        if todo_items:
            elements.append(Text(""))
            elements.append(_render_todo_panel(todo_items))

        for sa in subagents:
            if sa.tool_calls or sa.is_active:
                elements.extend(
                    _render_subagent_section(sa, compact=not sa.is_active)
                )

        if (
            is_processing
            and not is_thinking
            and not is_responding
            and not response_text
        ):
            if not any(sa.is_active for sa in subagents):
                elements.append(
                    Spinner("dots", text=" Analyzing results...", style="cyan")
                )

        if response_text and all_done:
            elements.append(Text(""))
            elements.append(response_markdown or Markdown(response_text))

    if not elements:
        return Group(Spinner("dots", text=" Processing...", style="cyan"))
    return Group(*elements)


# ---------------------------------------------------------------------------
# Final results display
# ---------------------------------------------------------------------------


def display_final_results(
    state: StreamState,
    thinking_max_length: int = THINKING_FINAL,
    show_thinking: bool = True,
    show_tools: bool = True,
) -> None:
    """Display final results after streaming completes."""
    if show_thinking and state.thinking_text:
        dt = state.thinking_text.rstrip()
        if len(dt) > thinking_max_length:
            half = thinking_max_length // 2
            dt = dt[:half] + "\n\n... (truncated) ...\n\n" + dt[-half:]
        console.print(
            Panel(Text(dt, style="dim"), title="Thinking", border_style="blue")
        )

    if state.summarization_text:
        sd = state.summarization_text.rstrip()
        if len(sd) > 500:
            sd = sd[:500] + " ..."
        console.print(
            Panel(
                Text(sd, style="dim italic"),
                title="Context Summarized",
                border_style="#f59e0b",
            )
        )

    if show_tools and state.tool_calls:
        shown_sa_names: set[str] = set()
        for i, tc in enumerate(state.tool_calls):
            has_result = i < len(state.tool_results)
            tr = state.tool_results[i] if has_result else None
            content = tr.get("content", "") if tr is not None else ""
            tool_name = tc.get("name", "")
            if tool_name in _INTERNAL_TOOLS:
                continue
            if tool_name.lower() == "task":
                console.print(_render_tool_call_line(tc, tr))
                sa_name = tc.get("args", {}).get("subagent_type", "")
                task_desc = tc.get("args", {}).get("description", "")
                matched_sa = None
                for sa in state.subagents:
                    if sa.name == sa_name or (
                        task_desc and task_desc in (sa.description or "")
                    ):
                        matched_sa = sa
                        break
                if matched_sa:
                    shown_sa_names.add(matched_sa.name)
                    for elem in _render_subagent_section(matched_sa, compact=True):
                        console.print(elem)
                continue
            console.print(_render_tool_call_line(tc, tr))
            if has_result and tr is not None:
                for elem in format_tool_result_compact(
                    tr["name"], content,
                    max_lines=10, tool_args=tc.get("args"),
                ):
                    console.print(elem)

        for sa in state.subagents:
            if sa.name not in shown_sa_names and (sa.tool_calls or sa.is_active):
                for elem in _render_subagent_section(sa, compact=True):
                    console.print(elem)
        console.print()

    if state.todo_items:
        console.print(_render_todo_panel(state.todo_items))
        console.print()

    if state.response_text:
        cr = state.response_text.strip()
        while cr.endswith("\n...") or cr.rstrip() == "...":
            cr = cr.rstrip().removesuffix("...").rstrip()
        console.print()
        console.print(Markdown(cr or state.response_text))

    if state.total_input_tokens or state.total_output_tokens:
        # Always show final token summary (controlled by _SHOW_TOKEN_DETAILS
        # only for intermediate streaming updates, not the final display)
        st = Text(justify="right")
        st.append("[", style="dim italic")
        st.append("Usage: ", style="dim italic")
        st.append(f"{state.total_input_tokens:,}", style="cyan italic")
        st.append(" in \u00b7 ", style="dim italic")
        st.append(f"{state.total_output_tokens:,}", style="green italic")
        st.append(" out", style="dim italic")
        st.append("]", style="dim italic")
        console.print(st)


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

_MAX_HITL_ITERATIONS = 50
_MAX_AUTONOMOUS_CONTINUATIONS = 32
_MAX_STALLED_AUTONOMOUS_TURNS = 3
_session_auto_approve = False


def set_session_auto_approve(enabled: bool) -> None:
    """Set the execution policy for the current CLI/Web runner process."""
    global _session_auto_approve
    _session_auto_approve = bool(enabled)


def _matches_shell_allow_list(command: str, allow_list: list[str]) -> bool:
    cmd = command.strip()
    return any(cmd.startswith(prefix) for prefix in allow_list)


def _resolve_hitl_approval(
    interrupt_data: dict,
    prompt_fn: Callable[[list], list[dict] | None] | None = None,
    interactive: bool = True,
) -> list[dict] | None:
    """Resolve HITL approval.

    BUG FIX: In non-interactive mode, auto-approve to prevent hanging.
    """
    global _session_auto_approve

    action_requests = interrupt_data.get("action_requests", [])
    if not action_requests:
        return [{"type": "approve"}]

    # A structured adapter (for example the Web runner) owns the approval
    # contract.  Consult it before terminal/global defaults so an unchecked Web
    # control can never be silently overridden by config.yaml.
    if prompt_fn is not None:
        return prompt_fn(action_requests)

    if _session_auto_approve:
        return [{"type": "approve"} for _ in action_requests]

    from ..config.settings import load_config
    cfg = load_config()
    if cfg.auto_approve:
        return [{"type": "approve"} for _ in action_requests]

    if not interactive:
        _logger.warning(
            "Non-interactive mode: auto-approving HITL interrupt (%d actions)",
            len(action_requests),
        )
        return [{"type": "approve"} for _ in action_requests]

    shell_allow_list = (
        [s.strip() for s in cfg.shell_allow_list.split(",") if s.strip()]
        if cfg.shell_allow_list
        else []
    )
    needs_prompt = False
    for req in action_requests:
        name = (
            req.get("name", "")
            if isinstance(req, dict)
            else getattr(req, "name", "")
        )
        args = (
            req.get("args", {})
            if isinstance(req, dict)
            else getattr(req, "args", {})
        )
        if name != "execute":
            continue
        command = args.get("command", "") if isinstance(args, dict) else ""
        if not _matches_shell_allow_list(command, shell_allow_list):
            needs_prompt = True
            break

    if not needs_prompt:
        return [{"type": "approve"} for _ in action_requests]

    return _prompt_hitl_approval(action_requests)


def _prompt_hitl_approval(action_requests: list) -> list[dict] | None:
    global _session_auto_approve

    console.print()
    panel_text = Text()
    for i, req in enumerate(action_requests):
        name = (
            req.get("name", "")
            if isinstance(req, dict)
            else getattr(req, "name", "")
        )
        args = (
            req.get("args", {})
            if isinstance(req, dict)
            else getattr(req, "args", {})
        )
        desc = _format_tool_compact(name, args if isinstance(args, dict) else {})
        if panel_text.plain:
            panel_text.append("\n")
        panel_text.append(f"  {i + 1}. {desc}", style="yellow")
    panel_text.append("\n\n")
    panel_text.append(
        "  [1] Approve  [2] Reject  [3] Approve all (session)", style="dim"
    )
    console.print(
        Panel(
            panel_text, title="Approval Required",
            border_style="yellow", padding=(0, 1),
        )
    )

    try:
        choice = input("  Choose [1/2/3, Enter=Approve]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]  Rejected.[/dim]")
        return None

    if choice == "1":
        return [{"type": "approve"} for _ in action_requests]
    elif choice == "3":
        _session_auto_approve = True
        return [{"type": "approve"} for _ in action_requests]
    else:
        console.print("[dim]  Rejected.[/dim]")
        return None


# ---------------------------------------------------------------------------
# ask_user prompt
# ---------------------------------------------------------------------------


def _resolve_ask_user_prompt(ask_user_data: dict) -> dict:
    global _session_auto_approve

    questions = ask_user_data.get("questions", [])
    if not questions:
        return {"answers": [], "status": "answered"}

    if _session_auto_approve:
        answers = []
        for question in questions:
            choices = question.get("choices", [])
            values = [str(choice.get("value", "")) for choice in choices]
            approved = next(
                (value for value in values if value.casefold() == "approve"),
                "",
            )
            delegated = "Proceed automatically using your best judgment."
            if question.get("type") == "multiple_choice":
                answers.append(approved or delegated)
            else:
                answers.append("" if question.get("required") is False else delegated)
        return {"answers": answers, "status": "answered"}

    console.print()
    console.print(
        Panel(
            Text("Quick check-in from AutoIdea", style="bold"),
            border_style="cyan", padding=(0, 1),
        )
    )
    console.print()

    answers: list[str] = []
    try:
        for i, q in enumerate(questions):
            q_text = q.get("question", "")
            q_type = q.get("type", "text")
            required = q.get("required", True)
            tag = " (optional)" if not required else ""
            console.print(f"  [bold]{i + 1}. {q_text}[/bold]{tag}")

            if q_type == "multiple_choice":
                choices = q.get("choices", [])
                for j, choice in enumerate(choices):
                    label = choice.get("value", str(choice))
                    letter = chr(ord("A") + j)
                    console.print(Text(f"     {letter}. {label}", style="dim"))
                other_letter = chr(ord("A") + len(choices))
                console.print(
                    Text(
                        f"     {other_letter}. Other (type your answer)",
                        style="dim",
                    )
                )
                letters = "/".join(
                    chr(ord("A") + k) for k in range(len(choices) + 1)
                )
                raw = input(f"  Choice [{letters}]: ").strip()
                if raw.upper() == other_letter:
                    raw = input("  > Your answer: ").strip()
                    answers.append(raw)
                elif len(raw) == 1 and raw.upper().isalpha():
                    idx = ord(raw.upper()) - ord("A")
                    if 0 <= idx < len(choices):
                        answers.append(choices[idx].get("value", raw))
                    else:
                        answers.append(raw)
                else:
                    answers.append(raw)
            else:
                raw = input("  > Answer: ").strip()
                answers.append(raw)
            console.print()
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]  Cancelled.[/dim]")
        return {"status": "cancelled"}

    if any(answer.strip().casefold() == "auto_continue" for answer in answers):
        _session_auto_approve = True

    return {"answers": answers, "status": "answered"}


# ---------------------------------------------------------------------------
# Async-to-sync bridge
# ---------------------------------------------------------------------------


def _create_event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def _get_event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = _create_event_loop()
    return loop


def _run_streaming(
    agent: Any,
    message: Any,
    thread_id: str,
    show_thinking: bool,
    interactive: bool,
    on_thinking: Callable[[str], None] | None = None,
    on_todo: Callable[[list[dict]], None] | None = None,
    on_file_write: Callable[[str], None] | None = None,
    metadata: dict | None = None,
    hitl_prompt_fn: Callable[[list], list[dict] | None] | None = None,
    ask_user_prompt_fn: Callable[[dict], dict] | None = None,
    progress_tracker: Any | None = None,
    *,
    _state: StreamState | None = None,
    _hitl_depth: int = 0,
    _media_sent: set[str] | None = None,
) -> str:
    """Run async streaming and render with Rich Live display."""
    import nest_asyncio  # type: ignore[import-untyped]
    nest_asyncio.apply()

    state = _state if _state is not None else StreamState()
    pipeline_progress = (
        progress_tracker.start()
        if progress_tracker is not None and _state is None
        else progress_tracker.snapshot() if progress_tracker is not None else None
    )

    # Session log: write streaming events to workspace/output/session_log.md
    _session_logger: SessionLogger | None = None
    if _state is None:  # Only create on first call, not HITL recursions
        _log_path = _get_session_log_path(metadata)
        if _log_path:
            try:
                _session_logger = SessionLogger(_log_path)
                if isinstance(message, str):
                    _session_logger.log_user_input(message)
            except Exception:
                _session_logger = None

    _thinking_sent = False
    _todo_sent = False
    if _media_sent is None:
        _media_sent = set()
    _MIN_THINKING_LEN = 200

    async def _consume() -> None:
        nonlocal _thinking_sent, _todo_sent, pipeline_progress
        async for event in stream_agent_events(
            agent, message, thread_id, metadata=metadata
        ):
            event_type = state.handle_event(event)
            if progress_tracker is not None:
                observed = progress_tracker.observe(event)
                if observed is not None:
                    pipeline_progress = observed

            # Write event to session log file
            if _session_logger is not None:
                _session_logger.log_event(event, event_type)

            if (
                on_thinking
                and not _thinking_sent
                and state.thinking_text
                and event_type != "thinking"
                and len(state.thinking_text) >= _MIN_THINKING_LEN
            ):
                on_thinking(state.thinking_text.rstrip())
                _thinking_sent = True

            if (
                on_todo
                and not _todo_sent
                and event_type == "tool_call"
                and event.get("name") == "write_todos"
                and state.todo_items
            ):
                if (
                    on_thinking
                    and not _thinking_sent
                    and state.thinking_text
                    and len(state.thinking_text) >= _MIN_THINKING_LEN
                ):
                    on_thinking(state.thinking_text.rstrip())
                    _thinking_sent = True
                on_todo(state.todo_items)
                _todo_sent = True

            if (
                on_file_write
                and event_type == "tool_result"
                and event.get("name") == "write_file"
                and event.get("success")
            ):
                wf_path = ""
                for tc in reversed(state.tool_calls):
                    if tc.get("name") == "write_file":
                        p = tc.get("args", {}).get("path", "")
                        if p and p not in _media_sent:
                            wf_path = p
                            break
                if wf_path:
                    ext = os.path.splitext(wf_path)[1].lower()
                    if ext in _MEDIA_EXTENSIONS:
                        real_path = str(resolve_virtual_path(wf_path))
                        if os.path.isfile(real_path):
                            _media_sent.add(wf_path)
                            on_file_write(real_path)

            live.update(
                create_streaming_display(
                    **state.get_display_args(),
                    show_thinking=show_thinking,
                    response_markdown=state.get_response_markdown(),
                    pipeline_progress=pipeline_progress,
                )
            )

    with Live(
        console=console,
        auto_refresh=False,
        transient=False,
        vertical_overflow="visible",
    ) as live:
        live.update(
            create_streaming_display(
                is_waiting=True,
                pipeline_progress=pipeline_progress,
            )
        )
        try:
            loop = _get_event_loop()
        except RuntimeError:
            loop = _create_event_loop()

        async def _run_with_live() -> None:
            # BUG FIX: Import Command at function start, not module level.
            from langgraph.types import Command  # type: ignore[import-untyped]  # noqa: F401

            async def _periodic_refresh() -> None:
                nonlocal pipeline_progress
                refresh_count = 0
                try:
                    while True:
                        await asyncio.sleep(0.05)
                        refresh_count += 1
                        if progress_tracker is not None and refresh_count % 20 == 0:
                            pipeline_progress = progress_tracker.snapshot()
                            live.update(
                                create_streaming_display(
                                    **state.get_display_args(),
                                    show_thinking=show_thinking,
                                    response_markdown=state.get_response_markdown(),
                                    pipeline_progress=pipeline_progress,
                                )
                            )
                        live.refresh()
                except asyncio.CancelledError:
                    pass

            refresh_task = asyncio.ensure_future(_periodic_refresh())
            try:
                await _consume()
            finally:
                refresh_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass
                if (
                    state.pending_interrupt is not None
                    or state.pending_ask_user is not None
                ):
                    final_display = create_streaming_display(
                        **state.get_display_args(),
                        show_thinking=show_thinking,
                        response_markdown=state.get_response_markdown(),
                        pipeline_progress=pipeline_progress,
                    )
                elif interactive:
                    final_display = create_streaming_display(
                        **state.get_display_args(),
                        show_thinking=show_thinking,
                        is_final=True,
                        final_show_thinking=False,
                        response_markdown=state.get_response_markdown(),
                        pipeline_progress=pipeline_progress,
                    )
                else:
                    final_display = create_streaming_display(
                        **state.get_display_args(),
                        show_thinking=show_thinking,
                        is_final=True,
                        final_show_thinking=True,
                        final_thinking_max_length=THINKING_FINAL,
                        response_markdown=state.get_response_markdown(),
                        pipeline_progress=pipeline_progress,
                    )
                live.update(final_display)
                live.refresh()

        loop.run_until_complete(_run_with_live())

    if on_thinking and not _thinking_sent and state.thinking_text:
        if len(state.thinking_text) >= _MIN_THINKING_LEN:
            on_thinking(state.thinking_text.rstrip())

    if state.pending_ask_user is not None and _hitl_depth < _MAX_HITL_ITERATIONS:
        if ask_user_prompt_fn is not None:
            result = ask_user_prompt_fn(state.pending_ask_user)
        else:
            result = _resolve_ask_user_prompt(state.pending_ask_user)
        from langgraph.types import Command  # type: ignore[import-untyped]
        state.pending_ask_user = None
        return _run_streaming(
            agent=agent,
            message=Command(resume=result),
            thread_id=thread_id,
            show_thinking=show_thinking,
            interactive=interactive,
            on_thinking=on_thinking,
            on_todo=on_todo,
            on_file_write=on_file_write,
            metadata=metadata,
            hitl_prompt_fn=hitl_prompt_fn,
            ask_user_prompt_fn=ask_user_prompt_fn,
            progress_tracker=progress_tracker,
            _state=state,
            _hitl_depth=_hitl_depth + 1,
            _media_sent=_media_sent,
        )

    if state.pending_interrupt is not None and _hitl_depth < _MAX_HITL_ITERATIONS:
        decisions = _resolve_hitl_approval(
            state.pending_interrupt,
            prompt_fn=hitl_prompt_fn,
            interactive=interactive,
        )
        if decisions is not None:
            from langgraph.types import Command  # type: ignore[import-untyped]
            state.pending_interrupt = None
            return _run_streaming(
                agent=agent,
                message=Command(resume={"decisions": decisions}),
                thread_id=thread_id,
                show_thinking=show_thinking,
                interactive=interactive,
                on_thinking=on_thinking,
                on_todo=on_todo,
                on_file_write=on_file_write,
                metadata=metadata,
                hitl_prompt_fn=hitl_prompt_fn,
                ask_user_prompt_fn=ask_user_prompt_fn,
                progress_tracker=progress_tracker,
                _state=state,
                _hitl_depth=_hitl_depth + 1,
                _media_sent=_media_sent,
            )
    elif state.pending_interrupt is not None:
        _logger.warning(
            "HITL loop reached max iterations (%d), stopping",
            _MAX_HITL_ITERATIONS,
        )

    # Close session log
    if _session_logger is not None:
        _session_logger.close()

    return (state.response_text or "").strip()


def _workspace_progress_fingerprint(
    workspace: Path,
    progress: dict[str, Any],
) -> str:
    """Hash material pipeline progress while ignoring heartbeat/log churn."""
    digest = hashlib.sha256()
    stable_progress = {
        key: value
        for key, value in progress.items()
        if key not in {"fingerprint", "snapshot"}
    }
    digest.update(
        json.dumps(stable_progress, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    ignored = {
        "pipeline_state.json",
        "run_status.json",
        "output/session_log.md",
    }
    if workspace.is_dir():
        for path in sorted(workspace.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md"}:
                continue
            try:
                relative = path.relative_to(workspace).as_posix()
                if relative in ignored:
                    continue
                digest.update(relative.encode("utf-8"))
                digest.update(path.read_bytes())
            except OSError:
                continue
    return digest.hexdigest()


def _inspect_autonomous_progress(
    workspace_dir: str | Path,
    *,
    checkpoint_events: list[str] | tuple[str, ...] = (),
    require_checkpoint_events: bool = False,
    audit_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the next unfinished stage and a stable completion proof."""
    from ..tools.pipeline_state import STAGES, _build_state
    from ..web.pipeline import REQUIRED_CHECKPOINTS, inspect_pipeline

    workspace = Path(workspace_dir).expanduser().resolve()
    parameters = dict(audit_parameters or {})
    if "target_paper_count" not in parameters:
        configured_target = os.getenv("AUTOIDEA_TARGET_PAPER_COUNT", "").strip()
        if configured_target:
            try:
                parameters["target_paper_count"] = int(configured_target)
            except ValueError:
                pass
    core_state = _build_state(
        workspace,
        target_paper_count=parameters.get("target_paper_count"),
    )
    recorded_checkpoints = sorted(set(checkpoint_events))
    snapshot = inspect_pipeline(
        workspace,
        checkpoint_events=recorded_checkpoints,
        include_audit=True,
        audit_parameters=parameters,
    )
    required_checkpoints = set(REQUIRED_CHECKPOINTS) if require_checkpoint_events else set()

    next_stage = ""
    reasons: list[str] = []
    for spec in STAGES:
        stage = core_state["stages"][spec.stage]
        if stage["status"] != "complete":
            next_stage = spec.stage
            missing = stage.get("missing_artifacts") or []
            if missing:
                reasons.append("missing artifact(s): " + ", ".join(missing))
            for issue in stage.get("validation_issues") or []:
                code = str(issue.get("code") or "INVALID_ARTIFACT")
                message = str(issue.get("message") or "artifact validation failed")
                reasons.append(f"{code}: {message}")
            break
        if not stage.get("has_reflection"):
            next_stage = spec.stage
            reasons.append(
                f"missing reflections/{spec.stage}_reflection.json after the stage artifact"
            )
            break
        if spec.stage in required_checkpoints and spec.stage not in recorded_checkpoints:
            next_stage = spec.stage
            reasons.append(
                "missing structured interaction_requested/interaction_resolved "
                "checkpoint approval"
            )
            break

    completion = snapshot["completion"]
    if not next_stage and completion.get("reflections_ready") is not True:
        next_stage = "stage_12"
        if completion.get("missing_gate_proofs"):
            reasons.append(
                "Stage 12 reflection does not contain proof of a successful stage gate"
            )
        else:
            reasons.append("the final stage reflection is missing or invalid")
    if not next_stage and completion.get("audit_passed") is not True:
        next_stage = "stage_12"
        errors = [
            issue
            for issue in completion.get("audit_issues") or []
            if str(issue.get("severity") or "").upper() == "ERROR"
        ]
        if errors:
            reasons.extend(
                f"{issue.get('code', 'AUDIT_ERROR')}: {issue.get('message', '')}"
                for issue in errors[:8]
            )
        else:
            reasons.append("final artifact audit has not passed")

    complete = bool(
        not next_stage
        and completion.get("required_artifacts_ready")
        and completion.get("final_report_present")
        and completion.get("reflections_ready") is True
        and completion.get("audit_passed") is True
    )
    progress = {
        "complete": complete,
        "next_stage": next_stage or "complete",
        "reasons": reasons,
        "completed_stages": snapshot.get("completed_count", 0),
        "total_stages": snapshot.get("total_stages", len(STAGES)),
        "checkpoint_events": recorded_checkpoints,
        "audit_passed": completion.get("audit_passed"),
    }
    progress["fingerprint"] = _workspace_progress_fingerprint(workspace, progress)
    progress["snapshot"] = snapshot
    return progress


def _automatic_continuation_message(progress: dict[str, Any]) -> str:
    stage = str(progress.get("next_stage") or "the next incomplete stage")
    reasons = progress.get("reasons") or []
    reason_text = "\n".join(f"- {reason}" for reason in reasons)
    if not reason_text:
        reason_text = "- the pipeline completion proof is not yet satisfied"
    return (
        "This is an automatic continuation inside the same AutoIdea run and thread. "
        "Do not ask the user for clarification and do not restart completed stages.\n\n"
        f"The earliest unfinished stage is {stage}. Current evidence:\n{reason_text}\n\n"
        "Complete only this stage now. Reuse valid workspace artifacts and repair or "
        "create only what is missing. Then run check_stage_gate, save the stage "
        "reflection, and mark the stage passed only after those operations succeed. "
        "For Stage 7, 9, or 10, call check_stage_gate so the configured checkpoint "
        "policy records its structured decision. Do not merely describe intended "
        "work. Return control after this stage; the runtime will continue the same "
        "run automatically."
    )


def _initial_pipeline_message(message: Any, progress: dict[str, Any]) -> Any:
    """Attach an explicit one-stage boundary to a new autonomous request."""
    if not isinstance(message, str) or progress.get("complete") is True:
        return message
    stage = str(progress.get("next_stage") or "stage_1")
    return (
        f"{message.rstrip()}\n\n"
        "[AutoIdea runtime execution boundary]\n"
        f"Begin with {stage} and complete only that stage in this model turn. "
        "Produce its required artifacts, run check_stage_gate, save its reflection, "
        "and mark it passed only after validation succeeds. Do not plan or call tools "
        "for later stages in this turn. Return control after the current stage; the "
        "runtime will automatically continue the same run and thread."
    )


def _run_streaming_to_pipeline_completion(
    agent: Any,
    message: Any,
    thread_id: str,
    show_thinking: bool,
    interactive: bool,
    progress_probe: Callable[[], dict[str, Any]],
    on_thinking: Callable[[str], None] | None = None,
    on_todo: Callable[[list[dict]], None] | None = None,
    on_file_write: Callable[[str], None] | None = None,
    metadata: dict | None = None,
    hitl_prompt_fn: Callable[[list], list[dict] | None] | None = None,
    ask_user_prompt_fn: Callable[[dict], dict] | None = None,
    on_continuation: Callable[[int, dict[str, Any]], None] | None = None,
    *,
    max_continuations: int = _MAX_AUTONOMOUS_CONTINUATIONS,
    max_stalled_turns: int = _MAX_STALLED_AUTONOMOUS_TURNS,
) -> str:
    """Keep one agent/thread running until deterministic pipeline proof passes."""
    if max_continuations < 0 or max_stalled_turns < 1:
        raise ValueError("Autonomous continuation limits must be positive.")

    progress = progress_probe()
    current_message = _initial_pipeline_message(message, progress)
    continuation_count = 0
    stalled_turns = 0
    response = ""

    while True:
        progress_tracker = None
        workspace = str((metadata or {}).get("workspace_dir") or "").strip()
        stage = str(progress.get("next_stage") or "")
        if workspace and stage and stage != "complete":
            from ..progress import RuntimeProgressTracker

            parameters = (metadata or {}).get("pipeline_parameters")
            progress_tracker = RuntimeProgressTracker(
                workspace,
                stage,
                parameters=parameters if isinstance(parameters, dict) else None,
            )
        response = _run_streaming(
            agent=agent,
            message=current_message,
            thread_id=thread_id,
            show_thinking=show_thinking,
            interactive=interactive,
            on_thinking=on_thinking,
            on_todo=on_todo,
            on_file_write=on_file_write,
            metadata=metadata,
            hitl_prompt_fn=hitl_prompt_fn,
            ask_user_prompt_fn=ask_user_prompt_fn,
            progress_tracker=progress_tracker,
        )
        updated = progress_probe()
        if updated.get("complete") is True:
            return response

        if updated.get("fingerprint") == progress.get("fingerprint"):
            stalled_turns += 1
        else:
            stalled_turns = 0
        if stalled_turns >= max_stalled_turns:
            stage = updated.get("next_stage") or "unknown"
            raise RuntimeError(
                "Autonomous pipeline made no material progress for "
                f"{stalled_turns} consecutive turns at {stage}."
            )
        if continuation_count >= max_continuations:
            stage = updated.get("next_stage") or "unknown"
            raise RuntimeError(
                "Autonomous pipeline did not complete within "
                f"{max_continuations} continuation turns; next stage is {stage}."
            )

        continuation_count += 1
        if on_continuation is not None:
            on_continuation(continuation_count, updated)
        current_message = _automatic_continuation_message(updated)
        progress = updated


# ---------------------------------------------------------------------------
# Thread-safe static streaming
# ---------------------------------------------------------------------------


async def _astream_to_console(
    agent: Any,
    message: str,
    thread_id: str,
    show_thinking: bool = True,
) -> str:
    """Stream agent events to console using static prints (thread-safe)."""
    state = StreamState()

    async for event in stream_agent_events(agent, message, thread_id):
        etype = state.handle_event(event)
        if etype == "subagent_start":
            name = event.get("name", "sub-agent")
            if name and name != "sub-agent":
                desc = event.get("description", "")
                line = Text()
                line.append("\u25b6 ", style="cyan bold")
                line.append(f"Cooking with {name}", style="cyan bold")
                if desc:
                    short = desc[:50] + "\u2026" if len(desc) > 50 else desc
                    line.append(f" \u2014 {short}", style="dim")
                console.print(line)

    if show_thinking and state.thinking_text:
        dt = state.thinking_text.rstrip()
        if len(dt) > 500:
            dt = dt[:250] + "\n\u2026truncated\u2026\n" + dt[-250:]
        console.print(
            Panel(Text(dt, style="dim"), title="Thinking", border_style="blue")
        )

    if state.summarization_text:
        st = state.summarization_text.rstrip()
        if len(st) > 500:
            st = st[:500] + " ..."
        console.print(
            Panel(
                Text(st, style="dim italic"),
                title="Context Summarized",
                border_style="#f59e0b",
            )
        )

    for i, tc in enumerate(state.tool_calls):
        if tc.get("name", "").lower() == "task":
            continue
        tr = state.tool_results[i] if i < len(state.tool_results) else None
        console.print(_render_tool_call_line(tc, tr))
        if tr and not _is_success(tr.get("content", "")):
            for elem in format_tool_result_compact(tr["name"], tr.get("content", "")):
                console.print(elem)

    if state.todo_items:
        console.print(_render_todo_panel(state.todo_items))
        console.print()

    for sa in state.subagents:
        if sa.tool_calls or not sa.is_active:
            for elem in _render_subagent_section(sa, compact=True):
                console.print(elem)

    if state.response_text:
        cr = state.response_text.strip()
        while cr.endswith("\n...") or cr.rstrip() == "...":
            cr = cr.rstrip().removesuffix("...").rstrip()
        console.print()
        console.print(Markdown(cr or state.response_text))
        console.print()

    return (state.response_text or "").strip()



# ---------------------------------------------------------------------------
# Session Logger -- writes streaming events to a human-readable log file
# ---------------------------------------------------------------------------

class SessionLogger:
    """Writes streaming events to a plain-text session log file.

    Creates a human-readable log capturing model output, tool calls,
    sub-agent activity, and thinking -- everything the user sees in
    the terminal during an AutoIdea run.

    Usage::

        logger = SessionLogger("/workspace/output/session_log.md")
        # Inside _consume loop:
        logger.log_event(event, event_type)
        # At end:
        logger.close()
    """

    def __init__(self, filepath: str):
        self._fp = open(filepath, "a", encoding="utf-8")
        from datetime import datetime
        self._fp.write(f"\n{'=' * 72}\n")
        self._fp.write(f"# AutoIdea Session Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._fp.write(f"{'=' * 72}\n\n")
        self._fp.flush()

    def log_event(self, event: dict, event_type: str) -> None:
        """Write a single event to the log file."""
        try:
            self._write_event(event, event_type)
        except Exception:
            pass  # Never let logging break the pipeline

    def _write_event(self, event: dict, event_type: str) -> None:
        w = self._fp.write

        if event_type == "thinking":
            content = event.get("content", "")
            if content:
                w(content)  # Accumulate thinking text inline

        elif event_type == "text":
            content = event.get("content", "")
            if content:
                w(content)  # Accumulate response text inline

        elif event_type == "tool_call":
            name = event.get("name", "unknown")
            args = event.get("args", {})
            # Claude native web_search — log with search icon
            if name == "web_search":
                query = (
                    args.get("query", "")
                    or args.get("search_query", "")
                    or args.get("q", "")
                )
                w(f"\n\n> 🔍 **Web Search**: `{query}`\n")
            else:
                w(f"\n\n> **Tool Call**: `{name}`\n")
            if args:
                import json
                try:
                    args_str = json.dumps(args, ensure_ascii=False, indent=2)
                    if len(args_str) > 2000:
                        args_str = args_str[:2000] + "\n... (display truncated)"
                    w(f"```json\n{args_str}\n```\n")
                except (TypeError, ValueError):
                    w(f"  Args: {str(args)[:500]}\n")

        elif event_type == "tool_result":
            name = event.get("name", "unknown")
            content = event.get("content", "")
            success = event.get("success", True)
            status = "OK" if success else "FAILED"
            # Claude native web_search results — summarize URLs
            if name == "web_search":
                w(f"\n> 🔍 **Web Search Result**: [{status}]\n")
                try:
                    import json
                    data = (
                        json.loads(content)
                        if content.strip().startswith(("{", "["))
                        else None
                    )
                    if data and isinstance(data, list):
                        for item in data[:5]:
                            title = item.get("title", "")
                            url = item.get("url", "")
                            w(f"  - [{title}]({url})\n")
                        if len(data) > 5:
                            w(f"  - ... +{len(data) - 5} more results\n")
                    elif content:
                        short = content[:3000]
                        if len(content) > 3000:
                            short += "\n... (display truncated)"
                        w(f"```\n{short}\n```\n")
                except (json.JSONDecodeError, TypeError, KeyError):
                    short = content[:3000]
                    if len(content) > 3000:
                        short += "\n... (display truncated)"
                    w(f"```\n{short}\n```\n")
            else:
                w(f"\n> **Tool Result** (`{name}`): [{status}]\n")
                if content:
                    short = content[:3000]
                    if len(content) > 3000:
                        short += "\n... (display truncated)"
                    w(f"```\n{short}\n```\n")

        elif event_type == "subagent_start":
            name = event.get("name", "sub-agent")
            desc = event.get("description", "")
            w(f"\n\n---\n### Sub-Agent: {name}\n")
            if desc:
                w(f"_{desc}_\n")
            w("\n")

        elif event_type == "subagent_tool_call":
            sa = event.get("subagent", "sub-agent")
            name = event.get("name", "unknown")
            w(f"  > [{sa}] Tool Call: `{name}`\n")

        elif event_type == "subagent_tool_result":
            sa = event.get("subagent", "sub-agent")
            name = event.get("name", "unknown")
            content = event.get("content", "")
            w(f"  > [{sa}] Tool Result: `{name}`")
            if content:
                short = content[:500]
                if len(content) > 500:
                    short += "..."
                w(f" — {short}")
            w("\n")

        elif event_type == "subagent_end":
            name = event.get("name", "sub-agent")
            w(f"\n  > [{name}] Done\n---\n\n")

        elif event_type == "usage_stats":
            # Only log token stats to session file when detail mode is on,
            # or when the counts are significant (reduces log noise).
            inp = event.get("input_tokens", 0)
            out = event.get("output_tokens", 0)
            if (inp or out) and (_SHOW_TOKEN_DETAILS or inp + out > 10000):
                w(f"\n> _Tokens: {inp:,} in / {out:,} out_\n")

        elif event_type == "error":
            msg = event.get("message", "Unknown error")
            w(f"\n> **ERROR**: {msg}\n")

        elif event_type == "done":
            w("\n\n--- Session complete ---\n")

        self._fp.flush()

    def log_user_input(self, message: str) -> None:
        """Log the user's input message."""
        self._fp.write(f"\n## User\n\n{message}\n\n## Assistant\n\n")
        self._fp.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass


def _get_session_log_path(metadata: dict | None) -> str | None:
    """Derive the session log file path from metadata or active workspace."""
    workspace = None
    if metadata:
        workspace = metadata.get("workspace_dir")
    if not workspace:
        try:
            from ..paths import get_active_workspace
            workspace = get_active_workspace()
        except Exception:
            pass
    if not workspace:
        return None

    import os
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "session_log.md")


# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------


def enable_file_logging(filepath: str) -> None:
    """Enable file logging with proper handle management.

    BUG FIX: Close previous file handle before opening new one to prevent
    file handle leak.
    """
    global _log_file_handle

    if _log_file_handle is not None:
        try:
            _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None

    try:
        _log_file_handle = open(filepath, "a", encoding="utf-8")
        handler = logging.StreamHandler(_log_file_handle)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger = logging.getLogger("autoidea")
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        _logger.info("File logging enabled: %s", filepath)
    except Exception as e:
        _logger.warning("Failed to enable file logging to %s: %s", filepath, e)
        if _log_file_handle is not None:
            try:
                _log_file_handle.close()
            except Exception:
                pass
            _log_file_handle = None
