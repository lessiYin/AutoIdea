"""Interactive REPL and single-shot execution for AutoIdea.

Provides the main interactive conversation loop (``cmd_interactive``) and
the single-shot runner (``cmd_run``).  The interactive mode uses
``prompt_toolkit`` for input with slash-command completion and
``asyncio`` for streaming agent responses.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import typer  # type: ignore[import-untyped]
from prompt_toolkit import PromptSession  # type: ignore[import-untyped]
from prompt_toolkit.completion import Completer, Completion  # type: ignore[import-untyped]
from prompt_toolkit.history import FileHistory  # type: ignore[import-untyped]
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # type: ignore[import-untyped]
from prompt_toolkit.formatted_text import HTML  # type: ignore[import-untyped]
from prompt_toolkit.shortcuts import CompleteStyle  # type: ignore[import-untyped]
from prompt_toolkit.styles import Style as PtStyle  # type: ignore[import-untyped]
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..sessions import (
    generate_thread_id,
    get_checkpointer,
    list_threads,
    thread_exists,
    find_similar_threads,
    delete_thread,
)
from ..utils import console
from .agent import _shorten_path, _load_agent


# =============================================================================
# Banner
# =============================================================================

# ASCII art logo with gradient colouring
_LOGO_LINES = [
    r"     _         _        ___     _              ",
    r"    / \  _   _| |_ ___ |_ _| __| | ___  __ _  ",
    r"   / _ \| | | | __/ _ \ | | / _` |/ _ \/ _` | ",
    r"  / ___ \ |_| | || (_) || || (_| |  __/ (_| | ",
    r" /_/   \_\__,_|\__\___/|___|\__,_|\___|\__,_| ",
]

_LOGO_GRADIENT = [
    "#80d0ff",
    "#60b0ff",
    "#4090ff",
    "#2070ff",
    "#0050ff",
]


def print_banner(
    thread_id: str,
    workspace_dir: str | None = None,
    model: str | None = None,
    provider: str | None = None,
):
    """Print welcome banner with ASCII art logo and session metadata."""
    console.print()
    for line, color in zip(_LOGO_LINES, _LOGO_GRADIENT):
        console.print(Text(line, style=f"{color} bold"))

    info = Text()
    info.append("  ", style="dim")

    parts: list[tuple[str, str]] = []
    if model:
        parts.append(("Model: ", model))
    if provider:
        parts.append(("Provider: ", provider))
    for i, (label, value) in enumerate(parts):
        if i > 0:
            info.append("  ", style="dim")
        info.append(label, style="dim")
        info.append(value, style="magenta")

    # Directory line
    effective_dir = workspace_dir or os.getcwd()
    home = os.path.expanduser("~")
    dir_display = (
        effective_dir.replace(home, "~", 1)
        if effective_dir.startswith(home)
        else effective_dir
    )
    info.append("\n  ", style="dim")
    info.append("Directory: ", style="dim")
    info.append(dir_display, style="magenta")

    # Thread hint
    info.append("\n  ", style="dim")
    info.append("Thread: ", style="dim")
    info.append(thread_id, style="yellow")

    info.append("\n  Type ", style="#ffe082")
    info.append("/", style="#ffe082 bold")
    info.append(" for commands", style="#ffe082")
    console.print(info)
    console.print()


# =============================================================================
# Slash-command completer
# =============================================================================

_SLASH_COMMANDS = [
    ("/current", "Show current session info"),
    ("/threads", "List recent sessions"),
    ("/resume", "Resume a previous session (prefix match)"),
    ("/delete", "Delete a saved session"),
    ("/new", "Start a new session"),
    ("/compact", "Compact conversation to free context"),
    ("/exit", "Quit AutoIdea"),
]

_COMPLETION_STYLE = PtStyle.from_dict(
    {
        "completion-menu": "bg:default noreverse nounderline noitalic",
        "completion-menu.completion": "bg:default #888888 noreverse",
        "completion-menu.completion.current": "bg:default default bold noreverse",
        "completion-menu.meta.completion": "bg:default #888888 noreverse",
        "completion-menu.meta.completion.current": "bg:default default bold noreverse",
        "scrollbar.background": "bg:default",
        "scrollbar.button": "bg:default",
    }
)


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands -- triggers when input starts with '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=f"{cmd:<30}",
                    display_meta=desc,
                )


# =============================================================================
# Streaming helper
# =============================================================================


def _astream_to_console(
    agent: Any,
    message: str,
    thread_id: str,
    show_thinking: bool = True,
    workspace_dir: str | None = None,
    model: str | None = None,
    interactive: bool = True,
    auto_complete: bool = False,
) -> str:
    """Stream agent response to console using the Rich Live streaming display.

    Delegates to ``stream.display._run_streaming`` which provides real-time
    token-by-token output, tool call rendering, sub-agent sections, and
    HITL interrupt handling via Rich Live.
    """
    from ..stream.display import (
        _inspect_autonomous_progress,
        _run_streaming,
        _run_streaming_to_pipeline_completion,
    )

    metadata: dict[str, Any] = {}
    if workspace_dir:
        metadata["workspace_dir"] = workspace_dir
    if model:
        metadata["model"] = model

    run = _run_streaming_to_pipeline_completion if auto_complete else _run_streaming
    kwargs: dict[str, Any] = {
        "agent": agent,
        "message": message,
        "thread_id": thread_id,
        "show_thinking": show_thinking,
        "interactive": interactive,
        "metadata": metadata if metadata else None,
    }
    if auto_complete:
        from ..paths import get_active_workspace

        workspace = workspace_dir or str(get_active_workspace())
        kwargs["progress_probe"] = lambda: _inspect_autonomous_progress(workspace)
    return run(**kwargs)


# =============================================================================
# Interactive & single-shot modes
# =============================================================================


def cmd_interactive(
    show_thinking: bool = True,
    workspace_dir: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    thread_id: str | None = None,
    auto_approve: bool = True,
    config: Any = None,
) -> None:
    """Interactive conversation mode with streaming output.

    Opens a persistent ``AsyncSqliteSaver`` checkpointer, loads the agent,
    prints the banner, and enters the main input loop.

    Args:
        show_thinking: Whether to display thinking panels.
        workspace_dir: Per-session workspace directory path.
        model: Model name to display in banner.
        provider: LLM provider name to display in banner.
        thread_id: Optional thread ID to resume a previous session.
        auto_approve: Whether to auto-approve HITL checkpoints.
        config: Pre-loaded AutoIdeaConfig instance.
    """
    import nest_asyncio  # type: ignore[import-untyped]
    from ..stream.display import set_session_auto_approve

    nest_asyncio.apply()
    set_session_auto_approve(auto_approve)

    from .. import paths
    from ..config.settings import get_state_dir

    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    history_file = str(state_dir / "history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(),
        complete_style=CompleteStyle.COLUMN,
        complete_while_typing=True,
        style=_COMPLETION_STYLE,
    )

    def _print_separator():
        """Print a horizontal separator line spanning the terminal width."""
        width = console.size.width
        console.print(Text("-" * width, style="dim"))

    # Mutable state for the async loop
    state: dict[str, Any] = {
        "agent": None,
        "thread_id": thread_id or generate_thread_id(),
        "workspace_dir": workspace_dir,
        "running": True,
        "resumed": bool(thread_id),
    }

    async def _resolve_thread_id(tid: str) -> str | None:
        """Resolve a (possibly partial) thread ID.  Returns full ID or None."""
        if await thread_exists(tid):
            return tid
        similar = await find_similar_threads(tid)
        if len(similar) == 1:
            return similar[0]
        if len(similar) > 1:
            console.print(
                f"[yellow]Ambiguous thread ID '{escape(tid)}'. Matches:[/yellow]"
            )
            for s in similar:
                console.print(f"  [cyan]{s}[/cyan]")
            return None
        console.print(f"[red]Thread '{escape(tid)}' not found.[/red]")
        return None

    async def _cmd_threads():
        """Handle /threads command -- show recent sessions."""
        threads = await list_threads(limit=20)
        if not threads:
            console.print("[yellow]No saved sessions.[/yellow]")
            return
        table = Table(title="Sessions", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="bold")
        for t in threads:
            tid = t["thread_id"]
            marker = " *" if tid == state["thread_id"] else ""
            table.add_row(f"{tid}{marker}")
        console.print()
        console.print(table)
        console.print(
            "[dim]  /resume <id> to continue  /delete <id> to remove  /new to start fresh[/dim]"
        )
        console.print()

    async def _cmd_resume(arg: str, checkpointer):
        """Handle /resume [id] -- resume a previous session."""
        if not arg:
            console.print("[red]Usage: /resume <thread-id>[/red]")
            return
        resolved = await _resolve_thread_id(arg)
        if not resolved:
            return

        state["thread_id"] = resolved
        state["resumed"] = True
        console.print("[dim]Loading session...[/dim]")
        state["agent"] = _load_agent(
            workspace_dir=state["workspace_dir"],
            checkpointer=checkpointer,
            config=config,
        )
        console.print(
            f"[green]Resumed session:[/green] [yellow]{resolved}[/yellow]"
        )
        if state["workspace_dir"]:
            console.print(
                f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]"
            )
        console.print()

    async def _cmd_delete(arg: str):
        """Handle /delete <id> -- delete a saved session."""
        if not arg:
            console.print("[red]Usage: /delete <thread-id>[/red]")
            return
        resolved = await _resolve_thread_id(arg)
        if not resolved:
            return
        if resolved == state["thread_id"]:
            console.print("[red]Cannot delete the current session.[/red]")
            return
        deleted = await delete_thread(resolved)
        if deleted:
            console.print(f"[green]Deleted session {resolved}.[/green]")
        else:
            console.print(f"[red]Session {resolved} not found.[/red]")

    async def _async_main_loop():
        """Async main loop with prompt_async."""
        async with get_checkpointer() as checkpointer:
            # Handle --thread-id resume
            if thread_id:
                resolved = await _resolve_thread_id(thread_id)
                if resolved:
                    state["thread_id"] = resolved
                    state["resumed"] = True

            console.print("[dim]Loading agent...[/dim]")
            state["agent"] = _load_agent(
                workspace_dir=state["workspace_dir"],
                checkpointer=checkpointer,
                config=config,
            )

            # Set active workspace
            if state["workspace_dir"]:
                paths.set_active_workspace(state["workspace_dir"])

            # Print banner
            print_banner(
                state["thread_id"],
                state["workspace_dir"],
                model,
                provider,
            )
            if state["resumed"]:
                console.print(
                    f"[green]Resumed session [yellow]{state['thread_id']}[/yellow][/green]\n"
                )

            try:
                _print_separator()
                while state["running"]:
                    try:
                        user_input = await session.prompt_async(
                            HTML("<ansiblue><b>></b></ansiblue> ")
                        )
                        user_input = user_input.strip()

                        if not user_input:
                            # Erase the empty prompt line
                            sys.stdout.write("\033[A\033[2K\r")
                            sys.stdout.flush()
                            continue

                        _print_separator()

                        # ---- Slash commands ----

                        if user_input.lower() in ("/exit", "/quit", "/q"):
                            console.print("[dim]Goodbye![/dim]")
                            state["running"] = False
                            break

                        if user_input.lower() == "/threads":
                            await _cmd_threads()
                            continue

                        if user_input.lower().startswith("/resume"):
                            arg = user_input[len("/resume"):].strip()
                            await _cmd_resume(arg, checkpointer)
                            continue

                        if user_input.lower().startswith("/delete"):
                            arg = user_input[len("/delete"):].strip()
                            await _cmd_delete(arg)
                            continue

                        if user_input.lower() == "/new":
                            state["thread_id"] = generate_thread_id()
                            state["resumed"] = False
                            console.print("[dim]Loading new session...[/dim]")
                            state["agent"] = _load_agent(
                                workspace_dir=state["workspace_dir"],
                                checkpointer=checkpointer,
                                config=config,
                            )
                            console.print(
                                f"[green]New session:[/green] [yellow]{state['thread_id']}[/yellow]"
                            )
                            if state["workspace_dir"]:
                                console.print(
                                    f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]\n"
                                )
                            continue

                        if user_input.lower() == "/current":
                            console.print(
                                f"[dim]Thread:[/dim] [yellow]{state['thread_id']}[/yellow]"
                            )
                            if state["workspace_dir"]:
                                console.print(
                                    f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]"
                                )
                            if model:
                                console.print(
                                    f"[dim]Model:[/dim] [cyan]{model}[/cyan]"
                                )
                            if provider:
                                console.print(
                                    f"[dim]Provider:[/dim] [cyan]{provider}[/cyan]"
                                )
                            console.print()
                            continue

                        if user_input.lower() == "/compact":
                            from .commands import (
                                compact_conversation,
                                render_compact_result,
                            )

                            with console.status(
                                "[cyan]Compacting conversation...[/cyan]"
                            ):
                                result = await compact_conversation(
                                    agent=state["agent"],
                                    thread_id=state["thread_id"],
                                )
                            console.print(render_compact_result(result))
                            continue

                        # ---- Regular message -> stream agent response ----

                        console.print()
                        _astream_to_console(
                            agent=state["agent"],
                            message=user_input,
                            thread_id=state["thread_id"],
                            show_thinking=show_thinking,
                            workspace_dir=state["workspace_dir"],
                            model=model,
                            interactive=True,
                            auto_complete=True,
                        )
                        console.print()
                        _print_separator()

                    except KeyboardInterrupt:
                        console.print("\n[dim]Goodbye![/dim]")
                        state["running"] = False
                        break
                    except EOFError:
                        console.print("\n[dim]Goodbye![/dim]")
                        state["running"] = False
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if (
                            "authentication" in error_msg.lower()
                            or "api_key" in error_msg.lower()
                        ):
                            console.print(
                                "[red]Error: API key not configured.[/red]"
                            )
                            console.print(
                                "[dim]Run [bold]autoidea config set[/bold] to configure your API key.[/dim]"
                            )
                            state["running"] = False
                            break
                        else:
                            console.print(
                                f"[red]Error: {escape(str(e))}[/red]"
                            )
            except Exception:
                pass

    # Run the async main loop
    try:
        asyncio.run(_async_main_loop())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


def cmd_run(
    agent: Any,
    prompt: str,
    thread_id: str | None = None,
    show_thinking: bool = True,
    workspace_dir: str | None = None,
    model: str | None = None,
    auto_approve: bool = True,
) -> None:
    """Single-shot execution with streaming display.

    Prints the prompt, streams the agent response once, and exits.

    Args:
        agent: Compiled agent graph.
        prompt: User prompt to execute.
        thread_id: Optional thread ID (generates new one if None).
        show_thinking: Whether to display thinking panels.
        workspace_dir: Per-session workspace directory path.
        model: Model name for checkpoint metadata.
        auto_approve: Whether to continue through checkpoints without prompting.
    """
    from ..stream.display import set_session_auto_approve

    set_session_auto_approve(auto_approve)
    thread_id = thread_id or generate_thread_id()

    width = console.size.width
    sep = Text("-" * width, style="dim")
    console.print(sep)
    console.print(Text(f"> {prompt}"))
    console.print(sep)
    console.print(f"[dim]Thread: {thread_id}[/dim]")
    if workspace_dir:
        console.print(f"[dim]Workspace: {_shorten_path(workspace_dir)}[/dim]")
    console.print()

    try:
        _astream_to_console(
            agent=agent,
            message=prompt,
            thread_id=thread_id,
            show_thinking=show_thinking,
            workspace_dir=workspace_dir,
            model=model,
            interactive=False,
            auto_complete=True,
        )
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            console.print("[red]Error: API key not configured.[/red]")
            console.print(
                "[dim]Run [bold]autoidea config set[/bold] to configure your API key.[/dim]"
            )
            raise typer.Exit(1)
        else:
            console.print(f"[red]Error: {escape(error_msg)}[/red]")
            raise typer.Exit(1)
