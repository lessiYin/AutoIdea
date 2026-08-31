"""Typer command registrations -- config, main callback, and helpers.

All ``@app.command`` and ``@config_app.command`` decorators live here so
that importing the ``commands`` module from ``__init__.py`` is sufficient
to wire up the entire CLI surface.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from importlib.metadata import version as _pkg_version
from pathlib import Path
from sys import version_info
from typing import Any, Optional

import typer  # type: ignore[import-untyped]
from rich.markup import escape
from rich.table import Table

from ..paths import ensure_dirs, set_workspace_root
from ..utils import console
from ._app import app, config_app
from .agent import (
    _load_agent,
)
from .interactive import cmd_interactive, cmd_run


# =============================================================================
# Web dashboard command
# =============================================================================


@app.command("web")
def web_dashboard(
    workspace: str = typer.Option(
        "workspace",
        "--workspace",
        "-w",
        help="AutoIdea workspace directory to visualize.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host interface for the dashboard server.",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="Port for the dashboard server.",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Do not open the dashboard URL in a browser.",
    ),
):
    """Start the browser dashboard for an AutoIdea workspace."""
    workspace_path = Path(workspace).expanduser().resolve()
    if workspace_path.exists() and not workspace_path.is_dir():
        console.print(
            f"[red]Workspace path is not a directory:[/red] {escape(str(workspace_path))}"
        )
        raise typer.Exit(1)
    try:
        workspace_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        console.print(
            f"[red]Unable to create workspace:[/red] {escape(str(exc))}"
        )
        raise typer.Exit(1) from exc

    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError:
        console.print(
            '[red]Missing web dependencies.[/red] Install with: pip install -e ".[web]"'
        )
        raise typer.Exit(1)

    from ..web.server import create_app
    from ..config import get_effective_config

    # Load a project-local .env before managed subprocesses are spawned.  Do
    # not push config/default values back into this server's environment: that
    # would make the Settings page mistake them for deployment overrides and
    # freeze stale defaults for the lifetime of the Web process.
    get_effective_config()

    url = f"http://{host}:{port}"
    console.print(f"[green]AutoIdea Research Console[/green] {url}")
    console.print(f"[dim]Workspace:[/dim] {workspace_path}")

    if not no_open:
        import webbrowser

        webbrowser.open(url)

    uvicorn.run(create_app(workspace_path), host=host, port=port)


@app.command("doctor")
def doctor() -> None:
    """Check Python, dependencies, and model-provider configuration."""
    from ..config import (
        get_config_path,
        get_effective_config,
        get_state_dir,
        validate_runtime_config,
    )

    config = get_effective_config()
    errors = validate_runtime_config(config)

    table = Table(title="AutoIdea environment", show_header=False)
    table.add_column("Check", style="cyan")
    table.add_column("Value")
    table.add_row("Python", f"{version_info.major}.{version_info.minor}.{version_info.micro}")
    table.add_row("Provider", config.provider)
    table.add_row("Model", config.model)
    table.add_row("Config", str(get_config_path()))
    table.add_row("State", str(get_state_dir()))
    console.print(table)

    if errors:
        for message in errors:
            console.print(f"[red]x[/red] {escape(message)}")
        console.print(
            "[dim]Copy .env.example to .env, fill the selected provider, "
            "then run autoidea doctor again.[/dim]"
        )
        raise typer.Exit(1)

    console.print("[green]OK: AutoIdea is ready to start CLI and Web runs.[/green]")


# =============================================================================
# Compact helper
# =============================================================================


class CompactResult:
    """Structured result from ``compact_conversation``.

    Attributes:
        status: ``"noop"`` (nothing to compact), ``"ok"`` (compacted),
            or ``"error"``.
        message: Short human-readable description.
        messages_compacted: Number of messages summarized (0 for noop/error).
        messages_kept: Number of messages unchanged.
        tokens_before: Total tokens before compaction.
        tokens_after: Total tokens after compaction.
        tokens_summarized: Tokens in the summarized portion (before).
        tokens_summary: Tokens in the summary message (after).
        pct_decrease: Percentage decrease.
    """

    __slots__ = (
        "status",
        "message",
        "messages_compacted",
        "messages_kept",
        "tokens_before",
        "tokens_after",
        "tokens_summarized",
        "tokens_summary",
        "pct_decrease",
    )

    def __init__(
        self,
        status: str,
        message: str,
        *,
        messages_compacted: int = 0,
        messages_kept: int = 0,
        tokens_before: int = 0,
        tokens_after: int = 0,
        tokens_summarized: int = 0,
        tokens_summary: int = 0,
        pct_decrease: int = 0,
    ):
        self.status = status
        self.message = message
        self.messages_compacted = messages_compacted
        self.messages_kept = messages_kept
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.tokens_summarized = tokens_summarized
        self.tokens_summary = tokens_summary
        self.pct_decrease = pct_decrease

    def __str__(self) -> str:
        return self.message


def render_compact_result(result: CompactResult):
    """Render a ``CompactResult`` as styled Rich Text.

    Uses cyan for numbers, green for savings, dim for labels.
    """
    from rich.text import Text

    output = Text()

    if result.status == "noop":
        output.append("  ", style="dim")
        output.append("Nothing to compact", style="dim")
        if result.tokens_before > 0:
            output.append(" -- conversation is ~", style="dim")
            output.append(f"{result.tokens_before:,}", style="cyan")
            output.append(" tokens, within retention budget", style="dim")
        elif result.message:
            output.append(
                f" -- {result.message.split('--')[-1].strip()}"
                if "--" in result.message
                else "",
                style="dim",
            )
        return output

    if result.status == "error":
        output.append("x ", style="red")
        output.append(result.message, style="red")
        return output

    # status == "ok"
    output.append("* ", style="green")
    output.append("Compacted ", style="dim")
    output.append(f"{result.messages_compacted}", style="bold")
    output.append(" messages", style="dim")
    output.append("  [", style="dim")
    output.append(f"{result.tokens_before:,}", style="cyan")
    output.append(" -> ", style="dim")
    output.append(f"{result.tokens_after:,}", style="green")
    output.append(" tokens", style="dim")
    output.append(f"  {result.pct_decrease}% decrease", style="green bold")
    output.append("]", style="dim")

    # Second line: detail breakdown
    output.append("\n  ", style="")
    output.append("Summarized: ", style="dim")
    output.append(f"{result.tokens_summarized:,}", style="cyan")
    output.append(" -> ", style="dim")
    output.append(f"{result.tokens_summary:,}", style="green")
    output.append("  |  ", style="dim")
    output.append("Kept: ", style="dim")
    output.append(f"{result.messages_kept}", style="cyan")
    output.append(" messages unchanged", style="dim")

    return output


async def compact_conversation(agent: Any, thread_id: str | None) -> CompactResult:
    """Compact the conversation by summarizing old messages.

    Reads the agent's checkpointed state, counts approximate tokens, and
    returns a structured ``CompactResult``.  If the ``deepagents``
    ``SummarizationMiddleware`` is available it will be used for a real
    compaction pass; otherwise the function reports token counts only.
    """
    if not agent or not thread_id:
        return CompactResult(
            "noop", "Nothing to compact -- start a conversation first."
        )

    try:
        from langchain_core.messages.utils import count_tokens_approximately
    except ImportError:
        return CompactResult(
            "error",
            "Cannot compact: langchain_core.messages.utils is unavailable.",
        )

    config = {"configurable": {"thread_id": thread_id}}

    try:
        state_snapshot = await agent.aget_state(config)
    except Exception as exc:
        return CompactResult("error", f"Failed to read state: {exc}")

    messages = state_snapshot.values.get("messages", [])
    if not messages:
        return CompactResult(
            "noop", "Nothing to compact -- no messages in conversation."
        )

    tokens_before = count_tokens_approximately(messages)

    # Try to use SummarizationMiddleware if available
    try:
        from autoidea.config import get_effective_config
        from autoidea.llm import get_chat_model
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
            compute_summarization_defaults,
        )

        runtime_config = get_effective_config()
        model = get_chat_model(
            getattr(runtime_config, "model", None),
            provider=getattr(runtime_config, "provider", None),
        )
        defaults = compute_summarization_defaults(model)
        middleware_kwargs: dict[str, Any] = {
            "model": model,
            "keep": defaults["keep"],
            "trim_tokens_to_summarize": None,
        }
        # deepagents 0.7 made the backend mandatory. Keep compatibility with
        # earlier supported releases whose constructor did not expose it.
        if "backend" in inspect.signature(SummarizationMiddleware).parameters:
            from deepagents.backends import StateBackend

            middleware_kwargs["backend"] = StateBackend()
        middleware = SummarizationMiddleware(**middleware_kwargs)

        # Rebuild effective message list accounting for prior compaction
        event = state_snapshot.values.get("_summarization_event")
        effective = middleware._apply_event_to_messages(messages, event)

        cutoff = middleware._determine_cutoff_index(effective)
        if cutoff == 0:
            conv_tokens = count_tokens_approximately(effective)
            return CompactResult(
                "noop",
                f"Nothing to compact -- conversation (~{conv_tokens:,} tokens) "
                f"is within the retention budget.",
                tokens_before=conv_tokens,
            )

        to_summarize, to_keep = middleware._partition_messages(effective, cutoff)

        tokens_summarized = count_tokens_approximately(to_summarize)
        tokens_kept = count_tokens_approximately(to_keep)
        tokens_total = tokens_summarized + tokens_kept

        # Skip trivial compaction
        if len(to_summarize) < 3 and tokens_summarized < tokens_total * 0.02:
            return CompactResult(
                "noop",
                f"Nothing to compact -- only {len(to_summarize)} message(s) "
                f"({tokens_summarized:,} tokens) would be summarized.",
                tokens_before=tokens_total,
            )

        # Generate summary (LLM call)
        summary = await middleware._acreate_summary(to_summarize)
        summary_msg = middleware._build_new_messages_with_path(summary, None)[0]

        tokens_summary = count_tokens_approximately([summary_msg])
        tokens_after = tokens_summary + tokens_kept
        pct = (
            round((tokens_total - tokens_after) / tokens_total * 100)
            if tokens_total > 0
            else 0
        )

        # Append savings note
        savings_note = (
            f"\n\n{len(to_summarize)} messages were compacted "
            f"({tokens_summarized:,} -> {tokens_summary:,} tokens). "
            f"Total context: {tokens_total:,} -> {tokens_after:,} tokens "
            f"({pct}% decrease), "
            f"{len(to_keep)} messages unchanged."
        )
        summary_msg.content += savings_note

        from deepagents.middleware.summarization import SummarizationEvent

        state_cutoff = middleware._compute_state_cutoff(event, cutoff)
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff,
            "summary_message": summary_msg,
            "file_path": None,
        }

        await agent.aupdate_state(config, {"_summarization_event": new_event})

        return CompactResult(
            "ok",
            f"Compacted {len(to_summarize)} messages "
            f"({tokens_total:,} -> {tokens_after:,} tokens, {pct}% decrease)",
            messages_compacted=len(to_summarize),
            messages_kept=len(to_keep),
            tokens_before=tokens_total,
            tokens_after=tokens_after,
            tokens_summarized=tokens_summarized,
            tokens_summary=tokens_summary,
            pct_decrease=pct,
        )

    except ImportError:
        # deepagents SummarizationMiddleware not available -- report counts only
        return CompactResult(
            "noop",
            f"Conversation has ~{tokens_before:,} tokens. "
            f"Install deepagents for full compaction support.",
            tokens_before=tokens_before,
        )
    except Exception as exc:
        return CompactResult("error", f"Compaction failed: {exc}")


# =============================================================================
# Config commands
# =============================================================================


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Configuration management commands."""
    if ctx.invoked_subcommand is None:
        config_list()


@config_app.command("list")
def config_list():
    """List saved defaults alongside their effective runtime values."""
    from ..config import (
        get_active_env_override,
        get_config_path,
        get_effective_config,
        load_config,
    )

    saved_config = load_config()
    effective_config = get_effective_config()

    table = Table(title="AutoIdea Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Saved default")
    table.add_column("Effective value")
    table.add_column("Source")

    def format_value(key: str, value: Any) -> str:
        if "api_key" in key and value:
            return "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
        if value == "":
            return "[dim](not set)[/dim]"
        return str(value)

    for key, value in vars(saved_config).items():
        env_override = get_active_env_override(key)
        table.add_row(
            key,
            format_value(key, value),
            format_value(key, getattr(effective_config, key)),
            env_override or "config/default",
        )

    console.print(table)
    console.print(f"\n[dim]Config file: {get_config_path()}[/dim]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Configuration key to get")):
    """Show the saved default and effective value for one setting."""
    from ..config import (
        get_active_env_override,
        get_config_value,
        get_effective_config,
    )

    try:
        value = get_config_value(key)
    except KeyError:
        console.print(f"[red]Unknown key: {escape(key)}[/red]")
        raise typer.Exit(1)

    effective_value = getattr(get_effective_config(), key)
    env_override = get_active_env_override(key)

    def format_value(candidate: Any) -> str:
        if "api_key" in key and candidate:
            return "***" + str(candidate)[-4:] if len(str(candidate)) > 4 else "***"
        return "(not set)" if candidate == "" else str(candidate)

    console.print(f"[cyan]{key}[/cyan]")
    console.print(f"  Saved default: {format_value(value)}")
    console.print(f"  Effective value: {format_value(effective_value)}")
    console.print(f"  Source: {env_override or 'config/default'}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="New value"),
):
    """Set a single configuration value."""
    from ..config import get_active_env_override, set_config_value

    try:
        set_config_value(key, value)
        env_override = get_active_env_override(key)
        if env_override:
            console.print(f"[green]Saved {escape(key)} as the user default.[/green]")
            console.print(
                f"[yellow]{escape(env_override)} currently overrides this saved value. "
                "Use a per-run CLI option or remove that environment override "
                "before starting a run.[/yellow]"
            )
        else:
            console.print(
                f"[green]Saved {escape(key)} as the effective default for future runs.[/green]"
            )
    except KeyError:
        console.print(f"[red]Invalid key: {escape(key)}[/red]")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset configuration to defaults."""
    from ..config import reset_config, get_config_path

    config_path = get_config_path()

    if not config_path.exists():
        console.print("[yellow]No config file to reset.[/yellow]")
        return

    if not yes:
        confirm = typer.confirm("Reset configuration to defaults?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    reset_config()
    console.print("[green]Configuration reset to defaults.[/green]")


@config_app.command("path")
def config_path():
    """Show the configuration file path."""
    from ..config import get_config_path

    path = get_config_path()
    exists = path.exists()
    status = "[green]exists[/green]" if exists else "[dim]not created yet[/dim]"
    console.print(f"{path} ({status})")


# =============================================================================
# Main callback (default behaviour)
# =============================================================================


def _version_callback(value: bool):
    if value:
        try:
            ver = _pkg_version("autoidea")
        except Exception:
            ver = "unknown"
        typer.echo(f"AutoIdea {ver}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "-V",
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="LLM model name (e.g. gpt-5.6-sol, claude-sonnet-4-5).",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider (anthropic, openai, google-genai, ollama, "
        "custom-openai, custom-anthropic).",
    ),
    prompt: Optional[str] = typer.Option(
        None, "-p", "--prompt", help="Query to execute (single-shot mode)."
    ),
    thread_id: Optional[str] = typer.Option(
        None, "-t", "--thread-id", help="Resume a specific conversation thread."
    ),
    workdir: Optional[str] = typer.Option(
        None, "-w", "--workdir", help="Override workspace directory for this session."
    ),
    seed_papers: Optional[str] = typer.Option(
        None,
        "--seed-papers",
        "-s",
        help="Path to a JSON file containing must-read seed papers. "
        "These papers will be mandatory references throughout the pipeline.",
    ),
    seed_ideas: Optional[str] = typer.Option(
        None,
        "--seed-ideas",
        "-i",
        help="Path to a file (.md/.txt/.json) containing the user's research "
        "ideas, brainstorming notes, or preliminary drafts. The system will "
        "analyze these to guide literature search and idea generation.",
    ),
    no_thinking: bool = typer.Option(
        False, "--no-thinking", help="Disable thinking display."
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Run fully automatically (the default; retained for compatibility).",
    ),
    manual_checkpoints: bool = typer.Option(
        False,
        "--manual-checkpoints",
        help="Pause for human review at the Stage 7, 9, and 10 checkpoints.",
    ),
    ui: Optional[str] = typer.Option(
        None,
        "--ui",
        help="UI backend (currently 'cli' only).",
    ),
):
    """AutoIdea -- Autonomous research idea generation with adversarial debate."""
    # If a subcommand was invoked, don't run the default behaviour
    if ctx.invoked_subcommand is not None:
        return

    # Load and apply configuration
    from ..config import (
        apply_config_to_env,
        get_effective_config,
        validate_runtime_config,
    )

    config = get_effective_config()
    # Apply CLI overrides onto config
    if model:
        config.model = model
    if provider:
        config.provider = provider
    if no_thinking:
        config.show_thinking = False
    if auto_approve and manual_checkpoints:
        raise typer.BadParameter(
            "--auto-approve and --manual-checkpoints cannot be used together."
        )
    if manual_checkpoints:
        config.auto_approve = False
    elif auto_approve:
        config.auto_approve = True
    if seed_papers:
        seed_papers_path = os.path.abspath(os.path.expanduser(seed_papers))
        if not os.path.isfile(seed_papers_path):
            raise typer.BadParameter(
                f"Seed papers file not found: {seed_papers_path}"
            )
        config.seed_papers_file = seed_papers_path
        console.print(
            f"[dim]Seed papers:[/dim] [cyan]{seed_papers_path}[/cyan]"
        )
    if seed_ideas:
        seed_ideas_path = os.path.abspath(os.path.expanduser(seed_ideas))
        if not os.path.isfile(seed_ideas_path):
            raise typer.BadParameter(
                f"Seed ideas file not found: {seed_ideas_path}"
            )
        config.seed_ideas_file = seed_ideas_path
        console.print(
            f"[dim]Seed ideas:[/dim] [cyan]{seed_ideas_path}[/cyan]"
        )

    apply_config_to_env(config)

    show_thinking = config.show_thinking

    config_errors = validate_runtime_config(config)
    if config_errors:
        console.print("[red]AutoIdea configuration is incomplete:[/red]")
        for message in config_errors:
            console.print(f"  [red]x[/red] {escape(message)}")
        console.print(
            "[dim]Copy .env.example to .env, fill the selected provider, "
            "and run autoidea doctor.[/dim]"
        )
        raise typer.Exit(2)

    # Validate options
    if ui and ui.lower() not in ("cli",):
        raise typer.BadParameter("--ui must be 'cli'.")

    # Resolve workspace directory
    if workdir:
        workspace_dir = os.path.abspath(os.path.expanduser(workdir))
        os.makedirs(workspace_dir, exist_ok=True)
        set_workspace_root(workspace_dir)
    elif config.workspace_dir:
        workspace_dir = os.path.abspath(os.path.expanduser(config.workspace_dir))
        os.makedirs(workspace_dir, exist_ok=True)
        set_workspace_root(workspace_dir)
    else:
        workspace_dir = os.path.join(os.getcwd(), "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        set_workspace_root(workspace_dir)

    # Ensure runtime subdirectories exist
    ensure_dirs()

    if prompt:
        # Single-shot mode: wrap in persistent checkpointer
        import nest_asyncio  # type: ignore[import-untyped]

        nest_asyncio.apply()

        from ..sessions import get_checkpointer, generate_thread_id

        async def _single_shot():
            async with get_checkpointer() as checkpointer:
                console.print("[dim]Loading agent...[/dim]")
                agent = _load_agent(
                    workspace_dir=workspace_dir,
                    checkpointer=checkpointer,
                    config=config,
                )
                tid = thread_id or generate_thread_id()
                cmd_run(
                    agent,
                    prompt,
                    thread_id=tid,
                    show_thinking=show_thinking,
                    workspace_dir=workspace_dir,
                    model=config.model,
                    auto_approve=config.auto_approve,
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.run_until_complete(_single_shot())
        else:
            asyncio.run(_single_shot())
    else:
        # Interactive mode (default) -- checkpointer managed inside cmd_interactive
        cmd_interactive(
            show_thinking=show_thinking,
            workspace_dir=workspace_dir,
            model=config.model,
            provider=config.provider,
            thread_id=thread_id,
            auto_approve=config.auto_approve,
            config=config,
        )


# =============================================================================
# Logging configuration
# =============================================================================


def _configure_logging():
    """Configure logging with Rich-based handler and dim warning formatting."""
    from rich.logging import RichHandler

    class DimWarningHandler(RichHandler):
        """Custom handler that renders warnings in dim style."""

        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno == logging.WARNING:
                msg = record.getMessage()
                console.print(
                    f"[dim yellow]Warning:[/dim yellow] [dim]{escape(msg)}[/dim]"
                )
            else:
                super().emit(record)

    # Configure root logger
    handler = DimWarningHandler(
        console=console, show_time=False, show_path=False, show_level=False
    )
    handler.setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)

    # Suppress noisy schema warnings from langchain providers
    logging.getLogger("langchain_google_genai._function_utils").setLevel(
        logging.ERROR
    )
