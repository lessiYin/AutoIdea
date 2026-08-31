"""Core agent construction for AutoIdea v3.0.

Provides ``create_cli_agent`` – the main factory that builds the
LangGraph-based research agent with 25-tool registry, 8 sub-agents,
middleware stack, and composite backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import get_effective_config, apply_config_to_env
from .prompts import get_system_prompt
from . import paths as _paths_mod
from .paths import set_active_workspace, set_workspace_root

logger = logging.getLogger(__name__)


# ── Claude Web Search helpers ────────────────────────────────────────────

def _is_claude_provider(cfg=None) -> bool:
    """Return whether the configured provider uses a Claude model."""
    if cfg is None:
        cfg = _ensure_config()
    provider = getattr(cfg, "provider", "openai")
    return provider in ("anthropic", "custom-anthropic")


def _build_web_search_tool_def(cfg=None) -> dict | None:
    """Build the native Claude Web Search tool definition.

    Return the definition only for a Claude provider when
    ``enable_web_search=True``; return ``None`` when it is disabled.
    """
    if cfg is None:
        cfg = _ensure_config()

    if not getattr(cfg, "enable_web_search", True):
        logger.debug("Claude Web Search disabled by config")
        return None

    if not _is_claude_provider(cfg):
        logger.debug(
            "Claude Web Search skipped: provider=%s is not Claude",
            getattr(cfg, "provider", "?"),
        )
        return None

    tool_def: dict = {
        "type": "web_search_20250305",
        "name": "web_search",
    }

    max_uses = getattr(cfg, "web_search_max_uses", 10)
    if max_uses > 0:
        tool_def["max_uses"] = max_uses

    allowed = getattr(cfg, "web_search_allowed_domains", "")
    if allowed and allowed.strip():
        tool_def["allowed_domains"] = [
            d.strip() for d in allowed.split(",") if d.strip()
        ]

    blocked = getattr(cfg, "web_search_blocked_domains", "")
    if blocked and blocked.strip():
        tool_def["blocked_domains"] = [
            d.strip() for d in blocked.split(",") if d.strip()
        ]

    logger.info("Claude Web Search tool built: %s", tool_def)
    return tool_def


# ── Module-level singletons ──────────────────────────────────────────────

_config = None


def _ensure_config(config=None):
    """Lazy singleton config with optional injection.

    - First call with no argument: loads from file/env/defaults.
    - Call with explicit config: replaces cache and re-applies.
    - Subsequent calls: returns cached config instantly.
    """
    global _config
    if config is not None:
        _config = config
        # Apply overrides from dict-like config
        if hasattr(config, "__dict__"):
            for key, value in vars(config).items():
                if not key.startswith("_") and value is not None:
                    pass  # env-level override handled by apply_config_to_env
        apply_config_to_env(_config)
    if _config is None:
        _config = get_effective_config()
        apply_config_to_env(_config)
    return _config


_chat_model = None


def _ensure_chat_model(model_name=None, provider=None):
    """Get or create the LLM chat model (lazy singleton)."""
    global _chat_model
    if _chat_model is not None and model_name is None:
        return _chat_model
    from .llm import get_chat_model
    cfg = _ensure_config()
    m = model_name or getattr(cfg, "model", None) or "claude-sonnet-4-20250514"
    p = provider or getattr(cfg, "provider", None)
    _chat_model = get_chat_model(m, provider=p)
    return _chat_model


# ── Tool Registry ────────────────────────────────────────────────────────

def _build_tool_registry(use_claude_web_search: bool = False):
    """Build the complete tool registry for AutoIdea.

    Args:
        use_claude_web_search: When True, skip registering the Tavily
            ``web_search`` tool because Claude's native web_search
            (injected via ``bind_tools``) takes priority and would
            conflict on the name ``web_search``.

    Returns:
        Tuple of (tool_registry dict, base_tools list).
        tool_registry maps tool names to tool objects for sub-agent wiring.
        base_tools is the list of tools given directly to the main agent.
    """
    # Import all tools
    from .tools.think import think, read_workspace_file, write_workspace_file
    from .tools.cite import cite_source
    from .tools.search import tavily_search, web_search, paper_lookup
    from .tools.scholar import (
        semantic_scholar_search,
        semantic_scholar_get_paper,
        arxiv_search,
        arxiv_get_paper,
        openalex_search,
        dblp_search,
        crossref_search,
        crossref_resolve_doi,
        pubmed_search,
        cvf_search,
        list_found_papers,
        multi_source_search,
        fetch_paper_fulltext,
    )
    from .tools.reranker import merge_and_rank_search_results
    from .tools.paper_content import fetch_paper_content, fetch_paper_section
    from .tools.idea_tournament import rank_ideas_tournament, generate_tournament_matchups
    from .tools.stage_gate import (
        check_stage_gate,
        save_stage_reflection,
        list_stage_reflections,
    )
    from .tools.artifact_writers import (
        write_design_space,
        write_evidence_db,
        write_research_gaps,
        write_raw_ideas,
        write_tournament_rankings,
        write_idea_reviews,
    )
    from .tools.batch_tasks import (
        create_search_batches,
        create_reading_batches,
        create_evidence_batches,
        record_batch_result,
        read_batch_manifest,
        merge_search_batches,
        merge_reading_batches,
        merge_evidence_batches,
    )
    from .tools.pipeline_state import inspect_pipeline_state
    from .tools.heartbeat import write_run_status, read_run_status
    from .tools.artifact_audit import audit_workspace_artifacts
    from .tools.evo_memory import (
        recall_ideation_memory,
        update_ideation_memory,
        get_memory_stats,
    )
    from .tools.seed_papers import list_seed_papers
    from .tools.seed_ideas import (
        list_seed_ideas,
        get_search_keywords_from_seeds,
        generate_seed_idea_analysis_report,
    )

    # Full tool registry (available to sub-agents via YAML tool references)
    tool_registry = {
        # Search tools
        "tavily_search": tavily_search,
        "paper_lookup": paper_lookup,
        # Scholar tools (6 sources)
        "semantic_scholar_search": semantic_scholar_search,
        "semantic_scholar_get_paper": semantic_scholar_get_paper,
        "arxiv_search": arxiv_search,
        "arxiv_get_paper": arxiv_get_paper,
        "openalex_search": openalex_search,
        "dblp_search": dblp_search,
        "crossref_search": crossref_search,
        "crossref_resolve_doi": crossref_resolve_doi,
        "pubmed_search": pubmed_search,
        "cvf_search": cvf_search,
        "list_found_papers": list_found_papers,
        # Multi-source search (recommended comprehensive search)
        "multi_source_search": multi_source_search,
        # Reranker
        "merge_and_rank_search_results": merge_and_rank_search_results,
        # Paper content
        "fetch_paper_content": fetch_paper_content,
        "fetch_paper_section": fetch_paper_section,
        "fetch_paper_fulltext": fetch_paper_fulltext,
        # Citation
        "cite_source": cite_source,
        # Thinking & workspace
        "think": think,
        "think_tool": think,  # alias used in subagent.yaml
        "read_workspace_file": read_workspace_file,
        "write_workspace_file": write_workspace_file,
        "write_design_space": write_design_space,
        "write_evidence_db": write_evidence_db,
        "write_research_gaps": write_research_gaps,
        "write_raw_ideas": write_raw_ideas,
        "write_tournament_rankings": write_tournament_rankings,
        "write_idea_reviews": write_idea_reviews,
        "create_search_batches": create_search_batches,
        "create_reading_batches": create_reading_batches,
        "create_evidence_batches": create_evidence_batches,
        "record_batch_result": record_batch_result,
        "read_batch_manifest": read_batch_manifest,
        "merge_search_batches": merge_search_batches,
        "merge_reading_batches": merge_reading_batches,
        "merge_evidence_batches": merge_evidence_batches,
        "inspect_pipeline_state": inspect_pipeline_state,
        "write_run_status": write_run_status,
        "read_run_status": read_run_status,
        # v3.0: Elo Tournament
        "rank_ideas_tournament": rank_ideas_tournament,
        "generate_tournament_matchups": generate_tournament_matchups,
        # v3.0: Stage Gate
        "check_stage_gate": check_stage_gate,
        "save_stage_reflection": save_stage_reflection,
        "list_stage_reflections": list_stage_reflections,
        "audit_workspace_artifacts": audit_workspace_artifacts,
        # v3.0: Persistent Memory
        "recall_ideation_memory": recall_ideation_memory,
        "update_ideation_memory": update_ideation_memory,
        "get_memory_stats": get_memory_stats,
        # Seed papers
        "list_seed_papers": list_seed_papers,
        # Seed ideas
        "list_seed_ideas": list_seed_ideas,
        "get_search_keywords_from_seeds": get_search_keywords_from_seeds,
        "generate_seed_idea_analysis_report": generate_seed_idea_analysis_report,
    }

    # Register Tavily web_search only when Claude native web_search is NOT active
    if not use_claude_web_search:
        tool_registry["web_search"] = web_search
    else:
        logger.info(
            "Tavily web_search tool skipped (Claude native web_search active)"
        )

    # Base tools: given directly to the main agent node
    base_tools = [
        think,
        cite_source,
        read_workspace_file,
        write_workspace_file,
        write_design_space,
        write_evidence_db,
        write_research_gaps,
        write_raw_ideas,
        write_tournament_rankings,
        write_idea_reviews,
        create_search_batches,
        create_reading_batches,
        create_evidence_batches,
        record_batch_result,
        read_batch_manifest,
        merge_search_batches,
        merge_reading_batches,
        merge_evidence_batches,
        inspect_pipeline_state,
        write_run_status,
        read_run_status,
        check_stage_gate,
        save_stage_reflection,
        audit_workspace_artifacts,
        list_seed_papers,
        list_seed_ideas,
        get_search_keywords_from_seeds,
        generate_seed_idea_analysis_report,
        multi_source_search,
        fetch_paper_fulltext,
        rank_ideas_tournament,
    ]

    return tool_registry, base_tools


# ── Sub-agent Loading ────────────────────────────────────────────────────

def _inject_subagent_middleware(subs: list[dict]) -> None:
    """Inject runtime hardening into every sub-agent."""
    from .middleware import ToolCallSerializationMiddleware, ToolErrorHandlerMiddleware

    for sa in subs:
        sa.setdefault("middleware", []).extend(
            [ToolCallSerializationMiddleware(), ToolErrorHandlerMiddleware()]
        )
        permissions = sa.setdefault("permissions", [])
        existing = {
            (
                getattr(rule, "mode", None),
                tuple(getattr(rule, "operations", []) or []),
                tuple(getattr(rule, "paths", []) or []),
            )
            for rule in permissions
        }
        for rule in _canonical_artifact_permissions():
            signature = (
                getattr(rule, "mode", None),
                tuple(getattr(rule, "operations", []) or []),
                tuple(getattr(rule, "paths", []) or []),
            )
            if signature not in existing:
                permissions.append(rule)
                existing.add(signature)


def _canonical_artifact_permissions():
    """Deny raw filesystem writes to artifacts that have validated writers."""
    try:
        from deepagents.middleware.filesystem import FilesystemPermission
    except ImportError:
        try:
            from deepagents.middleware.permissions import FilesystemPermission
        except ImportError:
            return []

    protected_paths = [
        # Canonical pipeline artifacts must be produced by validated
        # AutoIdea tools, not deepagents' raw write_file/edit_file tools.
        "/seed_idea_analysis.md",
        "/research_brief.md",
        "/task_formalization.md",
        "/paper_registry.json",
        "/literature_survey.md",
        "/paper_deep_reading.md",
        "/paper_positions.json",
        "/expanded_literature.md",
        "/evidence_db.json",
        "/knowledge_synthesis.md",
        "/research_gaps.json",
        "/design_space.json",
        "/raw_ideas.json",
        "/tournament_rankings.json",
        "/elo_rankings.json",
        "/debate_log.md",
        "/idea_reviews.json",
        "/feasibility_assessments.json",
        "/final_report.md",
        "/citations.json",
        "/fulltext_audit.json",
        "/batch_manifest.json",
        "/pipeline_state.json",
        "/run_status.json",
        "/stage_*_gate.json",
        "/reflections/**",
    ]
    return [
        FilesystemPermission(
            operations=["write"],
            paths=protected_paths,
            mode="deny",
        )
    ]


def _build_base_kwargs(backend=None, middleware=None):
    """Build kwargs dict for create_deep_agent from YAML + tool registry.

    Returns:
        Dict with keys: name, tools, subagents, middleware,
        system_prompt, backend.
    """
    from .utils import load_subagents_yaml, build_subagent_definitions

    cfg = _ensure_config()

    # Determine whether to use Claude native Web Search
    use_claude_ws = _is_claude_provider(cfg) and getattr(
        cfg, "enable_web_search", True
    )

    tool_registry, base_tools = _build_tool_registry(
        use_claude_web_search=use_claude_ws,
    )

    # Get model instance and optionally bind native Web Search tool
    chat_model = _ensure_chat_model()
    if use_claude_ws:
        ws_tool_def = _build_web_search_tool_def(cfg)
        if ws_tool_def:
            from .llm.models import bind_native_tools
            chat_model = bind_native_tools(chat_model, [ws_tool_def])
            logger.info("Claude native Web Search enabled for model")
        else:
            logger.debug("Claude Web Search tool def returned None; skipping")

    # Load sub-agent definitions from YAML
    yaml_path = Path(__file__).parent / "subagent.yaml"
    raw_subs = load_subagents_yaml(str(yaml_path))
    subs = build_subagent_definitions(raw_subs, tool_registry)
    _inject_subagent_middleware(subs)

    # Build system prompt with optional seed papers/ideas injection and pipeline params
    seed_section = _get_seed_papers_section()
    seed_ideas_section = _get_seed_ideas_section()
    pipeline_params = _get_pipeline_params()
    system_prompt = get_system_prompt(
        seed_papers_section=seed_section,
        seed_ideas_section=seed_ideas_section,
        pipeline_params=pipeline_params,
    )

    kwargs = {
        "name": "AutoIdea",
        "model": chat_model,
        "tools": base_tools,
        "subagents": subs,
        "middleware": middleware or [],
        "system_prompt": system_prompt,
        "backend": backend,
        "permissions": _canonical_artifact_permissions(),
    }
    return kwargs


def _get_seed_papers_section() -> str:
    """Load seed papers from config and format for prompt injection.

    Returns:
        Formatted markdown section for seed papers, or empty string
        if no seed papers are configured.
    """
    cfg = _ensure_config()
    seed_file = getattr(cfg, "seed_papers_file", "")
    if not seed_file:
        return ""

    try:
        from .tools.seed_papers import load_seed_papers, format_seed_papers_for_prompt
        papers = load_seed_papers(seed_file)
        logger.info("Loaded %d seed papers from %s", len(papers), seed_file)
        return format_seed_papers_for_prompt(papers)
    except FileNotFoundError:
        logger.warning("Seed papers file not found: %s", seed_file)
        return ""
    except ValueError as e:
        logger.warning("Failed to parse seed papers: %s", e)
        return ""
    except Exception as e:
        logger.warning("Unexpected error loading seed papers: %s", e)
        return ""


def _get_seed_ideas_section() -> str:
    """Load seed ideas from config and format for prompt injection.

    Uses **heuristic-only** extraction at startup to avoid blocking agent
    construction with LLM API calls.  Deep LLM-driven analysis happens
    later in Stage 0.5 when the agent calls
    ``generate_seed_idea_analysis_report``.

    Returns:
        Formatted markdown section for seed ideas, or empty string
        if no seed ideas are configured.
    """
    cfg = _ensure_config()
    seed_file = getattr(cfg, "seed_ideas_file", "")
    if not seed_file:
        return ""

    try:
        from .tools.seed_ideas import load_seed_ideas, format_seed_ideas_for_prompt

        # Startup: heuristic-only (no LLM calls) to keep loading fast.
        # The agent will perform LLM deep analysis in Stage 0.5.
        ideas = load_seed_ideas(seed_file, use_llm=False)
        logger.info("Loaded %d seed idea(s) from %s (heuristic)", len(ideas), seed_file)
        return format_seed_ideas_for_prompt(ideas)
    except FileNotFoundError:
        logger.warning("Seed ideas file not found: %s", seed_file)
        return ""
    except ValueError as e:
        logger.warning("Failed to parse seed ideas: %s", e)
        return ""
    except Exception as e:
        logger.warning("Unexpected error loading seed ideas: %s", e)
        return ""


def _get_pipeline_params() -> dict:
    """Extract pipeline parameters from config for prompt injection.

    Returns:
        Dict of parameter names to values.  Only includes parameters
        that differ from the prompt's built-in defaults so the agent
        sees explicit overrides.
    """
    cfg = _ensure_config()

    # Prompt built-in defaults (must match Section 9 of prompts.py AND
    # the defaults in config/settings.py AutoIdeaConfig)
    _PROMPT_DEFAULTS = {
        "max_search_queries": 50,
        "target_paper_count": 20,
        "max_ideas_to_generate": 10,
        "top_k_ranked": 20,
        "max_debate_rounds": 5,
        "deep_reading_top_k": 20,
    }

    params: dict = {}

    # Always inject all pipeline params so the agent has a clear picture
    for key, prompt_default in _PROMPT_DEFAULTS.items():
        value = getattr(cfg, key, prompt_default)
        params[key] = value

    # Also inject runtime context
    params["auto_approve"] = getattr(cfg, "auto_approve", True)
    params["provider"] = getattr(cfg, "provider", "openai")
    params["model"] = getattr(cfg, "model", "gpt-5.6-sol")

    return params


# ── Backend Construction ─────────────────────────────────────────────────

def _get_default_backend(workspace_dir: str, memory_dir: str):
    """Build the composite backend for file I/O routing.

    Routes:
    - Default: workspace files (sandbox with virtual_mode)
    - /memory/: shared cross-session memory
    """
    try:
        from deepagents.backends import FilesystemBackend, CompositeBackend
    except ImportError:
        try:
            from deepagents_langgraph.backends import (
                FilesystemBackend,
                CompositeBackend,
            )
        except ImportError:
            logger.warning(
                "deepagents backends not available; using simple filesystem backend"
            )
            return None

    ws = FilesystemBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
    )
    mem = FilesystemBackend(
        root_dir=memory_dir,
        virtual_mode=True,
    )

    return CompositeBackend(
        default=ws,
        routes={"/memory/": mem},
    )


# ── Middleware Construction ──────────────────────────────────────────────

def _get_default_middleware(memory_dir: str, cfg, backend=None):
    """Build the middleware stack.

    Returns list of: [AskUserMiddleware?, ToolCallSerializationMiddleware,
    ToolErrorHandlerMiddleware, MemoryMiddleware]

    The serialization middleware is required even when the graph runtime uses
    ``max_concurrency=1``.  Some OpenAI-compatible endpoints ignore
    ``parallel_tool_calls=False`` and emit dependent calls in one response.
    Executing those calls serially is still semantically unsafe because every
    argument was generated before any preceding tool result was observed.  By
    allowing one call per model turn, later calls are replanned from actual
    tool output.  ``max_concurrency=1`` remains a second safety boundary.
    """
    from .middleware import (
        ModelRetryMiddleware,
        ToolCallSerializationMiddleware,
        ToolErrorHandlerMiddleware,
        create_memory_middleware,
    )

    extraction_model = getattr(cfg, "extraction_model", None)
    memory_trigger = getattr(cfg, "memory_trigger_messages", 20) or 20
    mw = [
        ModelRetryMiddleware(),
        ToolCallSerializationMiddleware(),
        ToolErrorHandlerMiddleware(),
    ]

    # DeepAgents injects its own SummarizationMiddleware in create_deep_agent.
    # Do not add another one here: LangChain requires middleware names to be
    # unique and raises "Please remove duplicate middleware instances."
    # The built-in DeepAgents middleware provides model-aware auto compaction,
    # large tool-argument truncation, backend offload, and overflow retry.

    mw.append(
        create_memory_middleware(
            memory_dir=memory_dir,
            extraction_model=extraction_model,
            trigger=("messages", memory_trigger),
        )
    )

    enable_ask = getattr(cfg, "enable_ask_user", True)
    auto_approve = getattr(cfg, "auto_approve", True)

    if enable_ask and not auto_approve:
        from .middleware.ask_user import AskUserMiddleware
        mw.insert(0, AskUserMiddleware())

    return mw


# ── Default Agent (for langgraph dev / notebooks) ───────────────────────

_default_agent = None


def _get_default_agent():
    """Lazy default agent without checkpointer."""
    global _default_agent
    if _default_agent is not None:
        return _default_agent

    from .patches import apply_patches
    apply_patches()

    cfg = _ensure_config()
    workspace_dir = str(_paths_mod.WORKSPACE_ROOT)
    memory_dir = str(_paths_mod.MEMORY_DIR)

    be = _get_default_backend(workspace_dir, memory_dir)
    mw = _get_default_middleware(memory_dir, cfg, backend=be)
    kwargs = _build_base_kwargs(backend=be, middleware=mw)

    try:
        from deepagents import create_deep_agent
    except ImportError:
        from deepagents_langgraph import create_deep_agent

    _default_agent = create_deep_agent(**kwargs).with_config(
        {"recursion_limit": getattr(cfg, "recursion_limit", 1000)}
    )
    return _default_agent


# ── CLI Agent Factory ────────────────────────────────────────────────────

def create_cli_agent(
    workspace_dir: str | None = None,
    checkpointer=None,
    config=None,
):
    """Create the AutoIdea agent for CLI use.

    This is the primary entry point called by ``cli.agent._load_agent``.

    Args:
        workspace_dir: Per-session workspace directory. If None, uses
            paths.WORKSPACE_ROOT (possibly overridden by config).
        checkpointer: LangGraph checkpointer for multi-turn state.
            Defaults to InMemorySaver().
        config: Pre-loaded AutoIdeaConfig. If None, loaded from
            file/env/defaults.

    Returns:
        A compiled LangGraph agent with tools, sub-agents, middleware,
        and backend configured for the AutoIdea research pipeline.
    """

    from .patches import apply_patches
    apply_patches()

    cfg = _ensure_config(config)

    # Checkpointer fallback
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()

    # Workspace resolution
    from . import paths as _paths
    if workspace_dir is None:
        cfg_workspace = getattr(cfg, "workspace_dir", None)
        if cfg_workspace:
            set_workspace_root(cfg_workspace)
        workspace_dir = str(_paths.WORKSPACE_ROOT)

    set_active_workspace(workspace_dir)
    _paths.ensure_dirs()

    memory_dir = str(_paths.MEMORY_DIR)

    # Backend
    be = _get_default_backend(workspace_dir, memory_dir)

    # Middleware
    mw = _get_default_middleware(memory_dir, cfg, backend=be)

    # Build kwargs (tools, sub-agents, prompts, etc.)
    kwargs = _build_base_kwargs(backend=be, middleware=mw)

    # HITL interrupt configuration
    _interrupt_on = {}
    auto_approve = getattr(cfg, "auto_approve", True)
    if not auto_approve:
        _interrupt_on["execute"] = True

    # Create agent
    try:
        from deepagents import create_deep_agent
    except ImportError:
        from deepagents_langgraph import create_deep_agent

    agent = create_deep_agent(
        **kwargs,
        checkpointer=checkpointer,
        interrupt_on=_interrupt_on if _interrupt_on else None,
    ).with_config({"recursion_limit": getattr(cfg, "recursion_limit", 1000)})

    logger.info(
        "AutoIdea agent created | workspace=%s | model=%s | tools=%d | subagents=%d",
        workspace_dir,
        getattr(cfg, "model", "default"),
        len(kwargs.get("tools", [])),
        len(kwargs.get("subagents", [])),
    )

    return agent


# ── Module-level lazy attribute access ───────────────────────────────────

def __getattr__(name: str):
    """Support ``from autoidea.autoidea import autoidea_agent``."""
    if name == "autoidea_agent":
        return _get_default_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
