"""AutoIdea Memory Middleware -- Persistent long-term memory for the research pipeline.

Modeled after EvoScientist's memory middleware but adapted for AutoIdea's
12-stage research pipeline.  Provides two complementary mechanisms:

1. **Injection** (every LLM call): Reads ``/memory/MEMORY.md`` and injects it
   into the system prompt wrapped in ``<autoidea_memory>`` tags so the agent
   always has personalised context.
2. **Extraction** (threshold-triggered): When the conversation accumulates
   enough human messages, a secondary LLM call extracts structured facts
   (user profile, research preferences, experiment conclusions, learned
   preferences) and merges them into the appropriate MEMORY.md sections.

Usage
-----
::

    from autoidea.middleware.memory import create_memory_middleware

    middleware = create_memory_middleware(
        memory_dir="/path/to/memory",
        extraction_model=cheap_chat_model,
        trigger=("messages", 20),
    )
    agent = create_deep_agent(middleware=[middleware, ...])
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Annotated, NotRequired, cast

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# AgentMiddleware / AgentState / PrivateStateAttr -- imported from the
# deepagents-langgraph package with a graceful fallback so that the module
# can still be loaded (and tested) if the package is not installed.
# ---------------------------------------------------------------------------

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:
    try:
        from deepagents_langgraph.middleware import AgentMiddleware
    except ImportError:
        AgentMiddleware = object  # type: ignore[assignment,misc]

try:
    from langchain.agents.middleware.types import AgentState, PrivateStateAttr
except ImportError:
    try:
        from deepagents_langgraph import AgentState, PrivateStateAttr
    except ImportError:
        from typing import TypedDict

        AgentState = TypedDict  # type: ignore[assignment,misc]
        PrivateStateAttr = str  # type: ignore[assignment,misc]

from langchain_core.messages import AnyMessage, HumanMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level bookkeeping
# ---------------------------------------------------------------------------

#: Context variable holding the current MEMORY.md content for the active call.
_CURRENT_MEMORY: ContextVar[str] = ContextVar("autoidea_memory_current", default="")

#: The key used to store memory content in the agent state dict.
_STATE_MEMORY_KEY = "autoidea_memory_content"

#: Per-thread counter tracking how many human messages had been seen at the
#: last extraction.  Keyed by ``thread_id``.
_EXTRACTION_COUNTER: dict[str, int] = {}


# ============================================================================
# State Schema
# ============================================================================


class AutoIdeaMemoryState(AgentState):
    """State schema extension carrying the persisted memory content.

    The field uses ``PrivateStateAttr`` so it is transmitted between
    middleware hooks but **not** serialised into the LangGraph checkpoint
    (it lives on disk, not in the state store).
    """

    autoidea_memory_content: NotRequired[Annotated[str, PrivateStateAttr]]


# ============================================================================
# Pydantic extraction models
# ============================================================================


class UserProfile(BaseModel):
    """Extracted user profile information."""

    model_config = {"extra": "forbid"}

    name: str | None = Field(None, description="User's full name")
    role: str | None = Field(
        None, description="User's role (e.g. researcher, professor, student)"
    )
    institution: str | None = Field(
        None, description="User's institution or organisation"
    )
    language: str | None = Field(
        None, description="User's preferred language for communication"
    )


class ResearchPreferences(BaseModel):
    """Extracted research preference information."""

    model_config = {"extra": "forbid"}

    primary_domain: str | None = Field(
        None, description="Primary research domain (e.g. NLP, Computer Vision)"
    )
    sub_fields: str | None = Field(
        None, description="Research sub-fields (comma-separated)"
    )
    preferred_frameworks: str | None = Field(
        None,
        description="Preferred software frameworks (e.g. PyTorch, JAX, HuggingFace)",
    )
    preferred_models: str | None = Field(
        None,
        description="Preferred AI/ML models or model families (e.g. GPT-4, LLaMA)",
    )
    hardware: str | None = Field(
        None, description="Available hardware resources (GPUs, cluster details)"
    )
    constraints: str | None = Field(
        None,
        description="Resource, time, or other constraints on the research process",
    )


class ExperimentConclusion(BaseModel):
    """Extracted experiment conclusion.

    Only populated when a complete experiment or pipeline run was actually
    performed and concluded with actionable results.
    """

    model_config = {"extra": "forbid"}

    title: str = Field(description="Short experiment name or topic")
    question: str | None = Field(None, description="Research question investigated")
    method: str | None = Field(None, description="Brief method summary")
    key_result: str | None = Field(
        None, description="Primary metric, outcome, or finding"
    )
    conclusion: str | None = Field(
        None, description="One-line conclusion or take-away"
    )
    artifacts: str | None = Field(
        None,
        description="Paths to generated reports or artifacts (e.g. final_report.md)",
    )


class ExtractedMemory(BaseModel):
    """Aggregated structured output schema for memory extraction.

    Only fields that contain genuinely **new** information (not already
    present in MEMORY.md) should be populated.  Leave everything else as
    ``None`` / empty.
    """

    model_config = {"extra": "forbid"}

    user_profile: UserProfile | None = Field(
        None, description="Newly discovered user profile information"
    )
    research_preferences: ResearchPreferences | None = Field(
        None, description="Newly discovered research preferences"
    )
    experiment_conclusion: ExperimentConclusion | None = Field(
        None, description="Completed experiment conclusion (only if a run finished)"
    )
    learned_preferences: list[str] = Field(
        default_factory=list,
        description="New preferences, habits, or conventions observed from the user",
    )


# ============================================================================
# Prompts
# ============================================================================

EXTRACTION_PROMPT = """\
You are a memory extraction assistant for **AutoIdea**, an autonomous \
research idea generation agent that operates a 12-stage pipeline \
(Requirement Intake -> Task Formalization -> Literature Survey -> \
Position-First Analysis -> Hook-Driven Expansion -> Evidence Binding -> \
Knowledge Synthesis -> Design Space -> Idea Generation -> Elo Tournament -> \
Adversarial Debate -> Feasibility Assessment -> Final Report).

Analyze the following conversation and extract any NEW information that \
should be remembered long-term.  Only extract facts that are **not already \
present** in the current memory shown below.

<current_memory>
{current_memory}
</current_memory>

<conversation>
{conversation}
</conversation>

Rules:
- Only populate fields that contain genuinely new information.
- Leave fields as null / empty if there is nothing new to record.
- Do NOT repeat information already in <current_memory>.
- For ``experiment_conclusion``, only include it if a complete experiment \
  or pipeline run was actually executed and produced results.
- For ``learned_preferences``, include any specific habits, stylistic \
  choices, workflow preferences, or explicit "remember this" requests.
- Be concise.  Each value should be a short phrase, not a paragraph.
"""

MEMORY_INJECTION_TEMPLATE = """\
<autoidea_memory>
{memory_content}
</autoidea_memory>

<autoidea_memory_instructions>
The above <autoidea_memory> block contains your long-term memory about the \
user, their research preferences, past experiment conclusions, and learned \
conventions from prior AutoIdea sessions.

Use this memory to:
- Personalise responses and avoid re-asking known information.
- Tailor search queries and idea generation to the user's domain.
- Reference past experiment conclusions when relevant to the current pipeline \
  stage.
- Respect the user's preferred frameworks, models, and hardware constraints.

**AutoIdea's 12-stage pipeline context:**
Stages 1-2 (Intake & Formalisation), 3-5 (Literature & Analysis), \
6-7 (Evidence & Synthesis), 8-9 (Design Space & Ideas), 9.5 (Elo Tournament), \
10-11 (Debate & Feasibility), 12 (Final Report).

**When to update memory:**
- User shares name, role, institution, or language preference.
- User mentions research domain, preferred frameworks, models, or hardware.
- User explicitly asks you to remember something.
- A pipeline run (or significant experiment) completes with notable conclusions.

**How to update memory:**
- If ``/memory/MEMORY.md`` does not exist yet, use ``write_file`` to create it.
- If it already exists, use ``edit_file`` to update specific sections.
- Use this markdown structure:

```markdown
# AutoIdea Memory

## User Profile
- **Name**: ...
- **Role**: ...
- **Institution**: ...
- **Language**: ...

## Research Preferences
- **Primary Domain**: ...
- **Sub-fields**: ...
- **Preferred Frameworks**: ...
- **Preferred Models**: ...
- **Hardware**: ...
- **Constraints**: ...

## Experiment History
### [YYYY-MM-DD] Experiment Title
- **Question**: ...
- **Method**: ...
- **Key Result**: ...
- **Conclusion**: ...
- **Artifacts**: ...

## Learned Preferences
- ...
```

**Priority:** Update memory IMMEDIATELY when the user provides personal or \
research information -- before composing your main response.
</autoidea_memory_instructions>"""

DEFAULT_MEMORY_TEMPLATE = """\
# AutoIdea Memory

## User Profile
- **Name**: (unknown)
- **Role**: (unknown)
- **Institution**: (unknown)
- **Language**: (unknown)

## Research Preferences
- **Primary Domain**: (unknown)
- **Sub-fields**: (unknown)
- **Preferred Frameworks**: (unknown)
- **Preferred Models**: (unknown)
- **Hardware**: (unknown)
- **Constraints**: (unknown)

## Experiment History
(No experiments yet)

## Learned Preferences
- (none yet)
"""


# ============================================================================
# Helper functions
# ============================================================================


def _get_thread_id(runtime: Any) -> str:
    """Resolve the current thread ID from the LangGraph runtime config.

    Falls back to ``"default"`` when the thread ID cannot be determined.

    Parameters
    ----------
    runtime:
        The LangGraph ``Runtime`` object (or any object exposing a
        ``.config`` attribute that is a ``RunnableConfig``-like dict).

    Returns
    -------
    str
        The resolved thread identifier.
    """
    try:
        config = cast(dict, getattr(runtime, "config", {}))
        if isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id is not None:
                return str(thread_id)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to resolve thread_id from runtime config")
    return "default"


def _ensure_section(content: str, marker: str, body: str) -> str:
    """Ensure that a markdown section headed by *marker* exists in *content*.

    If *marker* (e.g. ``"## User Profile"``) is already present, the
    content is returned unchanged.  Otherwise the section is appended.

    Parameters
    ----------
    content:
        Existing markdown text.
    marker:
        The heading line to check for (e.g. ``"## Learned Preferences"``).
    body:
        Default body text to insert below the heading.

    Returns
    -------
    str
        Updated content with the section guaranteed to be present.
    """
    if marker in content:
        return content
    content = content.rstrip()
    if content:
        content += "\n\n"
    return f"{content}{marker}\n{body.rstrip()}\n"


def _ensure_memory_template(existing_md: str) -> str:
    """Guarantee that all expected sections are present in MEMORY.md.

    If *existing_md* is empty the full default template is returned.
    Otherwise each expected section is appended only when missing.

    Parameters
    ----------
    existing_md:
        Current content of MEMORY.md (may be empty).

    Returns
    -------
    str
        Content with all canonical sections present.
    """
    if not existing_md.strip():
        return DEFAULT_MEMORY_TEMPLATE

    result = existing_md

    # Ensure top-level heading
    if "# AutoIdea Memory" not in result:
        result = "# AutoIdea Memory\n\n" + result.lstrip()

    # User Profile
    result = _ensure_section(
        result,
        "## User Profile",
        "\n".join(
            [
                "- **Name**: (unknown)",
                "- **Role**: (unknown)",
                "- **Institution**: (unknown)",
                "- **Language**: (unknown)",
            ]
        ),
    )

    # Research Preferences
    result = _ensure_section(
        result,
        "## Research Preferences",
        "\n".join(
            [
                "- **Primary Domain**: (unknown)",
                "- **Sub-fields**: (unknown)",
                "- **Preferred Frameworks**: (unknown)",
                "- **Preferred Models**: (unknown)",
                "- **Hardware**: (unknown)",
                "- **Constraints**: (unknown)",
            ]
        ),
    )

    # Experiment History
    result = _ensure_section(result, "## Experiment History", "(No experiments yet)")

    # Learned Preferences
    result = _ensure_section(result, "## Learned Preferences", "- (none yet)")

    return result


def _section_bounds(content: str, marker: str) -> tuple[int | None, int | None]:
    """Locate the start and end character offsets of a markdown section.

    The *start* offset points to the character immediately **after** the
    marker line.  The *end* offset points to the beginning of the next
    ``## ``-level heading (or end-of-string).

    Parameters
    ----------
    content:
        Full markdown text.
    marker:
        Section heading to locate (e.g. ``"## Experiment History"``).

    Returns
    -------
    tuple[int | None, int | None]
        ``(start, end)`` character offsets, or ``(None, None)`` when the
        marker is not found.
    """
    idx = content.find(marker)
    if idx == -1:
        return None, None
    start = idx + len(marker)
    # Advance past newline immediately after marker
    if start < len(content) and content[start] == "\n":
        start += 1
    next_marker = content.find("\n## ", start)
    if next_marker == -1:
        next_marker = len(content)
    return start, next_marker


def _normalize_item(value: str) -> str:
    """Normalise a list item for deduplication comparison."""
    return re.sub(r"\s+", " ", value.strip().lower())


# ---------------------------------------------------------------------------
# Merge extracted JSON into MEMORY.md
# ---------------------------------------------------------------------------


def _merge_memory(existing_md: str, extracted: dict[str, Any]) -> str:
    """Merge extracted structured fields into the existing MEMORY.md content.

    Performs targeted regex replacements for scalar profile / preference
    fields, appends to "Experiment History" and "Learned Preferences"
    sections.  Unknown sections or empty extractions are left untouched.

    Parameters
    ----------
    existing_md:
        Current markdown content of MEMORY.md.
    extracted:
        Dictionary produced by ``ExtractedMemory.model_dump(exclude_none=True)``.

    Returns
    -------
    str
        Updated MEMORY.md content with new information merged.
    """
    if not extracted:
        return existing_md

    result = _ensure_memory_template(existing_md)

    # ----- User Profile -----
    profile = extracted.get("user_profile")
    if profile and isinstance(profile, dict):
        field_map = {
            "name": "Name",
            "role": "Role",
            "institution": "Institution",
            "language": "Language",
        }
        for key, label in field_map.items():
            value = profile.get(key)
            if value and str(value).lower() not in {"null", "none", ""}:
                pattern = rf"(- \*\*{re.escape(label)}\*\*: ).*"
                replacement_value = str(value)
                result = re.sub(
                    pattern,
                    lambda m, v=replacement_value: m.group(1) + v,
                    result,
                )

    # ----- Research Preferences -----
    prefs = extracted.get("research_preferences")
    if prefs and isinstance(prefs, dict):
        field_map = {
            "primary_domain": "Primary Domain",
            "sub_fields": "Sub-fields",
            "preferred_frameworks": "Preferred Frameworks",
            "preferred_models": "Preferred Models",
            "hardware": "Hardware",
            "constraints": "Constraints",
        }
        for key, label in field_map.items():
            value = prefs.get(key)
            if value and str(value).lower() not in {"null", "none", ""}:
                pattern = rf"(- \*\*{re.escape(label)}\*\*: ).*"
                replacement_value = str(value)
                result = re.sub(
                    pattern,
                    lambda m, v=replacement_value: m.group(1) + v,
                    result,
                )

    # ----- Experiment History (append new entries) -----
    exp = extracted.get("experiment_conclusion")
    should_add_exp = bool(exp and isinstance(exp, dict) and exp.get("title"))

    if should_add_exp and exp is not None:
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")
        title = str(exp.get("title", "Untitled")).strip()

        entry = f"\n### [{date_str}] {title}\n"
        entry += f"- **Question**: {exp.get('question', 'N/A')}\n"
        entry += f"- **Method**: {exp.get('method', 'N/A')}\n"
        entry += f"- **Key Result**: {exp.get('key_result', 'N/A')}\n"
        entry += f"- **Conclusion**: {exp.get('conclusion', 'N/A')}\n"
        if exp.get("artifacts"):
            entry += f"- **Artifacts**: {exp['artifacts']}\n"

        # Remove placeholder text "(No experiments yet)" if present
        exp_start, exp_end = _section_bounds(result, "## Experiment History")
        if exp_start is not None and exp_end is not None:
            exp_section = result[exp_start:exp_end]
            exp_lines = [
                line
                for line in exp_section.splitlines()
                if "(No experiments yet)" not in line
            ]
            result = (
                result[:exp_start]
                + "\n".join(exp_lines).strip("\n")
                + "\n"
                + result[exp_end:]
            )

        # Deduplicate by title -- do not add if an entry with the same
        # title already exists regardless of date.
        if re.search(rf"### \[[0-9-]+\] {re.escape(title)}\b", result):
            should_add_exp = False

    if should_add_exp and exp is not None and isinstance(exp, dict) and exp.get("title"):
        title = str(exp["title"]).strip()
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")

        entry = f"\n### [{date_str}] {title}\n"
        entry += f"- **Question**: {exp.get('question', 'N/A')}\n"
        entry += f"- **Method**: {exp.get('method', 'N/A')}\n"
        entry += f"- **Key Result**: {exp.get('key_result', 'N/A')}\n"
        entry += f"- **Conclusion**: {exp.get('conclusion', 'N/A')}\n"
        if exp.get("artifacts"):
            entry += f"- **Artifacts**: {exp['artifacts']}\n"

        # Insert before "## Learned Preferences" if possible
        marker = "## Learned Preferences"
        if marker in result:
            result = result.replace(marker, entry + "\n" + marker, 1)
        else:
            result = result.rstrip() + "\n" + entry

    # ----- Learned Preferences (append, deduplicated) -----
    learned = extracted.get("learned_preferences")
    if learned and isinstance(learned, list):
        marker = "## Learned Preferences"
        start, end = _section_bounds(result, marker)
        if start is None or end is None:
            result = _ensure_section(result, marker, "- (none yet)")
            start, end = _section_bounds(result, marker)

        if start is not None and end is not None:
            section = result[start:end]
            section_lines = [
                line
                for line in section.splitlines()
                if line.strip()
                and line.strip() not in {"- (none yet)", "- (none)", "(none yet)"}
            ]
            existing_items: set[str] = {
                _normalize_item(line.lstrip("- "))
                for line in section_lines
                if line.strip().startswith("- ")
            }
            new_lines: list[str] = []
            for item in learned:
                if not item:
                    continue
                normalized = _normalize_item(str(item))
                if normalized in existing_items:
                    continue
                existing_items.add(normalized)
                new_lines.append(f"- {item}")

            if new_lines:
                section_lines.extend(new_lines)
                rebuilt = "\n".join(line for line in section_lines if line.strip())
                result = result[:start] + rebuilt + "\n" + result[end:]

    return result


# ============================================================================
# Middleware class
# ============================================================================


class AutoIdeaMemoryMiddleware(AgentMiddleware):
    """Middleware that injects and auto-extracts long-term memory.

    Designed for the AutoIdea 12-stage research pipeline.  Reads
    ``MEMORY.md`` from a configurable backend before every LLM call,
    injects its content into the system prompt, and periodically
    extracts structured memories using a secondary LLM.

    Parameters
    ----------
    backend:
        A backend instance (or callable factory) supporting
        ``download_files`` / ``write`` / ``edit`` operations.
    memory_path:
        Virtual path to the MEMORY.md file inside the backend.
    extraction_model:
        Chat model used for memory extraction (should be a
        cheap / fast model such as ``claude-3-haiku`` or ``gpt-4o-mini``).
        When ``None``, automatic extraction is disabled and only
        prompt-guided manual ``edit_file`` operations work.
    trigger:
        When to run automatic extraction.  Currently supports
        ``("messages", N)`` which triggers every *N* new human messages.
    """

    state_schema = AutoIdeaMemoryState

    def __init__(
        self,
        *,
        backend: Any,
        memory_path: str = "/memory/MEMORY.md",
        extraction_model: BaseChatModel | None = None,
        trigger: tuple[str, int] = ("messages", 20),
    ) -> None:
        self._backend = backend
        self._memory_path = memory_path
        self._extraction_model = extraction_model
        self._trigger = trigger
        # Local instance-level tracking as well (supplements module-level
        # ``_EXTRACTION_COUNTER`` for multi-instance scenarios).
        self._last_extraction_at: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _get_backend(self, state: Any, runtime: Any) -> Any:
        """Resolve the backend, calling the factory if necessary."""
        if callable(self._backend) and not hasattr(self._backend, "download_files"):
            try:
                from langchain_core.runnables.config import RunnableConfig

                config = cast(RunnableConfig, getattr(runtime, "config", {}))
                # Some backend factories expect a ToolRuntime-like object.
                try:
                    from langchain.tools import ToolRuntime

                    tool_runtime = ToolRuntime(
                        state=state,
                        context=getattr(runtime, "context", None),
                        stream_writer=getattr(runtime, "stream_writer", None),
                        store=getattr(runtime, "store", None),
                        config=config,
                        tool_call_id=None,
                    )
                    return self._backend(tool_runtime)
                except (ImportError, TypeError):
                    return self._backend(config)
            except Exception:  # noqa: BLE001
                return self._backend()
        return self._backend

    # ------------------------------------------------------------------
    # Agent-level preload hooks
    # ------------------------------------------------------------------

    def before_agent(
        self,
        state: Any,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Preload MEMORY.md into the agent state on the first invocation."""
        if isinstance(state, dict) and state.get(_STATE_MEMORY_KEY) is not None:
            return None
        backend = self._get_backend(state, runtime)
        memory = self._read_memory(backend)
        _CURRENT_MEMORY.set(memory)
        return {_STATE_MEMORY_KEY: memory}

    async def abefore_agent(
        self,
        state: Any,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`before_agent`."""
        if isinstance(state, dict) and state.get(_STATE_MEMORY_KEY) is not None:
            return None
        backend = self._get_backend(state, runtime)
        memory = await self._aread_memory(backend)
        _CURRENT_MEMORY.set(memory)
        return {_STATE_MEMORY_KEY: memory}

    # ------------------------------------------------------------------
    # Read / write helpers
    # ------------------------------------------------------------------

    def _read_memory(self, backend: Any) -> str:
        """Read MEMORY.md content from the backend (bytes -> str)."""
        try:
            responses = backend.download_files([self._memory_path])
            if (
                responses
                and responses[0].content is not None
                and getattr(responses[0], "error", None) is None
            ):
                raw = responses[0].content
                if isinstance(raw, (bytes, bytearray)):
                    return raw.decode("utf-8")
                return str(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read memory at %s: %s", self._memory_path, exc)
        return ""

    async def _aread_memory(self, backend: Any) -> str:
        """Async: read MEMORY.md content from the backend."""
        try:
            responses = await backend.adownload_files([self._memory_path])
            if (
                responses
                and responses[0].content is not None
                and getattr(responses[0], "error", None) is None
            ):
                raw = responses[0].content
                if isinstance(raw, (bytes, bytearray)):
                    return raw.decode("utf-8")
                return str(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read memory at %s: %s", self._memory_path, exc)
        return ""

    def _write_memory(
        self,
        backend: Any,
        old_content: str,
        new_content: str,
    ) -> None:
        """Write updated MEMORY.md content to the backend.

        Uses ``edit`` when old content exists, ``write`` for initial creation.
        """
        try:
            if old_content:
                result = backend.edit(self._memory_path, old_content, new_content)
            else:
                result = backend.write(self._memory_path, new_content)
            if result and getattr(result, "error", None):
                logger.warning("Failed to write memory: %s", result.error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exception writing memory: %s", exc)

    async def _awrite_memory(
        self,
        backend: Any,
        old_content: str,
        new_content: str,
    ) -> None:
        """Async: write updated MEMORY.md content to the backend."""
        try:
            if old_content:
                result = await backend.aedit(
                    self._memory_path, old_content, new_content
                )
            else:
                result = await backend.awrite(self._memory_path, new_content)
            if result and getattr(result, "error", None):
                logger.warning("Failed to write memory: %s", result.error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exception writing memory: %s", exc)

    # ------------------------------------------------------------------
    # Extraction threshold
    # ------------------------------------------------------------------

    def _should_extract(self, thread_id: str, messages: list[AnyMessage]) -> bool:
        """Determine whether automatic extraction should run.

        The check counts the number of ``HumanMessage`` instances in
        *messages* and compares against the last recorded extraction
        count for this thread.  Returns ``True`` when the difference
        meets or exceeds the trigger threshold.

        Parameters
        ----------
        thread_id:
            Conversation thread identifier.
        messages:
            Current message history.

        Returns
        -------
        bool
        """
        if self._extraction_model is None:
            return False

        trigger_type, trigger_value = self._trigger
        if trigger_type != "messages":
            logger.warning("Unsupported trigger type: %s", trigger_type)
            return False

        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))

        # Check module-level counter first, then instance-level fallback
        last_module = _EXTRACTION_COUNTER.get(thread_id, 0)
        last_instance = self._last_extraction_at.get(thread_id, 0)
        last = max(last_module, last_instance)

        return (human_count - last) >= trigger_value

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_extraction_prompt(memory: str, messages: list[AnyMessage]) -> str:
        """Build the extraction prompt from recent conversation messages.

        Selects the last 30 messages, filtering to human and AI only,
        and formats them alongside the current memory for the extraction
        LLM.
        """
        # Take a window of recent messages (cap to avoid token overflow)
        recent = messages[-30:]
        conv_parts: list[str] = []
        for msg in recent:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            # Skip empty tool-call-only messages
            if not content.strip():
                continue
            conv_parts.append(f"[{role}]: {content}")

        return EXTRACTION_PROMPT.format(
            current_memory=memory or "(empty -- no memory saved yet)",
            conversation="\n".join(conv_parts) or "(no conversation yet)",
        )

    @staticmethod
    def _disable_thinking(model: BaseChatModel) -> BaseChatModel:
        """Return a copy of *model* with thinking / reasoning features disabled.

        Anthropic's API does not allow extended thinking when
        ``tool_choice`` forces tool use (as ``with_structured_output``
        does).  Similarly, OpenAI reasoning models can conflict.
        This helper strips ``thinking`` / ``reasoning`` settings so
        extraction works reliably across providers.

        Uses ``model_copy()`` (Pydantic v2) to produce a real new
        instance rather than ``bind()``, which only wraps in a
        ``RunnableBinding`` and does not override first-class fields.
        """
        updates: dict[str, Any] = {}
        model_kwargs = getattr(model, "model_kwargs", {}) or {}

        # Anthropic extended thinking
        if getattr(model, "thinking", None) or "thinking" in model_kwargs:
            updates["thinking"] = None
        # OpenAI reasoning
        if getattr(model, "reasoning", None) or "reasoning" in model_kwargs:
            updates["reasoning"] = None
        # Some models expose a reasoning_effort knob
        if getattr(model, "reasoning_effort", None) or "reasoning_effort" in model_kwargs:
            updates["reasoning_effort"] = None

        if not updates:
            return model

        try:
            return model.model_copy(update=updates)
        except Exception:  # noqa: BLE001
            # Fallback for non-Pydantic or unusual model classes
            non_none = {k: v for k, v in updates.items() if v is not None}
            if non_none:
                return model.bind(**non_none)
            return model

    @staticmethod
    def _structured_output_kwargs(model: BaseChatModel) -> dict[str, Any]:
        """Return extra kwargs for ``with_structured_output`` based on provider.

        OpenAI's Structured Outputs mode (default since ``langchain-openai``
        0.3) requires ``additionalProperties: false`` and all-required fields.
        ``ExtractedMemory`` uses ``Optional`` unions that violate these rules,
        so we fall back to ``function_calling`` for OpenAI-family models.
        """
        model_module = type(model).__module__ or ""
        if model_module.startswith("langchain_openai"):
            return {"method": "function_calling"}
        return {}

    def _extract(
        self,
        model: BaseChatModel,
        memory: str,
        messages: list[AnyMessage],
    ) -> dict[str, Any]:
        """Run synchronous LLM extraction on recent messages.

        Returns a dict of extracted fields (empty on failure).
        """
        prompt = self._build_extraction_prompt(memory, messages)
        try:
            plain_model = self._disable_thinking(model)
            so_kwargs = self._structured_output_kwargs(plain_model)
            structured_model = plain_model.with_structured_output(
                ExtractedMemory, **so_kwargs
            )
            result = structured_model.invoke(prompt)
            if result is None:
                return {}
            return result.model_dump(exclude_none=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory extraction failed: %s", exc)
            return {}

    async def _aextract(
        self,
        model: BaseChatModel,
        memory: str,
        messages: list[AnyMessage],
    ) -> dict[str, Any]:
        """Run async LLM extraction on recent messages.

        Returns a dict of extracted fields (empty on failure).
        """
        prompt = self._build_extraction_prompt(memory, messages)
        try:
            plain_model = self._disable_thinking(model)
            so_kwargs = self._structured_output_kwargs(plain_model)
            structured_model = plain_model.with_structured_output(
                ExtractedMemory, **so_kwargs
            )
            result = await structured_model.ainvoke(prompt)
            if result is None:
                return {}
            return result.model_dump(exclude_none=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory extraction failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Request modification -- inject memory into system message
    # ------------------------------------------------------------------

    def modify_request(self, request: Any) -> Any:
        """Inject memory content and instructions into the system message.

        Always injects ``<autoidea_memory_instructions>`` so the agent
        knows it can save memories, even when MEMORY.md does not exist yet.
        """
        state = getattr(request, "state", None) or {}
        memory_content: str = ""

        # 1. Try from state
        if isinstance(state, dict):
            memory_content = state.get(_STATE_MEMORY_KEY, "")

        # 2. Try from context var
        if not memory_content:
            memory_content = _CURRENT_MEMORY.get()

        # 3. Try reading from backend
        if not memory_content and getattr(request, "runtime", None) is not None:
            try:
                backend = self._get_backend(state, request.runtime)
                memory_content = self._read_memory(backend)
                _CURRENT_MEMORY.set(memory_content)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Failed to load memory during modify_request: %s", exc
                )

        # 4. Use placeholder when nothing exists
        if not memory_content:
            memory_content = (
                "(No memory saved yet. Create ``/memory/MEMORY.md`` when you "
                "learn important information about the user or their research.)"
            )

        injection = MEMORY_INJECTION_TEMPLATE.format(memory_content=memory_content)

        # Append to existing system message
        # NOTE: Do NOT use ``+`` operator on BaseMessage objects — it returns
        # a ChatPromptTemplate instead of a SystemMessage, which breaks the
        # downstream message coercion in langchain-core.
        current_system = getattr(request, "system_message", None)
        if current_system is not None:
            if hasattr(current_system, "content"):
                base_text = current_system.content if isinstance(current_system.content, str) else str(current_system.content)
            else:
                base_text = str(current_system) if current_system else ""
            from langchain_core.messages import SystemMessage as _SM
            new_system = _SM(content=base_text + "\n\n" + injection)
        else:
            from langchain_core.messages import SystemMessage as _SM
            new_system = _SM(content=injection)

        # Use the request's override() if available, otherwise try attribute
        if hasattr(request, "override"):
            return request.override(system_message=new_system)
        elif hasattr(request, "_replace"):
            return request._replace(system_message=new_system)
        else:
            try:
                request.system_message = new_system
            except AttributeError:
                logger.debug("Cannot inject memory into request: no override method")
            return request

    # ------------------------------------------------------------------
    # Model call wrappers
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[..., Any],
    ) -> Any:
        """Synchronous wrapper: inject memory then delegate to handler."""
        modified = self.modify_request(request)
        return handler(modified)

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[..., Awaitable[Any]],
    ) -> Any:
        """Async wrapper: inject memory then delegate to handler."""
        modified = self.modify_request(request)
        return await handler(modified)

    # ------------------------------------------------------------------
    # Before-model hooks (read + extract + merge + write)
    # ------------------------------------------------------------------

    def before_model(
        self,
        state: Any,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Read memory and optionally run extraction before each LLM call.

        Steps:
        1. Read current MEMORY.md from the backend.
        2. Set the context variable so ``modify_request`` can inject it.
        3. If the extraction threshold is reached, call the extraction
           LLM, merge results, and write the updated memory.
        """
        backend = self._get_backend(state, runtime)
        messages: list[AnyMessage] = (
            state["messages"] if isinstance(state, dict) else []
        )
        thread_id = _get_thread_id(runtime)

        # Always refresh memory for injection
        memory = self._read_memory(backend)
        _CURRENT_MEMORY.set(memory)
        state_update: dict[str, Any] | None = None
        if isinstance(state, dict) and state.get(_STATE_MEMORY_KEY) != memory:
            state_update = {_STATE_MEMORY_KEY: memory}

        # Conditional extraction
        if self._should_extract(thread_id, messages):
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            extracted = self._extract(self._extraction_model, memory, messages)
            if extracted:
                new_memory = _merge_memory(memory, extracted)
                if new_memory != memory:
                    self._write_memory(backend, memory, new_memory)
                    _CURRENT_MEMORY.set(new_memory)
                    logger.info(
                        "Auto-extracted and updated memory for thread '%s'",
                        thread_id,
                    )
                    state_update = {_STATE_MEMORY_KEY: new_memory}
            # Update both module-level and instance-level counters
            _EXTRACTION_COUNTER[thread_id] = human_count
            self._last_extraction_at[thread_id] = human_count

        return state_update

    async def abefore_model(
        self,
        state: Any,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`before_model`."""
        backend = self._get_backend(state, runtime)
        messages: list[AnyMessage] = (
            state["messages"] if isinstance(state, dict) else []
        )
        thread_id = _get_thread_id(runtime)

        memory = await self._aread_memory(backend)
        _CURRENT_MEMORY.set(memory)
        state_update: dict[str, Any] | None = None
        if isinstance(state, dict) and state.get(_STATE_MEMORY_KEY) != memory:
            state_update = {_STATE_MEMORY_KEY: memory}

        if self._should_extract(thread_id, messages):
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            extracted = await self._aextract(
                self._extraction_model, memory, messages
            )
            if extracted:
                new_memory = _merge_memory(memory, extracted)
                if new_memory != memory:
                    await self._awrite_memory(backend, memory, new_memory)
                    _CURRENT_MEMORY.set(new_memory)
                    logger.info(
                        "Auto-extracted and updated memory for thread '%s'",
                        thread_id,
                    )
                    state_update = {_STATE_MEMORY_KEY: new_memory}
            _EXTRACTION_COUNTER[thread_id] = human_count
            self._last_extraction_at[thread_id] = human_count

        return state_update


# ============================================================================
# Factory
# ============================================================================


def create_memory_middleware(
    memory_dir: str | None = None,
    extraction_model: BaseChatModel | None = None,
    trigger: tuple[str, int] = ("messages", 20),
) -> AutoIdeaMemoryMiddleware:
    """Create an :class:`AutoIdeaMemoryMiddleware` backed by the filesystem.

    Uses a ``FilesystemBackend`` rooted at the given directory so that
    memory persists across threads and sessions.

    Parameters
    ----------
    memory_dir:
        Path to the shared memory directory (not per-session).  Defaults
        to :data:`autoidea.paths.MEMORY_DIR`.
    extraction_model:
        Chat model for auto-extraction (optional).  When ``None``, only
        prompt-guided manual memory updates via ``edit_file`` will work.
    trigger:
        When to auto-extract.  Default: every 20 new human messages.

    Returns
    -------
    AutoIdeaMemoryMiddleware
        A fully configured middleware instance ready to be passed to the
        agent builder.
    """
    try:
        from deepagents_langgraph.backends import FilesystemBackend
    except ImportError:
        from deepagents.backends import FilesystemBackend  # type: ignore[no-redef]

    from ..paths import MEMORY_DIR as _DEFAULT_MEMORY_DIR

    if memory_dir is None:
        memory_dir = str(_DEFAULT_MEMORY_DIR)

    memory_backend = FilesystemBackend(
        root_dir=memory_dir,
        virtual_mode=True,
    )
    return AutoIdeaMemoryMiddleware(
        backend=memory_backend,
        memory_path="/MEMORY.md",
        extraction_model=extraction_model,
        trigger=trigger,
    )
