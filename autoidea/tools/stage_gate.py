"""Stage Gate Validation for AutoIdea v3.0.

Provides code-level enforcement of stage transition conditions.
Each stage has explicit gate criteria that must be met before
the pipeline can proceed to the next stage.

Includes a per-stage retry counter to prevent infinite loops when
the agent repeatedly fails a gate check.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Evidence helpers ──────────────────────────────────────────────────────

def _get_evidence_int(evidence: dict, *keys, default: int = 0) -> int:
    """Get an integer value from evidence dict, trying multiple key aliases.

    Handles str→int and float→int coercion gracefully.
    """
    for key in keys:
        val = evidence.get(key)
        if val is not None:
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                try:
                    return int(val)
                except ValueError:
                    continue
            if isinstance(val, float):
                return int(val)
    return default


def _get_evidence_list(evidence: dict, *keys, default: list | None = None) -> list:
    """Get a list value from evidence dict, trying multiple key aliases.

    If the value is a JSON-encoded string, it is parsed automatically.
    A bare string is wrapped in a single-element list.
    """
    if default is None:
        default = []
    for key in keys:
        val = evidence.get(key)
        if val is not None:
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    return [val]  # Wrap single string in list
    return default

# ── Retry tracking ────────────────────────────────────────────────────────

# Maximum number of consecutive FAIL results per stage before the gate
# auto-passes with a warning.  This prevents the catastrophic infinite
# loop observed in production where Stage 2 failed 56+ times.
MAX_GATE_RETRIES: int = 5

# Per-stage failure counter: stage_id -> consecutive_fail_count
_gate_fail_counts: dict[str, int] = {}
# A successful gate check grants one in-process, workspace-scoped permission to
# save that stage's reflection.  This prevents the agent from recording a stage
# as complete after a FAIL result while keeping Web and CLI on the same path.
_gate_passed_stages: set[tuple[str, str]] = set()
_web_checkpoint_events_file: Path | None = None
_web_approved_checkpoints: set[str] = set()

_WEB_CHECKPOINT_QUESTIONS: dict[str, tuple[str, str]] = {
    "stage_7": (
        "Stage 7 / 研究空白审查",
        "Approve the identified research gaps before Stage 8? / 是否批准当前研究空白并进入 Stage 8？",
    ),
    "stage_9": (
        "Stage 9 / 候选想法审查",
        "Approve the candidate research ideas before ranking? / 是否批准候选研究想法并进入排序？",
    ),
    "stage_10": (
        "Stage 10 / 对抗评审结论",
        "Accept the debate verdicts before feasibility analysis? / 是否接受对抗评审结论并进入可行性分析？",
    ),
}


def configure_web_checkpoint_events(events_file: str | Path | None) -> None:
    """Enable mandatory Web checkpoints and load approvals from earlier runs."""
    global _web_checkpoint_events_file
    _web_checkpoint_events_file = (
        Path(events_file).expanduser().resolve() if events_file else None
    )
    _web_approved_checkpoints.clear()
    if _web_checkpoint_events_file is None:
        return
    requested: dict[str, str] = {}
    try:
        lines = _web_checkpoint_events_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        interaction_id = str(event.get("interaction_id") or "")
        if event.get("type") == "interaction_requested" and interaction_id:
            requested[interaction_id] = str(event.get("checkpoint_stage") or "")
        elif event.get("type") == "interaction_resolved" and requested.get(interaction_id):
            response = event.get("response")
            if isinstance(response, dict) and response.get("approved") is True:
                _web_approved_checkpoints.add(requested[interaction_id])


def _web_checkpoint_is_approved(stage: str) -> bool:
    return stage in _web_approved_checkpoints


def _auto_approve_enabled() -> bool:
    value = os.getenv("AUTOIDEA_AUTO_APPROVE", "true")
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _request_web_checkpoint(stage: str) -> tuple[bool, str]:
    """Pause the LangGraph tool call for a mandatory browser decision."""
    from langgraph.types import interrupt

    title, question = _WEB_CHECKPOINT_QUESTIONS[stage]
    response = interrupt(
        {
            "type": "ask_user",
            "tool_call_id": f"autoidea-checkpoint:{stage}",
            "questions": [
                {
                    "question": f"{title}: {question}",
                    "type": "multiple_choice",
                    "choices": [
                        {"value": "approve", "label": "Approve / 批准"},
                        {
                            "value": "auto_continue",
                            "label": "Continue automatically / 不回答，后续全自动",
                        },
                        {"value": "revise", "label": "Revise / 修改"},
                        {"value": "rerun", "label": "Re-run / 重新运行"},
                    ],
                    "required": True,
                },
                {
                    "question": "Optional feedback / 可选反馈",
                    "type": "text",
                    "required": False,
                },
            ],
        }
    )
    if not isinstance(response, dict) or response.get("status") != "answered":
        return False, "Checkpoint response was cancelled or invalid."
    answers = response.get("answers")
    if not isinstance(answers, list) or not answers:
        return False, "Checkpoint response did not include a decision."
    decision = str(answers[0]).strip().casefold()
    feedback = str(answers[1]).strip() if len(answers) > 1 else ""
    approved = decision in {"approve", "auto_continue"}
    if approved:
        _web_approved_checkpoints.add(stage)
    return approved, feedback


def _configured_deep_reading_top_k() -> int:
    env_value = os.getenv("AUTOIDEA_DEEP_READING_TOP_K")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    try:
        from autoidea.config import load_config

        return int(getattr(load_config(), "deep_reading_top_k", 20) or 20)
    except Exception:
        return 20


def _configured_max_ideas_to_generate() -> int:
    """Return the current Stage 9 hard cap, honoring runtime overrides."""
    env_value = os.getenv("AUTOIDEA_MAX_IDEAS_TO_GENERATE")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    try:
        from autoidea.config import load_config

        configured = getattr(load_config(), "max_ideas_to_generate", 10) or 10
        return max(1, int(configured))
    except Exception:
        return 10


def reset_gate_counters() -> None:
    """Reset gate retry counters and unconsumed pass permissions."""
    _gate_fail_counts.clear()
    _gate_passed_stages.clear()


def _gate_pass_key(stage: str) -> tuple[str, str]:
    """Return a workspace-scoped key for a stage-gate pass."""
    from autoidea.paths import get_active_workspace

    workspace = Path(get_active_workspace()).expanduser().resolve()
    return str(workspace), stage


# ── Gate Definitions ──────────────────────────────────────────────────────

STAGE_GATES: dict[str, dict[str, Any]] = {
    "stage_0.5": {
        "name": "Seed Idea Analysis",
        "required_fields": ["ideas_analyzed", "has_synthesis"],
        "min_word_count": 30,
        "description": "Seed ideas analyzed with assessments and cross-idea synthesis.",
    },
    "stage_1": {
        "name": "Research Brief",
        "required_fields": ["topic", "domain", "scope"],
        "min_word_count": 30,
        "description": "User provides clear research brief with topic, domain, and scope.",
    },
    "stage_2": {
        "name": "Task Formalization",
        "required_fields": ["research_question", "keywords", "constraints"],
        "min_word_count": 50,
        "description": "Agent formalizes the research question with keywords and constraints.",
    },
    "stage_3": {
        "name": "Literature Survey",
        "required_fields": [],
        "min_papers": 10,
        "description": "At least 10 relevant papers surveyed from multiple sources.",
    },
    "stage_3.5": {
        "name": "Paper Deep Reading",
        "required_fields": [],
        "min_papers_read": 5,
        "description": "At least 5 papers deeply read and summarized (full-text or abstract-based).",
    },
    "stage_4": {
        "name": "Positioning Analysis",
        "required_fields": [],
        "min_papers_positioned": 5,
        "description": "At least 5 papers analyzed with Critique-First protocol.",
    },
    "stage_5": {
        "name": "Hook-Driven Exploration",
        "required_fields": [],
        "min_hooks": 3,
        "description": "At least 3 research hooks identified from positioned papers.",
    },
    "stage_6": {
        "name": "Evidence Synthesis",
        "required_fields": [],
        "min_citations": 5,
        "description": "At least 5 verified citations collected.",
    },
    "stage_7": {
        "name": "Gap Identification",
        "required_fields": ["gaps", "evidence_gap_links"],
        "min_gaps": 3,
        "min_evidence_gap_links": 3,
        "hitl_checkpoint": True,
        "description": (
            "OSMOSIS gap analysis with at least 3 gaps and explicit "
            "Claim-to-Gap provenance. HITL checkpoint."
        ),
    },
    "stage_8": {
        "name": "Design Space Mapping",
        "required_fields": ["axes", "combinations"],
        "min_axes": 2,
        "description": "At least 2 design axes with promising combinations.",
    },
    "stage_9": {
        "name": "Idea Generation",
        "required_fields": ["ideas"],
        "min_ideas": 5,
        "hitl_checkpoint": True,
        "description": (
            "Generate a bounded set of research ideas using distinct Nova methods. "
            "The configured max_ideas_to_generate value is a hard cap. HITL checkpoint."
        ),
    },
    "stage_9.5": {
        "name": "Elo Tournament",
        "required_fields": ["rankings"],
        "min_comparisons": 3,
        "description": "Elo-based tournament ranking of generated ideas.",
    },
    "stage_10": {
        "name": "Multi-Round Debate",
        "required_fields": [],
        "min_rounds": 1,
        "max_rounds": 3,
        "hitl_checkpoint": True,
        "description": "ATTACK/DEFEND/RE-EVAL protocol. HITL checkpoint.",
    },
    "stage_11": {
        "name": "Feasibility Assessment",
        "required_fields": ["assessment"],
        "description": "Feasibility assessment covering resources, timeline, risks.",
    },
    "stage_12": {
        "name": "Final Report",
        "required_fields": [],
        "description": "Complete research proposal with [Pn]+[Cn] evidence tags.",
    },
}


def _get_expected_fields(stage: str) -> str:
    """Return a hint about expected evidence_json fields for a stage."""
    _FIELD_HINTS = {
        "stage_0.5": '{"ideas_analyzed": <int>, "has_synthesis": true}',
        "stage_1": '{"topic": "...", "domain": "...", "scope": "..."}',
        "stage_2": '{"research_question": "...", "keywords": [...], "constraints": "..."}',
        "stage_3": '{"papers_found": <int>, "sources_used": ["s2", "arxiv", ...]}',
        "stage_3.5": '{"papers_read": <int>, "fulltext_count": <int>}',
        "stage_4": '{"papers_positioned": <int>}',
        "stage_5": '{"hooks": ["hook1", "hook2", ...]}',
        "stage_6": '{"citations_count": <int>}',
        "stage_7": '{"gaps": ["G1", "G2", "G3"], "evidence_gap_links": <int>}',
        "stage_8": '{"axes": [...], "combinations": [...]}',
        "stage_9": '{"ideas": ["idea1", ...]}',
        "stage_9.5": '{"rankings": [...], "comparisons": <int>}',
        "stage_10": '{"debate_rounds": <int>}',
        "stage_11": '{"assessment": "..."}',
    }
    return _FIELD_HINTS.get(stage, "")


def _get_reflections_dir() -> Path:
    """Get the reflections directory for the current workspace."""
    from autoidea.paths import get_active_workspace
    workspace = get_active_workspace()
    reflections_dir = Path(workspace) / "reflections"
    reflections_dir.mkdir(parents=True, exist_ok=True)
    return reflections_dir


def _coerce_artifact_int(value: Any, field: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"`{field}` must be an integer, not a boolean."
    if isinstance(value, int):
        return value, None
    if isinstance(value, float) and value.is_integer():
        return int(value), None
    if isinstance(value, str):
        try:
            return int(value), None
        except ValueError:
            return None, f"`{field}` must be an integer-compatible value, got {value!r}."
    return None, f"`{field}` must be an integer-compatible value, got {type(value).__name__}."


def _reflection_artifact_count_errors(stage: str, artifacts: dict[str, Any], workspace: Path) -> list[str]:
    """Validate reflection counts against canonical artifacts before writing.

    The stage gate runs artifact audit before the reflection is saved.  Without
    this pre-write check, a bad reflection can make the workspace invalid after
    the gate has already passed.
    """
    errors: list[str] = []

    def check(field: str, actual: int, source: str) -> None:
        if field not in artifacts or artifacts[field] is None:
            return
        reported, error = _coerce_artifact_int(artifacts[field], field)
        if error:
            errors.append(error)
            return
        if reported != actual:
            errors.append(
                f"`{field}` reports {reported}, but `{source}` has {actual}."
            )

    registry_path = workspace / "paper_registry.json"
    if stage == "stage_3" or "papers_found" in artifacts:
        try:
            registry = (
                json.loads(registry_path.read_text(encoding="utf-8"))
                if registry_path.exists()
                else None
            )
        except Exception as exc:
            errors.append(
                f"Could not validate `papers_found` from `paper_registry.json`: {exc}"
            )
        else:
            actual = len(registry) if isinstance(registry, list) else 0
            check("papers_found", actual, "paper_registry.json")

    deep_path = workspace / "paper_deep_reading.md"
    if stage == "stage_3.5" or "papers_read" in artifacts or "fulltext_count" in artifacts:
        deep_text = deep_path.read_text(encoding="utf-8", errors="replace") if deep_path.exists() else ""
        papers_read = len(re.findall(r"Full-text status\*\*:\s*(?:FULL-TEXT|ABSTRACT-ONLY)", deep_text, re.I))
        fulltext_count = len(re.findall(r"Full-text status\*\*:\s*FULL-TEXT", deep_text, re.I))
        check("papers_read", papers_read, "paper_deep_reading.md")
        check("fulltext_count", fulltext_count, "paper_deep_reading.md")

    positions_path = workspace / "paper_positions.json"
    if "papers_positioned" in artifacts:
        try:
            positions = json.loads(positions_path.read_text(encoding="utf-8")) if positions_path.exists() else None
        except Exception as exc:
            errors.append(f"Could not validate `papers_positioned` from `paper_positions.json`: {exc}")
        else:
            actual = len(positions) if isinstance(positions, list) else 0
            check("papers_positioned", actual, "paper_positions.json")

    evidence_path = workspace / "evidence_db.json"
    if "citations_count" in artifacts:
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else None
        except Exception as exc:
            errors.append(f"Could not validate `citations_count` from `evidence_db.json`: {exc}")
        else:
            claims = evidence.get("claims") if isinstance(evidence, dict) else None
            actual = len({c.get("citation_id") for c in claims if isinstance(c, dict) and c.get("citation_id")}) if isinstance(claims, list) else 0
            check("citations_count", actual, "evidence_db.json")

    gap_path = workspace / "research_gaps.json"
    if (
        stage == "stage_7"
        or "gaps_count" in artifacts
        or "evidence_gap_links" in artifacts
    ):
        try:
            gap_data = (
                json.loads(gap_path.read_text(encoding="utf-8"))
                if gap_path.exists()
                else None
            )
        except Exception as exc:
            errors.append(
                "Could not validate Stage 7 counts from "
                f"research_gaps.json: {exc}"
            )
        else:
            gaps = gap_data.get("gaps") if isinstance(gap_data, dict) else None
            actual_gaps = len(gaps) if isinstance(gaps, list) else 0
            actual_links = sum(
                len(gap.get("evidence_links", []))
                for gap in gaps or []
                if isinstance(gap, dict)
                and isinstance(gap.get("evidence_links"), list)
            )
            check("gaps_count", actual_gaps, "research_gaps.json")
            check(
                "evidence_gap_links",
                actual_links,
                "research_gaps.json",
            )

    return errors


@tool(parse_docstring=True)
def check_stage_gate(
    stage: str,
    evidence_json: str = "{}",
) -> str:
    """Check whether gate criteria for a pipeline stage are met.

    Validates the provided evidence against the gate criteria for the
    specified stage. Returns PASS or FAIL with detailed reasoning.

    Gate criteria include required fields, minimum counts, and
    quality thresholds specific to each stage.

    Expected ``evidence_json`` fields per stage:

    - **stage_1**:  ``{"topic": "...", "domain": "...", "scope": "..."}``
    - **stage_2**:  ``{"research_question": "...", "keywords": [...], "constraints": "..."}``
    - **stage_3**:  ``{"papers_found": <int>, "sources_used": ["s2", "arxiv", ...]}``
    - **stage_3.5**: ``{"papers_read": <int>, "fulltext_count": <int>}``
    - **stage_4**:  ``{"papers_positioned": <int>}``
    - **stage_5**:  ``{"hooks": ["hook1", "hook2", ...]}``
    - **stage_6**:  ``{"citations_count": <int>}``
    - **stage_7**:  ``{"gaps": ["G1", "G2", "G3"], "evidence_gap_links": <int>}``
    - **stage_8**:  ``{"axes": [...], "combinations": [...]}``
    - **stage_9**:  ``{"ideas": ["idea1", ...]}``
    - **stage_9.5**: ``{"rankings": [...], "comparisons": <int>}``
    - **stage_10**: ``{"debate_rounds": <int>}``
    - **stage_11**: ``{"assessment": "..."}``

    Numeric fields accept aliases (e.g. ``papers_found`` / ``paper_count`` /
    ``total_papers``) and string-encoded integers are coerced automatically.
    List fields accept JSON-encoded strings as well as native lists.

    Args:
        stage: Stage identifier (e.g. "stage_1", "stage_7", "stage_9.5").
        evidence_json: JSON object with evidence of stage completion. See the per-stage field list above.

    Returns:
        Markdown-formatted gate check result (PASS/FAIL with details).
    """
    gate = STAGE_GATES.get(stage)
    if gate is None:
        return f"Error: Unknown stage '{stage}'. Valid stages: {', '.join(STAGE_GATES.keys())}"

    gate_pass_key = _gate_pass_key(stage)
    # Every new check supersedes any earlier, unconsumed result for this stage.
    # A later malformed or failing retry must never leave a stale PASS usable.
    _gate_passed_stages.discard(gate_pass_key)
    if stage == "stage_3.5":
        gate = dict(gate)
        gate["min_papers_read"] = _configured_deep_reading_top_k()
    elif stage == "stage_9":
        gate = dict(gate)
        configured_max = _configured_max_ideas_to_generate()
        gate["min_ideas"] = min(int(gate.get("min_ideas", 5)), configured_max)
        gate["max_ideas"] = configured_max

    try:
        evidence = json.loads(evidence_json)
    except json.JSONDecodeError as e:
        return f"Error parsing evidence_json: {e}"

    failures: list[str] = []
    warnings: list[str] = []

    # Check required fields
    for field in gate.get("required_fields", []):
        if field not in evidence or not evidence[field]:
            failures.append(f"Missing required field: `{field}`")

    # Check minimum word count
    min_words = gate.get("min_word_count")
    if min_words:
        total_words = sum(
            len(str(v).split()) for v in evidence.values()
        )
        if total_words < min_words:
            failures.append(
                f"Insufficient detail: {total_words} words (minimum: {min_words})"
            )

    # Check minimum paper counts (aliases: papers_found, paper_count, total_papers, papers)
    min_papers = gate.get("min_papers")
    if min_papers:
        papers_found = _get_evidence_int(
            evidence, "papers_found", "paper_count", "total_papers", "papers",
        )
        if papers_found < min_papers:
            failures.append(
                f"Insufficient papers: {papers_found} (minimum: {min_papers})"
            )
        if stage == "stage_3":
            from autoidea.paths import get_active_workspace

            registry_path = Path(get_active_workspace()) / "paper_registry.json"
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                actual_papers = len(registry) if isinstance(registry, list) else 0
            except (OSError, UnicodeError, json.JSONDecodeError):
                actual_papers = 0
            if actual_papers < min_papers:
                failures.append(
                    "Canonical paper registry is incomplete: "
                    f"{actual_papers} paper(s) on disk (minimum: {min_papers})."
                )
            if papers_found != actual_papers:
                failures.append(
                    f"`papers_found` reports {papers_found}, but "
                    f"`paper_registry.json` has {actual_papers}."
                )

    # Check minimum papers read (aliases: papers_read, papers_summarized, deep_read_count)
    min_papers_read = gate.get("min_papers_read")
    if min_papers_read:
        papers_read = _get_evidence_int(
            evidence, "papers_read", "papers_summarized", "deep_read_count",
        )
        if papers_read < min_papers_read:
            failures.append(
                f"Insufficient papers read: {papers_read} (minimum: {min_papers_read})"
            )

    # Check positioned papers (aliases: papers_positioned, positioned_count, papers_analyzed, positioned)
    min_positioned = gate.get("min_papers_positioned")
    if min_positioned:
        positioned = _get_evidence_int(
            evidence, "papers_positioned", "positioned_count",
            "papers_analyzed", "positioned",
        )
        if positioned < min_positioned:
            failures.append(
                f"Insufficient positioned papers: {positioned} (minimum: {min_positioned})"
            )

    # Check minimum hooks (aliases: hooks, research_hooks, hook_list)
    min_hooks = gate.get("min_hooks")
    if min_hooks:
        hooks_list = _get_evidence_list(
            evidence, "hooks", "research_hooks", "hook_list",
        )
        if len(hooks_list) < min_hooks:
            failures.append(
                f"Insufficient hooks: {len(hooks_list)} (minimum: {min_hooks})"
            )

    # Check minimum citations (aliases: citations_count, citations_registered, total_citations, citations)
    min_citations = gate.get("min_citations")
    if min_citations:
        citations = _get_evidence_int(
            evidence, "citations_count", "citations_registered",
            "total_citations", "citations",
        )
        if citations < min_citations:
            failures.append(
                f"Insufficient citations: {citations} (minimum: {min_citations})"
            )

    # Check minimum gaps (aliases: gaps, identified_gaps, gap_list)
    min_gaps = gate.get("min_gaps")
    if min_gaps:
        gaps_list = _get_evidence_list(
            evidence, "gaps", "identified_gaps", "gap_list",
        )
        if len(gaps_list) < min_gaps:
            failures.append(
                f"Insufficient gaps: {len(gaps_list)} (minimum: {min_gaps})"
            )

    min_gap_links = gate.get("min_evidence_gap_links")
    if min_gap_links:
        gap_links = _get_evidence_int(
            evidence,
            "evidence_gap_links",
            "gap_links_count",
        )
        if gap_links < min_gap_links:
            failures.append(
                "Insufficient Evidence-to-Gap links: "
                f"{gap_links} (minimum: {min_gap_links})"
            )

    if stage == "stage_7":
        from autoidea.paths import get_active_workspace
        from autoidea.tools.artifact_audit import validate_stage7_artifacts

        workspace = Path(get_active_workspace())
        stage7_issues = validate_stage7_artifacts(workspace)
        for issue in stage7_issues:
            message = f"{issue.get('code')}: {issue.get('message')}"
            if issue.get("severity") == "ERROR":
                failures.append(message)
            else:
                warnings.append(message)

        gap_path = workspace / "research_gaps.json"
        if gap_path.is_file() and not any(
            issue.get("severity") == "ERROR" for issue in stage7_issues
        ):
            try:
                gap_data = json.loads(gap_path.read_text(encoding="utf-8"))
                actual_gaps = gap_data.get("gaps", [])
                actual_gap_ids = [
                    str(gap.get("gap_id") or "")
                    for gap in actual_gaps
                    if isinstance(gap, dict)
                ]
                actual_links = sum(
                    len(gap.get("evidence_links", []))
                    for gap in actual_gaps
                    if isinstance(gap, dict)
                    and isinstance(gap.get("evidence_links"), list)
                )
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                pass
            else:
                reported_gaps = [str(value) for value in gaps_list]
                if reported_gaps != actual_gap_ids:
                    failures.append(
                        "`gaps` must exactly match the ordered gap_id values in "
                        "research_gaps.json."
                    )
                if gap_links != actual_links:
                    failures.append(
                        "`evidence_gap_links` reports "
                        f"{gap_links}, but research_gaps.json has {actual_links}."
                    )

    # Check minimum ideas (aliases: ideas, generated_ideas, idea_list)
    min_ideas = gate.get("min_ideas")
    if min_ideas:
        ideas_list = _get_evidence_list(
            evidence, "ideas", "generated_ideas", "idea_list",
        )
        if len(ideas_list) < min_ideas:
            failures.append(
                f"Insufficient ideas: {len(ideas_list)} (minimum: {min_ideas})"
            )

    max_ideas = gate.get("max_ideas")
    if max_ideas and len(ideas_list) > max_ideas:
        failures.append(
            f"Too many ideas: {len(ideas_list)} "
            f"(configured maximum: {max_ideas})"
        )

    # Check minimum axes (aliases: axes, design_axes, axis_list)
    min_axes = gate.get("min_axes")
    if min_axes:
        axes_list = _get_evidence_list(
            evidence, "axes", "design_axes", "axis_list",
        )
        if len(axes_list) < min_axes:
            failures.append(
                f"Insufficient design axes: {len(axes_list)} (minimum: {min_axes})"
            )

    # Check minimum comparisons for Elo (aliases: comparisons, comparison_count, total_comparisons)
    min_comparisons = gate.get("min_comparisons")
    if min_comparisons:
        comparisons = _get_evidence_int(
            evidence, "comparisons", "comparison_count", "total_comparisons",
        )
        if comparisons < min_comparisons:
            failures.append(
                f"Insufficient comparisons: {comparisons} (minimum: {min_comparisons})"
            )

    # Check debate rounds (aliases: debate_rounds, rounds, num_rounds)
    min_rounds = gate.get("min_rounds")
    if min_rounds:
        rounds = _get_evidence_int(
            evidence, "debate_rounds", "rounds", "num_rounds",
        )
        if rounds < min_rounds:
            failures.append(
                f"Insufficient debate rounds: {rounds} (minimum: {min_rounds})"
            )

    # In regular CLI mode the agent still owns the checkpoint prompt. The
    # structured Web runner enables a code-level interrupt below so tool
    # auto-approval can never bypass research decisions.
    if (
        gate.get("hitl_checkpoint")
        and _web_checkpoint_events_file is None
        and not _auto_approve_enabled()
        and not evidence.get("user_approved")
    ):
        warnings.append("HITL checkpoint: awaiting user approval before proceeding.")

    # Build result
    passed = len(failures) == 0

    # ── Retry counter: prevent infinite loops ─────────────────────────
    if not passed:
        _gate_fail_counts[stage] = _gate_fail_counts.get(stage, 0) + 1
        fail_count = _gate_fail_counts[stage]
        logger.warning(
            "Stage gate %s FAIL (attempt %d/%d)",
            stage, fail_count, MAX_GATE_RETRIES,
        )

        if fail_count >= MAX_GATE_RETRIES:
            # Force-pass to break the loop
            logger.error(
                "Stage gate %s exceeded MAX_GATE_RETRIES=%d — force-passing "
                "to prevent infinite loop. Original failures: %s",
                stage, MAX_GATE_RETRIES, "; ".join(failures),
            )
            passed = True
            warnings.append(
                f"⚠️ FORCE-PASSED after {fail_count} consecutive failures "
                f"(max retries: {MAX_GATE_RETRIES}). The following issues "
                f"were NOT resolved but the pipeline will continue to avoid "
                f"an infinite loop:"
            )
            for f_msg in failures:
                warnings.append(f"  - {f_msg}")
            failures = []  # Clear failures since we're force-passing
    else:
        # Reset counter on success
        _gate_fail_counts.pop(stage, None)

    checkpoint_confirmed = False
    if passed and gate.get("hitl_checkpoint") and _web_checkpoint_events_file is not None:
        if _web_checkpoint_is_approved(stage):
            checkpoint_confirmed = True
        else:
            approved, feedback = _request_web_checkpoint(stage)
            if approved:
                checkpoint_confirmed = True
            else:
                passed = False
                detail = feedback or "No revision details were supplied."
                failures.append(
                    "Mandatory human checkpoint requested a revision or re-run. "
                    f"User feedback: {detail}"
                )

    parts = [
        f"## Stage Gate Check: {stage}",
        f"**Stage**: {gate['name']}",
        f"**Status**: {'PASS ✓' if passed else 'FAIL ✗'}",
        "",
    ]

    # Show retry count when failing
    if not passed:
        fail_count = _gate_fail_counts.get(stage, 0)
        if fail_count:
            parts.append(
                f"**Attempt**: {fail_count}/{MAX_GATE_RETRIES} "
                f"(will force-pass after {MAX_GATE_RETRIES} failures)\n"
            )

    if failures:
        parts.append("### Failures")
        for f in failures:
            parts.append(f"- {f}")
        parts.append("")

        # ── Remediation guidance with expected field names ────────────
        hint = _get_expected_fields(stage)
        parts.append("### How to Fix")
        parts.append(
            "Fix the specific issues listed above, then call "
            "`check_stage_gate` again with corrected `evidence_json`."
        )
        if hint:
            parts.append(f"\n**Expected evidence_json format for {stage}**:")
            parts.append(f"```json\n{hint}\n```")
        parts.append(
            "\n> **Tip**: Do NOT re-read workspace files or rewrite MEMORY.md "
            "between retries. Just fix the evidence and retry immediately."
        )
        parts.append("")

    if warnings:
        parts.append("### Warnings")
        for w in warnings:
            parts.append(f"- {w}")
        parts.append("")

    if checkpoint_confirmed:
        parts.append("Mandatory Web checkpoint: APPROVED by the user.")
        parts.append("")

    # Artifact integrity is checked from files on disk, not from self-reported
    # evidence_json.  These stages are where historical regressions corrupted
    # final reports: paper-ID drift, unverifiable full-text claims, local
    # citation IDs, count mismatches, and nested workspace ambiguity.
    if passed and stage in {"stage_3", "stage_3.5", "stage_4", "stage_6", "stage_7", "stage_12"}:
        try:
            from autoidea.paths import get_active_workspace
            from autoidea.tools.artifact_audit import audit_workspace

            artifact_report = audit_workspace(
                get_active_workspace(),
                verify_urls=stage in {"stage_6", "stage_12"},
            )
            if artifact_report.has_errors:
                passed = False
                parts[2] = "**Status**: FAIL ✗"
                parts.append("")
                parts.append("### Artifact Integrity Failures")
                for issue in artifact_report.issues:
                    if issue.severity.value == "ERROR":
                        path = f" (`{issue.path}`)" if issue.path else ""
                        parts.append(f"- **{issue.code}**{path}: {issue.message}")
                parts.append("")
                parts.append(
                    "Fix the artifact files and run `audit_workspace_artifacts` "
                    "before retrying this stage gate."
                )
        except Exception as exc:
            passed = False
            parts[2] = "**Status**: FAIL ✗"
            parts.append("")
            parts.append("### Artifact Integrity Failures")
            parts.append(f"- Artifact audit could not run: {exc}")

    if passed:
        _gate_passed_stages.add(gate_pass_key)
        if not warnings:
            parts.append("All gate criteria met. Ready to proceed to next stage.")
    else:
        _gate_passed_stages.discard(gate_pass_key)

    return "\n".join(parts)


@tool(parse_docstring=True)
def save_stage_reflection(
    stage: str,
    reflection: str,
    artifacts_json: str = "{}",
) -> str:
    """Save a reflection and artifacts for a completed pipeline stage.

    Records what was learned, decisions made, and key artifacts produced
    at each stage. This creates an audit trail for the research process.

    Args:
        stage: Stage identifier (e.g. "stage_1", "stage_7").
        reflection: Free-text reflection on what was accomplished, challenges encountered, and decisions made at this stage.
        artifacts_json: JSON object listing key artifacts produced, e.g. {"papers_found": 15, "top_papers": ["title1", "title2"]}.

    Returns:
        Confirmation message with file path.
    """
    try:
        artifacts = json.loads(artifacts_json)
    except json.JSONDecodeError:
        artifacts = {}
    if not isinstance(artifacts, dict):
        artifacts = {}

    reflections_dir = _get_reflections_dir()
    workspace = reflections_dir.parent
    count_errors = _reflection_artifact_count_errors(stage, artifacts, workspace)
    if count_errors:
        details = "\n".join(f"- {error}" for error in count_errors)
        return (
            f"Error: Refusing to save inconsistent reflection for **{stage}**.\n"
            f"{details}\n"
            "Fix `artifacts_json` to match the canonical artifact files, then retry."
        )

    gate_pass_key = _gate_pass_key(stage)
    if gate_pass_key not in _gate_passed_stages:
        return (
            f"Error: Refusing to save reflection for **{stage}** because its "
            "stage gate has not passed.\n"
            "Run `check_stage_gate` for this stage and save the reflection only "
            "after it returns PASS."
        )

    record = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate_passed": True,
        "reflection": reflection,
        "artifacts": artifacts,
    }

    filepath = reflections_dir / f"{stage}_reflection.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # The permission is single-use.  Any later reflection rewrite must be
    # preceded by a fresh gate check against the then-current artifacts.
    _gate_passed_stages.discard(gate_pass_key)

    return (
        f"Stage reflection saved for **{stage}** "
        f"({STAGE_GATES.get(stage, {}).get('name', 'Unknown')}).\n"
        f"File: `{filepath}`"
    )


@tool(parse_docstring=True)
def list_stage_reflections() -> str:
    """List all saved stage reflections for the current session.

    Returns a summary of all stage reflections that have been saved,
    showing which stages have been completed and their key insights.

    Returns:
        Markdown-formatted list of stage reflections.
    """
    reflections_dir = _get_reflections_dir()

    if not reflections_dir.exists():
        return "No stage reflections found for this session."

    files = sorted(reflections_dir.glob("stage_*_reflection.json"))
    if not files:
        return "No stage reflections found for this session."

    parts = ["## Stage Reflections\n"]
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            stage = data.get("stage", "unknown")
            timestamp = data.get("timestamp", "")
            reflection = data.get("reflection", "")[:200]
            parts.append(
                f"### {stage} — {STAGE_GATES.get(stage, {}).get('name', 'Unknown')}\n"
                f"*{timestamp}*\n\n"
                f"{reflection}{'...' if len(data.get('reflection', '')) > 200 else ''}\n"
            )
        except Exception:
            parts.append(f"- Error reading {f.name}")

    return "\n".join(parts)
