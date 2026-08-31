"""Deterministic pipeline state inspection for AutoIdea workspaces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


@dataclass(frozen=True)
class StageSpec:
    stage: str
    name: str
    artifacts: tuple[str, ...]


STAGES: tuple[StageSpec, ...] = (
    StageSpec("stage_1", "Requirement Intake", ("research_brief.md",)),
    StageSpec("stage_2", "Task Formalization", ("task_formalization.md",)),
    StageSpec("stage_3", "Literature Survey", ("literature_survey.md", "paper_registry.json")),
    StageSpec("stage_3.5", "Paper Deep Reading", ("paper_deep_reading.md",)),
    StageSpec("stage_4", "Position-First Analysis", ("paper_positions.json",)),
    StageSpec("stage_5", "Hook-Driven Expansion", ("expanded_literature.md",)),
    StageSpec("stage_6", "Evidence Binding", ("evidence_db.json",)),
    StageSpec(
        "stage_7",
        "Knowledge Synthesis",
        ("knowledge_synthesis.md", "research_gaps.json"),
    ),
    StageSpec("stage_8", "Design Space Definition", ("design_space.json",)),
    StageSpec("stage_9", "Idea Generation", ("raw_ideas.json",)),
    StageSpec("stage_9.5", "Elo Tournament", ("tournament_rankings.json",)),
    StageSpec("stage_10", "Adversarial Debate", ("debate_log.md", "idea_reviews.json")),
    StageSpec("stage_11", "Feasibility Assessment", ("feasibility_assessments.json",)),
    StageSpec("stage_12", "Final Report", ("final_report.md",)),
)


def _workspace() -> Path:
    from autoidea.paths import get_active_workspace

    return Path(get_active_workspace())


def _reflection_path(stage: str, workspace: Path | None = None) -> Path:
    root = workspace if workspace is not None else _workspace()
    return root / "reflections" / f"{stage}_reflection.json"


def _artifact_exists(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _stage3_issues(
    ws: Path,
    expected_paper_count: int | None = None,
) -> list[dict[str, str]]:
    try:
        from autoidea.tools.artifact_audit import validate_stage3_artifacts

        issues = validate_stage3_artifacts(ws)
        if expected_paper_count is not None:
            expected = max(1, int(expected_paper_count))
            registry = json.loads(
                (ws / "paper_registry.json").read_text(encoding="utf-8")
            )
            actual = len(registry) if isinstance(registry, list) else 0
            if actual < expected:
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": "PAPER_REGISTRY_BELOW_TARGET",
                        "message": (
                            f"paper_registry.json has {actual} paper(s), "
                            f"expected at least {expected}."
                        ),
                        "path": str(ws / "paper_registry.json"),
                    }
                )
        return issues
    except Exception as exc:
        return [
            {
                "severity": "ERROR",
                "code": "STAGE3_VALIDATION_ERROR",
                "message": f"Could not validate Stage 3 artifacts: {exc}",
                "path": str(ws),
            }
        ]


def _stage35_issues(ws: Path) -> list[dict[str, str]]:
    path = ws / "paper_deep_reading.md"
    if not path.exists():
        return []
    try:
        import os

        env_value = os.getenv("AUTOIDEA_DEEP_READING_TOP_K")
        if env_value:
            expected = int(env_value)
        else:
            from autoidea.config import load_config

            expected = int(getattr(load_config(), "deep_reading_top_k", 20) or 20)
    except Exception:
        expected = 20

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [
            {
                "severity": "ERROR",
                "code": "STAGE35_VALIDATION_ERROR",
                "message": f"Could not read paper_deep_reading.md: {exc}",
                "path": str(path),
            }
        ]
    actual = len(
        re.findall(
            r"Full-text status\*\*:\s*(?:FULL-TEXT|ABSTRACT-ONLY)",
            text,
            re.I,
        )
    )
    if actual < expected:
        return [
            {
                "severity": "ERROR",
                "code": "DEEP_READING_INCOMPLETE",
                "message": (
                    f"paper_deep_reading.md has {actual} reading block(s), "
                    f"expected at least {expected} from deep_reading_top_k."
                ),
                "path": str(path),
            }
        ]
    return []


def _stage7_issues(ws: Path) -> list[dict[str, str]]:
    try:
        from autoidea.tools.artifact_audit import validate_stage7_artifacts

        return [
            issue
            for issue in validate_stage7_artifacts(ws)
            if issue.get("severity") == "ERROR"
        ]
    except Exception as exc:
        return [
            {
                "severity": "ERROR",
                "code": "STAGE7_VALIDATION_ERROR",
                "message": f"Could not validate Stage 7 artifacts: {exc}",
                "path": str(ws),
            }
        ]


def _build_state(
    workspace: str | Path | None = None,
    *,
    target_paper_count: int | None = None,
) -> dict[str, Any]:
    ws = (
        Path(workspace).expanduser().resolve()
        if workspace is not None
        else _workspace()
    )
    stages: dict[str, Any] = {}
    last_completed = ""
    next_stage = ""
    contiguous_open = True

    for spec in STAGES:
        missing = [
            artifact
            for artifact in spec.artifacts
            if not _artifact_exists(ws / artifact)
        ]
        reflection = _reflection_path(spec.stage, ws)
        has_reflection = _artifact_exists(reflection)
        complete = not missing
        validation_issues: list[dict[str, str]] = []
        if spec.stage == "stage_3" and complete:
            validation_issues = _stage3_issues(ws, target_paper_count)
            complete = not validation_issues
        if spec.stage == "stage_3.5" and complete:
            validation_issues = _stage35_issues(ws)
            complete = not validation_issues
        if spec.stage == "stage_7" and complete:
            validation_issues = _stage7_issues(ws)
            complete = not validation_issues
        status = "complete" if complete else "pending"
        if validation_issues:
            status = "invalid"
        if complete and contiguous_open:
            last_completed = spec.stage
        elif not complete and not next_stage:
            next_stage = spec.stage
            contiguous_open = False
        stages[spec.stage] = {
            "name": spec.name,
            "status": status,
            "required_artifacts": list(spec.artifacts),
            "missing_artifacts": missing,
            "has_reflection": has_reflection,
            "reflection_file": str(reflection.relative_to(ws)) if has_reflection else "",
            "validation_issues": validation_issues,
        }

    if not next_stage:
        next_stage = "complete"

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(ws),
        "last_completed_stage": last_completed,
        "next_stage": next_stage,
        "stages": stages,
    }


def _write_state(state: dict[str, Any]) -> Path:
    path = _workspace() / "pipeline_state.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _format_state_markdown(state: dict[str, Any]) -> str:
    next_stage = state["next_stage"]
    lines = [
        "# Pipeline State",
        "",
        f"- Last completed stage: {state['last_completed_stage'] or 'none'}",
        f"- Next stage: {next_stage}",
        "- State file: pipeline_state.json",
        "",
        "## Stage Status",
    ]
    for stage, info in state["stages"].items():
        missing = info["missing_artifacts"]
        missing_text = ", ".join(missing) if missing else "none"
        lines.append(
            f"- {stage} ({info['name']}): {info['status']}; "
            f"missing_artifacts={missing_text}; "
            f"reflection={'yes' if info['has_reflection'] else 'no'}"
        )
        if stage == "stage_3" and info.get("validation_issues"):
            lines.append("  Stage 3 artifacts are structurally invalid:")
            for issue in info["validation_issues"][:5]:
                lines.append(f"  - {issue.get('code')}: {issue.get('message')}")
        if stage in {"stage_3.5", "stage_7"} and info.get("validation_issues"):
            lines.append(f"  {stage.replace('_', ' ').title()} artifacts are structurally invalid:")
            for issue in info["validation_issues"][:5]:
                lines.append(f"  - {issue.get('code')}: {issue.get('message')}")
        if stage == next_stage:
            break
    if next_stage != "complete":
        artifacts = state["stages"][next_stage]["required_artifacts"]
        lines.extend(
            [
                "",
                "## Resume Instruction",
                (
                    f"Resume from {next_stage}. Produce required artifact(s): "
                    f"{', '.join(artifacts)}. Do not rerun completed stages unless "
                    "their artifacts fail audit."
                ),
            ]
        )
    return "\n".join(lines)


@tool(parse_docstring=True)
def inspect_pipeline_state() -> str:
    """Inspect workspace artifacts and write deterministic pipeline_state.json.

    Use this before recovery or long unattended runs. It determines the next
    stage from files on disk rather than relying on old chat history.

    Returns:
        Concise markdown summary with the last completed stage and next stage.
    """
    state = _build_state()
    _write_state(state)
    return _format_state_markdown(state)
