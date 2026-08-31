"""Observed pipeline state and completion proofs for the Web workbench.

The CLI's ``pipeline_state.json`` is useful provenance, but it can be stale when a
process crashes between artifact writes.  The Web UI therefore derives its primary
status from the selected run directory and treats persisted pipeline state as
supporting evidence only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..progress import PIPELINE_STAGES, build_stage_progress

REQUIRED_CHECKPOINTS: tuple[str, ...] = ("stage_7", "stage_9", "stage_10")

_MIN_TEXT_LENGTHS: dict[str, int] = {
    "research_brief.md": 80,
    "task_formalization.md": 80,
    "literature_survey.md": 80,
    "paper_deep_reading.md": 200,
    "expanded_literature.md": 200,
    "knowledge_synthesis.md": 200,
    "debate_log.md": 120,
    "final_report.md": 200,
}


def inspect_pipeline(
    workspace: str | Path,
    *,
    run_status: str = "",
    checkpoint_events: Iterable[str] = (),
    include_audit: bool = False,
    audit_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, JSON-safe view of pipeline progress.

    ``checkpoint_events`` contains checkpoint stage IDs that have a persisted
    structured response.  They are required for a Web run to be called fully
    verified, even when all files happen to exist.
    """
    root = Path(workspace).expanduser().resolve()
    persisted = _load_object(root / "pipeline_state.json")
    run_marker = _load_object(root / "run_status.json")
    marker_stage = str(run_marker.get("stage") or "")
    marker_detail = str(run_marker.get("detail") or "")
    marker_status = str(run_marker.get("status") or "").casefold()
    run_status = str(run_status or marker_status).casefold()
    active_detail = ""
    observed_checkpoint_events = set(checkpoint_events)

    stage_rows: list[dict[str, Any]] = []
    first_incomplete = ""
    completed_count = 0
    for spec in PIPELINE_STAGES:
        artifacts = list(spec["artifacts"])
        checks = [_artifact_check(root / name) for name in artifacts]
        artifacts_ready = all(check["ready"] for check in checks)
        reflection_name = f"reflections/{spec['id']}_reflection.json"
        reflection = _artifact_check(root / reflection_name)
        reflection_record = _load_object(root / reflection_name)
        gate_passed = bool(
            reflection["ready"] and reflection_record.get("gate_passed") is True
        )
        validation_issues: list[dict[str, str]] = []
        invalid_artifacts = [
            check["name"]
            for check in checks
            if check["exists"] and not check["ready"]
        ]
        if spec["id"] == "stage_7" and artifacts_ready:
            try:
                from autoidea.tools.artifact_audit import validate_stage7_artifacts

                validation_issues = validate_stage7_artifacts(root)
            except Exception as exc:  # noqa: BLE001  # pragma: no cover
                validation_issues = [
                    {
                        "severity": "ERROR",
                        "code": "STAGE7_VALIDATION_ERROR",
                        "message": str(exc),
                        "path": str(root),
                    }
                ]
            if any(issue.get("severity") == "ERROR" for issue in validation_issues):
                artifacts_ready = False
                invalid_artifacts.append("research_gaps.json")
        stage_complete = artifacts_ready and (
            spec["id"] != "stage_12" or gate_passed
        )
        if stage_complete:
            completed_count += 1
        elif not first_incomplete:
            first_incomplete = str(spec["id"])
        stage_rows.append(
            {
                "id": spec["id"],
                "number": spec["number"],
                "name": spec["name"],
                "checkpoint": bool(spec.get("checkpoint")),
                "checkpoint_recorded": spec["id"] in observed_checkpoint_events,
                "status": "complete" if stage_complete else "pending",
                "required_artifacts": artifacts,
                "missing_artifacts": [
                    check["name"] for check in checks if not check["exists"]
                ],
                "invalid_artifacts": sorted(set(invalid_artifacts)),
                "validation_issues": validation_issues,
                "reflection_present": bool(reflection["ready"]),
                "gate_passed": gate_passed,
                "reflection_path": reflection_name if reflection["exists"] else "",
            }
        )

    stage_ids = [str(spec["id"]) for spec in PIPELINE_STAGES]
    active_stage = first_incomplete
    if marker_stage in stage_ids and first_incomplete:
        marker_index = stage_ids.index(marker_stage)
        incomplete_index = stage_ids.index(first_incomplete)
        marker_is_finished = marker_status in {"passed", "complete", "completed"}
        if marker_index < incomplete_index and not marker_is_finished:
            active_stage = marker_stage
        elif marker_index == incomplete_index:
            active_stage = marker_stage
    if active_stage == marker_stage:
        active_detail = marker_detail
    if run_status in {"running", "queued"} and active_stage:
        _set_stage_status(stage_rows, active_stage, "running")
    elif run_status in {"waiting_for_input", "checkpoint_reached"}:
        checkpoint_stage = infer_checkpoint_stage(root)
        if checkpoint_stage:
            _set_stage_status(stage_rows, checkpoint_stage, "waiting_for_input")
            active_stage = checkpoint_stage
    elif run_status in {"failed", "stopped", "stale"} and active_stage:
        _set_stage_status(stage_rows, active_stage, run_status)

    final_report = _artifact_check(root / "final_report.md")
    required_artifacts_ready = all(
        not row["missing_artifacts"] and not row["invalid_artifacts"]
        for row in stage_rows
    )
    missing_reflections = [
        str(row["id"])
        for row in stage_rows
        if not row["reflection_present"]
    ]
    missing_gate_proofs = [
        "stage_12"
        for row in stage_rows
        if row["id"] == "stage_12" and not row["gate_passed"]
    ]
    missing_checkpoints = [
        stage for stage in REQUIRED_CHECKPOINTS if stage not in observed_checkpoint_events
    ]

    audit_issues: list[dict[str, str]] = []
    audit_passed: bool | None = None
    if include_audit and required_artifacts_ready:
        try:
            from autoidea.tools.artifact_audit import audit_workspace

            expected_top_k: int | None = None
            if audit_parameters:
                candidate = audit_parameters.get("deep_reading_top_k")
                try:
                    parsed = int(candidate)
                except (TypeError, ValueError):
                    parsed = 0
                if parsed > 0:
                    expected_top_k = parsed
            report = audit_workspace(
                root,
                verify_urls=False,
                deep_reading_top_k=expected_top_k,
            )
            audit_issues = [
                {
                    "severity": issue.severity.value,
                    "code": issue.code,
                    "message": issue.message,
                    "path": issue.path,
                }
                for issue in report.issues
            ]
            audit_passed = not report.has_errors
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - external audit guard
            audit_passed = False
            audit_issues = [
                {
                    "severity": "ERROR",
                    "code": "AUDIT_UNAVAILABLE",
                    "message": str(exc),
                    "path": str(root),
                }
            ]

    finalization_ready = bool(
        required_artifacts_ready
        and final_report["ready"]
        and not missing_reflections
        and not missing_gate_proofs
        and audit_passed is True
    )
    verified = bool(finalization_ready and not missing_checkpoints)
    if not finalization_ready and completed_count == len(stage_rows):
        # The report file appearing on disk is not the end of Stage 12. Keep
        # the final stage active until its reflection and deterministic audit
        # both pass, so the UI never advertises a premature 14/14 completion.
        final_status = (
            "running"
            if run_status in {"running", "queued"}
            else run_status
            if run_status in {"failed", "stopped", "stale"}
            else "pending"
        )
        _set_stage_status(stage_rows, "stage_12", final_status)
        completed_count -= 1
        first_incomplete = "stage_12"
        if not active_stage:
            active_stage = "stage_12"
            if marker_stage == "stage_12":
                active_detail = marker_detail
    if verified:
        # A stale run_status.json often still names stage_12 because the agent
        # writes it before producing and auditing the final report.  Completion
        # proof is authoritative once every artifact, checkpoint, and audit is
        # satisfied, so do not keep presenting that stale marker as active.
        active_stage = ""
        active_detail = ""
    persisted_next = str(persisted.get("next_stage") or "")

    active_row = next((row for row in stage_rows if row["id"] == active_stage), None)
    active_progress = build_stage_progress(
        root,
        active_stage,
        status=str(active_row.get("status") if active_row else run_status),
        parameters=audit_parameters,
        runtime=run_marker.get("progress") if isinstance(run_marker.get("progress"), dict) else None,
    ) if active_stage else {}
    if active_row is not None:
        active_row["progress"] = active_progress

    return {
        "source": "observed",
        "workspace": str(root),
        "updated_at": str(run_marker.get("updated_at") or persisted.get("updated_at") or ""),
        "active_stage": active_stage,
        "active_detail": active_detail,
        "active_progress": active_progress,
        "last_completed_stage": _last_contiguous_complete(stage_rows),
        "next_stage": first_incomplete or "complete",
        "persisted_next_stage": persisted_next,
        "persisted_state_stale": bool(
            persisted_next and persisted_next != (first_incomplete or "complete")
        ),
        "completed_count": completed_count,
        "total_stages": len(stage_rows),
        "percent": round((completed_count / len(stage_rows)) * 100),
        "stages": stage_rows,
        "completion": {
            "verified": verified,
            "required_artifacts_ready": required_artifacts_ready,
            "final_report_present": bool(final_report["ready"]),
            "final_report_path": "final_report.md" if final_report["exists"] else "",
            "checkpoint_events": sorted(observed_checkpoint_events),
            "missing_checkpoints": missing_checkpoints,
            "reflections_ready": not missing_reflections and not missing_gate_proofs,
            "missing_reflections": missing_reflections,
            "stage_12_gate_passed": not missing_gate_proofs,
            "missing_gate_proofs": missing_gate_proofs,
            "audit_passed": audit_passed,
            "audit_issues": audit_issues,
        },
    }


def infer_checkpoint_stage(workspace: str | Path) -> str:
    """Infer which mandatory checkpoint a newly emitted interaction belongs to."""
    root = Path(workspace).expanduser().resolve()
    if _artifact_check(root / "debate_log.md")["ready"] and _artifact_check(
        root / "idea_reviews.json"
    )["ready"] and not _artifact_check(root / "feasibility_assessments.json")["ready"]:
        return "stage_10"
    if _artifact_check(root / "raw_ideas.json")["ready"] and not _artifact_check(
        root / "tournament_rankings.json"
    )["ready"]:
        return "stage_9"
    stage7_ready = (
        _artifact_check(root / "knowledge_synthesis.md")["ready"]
        and _artifact_check(root / "research_gaps.json")["ready"]
    )
    if stage7_ready:
        try:
            from autoidea.tools.artifact_audit import validate_stage7_artifacts

            stage7_ready = not any(
                issue.get("severity") == "ERROR"
                for issue in validate_stage7_artifacts(root)
            )
        except Exception:  # pragma: no cover - conservative checkpoint guard
            stage7_ready = False
    if stage7_ready and not _artifact_check(root / "design_space.json")["ready"]:
        return "stage_7"
    return ""


def checkpoint_events_from_events(events: Iterable[dict[str, Any]]) -> list[str]:
    """Return checkpoint IDs whose structured interaction was resolved."""
    requested: dict[str, str] = {}
    resolved: set[str] = set()
    for event in events:
        event_id = str(event.get("interaction_id") or "")
        stage = str(event.get("checkpoint_stage") or "")
        if event.get("type") == "interaction_requested" and event_id and stage:
            requested[event_id] = stage
        elif event.get("type") == "interaction_resolved" and event_id in requested:
            response = event.get("response")
            if not isinstance(response, dict) or response.get("approved") is not False:
                resolved.add(requested[event_id])
    return sorted(resolved)


def _artifact_check(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    result = {"name": path.name, "exists": exists, "ready": False, "reason": "missing"}
    if not exists:
        return result
    try:
        size = path.stat().st_size
        if size <= 0:
            result["reason"] = "empty"
            return result
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if data in ({}, []):
                result["reason"] = "empty_data"
                return result
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            minimum = _MIN_TEXT_LENGTHS.get(path.name, 1)
            if len(text) < minimum:
                result["reason"] = "too_small"
                return result
    except (OSError, json.JSONDecodeError, UnicodeError):
        result["reason"] = "invalid"
        return result
    result["ready"] = True
    result["reason"] = "ready"
    return result


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _set_stage_status(rows: list[dict[str, Any]], stage_id: str, status: str) -> None:
    for row in rows:
        if row["id"] == stage_id:
            row["status"] = status
            return


def _last_contiguous_complete(rows: list[dict[str, Any]]) -> str:
    last = ""
    for row in rows:
        if row["status"] not in {"complete", "waiting_for_input"}:
            break
        last = str(row["id"])
    return last
