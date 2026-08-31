"""Shared, artifact-grounded progress reporting for CLI and Web runs."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


PIPELINE_STAGES: tuple[dict[str, Any], ...] = (
    {"id": "stage_1", "number": "01", "name": "Requirement intake", "artifacts": ("research_brief.md",)},
    {"id": "stage_2", "number": "02", "name": "Task formalization", "artifacts": ("task_formalization.md",)},
    {"id": "stage_3", "number": "03", "name": "Literature survey", "artifacts": ("literature_survey.md", "paper_registry.json")},
    {"id": "stage_3.5", "number": "03.5", "name": "Paper deep reading", "artifacts": ("paper_deep_reading.md",)},
    {"id": "stage_4", "number": "04", "name": "Position analysis", "artifacts": ("paper_positions.json",)},
    {"id": "stage_5", "number": "05", "name": "Hook-driven expansion", "artifacts": ("expanded_literature.md",)},
    {"id": "stage_6", "number": "06", "name": "Evidence binding", "artifacts": ("evidence_db.json",)},
    {"id": "stage_7", "number": "07", "name": "Knowledge synthesis", "artifacts": ("knowledge_synthesis.md", "research_gaps.json"), "checkpoint": True},
    {"id": "stage_8", "number": "08", "name": "Design space", "artifacts": ("design_space.json",)},
    {"id": "stage_9", "number": "09", "name": "Idea generation", "artifacts": ("raw_ideas.json",), "checkpoint": True},
    {"id": "stage_9.5", "number": "09.5", "name": "Elo tournament", "artifacts": ("tournament_rankings.json",)},
    {"id": "stage_10", "number": "10", "name": "Adversarial debate", "artifacts": ("debate_log.md", "idea_reviews.json"), "checkpoint": True},
    {"id": "stage_11", "number": "11", "name": "Feasibility assessment", "artifacts": ("feasibility_assessments.json",)},
    {"id": "stage_12", "number": "12", "name": "Final report", "artifacts": ("final_report.md",)},
)

_PIPELINE_STAGE_IDS = {str(stage["id"]) for stage in PIPELINE_STAGES}

STAGE_PHASES: dict[str, str] = {
    "stage_1": "defining_requirements",
    "stage_2": "formalizing_problem",
    "stage_3": "surveying_literature",
    "stage_3.5": "reading_papers",
    "stage_4": "positioning_papers",
    "stage_5": "expanding_literature",
    "stage_6": "binding_evidence",
    "stage_7": "synthesizing_gaps",
    "stage_8": "mapping_design_space",
    "stage_9": "generating_ideas",
    "stage_9.5": "ranking_ideas",
    "stage_10": "debating_ideas",
    "stage_11": "assessing_feasibility",
    "stage_12": "writing_report",
}

_DEFAULT_PARAMETERS = {
    "target_paper_count": 20,
    "deep_reading_top_k": 20,
    "max_ideas_to_generate": 10,
    "top_k_ranked": 20,
    "max_debate_rounds": 5,
}

_BATCH_STAGE = {
    "stage_3": "stage_3_search",
    "stage_3.5": "stage_3_5_reading",
    "stage_6": "stage_6_evidence",
}

_TOOL_PHASES = {
    "create_search_batches": "preparing_batches",
    "create_reading_batches": "preparing_batches",
    "create_evidence_batches": "preparing_batches",
    "record_batch_result": "processing_batch",
    "read_batch_manifest": "checking_batches",
    "merge_search_batches": "merging_batches",
    "merge_reading_batches": "merging_batches",
    "merge_evidence_batches": "merging_batches",
    "semantic_scholar_search": "searching_sources",
    "arxiv_search": "searching_sources",
    "openalex_search": "searching_sources",
    "dblp_search": "searching_sources",
    "crossref_search": "searching_sources",
    "pubmed_search": "searching_sources",
    "cvf_search": "searching_sources",
    "multi_source_search": "searching_sources",
    "tavily_search": "searching_sources",
    "web_search": "searching_sources",
    "fetch_paper_fulltext": "retrieving_full_text",
    "fetch_paper_content": "retrieving_full_text",
    "fetch_paper_section": "retrieving_full_text",
    "rank_ideas_tournament": "ranking_ideas",
    "generate_tournament_matchups": "ranking_ideas",
    "check_stage_gate": "validating_stage",
    "save_stage_reflection": "recording_reflection",
    "write_workspace_file": "writing_artifact",
    "write_design_space": "writing_artifact",
    "write_evidence_db": "writing_artifact",
    "write_research_gaps": "writing_artifact",
    "write_raw_ideas": "writing_artifact",
    "write_tournament_rankings": "writing_artifact",
    "write_idea_reviews": "writing_artifact",
    "task": "running_subagent",
    "think": "reasoning",
    "think_tool": "reasoning",
}


def build_stage_progress(
    workspace: str | Path,
    stage: str,
    *,
    status: str = "running",
    parameters: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe stage snapshot from durable workspace evidence."""
    root = Path(workspace).expanduser().resolve()
    spec = next((item for item in PIPELINE_STAGES if item["id"] == stage), None)
    if spec is None:
        return {}
    params = _parameters(parameters)
    live = dict(runtime or {}) if runtime and runtime.get("stage") == stage else {}
    activity_state = str(live.get("state") or status or "running")
    if status in {"failed", "stopped", "stale", "waiting_for_input", "checkpoint_reached"}:
        activity_state = status
    snapshot: dict[str, Any] = {
        "stage": stage,
        "number": spec["number"],
        "name": spec["name"],
        "index": next(index for index, item in enumerate(PIPELINE_STAGES, 1) if item["id"] == stage),
        "total_stages": len(PIPELINE_STAGES),
        "status": status,
        "phase": str(live.get("phase") or STAGE_PHASES[stage]),
        "activity": str(live.get("activity") or ""),
        "subject": str(live.get("subject") or ""),
        "activity_state": activity_state,
        "activity_started_at": str(live.get("activity_started_at") or ""),
        "updated_at": str(live.get("updated_at") or _latest_mtime(root)),
        "current": None,
        "total": None,
        "unit": "",
        "percent": None,
        "indeterminate": True,
        "counts": {},
    }

    papers = _list_count(_load_json(root / "paper_registry.json"), ())
    if stage == "stage_3":
        batches = _batch_counts(root, stage)
        _set_measure(snapshot, papers, params["target_paper_count"], "papers_collected")
        snapshot["counts"] = {"papers": papers, **batches}
    elif stage == "stage_3.5":
        records = _records(_load_json(root / "fulltext_audit.json"), "records")
        full_text = sum(str(item.get("status") or "") == "full_text" for item in records)
        failed = sum(str(item.get("status") or "") == "failed" for item in records)
        _set_measure(snapshot, len(records), params["deep_reading_top_k"], "papers_processed")
        snapshot["counts"] = {
            "full_text": full_text,
            "failed": failed,
            **_batch_counts(root, stage),
        }
    elif stage == "stage_4":
        positions = _list_count(_load_json(root / "paper_positions.json"), ("positions",))
        if papers:
            _set_measure(snapshot, positions, papers, "papers_positioned")
        snapshot["counts"] = {"positions": positions, "papers": papers}
    elif stage == "stage_6":
        batches = _batch_counts(root, stage)
        if batches.get("batches_total", 0):
            _set_measure(snapshot, batches["batches_completed"], batches["batches_total"], "batches")
        claims = _list_count(_load_json(root / "evidence_db.json"), ("claims",))
        snapshot["counts"] = {"claims": claims, **batches}
    elif stage == "stage_7":
        gaps = _records(_load_json(root / "research_gaps.json"), "gaps")
        links = sum(len(item.get("evidence_links") or []) for item in gaps)
        snapshot["counts"] = {"gaps": len(gaps), "evidence_links": links}
    elif stage == "stage_8":
        design = _load_json(root / "design_space.json")
        snapshot["counts"] = {
            "axes": _list_count(design, ("axes",)),
            "combinations": _list_count(design, ("promising_combinations",)),
        }
    elif stage == "stage_9":
        ideas = _list_count(_load_json(root / "raw_ideas.json"), ("ideas",))
        _set_measure(snapshot, ideas, params["max_ideas_to_generate"], "ideas_generated")
        snapshot["counts"] = {"ideas": ideas}
    elif stage == "stage_9.5":
        ideas = _list_count(_load_json(root / "raw_ideas.json"), ("ideas",))
        rankings_data = _load_json(root / "tournament_rankings.json")
        rankings = _list_count(rankings_data, ("rankings",))
        target = min(ideas, params["top_k_ranked"]) if ideas else 0
        if target:
            _set_measure(snapshot, rankings, target, "ideas_ranked")
        snapshot["counts"] = {
            "rankings": rankings,
            "comparisons": _field_count(
                rankings_data,
                ("comparisons", "comparison_count", "total_comparisons", "matches", "matchups"),
            ),
        }
    elif stage == "stage_10":
        reviews_data = _load_json(root / "idea_reviews.json")
        reviews = _list_count(reviews_data, ("reviews",))
        target = _ranking_target(root)
        if target:
            _set_measure(snapshot, reviews, target, "ideas_reviewed")
        snapshot["counts"] = {
            "reviews": reviews,
            "debate_rounds": _largest_int(
                reviews_data,
                ("debate_rounds", "rounds", "num_rounds", "round_number"),
            ),
            "round_target": params["max_debate_rounds"],
        }
    elif stage == "stage_11":
        assessments = _list_count(
            _load_json(root / "feasibility_assessments.json"),
            ("assessments", "ideas"),
        )
        target = _ranking_target(root)
        if target:
            _set_measure(snapshot, assessments, target, "ideas_assessed")
        snapshot["counts"] = {"assessments": assessments}

    if status == "complete":
        snapshot.update({"current": 1, "total": 1, "unit": "stage", "percent": 100, "indeterminate": False})
    return snapshot


class RuntimeProgressTracker:
    """Observe shared stream events and persist the active stage for both UIs."""

    def __init__(self, workspace: str | Path, stage: str, *, parameters: Mapping[str, Any] | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.stage = stage
        self.parameters = dict(parameters or {})
        self.runtime: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        self._record(phase=STAGE_PHASES.get(self.stage, "working"), activity="stage_started")
        return self.snapshot()

    def observe(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        event_type = str(event.get("type") or "")
        if event_type not in {
            "tool_call", "tool_result", "subagent_start", "subagent_tool_call",
            "subagent_tool_result", "subagent_end", "error",
        }:
            return None
        name = str(event.get("name") or "")
        if name == "write_run_status":
            args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
            reported_stage = _normalize_stage_id(args.get("stage"))
            if reported_stage:
                self.stage = reported_stage
            return None
        if event_type == "subagent_start":
            phase, activity = "running_subagent", "task"
        elif event_type == "subagent_end":
            phase, activity = "integrating_subagent", "task"
        elif event_type == "error":
            phase, activity = "runtime_error", "error"
        else:
            phase = _TOOL_PHASES.get(name, STAGE_PHASES.get(self.stage, "working"))
            activity = name
        args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
        subject = _activity_subject(activity, args)
        state = "complete" if event_type.endswith("result") or event_type == "subagent_end" else "running"
        self._record(phase=phase, activity=activity, subject=subject, state=state)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return build_stage_progress(
            self.workspace,
            self.stage,
            parameters=self.parameters,
            runtime=self.runtime,
        )

    def _record(self, *, phase: str, activity: str, subject: str = "", state: str = "running") -> None:
        now = _now()
        previous = _load_json(self.workspace / "run_status.json")
        old_runtime = previous.get("progress") if isinstance(previous.get("progress"), dict) else {}
        same_activity = (
            old_runtime.get("stage") == self.stage
            and old_runtime.get("activity") == activity
            and old_runtime.get("subject") == subject
        )
        self.runtime = {
            "stage": self.stage,
            "phase": phase,
            "activity": activity,
            "subject": subject,
            "state": state,
            "activity_started_at": str(old_runtime.get("activity_started_at") or now) if same_activity else now,
            "updated_at": now,
        }
        record = dict(previous)
        record.update({"updated_at": now, "pid": os.getpid(), "stage": self.stage, "status": "running", "progress": self.runtime})
        if previous.get("stage") != self.stage:
            record["detail"] = ""
        _atomic_write_json(self.workspace / "run_status.json", record)


def write_reported_status(path: Path, data: dict[str, Any]) -> None:
    """Atomically write model-reported status without dropping live progress."""
    previous = _load_json(path)
    runtime = previous.get("progress")
    if isinstance(runtime, dict) and runtime.get("stage") == data.get("stage"):
        data = {**data, "progress": runtime}
    _atomic_write_json(path, data)


def _set_measure(snapshot: dict[str, Any], current: int, total: int, unit: str) -> None:
    if total <= 0:
        return
    snapshot.update({
        "current": max(0, current),
        "total": total,
        "unit": unit,
        "percent": min(100, round((max(0, current) / total) * 100)),
        "indeterminate": False,
    })


def _parameters(values: Mapping[str, Any] | None) -> dict[str, int]:
    result = dict(_DEFAULT_PARAMETERS)
    for key, default in _DEFAULT_PARAMETERS.items():
        env_key = f"AUTOIDEA_{key.upper()}"
        candidate = (values or {}).get(key, os.getenv(env_key, default))
        try:
            parsed = int(candidate)
        except (TypeError, ValueError):
            parsed = default
        result[key] = parsed if parsed > 0 else default
    return result


def _batch_counts(root: Path, stage: str) -> dict[str, int]:
    stage_name = _BATCH_STAGE.get(stage, "")
    manifest = _load_json(root / "batch_manifest.json")
    batches = [
        item for item in manifest.get("batches", [])
        if isinstance(item, dict) and item.get("stage") == stage_name
    ] if isinstance(manifest, dict) else []
    completed = sum(str(item.get("status") or "") in {"passed", "failed"} for item in batches)
    return {
        "batches_completed": completed,
        "batches_total": len(batches),
        "batches_failed": sum(str(item.get("status") or "") == "failed" for item in batches),
    }


def _ranking_target(root: Path) -> int:
    rankings = _list_count(_load_json(root / "tournament_rankings.json"), ("rankings",))
    return rankings or _list_count(_load_json(root / "raw_ideas.json"), ("ideas",))


def _records(value: Any, key: str) -> list[dict[str, Any]]:
    items = value.get(key, []) if isinstance(value, dict) else []
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _list_count(value: Any, keys: tuple[str, ...]) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in keys:
            if isinstance(value.get(key), list):
                return len(value[key])
    return 0


def _field_count(value: Any, keys: tuple[str, ...]) -> int:
    if not isinstance(value, dict):
        return 0
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, list):
            return len(candidate)
        try:
            return max(0, int(candidate))
        except (TypeError, ValueError):
            continue
    return 0


def _largest_int(value: Any, keys: tuple[str, ...]) -> int:
    found: list[int] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                if key in keys:
                    try:
                        found.append(int(child))
                    except (TypeError, ValueError):
                        pass
                if isinstance(child, (dict, list)):
                    stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)
    return max(found, default=0)


def _activity_subject(activity: str, args: Mapping[str, Any]) -> str:
    keys = {
        "fetch_paper_fulltext": ("identifier", "paper_id", "url"),
        "fetch_paper_content": ("paper_id", "url"),
        "record_batch_result": ("batch_id",),
        "write_workspace_file": ("file_path", "path"),
        "task": ("description", "subagent_type"),
    }.get(activity, ("query", "stage", "batch_id"))
    for key in keys:
        value = args.get(key)
        if value:
            return " ".join(str(value).split())[:120]
    return ""


def _normalize_stage_id(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    if not raw:
        return ""
    candidate = raw if raw.startswith("stage_") else f"stage_{raw}"
    return candidate if candidate in _PIPELINE_STAGE_IDS else ""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return {}


def _latest_mtime(root: Path) -> str:
    latest = 0.0
    for name in ("run_status.json", "batch_manifest.json", "fulltext_audit.json"):
        try:
            latest = max(latest, (root / name).stat().st_mtime)
        except OSError:
            pass
    return datetime.fromtimestamp(latest, UTC).isoformat() if latest else ""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
