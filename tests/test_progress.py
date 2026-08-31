from __future__ import annotations

import json
from pathlib import Path

from autoidea.progress import (
    PIPELINE_STAGES,
    STAGE_PHASES,
    RuntimeProgressTracker,
    build_stage_progress,
)
from autoidea.stream.display import (
    _inspect_autonomous_progress,
    _render_pipeline_progress,
)
from autoidea.web.pipeline import inspect_pipeline


def _write_text(path: Path, title: str) -> None:
    path.write_text(f"# {title}\n\n" + ("Grounded research content. " * 12), encoding="utf-8")


def test_all_fourteen_stages_have_a_shared_progress_descriptor(tmp_path: Path) -> None:
    assert len(PIPELINE_STAGES) == 14
    assert set(STAGE_PHASES) == {str(stage["id"]) for stage in PIPELINE_STAGES}

    for index, stage in enumerate(PIPELINE_STAGES, start=1):
        progress = build_stage_progress(tmp_path, str(stage["id"]))
        assert progress["stage"] == stage["id"]
        assert progress["number"] == stage["number"]
        assert progress["index"] == index
        assert progress["total_stages"] == 14
        assert progress["phase"]
        assert isinstance(progress["indeterminate"], bool)

    failed = build_stage_progress(tmp_path, "stage_3.5", status="failed")
    assert failed["activity_state"] == "failed"


def test_stale_passed_marker_advances_to_observed_stage_3_5(tmp_path: Path) -> None:
    _write_text(tmp_path / "research_brief.md", "Research brief")
    _write_text(tmp_path / "task_formalization.md", "Task formalization")
    _write_text(tmp_path / "literature_survey.md", "Literature survey")
    (tmp_path / "paper_registry.json").write_text(
        json.dumps([{"paper_id": "P1", "title": "Paper"}]),
        encoding="utf-8",
    )
    (tmp_path / "run_status.json").write_text(
        json.dumps({"stage": "stage_3", "status": "passed", "detail": "done"}),
        encoding="utf-8",
    )
    reflection_dir = tmp_path / "reflections"
    reflection_dir.mkdir()
    for stage in ("stage_1", "stage_2", "stage_3"):
        (reflection_dir / f"{stage}_reflection.json").write_text(
            json.dumps({"stage": stage, "reflection": "complete"}),
            encoding="utf-8",
        )

    pipeline = inspect_pipeline(tmp_path, run_status="running")

    assert pipeline["completed_count"] == 3
    assert pipeline["active_stage"] == "stage_3.5"
    assert pipeline["active_detail"] == ""
    assert pipeline["active_progress"]["stage"] == "stage_3.5"


def test_final_stage_requires_reflection_and_audit_before_14_of_14(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for stage in PIPELINE_STAGES:
        for artifact in stage["artifacts"]:
            target = tmp_path / str(artifact)
            if target.suffix == ".json":
                target.write_text(json.dumps({"ready": True}), encoding="utf-8")
            else:
                _write_text(target, str(stage["name"]))

    reflection_dir = tmp_path / "reflections"
    reflection_dir.mkdir()
    for stage in PIPELINE_STAGES[:-1]:
        stage_id = str(stage["id"])
        (reflection_dir / f"{stage_id}_reflection.json").write_text(
            json.dumps({"stage": stage_id, "reflection": "complete"}),
            encoding="utf-8",
        )

    class PassingAudit:
        has_errors = False
        issues: list = []

    monkeypatch.setattr(
        "autoidea.tools.artifact_audit.validate_stage7_artifacts",
        lambda _workspace: [],
    )
    monkeypatch.setattr(
        "autoidea.tools.artifact_audit.audit_workspace",
        lambda *_args, **_kwargs: PassingAudit(),
    )
    checkpoints = ["stage_7", "stage_9", "stage_10"]

    missing_reflection = inspect_pipeline(
        tmp_path,
        run_status="running",
        checkpoint_events=checkpoints,
        include_audit=True,
    )
    assert missing_reflection["completed_count"] == 13
    assert missing_reflection["active_stage"] == "stage_12"
    assert missing_reflection["completion"]["verified"] is False
    assert missing_reflection["completion"]["missing_reflections"] == ["stage_12"]

    (reflection_dir / "stage_12_reflection.json").write_text(
        json.dumps({"stage": "stage_12", "reflection": "complete"}),
        encoding="utf-8",
    )
    missing_gate_proof = inspect_pipeline(
        tmp_path,
        run_status="completed",
        checkpoint_events=checkpoints,
        include_audit=True,
    )
    assert missing_gate_proof["completed_count"] == 13
    assert missing_gate_proof["active_stage"] == "stage_12"
    assert missing_gate_proof["completion"]["reflections_ready"] is False
    assert missing_gate_proof["completion"]["missing_gate_proofs"] == ["stage_12"]

    (reflection_dir / "stage_12_reflection.json").write_text(
        json.dumps(
            {
                "stage": "stage_12",
                "reflection": "complete",
                "gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    audit_pending = inspect_pipeline(
        tmp_path,
        run_status="running",
        checkpoint_events=checkpoints,
        include_audit=False,
    )
    assert audit_pending["completed_count"] == 13
    assert audit_pending["active_stage"] == "stage_12"
    assert audit_pending["completion"]["audit_passed"] is None

    completed = inspect_pipeline(
        tmp_path,
        run_status="completed",
        checkpoint_events=checkpoints,
        include_audit=True,
    )
    assert completed["completed_count"] == 14
    assert completed["active_stage"] == ""
    assert completed["next_stage"] == "complete"
    assert completed["completion"]["verified"] is True


def test_stage_3_5_reports_fulltext_and_batch_progress(tmp_path: Path) -> None:
    (tmp_path / "fulltext_audit.json").write_text(
        json.dumps(
            {
                "records": [
                    *[{"status": "full_text"} for _ in range(9)],
                    *[{"status": "failed"} for _ in range(2)],
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "batch_manifest.json").write_text(
        json.dumps(
            {
                "batches": [
                    {"stage": "stage_3_5_reading", "status": "passed"},
                    *[
                        {"stage": "stage_3_5_reading", "status": "pending"}
                        for _ in range(3)
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )

    progress = build_stage_progress(
        tmp_path,
        "stage_3.5",
        parameters={"deep_reading_top_k": 20},
    )

    assert progress["current"] == 11
    assert progress["total"] == 20
    assert progress["percent"] == 55
    assert progress["counts"] == {
        "full_text": 9,
        "failed": 2,
        "batches_completed": 1,
        "batches_total": 4,
        "batches_failed": 0,
    }


def test_later_stage_progress_uses_canonical_artifacts(tmp_path: Path) -> None:
    (tmp_path / "research_gaps.json").write_text(
        json.dumps(
            {
                "gaps": [
                    {"evidence_links": [{"citation_id": "C1"}, {"citation_id": "C2"}]},
                    {"evidence_links": [{"citation_id": "C3"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "design_space.json").write_text(
        json.dumps({"axes": [{}, {}], "promising_combinations": [{}, {}, {}]}),
        encoding="utf-8",
    )
    (tmp_path / "raw_ideas.json").write_text(
        json.dumps({"ideas": [{"idea_id": f"I{i}"} for i in range(5)]}),
        encoding="utf-8",
    )
    (tmp_path / "tournament_rankings.json").write_text(
        json.dumps({"rankings": [{"idea_id": f"I{i}"} for i in range(5)], "comparisons": 10}),
        encoding="utf-8",
    )
    (tmp_path / "idea_reviews.json").write_text(
        json.dumps({"reviews": [{"idea_id": f"I{i}", "round_number": i} for i in range(1, 4)]}),
        encoding="utf-8",
    )
    (tmp_path / "feasibility_assessments.json").write_text(
        json.dumps({"assessments": [{"idea_id": f"I{i}"} for i in range(3)]}),
        encoding="utf-8",
    )

    assert build_stage_progress(tmp_path, "stage_7")["counts"] == {
        "gaps": 2,
        "evidence_links": 3,
    }
    assert build_stage_progress(tmp_path, "stage_8")["counts"] == {
        "axes": 2,
        "combinations": 3,
    }
    assert build_stage_progress(tmp_path, "stage_9.5")["counts"] == {
        "rankings": 5,
        "comparisons": 10,
    }
    debate = build_stage_progress(tmp_path, "stage_10", parameters={"max_debate_rounds": 5})
    assert debate["counts"]["debate_rounds"] == 3
    assert debate["counts"]["round_target"] == 5
    assert build_stage_progress(tmp_path, "stage_11")["current"] == 3


def test_cli_progress_panel_includes_later_stage_metrics(tmp_path: Path) -> None:
    progress = build_stage_progress(tmp_path, "stage_7")
    progress["counts"] = {
        "gaps": 3,
        "evidence_links": 11,
        "axes": 4,
        "combinations": 6,
        "comparisons": 10,
        "debate_rounds": 2,
        "round_target": 5,
    }

    rendered = _render_pipeline_progress(progress).renderable.plain

    assert "3 gaps" in rendered
    assert "11 evidence links" in rendered
    assert "4 design axes" in rendered
    assert "6 combinations" in rendered
    assert "10 comparisons" in rendered
    assert "2/5 debate rounds" in rendered


def test_runtime_tracker_persists_activity_and_cli_web_share_snapshot(tmp_path: Path) -> None:
    tracker = RuntimeProgressTracker(
        tmp_path,
        "stage_1",
        parameters={"target_paper_count": 7},
    )
    tracker.start()
    observed = tracker.observe(
        {
            "type": "tool_call",
            "name": "write_workspace_file",
            "args": {"file_path": "research_brief.md"},
        }
    )
    marker = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))

    assert observed is not None
    assert observed["phase"] == "writing_artifact"
    assert observed["subject"] == "research_brief.md"
    assert marker["stage"] == "stage_1"
    assert marker["progress"]["activity"] == "write_workspace_file"

    web_snapshot = inspect_pipeline(tmp_path, run_status="running")
    cli_snapshot = _inspect_autonomous_progress(tmp_path)["snapshot"]
    assert cli_snapshot["active_progress"] == web_snapshot["active_progress"]


def test_runtime_progress_tracker_follows_reported_stage_transition(
    tmp_path: Path,
) -> None:
    tracker = RuntimeProgressTracker(tmp_path, "stage_9.5")
    tracker.start()

    assert tracker.observe(
        {
            "type": "tool_call",
            "name": "write_run_status",
            "args": {"stage": "stage_10", "status": "running"},
        }
    ) is None
    observed = tracker.observe(
        {
            "type": "tool_call",
            "name": "task",
            "args": {"description": "Run the adversarial review."},
        }
    )
    marker = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))

    assert observed is not None
    assert observed["stage"] == "stage_10"
    assert observed["phase"] == "running_subagent"
    assert marker["stage"] == "stage_10"
