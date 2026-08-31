from __future__ import annotations

import json
from pathlib import Path

from autoidea.paths import get_active_workspace, set_active_workspace
from autoidea.progress import RuntimeProgressTracker
from autoidea.tools.heartbeat import read_run_status, write_run_status


def test_write_and_read_run_status(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)

        response = write_run_status.invoke(
            {
                "stage": "stage_9",
                "status": "running",
                "detail": "generating raw ideas",
            }
        )

        assert "run_status.json" in response
        data = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
        assert data["stage"] == "stage_9"
        assert data["status"] == "running"
        assert data["detail"] == "generating raw ideas"
        assert isinstance(data["pid"], int)
        assert data["updated_at"]

        pipeline = json.loads(
            (tmp_path / "pipeline_state.json").read_text(encoding="utf-8")
        )
        assert pipeline["next_stage"] == "stage_1"

        read_response = read_run_status.invoke({})
        assert "stage_9" in read_response
        assert "generating raw ideas" in read_response
    finally:
        set_active_workspace(old_workspace)


def test_passed_status_is_rejected_until_required_artifact_exists(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        write_run_status.invoke(
            {"stage": "stage_1", "status": "running", "detail": "working"}
        )

        rejected = write_run_status.invoke(
            {"stage": "stage_1", "status": "passed", "detail": "too early"}
        )

        assert "Refused to mark stage_1 as passed" in rejected
        status = json.loads(
            (tmp_path / "run_status.json").read_text(encoding="utf-8")
        )
        assert status["status"] == "running"

        (tmp_path / "research_brief.md").write_text(
            "# Research brief\n\nComplete.",
            encoding="utf-8",
        )
        accepted = write_run_status.invoke(
            {"stage": "stage_1", "status": "passed", "detail": "done"}
        )

        assert "stage_1 / passed" in accepted
    finally:
        set_active_workspace(old_workspace)


def test_model_heartbeat_preserves_runtime_progress_for_the_same_stage(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        RuntimeProgressTracker(tmp_path, "stage_2").start()

        write_run_status.invoke(
            {"stage": "stage_2", "status": "running", "detail": "formalizing"}
        )

        data = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
        assert data["detail"] == "formalizing"
        assert data["progress"]["stage"] == "stage_2"
        assert data["progress"]["phase"] == "formalizing_problem"
    finally:
        set_active_workspace(old_workspace)
