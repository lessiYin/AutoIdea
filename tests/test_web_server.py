from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoidea.web.models import RunRecord
from autoidea.web.runs import WebRunManager
from autoidea.web.server import create_app

FIXTURE = Path(__file__).parent / "fixtures" / "sample_workspace"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    for path in FIXTURE.iterdir():
        if path.is_file():
            shutil.copy2(path, root / path.name)
    return root


def test_web_server_exposes_health_and_snapshot(workspace: Path) -> None:
    client = TestClient(create_app(workspace))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    snapshot = client.get("/api/snapshot")
    assert snapshot.status_code == 200
    data = snapshot.json()
    assert data["workspace"]["name"] == workspace.name
    assert data["counts"]["papers"] == 2
    assert data["graph"]["nodes"]


def test_web_server_serves_static_dashboard(workspace: Path) -> None:
    client = TestClient(create_app(workspace))

    response = client.get("/")
    icon = client.get("/static/autoidea-mark.svg")

    assert response.status_code == 200
    assert icon.status_code == 200
    assert icon.headers["content-type"].startswith("image/svg+xml")
    assert "AutoIdea Research Observatory" in response.text
    assert 'rel="icon" href="/static/autoidea-mark.svg"' in response.text
    assert 'class="brand-mark" src="/static/autoidea-mark.svg"' in response.text
    assert "Research Observatory" in response.text
    assert "Literature map" in response.text
    assert 'class="skip-link"' in response.text
    assert '<button id="languageToggle"' in response.text
    assert 'href="/?lang=zh"' not in response.text
    assert 'id="sidebarLanguageToggle"' not in response.text
    assert "中文" in response.text


def test_web_server_disables_browser_cache_for_workbench_assets(workspace: Path) -> None:
    client = TestClient(create_app(workspace))

    index = client.get("/")
    script = client.get("/static/app.js")

    assert "no-store" in index.headers["cache-control"]
    assert "no-store" in script.headers["cache-control"]


def test_web_server_exposes_artifact_content(workspace: Path) -> None:
    client = TestClient(create_app(workspace))

    response = client.get("/api/artifacts/literature_survey.md")

    assert response.status_code == 200
    data = response.json()
    assert data["path"] == "literature_survey.md"
    assert data["kind"] == "md"
    assert "<h1>Literature Survey</h1>" in data["html"]


def test_web_server_blocks_artifact_path_traversal(workspace: Path) -> None:
    client = TestClient(create_app(workspace))

    response = client.get("/api/artifacts/../paper_registry.json")

    assert response.status_code == 404


def test_web_server_exposes_run_collection(workspace: Path) -> None:
    client = TestClient(create_app(workspace))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == []


def test_web_server_returns_readable_json_when_run_process_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    client = TestClient(create_app(workspace))

    def fail_popen(*_args, **_kwargs):
        raise FileNotFoundError("missing autoidea executable")

    monkeypatch.setattr("autoidea.web.runs.subprocess.Popen", fail_popen)

    response = client.post("/api/runs", json={"prompt": "Hello"})

    assert response.status_code == 500
    assert "Unable to start AutoIdea process" in response.json()["detail"]


def test_web_server_accepts_run_input(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    sent: list[tuple[str, dict]] = []
    client = TestClient(create_app(workspace))

    def send_input(run_id: str, value: dict):
        sent.append((run_id, value))
        return {"run_id": run_id, "status": "sent"}

    monkeypatch.setattr(
        "autoidea.web.server.WebRunManager.send_input",
        lambda self, run_id, value: send_input(run_id, value),
    )

    response = client.post("/api/runs/run123/input", json={"value": "B"})

    assert response.status_code == 200
    assert response.json() == {"run_id": "run123", "status": "sent"}
    assert sent == [("run123", {"value": "B"})]


def test_web_server_starts_followup_run_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    started = []
    client = TestClient(create_app(workspace))

    def fake_get_run(self, run_id: str):
        from autoidea.web.models import RunRecord

        assert run_id == "checkpoint"
        return RunRecord(
            run_id="checkpoint",
            status="completed",
            prompt="长视频理解",
            workspace="/tmp/workspace/runs/lvu",
            run_name="lvu",
            thread_id="abc123",
            model="qwen-test",
            provider="custom-openai",
            seed_papers="/tmp/papers.json",
            seed_ideas="/tmp/ideas.md",
            auto_approve=True,
            show_thinking=False,
            log_path="/tmp/log.txt",
            interaction={
                "kind": "checkpoint_review",
                "question": "请审阅以上研究想法。您希望：",
                "options": [],
                "allows_other": True,
            },
        )

    def fake_start_run(self, request):
        from autoidea.web.models import RunRecord

        started.append(request)
        return RunRecord(
            run_id="followup",
            status="running",
            prompt=request.prompt,
            workspace=request.workspace,
            run_name=request.run_name,
            thread_id=request.thread_id,
        )

    monkeypatch.setattr("autoidea.web.server.WebRunManager.get_run", fake_get_run)
    monkeypatch.setattr("autoidea.web.server.WebRunManager.start_run", fake_start_run)

    response = client.post(
        "/api/runs/checkpoint/followup",
        json={"action": "approve", "feedback": "继续"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "followup"
    assert len(started) == 1
    request = started[0]
    assert request.run_name == "lvu"
    assert request.workspace == "/tmp/workspace/runs/lvu"
    assert request.thread_id == "abc123"
    assert request.model == "qwen-test"
    assert request.provider == "custom-openai"
    assert request.seed_papers == "/tmp/papers.json"
    assert request.seed_ideas == "/tmp/ideas.md"
    assert request.auto_approve is True
    assert request.show_thinking is False
    assert "批准" in request.prompt
    assert "继续" in request.prompt


def test_web_server_scopes_snapshot_artifact_and_events_to_run(
    workspace: Path,
    tmp_path: Path,
) -> None:
    run_workspace = tmp_path / "isolated-run"
    run_workspace.mkdir()
    (run_workspace / "research_brief.md").write_text(
        "# Run-only brief\n\n" + "isolated evidence " * 10,
        encoding="utf-8",
    )
    (run_workspace / "final_report.md").write_text(
        "# Run-only final report\n\n" + "verified result " * 30,
        encoding="utf-8",
    )
    manager = WebRunManager(workspace)
    events_path = manager.events_dir / "scoped.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        (
            '{"type":"runner_started","run_id":"scoped"}\n'
            '{"type":"interaction_requested","interaction_id":"s7",'
            '"checkpoint_stage":"stage_7"}\n'
            '{"type":"interaction_resolved","interaction_id":"s7",'
            '"response":{"approved":true}}\n'
        ),
        encoding="utf-8",
    )
    manager._update_record(
        RunRecord(
            run_id="scoped",
            status="failed",
            prompt="isolated",
            workspace=str(run_workspace),
            run_name="isolated-run",
            events_path=str(events_path),
            status_detail="fixture",
        )
    )
    client = TestClient(create_app(workspace, run_manager=manager))

    snapshot = client.get("/api/runs/scoped/snapshot")
    artifact = client.get("/api/runs/scoped/artifacts/final_report.md")
    events = client.get("/api/runs/scoped/events")

    assert snapshot.status_code == 200
    assert snapshot.json()["workspace"]["path"] == str(run_workspace.resolve())
    assert {item["path"] for item in snapshot.json()["artifacts"]} == {
        "research_brief.md",
        "final_report.md",
    }
    assert snapshot.json()["pipeline"]["completion"]["checkpoint_events"] == [
        "stage_7"
    ]
    stage_7 = next(
        stage
        for stage in snapshot.json()["pipeline"]["stages"]
        if stage["id"] == "stage_7"
    )
    assert stage_7["checkpoint_recorded"] is True
    assert artifact.status_code == 200
    assert artifact.json()["title"] == "Final Report"
    assert "Run-only final report" in artifact.json()["html"]
    assert len(events.json()) == 3
    assert events.json()[-1]["response"]["approved"] is True
