"""FastAPI application factory for the AutoIdea browser dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactAccessError, read_artifact
from .configuration import get_config_payload, reset_config_values, update_config_values
from .models import RunRecord
from .pipeline import checkpoint_events_from_events, inspect_pipeline
from .runs import RunRequest, WebRunManager, pipeline_parameters_for_record
from .workspace import load_workspace_snapshot


def create_app(
    workspace: str | Path,
    *,
    run_manager: WebRunManager | None = None,
):
    """Create a FastAPI app bound to an AutoIdea workspace directory."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised through CLI messaging
        raise RuntimeError(
            "The web dashboard requires the optional web dependencies. "
            'Install them with: pip install -e ".[web]"'
        ) from exc

    workspace_path = Path(workspace).expanduser().resolve()
    static_dir = Path(__file__).parent / "static"
    manager = run_manager or WebRunManager(workspace_path)

    class NoStoreStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope: dict[str, Any]):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

    app = FastAPI(
        title="AutoIdea Research Console",
        description="Browser dashboard for AutoIdea research workspaces.",
        version="0.1.0",
    )

    app.mount("/static", NoStoreStaticFiles(directory=static_dir), name="static")

    def run_snapshot_payload(record: RunRecord) -> dict:
        payload = load_workspace_snapshot(record.workspace).to_dict()
        checkpoints = checkpoint_events_from_events(manager.list_events(record.run_id))
        payload["pipeline"] = inspect_pipeline(
            record.workspace,
            run_status=record.status,
            checkpoint_events=checkpoints,
            include_audit=record.status
            in {
                "completed",
                "pipeline_completed",
                "failed",
                "stopped",
                "stale",
                "checkpoint_reached",
            },
            audit_parameters=pipeline_parameters_for_record(record),
        )
        return payload

    @app.get("/")
    def index():
        return FileResponse(
            static_dir / "index.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "workspace": str(workspace_path)}

    @app.get("/api/snapshot")
    def snapshot(run_id: str | None = None) -> dict:
        if not run_id:
            return load_workspace_snapshot(workspace_path).to_dict()
        record = manager.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return run_snapshot_payload(record)

    @app.get("/api/config")
    def config() -> dict:
        return get_config_payload()

    @app.patch("/api/config")
    def update_config(request: dict[str, Any]) -> dict:
        try:
            return update_config_values(request.get("values", {}))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/config/reset")
    def reset_config() -> dict:
        return reset_config_values()

    @app.get("/api/artifacts/{artifact_path:path}")
    def artifact(artifact_path: str) -> dict:
        from dataclasses import asdict

        try:
            return asdict(read_artifact(workspace_path, artifact_path))
        except ArtifactAccessError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs")
    def runs() -> list[dict]:
        from dataclasses import asdict

        return [asdict(record) for record in manager.list_runs()]

    @app.post("/api/runs")
    def start_run(request: RunRequest) -> dict:
        from dataclasses import asdict

        try:
            return asdict(manager.start_run(request))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to start AutoIdea process: {exc}",
            ) from exc

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        from dataclasses import asdict

        record = manager.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return asdict(record)

    @app.get("/api/runs/{run_id}/snapshot")
    def run_snapshot(run_id: str) -> dict:
        record = manager.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return run_snapshot_payload(record)

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def run_artifact(run_id: str, artifact_path: str) -> dict:
        from dataclasses import asdict

        record = manager.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        try:
            return asdict(read_artifact(record.workspace, artifact_path))
        except ArtifactAccessError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> list[dict[str, Any]]:
        try:
            return manager.list_events(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

    @app.post("/api/runs/{run_id}/input")
    def send_run_input(run_id: str, request: dict[str, Any]) -> dict:
        try:
            return manager.send_input(run_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/followup")
    def start_followup_run(run_id: str, request: dict[str, Any]) -> dict:
        from dataclasses import asdict

        try:
            return asdict(
                manager.start_followup(
                    run_id,
                    str(request.get("action", "")),
                    str(request.get("feedback", "")),
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict:
        from dataclasses import asdict

        try:
            return asdict(manager.stop_run(run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

    return app
