"""Heartbeat tools for exposing long-run status on disk."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool


def _workspace() -> Path:
    from autoidea.paths import get_active_workspace

    return Path(get_active_workspace())


def _status_path() -> Path:
    return _workspace() / "run_status.json"


@tool(parse_docstring=True)
def write_run_status(stage: str, status: str, detail: str = "") -> str:
    """Write a heartbeat status record to run_status.json.

    Args:
        stage: Current pipeline stage, e.g. stage_3.5 or stage_9.
        status: Current status, e.g. starting, running, waiting_model,
            waiting_user, passed, failed.
        detail: Short human-readable detail.

    Returns:
        Confirmation with the status file path.
    """
    from autoidea.tools.pipeline_state import STAGES, _build_state, _write_state

    normalized_status = status.strip().casefold()
    if normalized_status in {"passed", "complete", "completed"}:
        state = _build_state()
        if stage == "complete":
            next_stage = str(state.get("next_stage") or "")
            blocked_stage = next_stage if next_stage != "complete" else ""
        else:
            stage_names = {item.stage for item in STAGES}
            blocked_stage = (
                stage
                if stage in stage_names
                and state["stages"][stage]["status"] != "complete"
                else ""
            )
        if blocked_stage:
            _write_state(state)
            info = state["stages"][blocked_stage]
            missing = ", ".join(info.get("missing_artifacts", [])) or "invalid artifacts"
            return (
                f"Refused to mark {stage} as {status}: {blocked_stage} is not "
                f"complete ({missing}). Produce and validate its required "
                "artifacts before advancing."
            )

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "stage": stage,
        "status": status,
        "detail": detail,
    }
    path = _status_path()
    from autoidea.progress import write_reported_status

    write_reported_status(path, data)
    # Keep the persisted pipeline snapshot synchronized with the artifacts at
    # every heartbeat. The Web UI derives truth from disk, but an obsolete
    # pipeline_state.json otherwise produces a confusing stale-state warning
    # even after a fully verified run.
    _write_state(_build_state())
    return f"Run status written to run_status.json: {stage} / {status}"


@tool(parse_docstring=True)
def read_run_status() -> str:
    """Read run_status.json.

    Returns:
        Concise markdown status, or a message if no heartbeat exists.
    """
    path = _status_path()
    if not path.exists():
        return "No run_status.json found."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Could not read run_status.json: {exc}"
    return (
        "# Run Status\n\n"
        f"- updated_at: {data.get('updated_at', '')}\n"
        f"- pid: {data.get('pid', '')}\n"
        f"- stage: {data.get('stage', '')}\n"
        f"- status: {data.get('status', '')}\n"
        f"- detail: {data.get('detail', '')}"
    )
