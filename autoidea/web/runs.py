"""Managed local AutoIdea runs for the browser workbench."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import RunRecord
from .pipeline import checkpoint_events_from_events, inspect_pipeline


@dataclass(frozen=True)
class RunRequest:
    prompt: str
    run_name: str = ""
    workspace: str = ""
    model: str = ""
    provider: str = ""
    thread_id: str = ""
    seed_papers: str = ""
    seed_ideas: str = ""
    auto_approve: bool = True
    show_thinking: bool = True
    mode: str = "new"
    parent_run_id: str = ""


def _resolve_run_request_defaults(request: RunRequest) -> RunRequest:
    """Snapshot effective model defaults when a Web request leaves them blank."""
    from autoidea.config import get_effective_config

    config = get_effective_config()
    return replace(
        request,
        model=request.model.strip() or str(config.model).strip(),
        provider=request.provider.strip() or str(config.provider).strip(),
    )


_PIPELINE_PARAMETER_DEFAULTS: dict[str, int] = {
    "max_search_queries": 50,
    "target_paper_count": 20,
    "deep_reading_top_k": 20,
    "max_ideas_to_generate": 10,
    "top_k_ranked": 20,
    "max_debate_rounds": 5,
    "elo_initial_score": 1500,
    "elo_k_factor": 32,
}


class WebRunManager:
    """Start, stop, and persist local AutoIdea CLI runs."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        autoidea_executable: str | None = None,
        runner_command: list[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.state_dir = self.workspace / ".autoidea_web"
        self.log_dir = self.state_dir / "logs"
        self.events_dir = self.state_dir / "events"
        self.responses_dir = self.state_dir / "responses"
        self.runs_file = self.state_dir / "runs.json"
        if autoidea_executable:
            self.autoidea_command = [autoidea_executable]
        elif found_autoidea := shutil.which("autoidea"):
            self.autoidea_command = [found_autoidea]
        else:
            self.autoidea_command = [sys.executable, "-m", "autoidea"]
        self.autoidea_executable = self.autoidea_command[0]
        self.runner_command = runner_command or [
            sys.executable,
            "-m",
            "autoidea.web.runner",
        ]
        self._processes: dict[str, subprocess.Popen] = {}
        self._stdin_pipes: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._reconcile_persisted_runs()

    def build_command(
        self,
        request: RunRequest,
        workspace_dir: Path,
        *,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        command = list(self.autoidea_command)
        if extra_args:
            command.extend(extra_args)
        command.extend(["--prompt", request.prompt])
        command.extend(["--workdir", str(workspace_dir)])
        if request.model:
            command.extend(["--model", request.model])
        if request.provider:
            command.extend(["--provider", request.provider])
        if request.thread_id:
            command.extend(["--thread-id", request.thread_id])
        if request.seed_papers:
            command.extend(["--seed-papers", request.seed_papers])
        if request.seed_ideas:
            command.extend(["--seed-ideas", request.seed_ideas])
        if request.auto_approve:
            command.append("--auto-approve")
        else:
            command.append("--manual-checkpoints")
        if not request.show_thinking:
            command.append("--no-thinking")
        return command

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            records = [self._record_from_dict(item) for item in self._read_metadata()]
        return [self._refresh_record(record) for record in records]

    def get_run(self, run_id: str) -> RunRecord | None:
        for record in self.list_runs():
            if record.run_id == run_id:
                return record
        return None

    def start_run(
        self,
        request: RunRequest,
        *,
        extra_args: list[str] | None = None,
    ) -> RunRecord:
        if not request.prompt.strip():
            raise ValueError("Prompt is required.")
        if request.mode not in {"new", "resume", "followup"}:
            raise ValueError("Run mode must be new, resume, or followup.")
        request = _resolve_run_request_defaults(request)

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

        run_id = uuid.uuid4().hex[:10]
        safe_name = _safe_name(request.run_name or run_id)
        if request.workspace:
            requested_workspace = Path(request.workspace).expanduser().resolve()
            target_workspace = (
                requested_workspace
                if request.mode in {"resume", "followup"}
                else _unique_workspace(requested_workspace)
            )
        else:
            target_workspace = _unique_workspace(self.workspace / "runs" / safe_name)
        target_workspace.mkdir(parents=True, exist_ok=True)

        log_path = self.log_dir / f"{run_id}.log"
        events_path = self.events_dir / f"{run_id}.jsonl"
        response_dir = self.responses_dir / run_id
        response_dir.mkdir(parents=True, exist_ok=True)
        thread_id = request.thread_id.strip() or uuid.uuid4().hex[:8]
        normalized_request = RunRequest(
            **{
                **asdict(request),
                "run_name": target_workspace.name,
                "thread_id": thread_id,
            }
        )
        pipeline_parameters = _current_pipeline_parameters()
        if normalized_request.parent_run_id:
            parent = self.get_run(normalized_request.parent_run_id)
            if parent is not None:
                inherited_parameters = pipeline_parameters_for_record(parent)
                if inherited_parameters:
                    pipeline_parameters = inherited_parameters
        if (
            not extra_args
            and normalized_request.mode in {"resume", "followup"}
            and normalized_request.parent_run_id
        ):
            self._inherit_approved_checkpoints(
                normalized_request.parent_run_id,
                events_path,
                workspace=target_workspace,
                thread_id=thread_id,
            )
        if extra_args:
            command = self.build_command(
                normalized_request,
                target_workspace,
                extra_args=extra_args,
            )
        else:
            command = self._build_structured_command(
                normalized_request,
                target_workspace,
                run_id=run_id,
                events_path=events_path,
                response_dir=response_dir,
            )

        queued = RunRecord(
            run_id=run_id,
            status="queued",
            status_detail="Preparing the local research process.",
            prompt=normalized_request.prompt,
            workspace=str(target_workspace),
            run_name=target_workspace.name,
            mode=normalized_request.mode,
            parent_run_id=normalized_request.parent_run_id,
            model=normalized_request.model,
            provider=normalized_request.provider,
            thread_id=thread_id,
            seed_papers=normalized_request.seed_papers,
            seed_ideas=normalized_request.seed_ideas,
            auto_approve=normalized_request.auto_approve,
            show_thinking=normalized_request.show_thinking,
            pipeline_parameters=pipeline_parameters,
            started_at=_now(),
            log_path=str(log_path),
            events_path=str(events_path) if not extra_args else "",
            response_dir=str(response_dir) if not extra_args else "",
            command=command,
        )
        with self._lock:
            records = self._read_metadata()
            records.insert(0, self._persisted_dict(queued))
            self._write_metadata(records)

        log_file = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.workspace),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            log_file.close()
            self._update_record(
                RunRecord(
                    **{
                        **asdict(queued),
                        "status": "failed",
                        "status_detail": f"Unable to start the process: {exc}",
                        "finished_at": _now(),
                    }
                )
            )
            raise

        record = RunRecord(
            **{
                **asdict(queued),
                "status": "running",
                "status_detail": "Research process is running.",
                "pid": process.pid,
            }
        )
        with self._lock:
            self._processes[run_id] = process
            if process.stdin is not None:
                self._stdin_pipes[run_id] = process.stdin
            records = self._read_metadata()
            for index, item in enumerate(records):
                if item.get("run_id") == run_id:
                    records[index] = self._persisted_dict(record)
                    break
            self._write_metadata(records)
        threading.Thread(
            target=self._watch_process,
            args=(run_id, process, log_file),
            daemon=True,
        ).start()
        return self._with_runtime(record)

    def send_input(self, run_id: str, value: str | dict[str, Any]) -> dict[str, str]:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        if record.status not in {"running", "waiting_for_input"}:
            raise RuntimeError("Run is not accepting input.")

        interaction = record.interaction or {}
        interaction_id = str(interaction.get("interaction_id") or "")
        if record.response_dir and interaction_id:
            response_dir = Path(record.response_dir)
            response_dir.mkdir(parents=True, exist_ok=True)
            response_path = response_dir / f"{interaction_id}.json"
            if response_path.exists():
                raise RuntimeError("This interaction already has a response.")
            payload = value if isinstance(value, dict) else {"value": str(value)}
            payload = {**payload, "interaction_id": interaction_id}
            temporary = response_dir / f".{interaction_id}.{uuid.uuid4().hex}.tmp"
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, response_path)
            return {"run_id": run_id, "status": "accepted"}

        process = self._processes.get(run_id)
        pipe = self._stdin_pipes.get(run_id)
        if process is None or pipe is None or process.poll() is not None:
            raise RuntimeError("Run input is no longer available.")

        text = str(value.get("value", "")) if isinstance(value, dict) else str(value)
        if not text.endswith("\n"):
            text += "\n"
        try:
            pipe.write(text)
            pipe.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Unable to send input to run: {exc}") from exc
        return {"run_id": run_id, "status": "sent"}

    def start_followup(self, run_id: str, action: str, feedback: str = "") -> RunRecord:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        interaction = record.interaction or {}
        if interaction.get("kind") != "checkpoint_review":
            raise RuntimeError("Run does not have a checkpoint review to continue.")

        prompt = _followup_prompt(action, feedback)
        return self.start_run(
            RunRequest(
                prompt=prompt,
                run_name=record.run_name,
                workspace=record.workspace,
                mode="followup",
                parent_run_id=record.run_id,
                model=record.model,
                provider=record.provider,
                thread_id=record.thread_id,
                seed_papers=record.seed_papers,
                seed_ideas=record.seed_ideas,
                auto_approve=record.auto_approve or action == "auto_continue",
                show_thinking=record.show_thinking,
            )
        )

    def stop_run(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        process = self._processes.get(run_id)
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        elif record.pid and self._pid_matches(record):
            try:
                os.killpg(record.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        updated = RunRecord(
            **{
                **asdict(record),
                "status": "stopped",
                "exit_code": process.returncode if process else record.exit_code,
                "status_detail": "Stopped by the user.",
                "finished_at": record.finished_at or _now(),
            }
        )
        self._update_record(updated)
        return self._with_runtime(updated)

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return self._read_events(record)

    def _build_structured_command(
        self,
        request: RunRequest,
        workspace_dir: Path,
        *,
        run_id: str,
        events_path: Path,
        response_dir: Path,
    ) -> list[str]:
        command = [
            *self.runner_command,
            "--prompt",
            request.prompt,
            "--workdir",
            str(workspace_dir),
            "--thread-id",
            request.thread_id,
            "--run-id",
            run_id,
            "--events-file",
            str(events_path),
            "--responses-dir",
            str(response_dir),
        ]
        if request.model:
            command.extend(["--model", request.model])
        if request.provider:
            command.extend(["--provider", request.provider])
        if request.seed_papers:
            command.extend(["--seed-papers", request.seed_papers])
        if request.seed_ideas:
            command.extend(["--seed-ideas", request.seed_ideas])
        if request.auto_approve:
            command.append("--auto-approve")
        if not request.show_thinking:
            command.append("--no-thinking")
        return command

    def _watch_process(
        self,
        run_id: str,
        process: subprocess.Popen,
        log_file: Any,
    ) -> None:
        exit_code = process.wait()
        log_file.close()
        try:
            # Keep the completed Popen registered until final metadata has
            # been persisted.  Otherwise a concurrent poll can see neither a
            # live process nor an exit code and briefly report a completed run
            # with ``exit_code=None``.
            record = self.get_run(run_id)
            if record is None or record.status == "stopped":
                return
            updated = self._finalized_record(record, exit_code)
            self._update_record(updated)
        finally:
            with self._lock:
                self._processes.pop(run_id, None)
                self._stdin_pipes.pop(run_id, None)

    def _refresh_record(self, record: RunRecord) -> RunRecord:
        process = self._processes.get(record.run_id)
        if process and process.poll() is not None and record.status in {
            "queued",
            "running",
            "waiting_for_input",
        }:
            refreshed = self._finalized_record(record, process.returncode)
            self._update_record(refreshed)
            return self._with_runtime(refreshed)
        runtime = self._with_runtime(record)
        if (
            process is None
            and record.status in {"queued", "running", "waiting_for_input", "completed"}
            and not self._pid_matches(record)
        ):
            refreshed = self._reconcile_dead_record(runtime)
            self._update_record(refreshed)
            return refreshed
        return runtime

    def _reconcile_dead_record(self, record: RunRecord) -> RunRecord:
        """Resolve a persisted active run after its detached process exits."""
        events = self._read_events(record)
        checkpoints = checkpoint_events_from_events(events)
        pipeline = inspect_pipeline(
            record.workspace,
            run_status="completed",
            checkpoint_events=checkpoints,
            include_audit=True,
            audit_parameters=pipeline_parameters_for_record(record),
        )
        if pipeline["completion"]["verified"]:
            status = "pipeline_completed"
            detail = "Recovered completion proof after the Web service restarted."
            exit_code = record.exit_code
        elif record.interaction is not None:
            status = "checkpoint_reached"
            detail = "The original process is gone; resume this checkpoint explicitly."
            exit_code = record.exit_code
        else:
            terminal_event = next(
                (
                    str(event.get("type") or "")
                    for event in reversed(events)
                    if event.get("type") in {"runner_finished", "runner_failed"}
                ),
                "",
            )
            if terminal_event == "runner_finished":
                return self._finalized_record(record, 0)
            if terminal_event == "runner_failed":
                return self._finalized_record(record, 1)
            status = "stale"
            detail = (
                "The Web service restarted and the recorded process is no longer running."
            )
            exit_code = record.exit_code
        return RunRecord(
            **{
                **asdict(record),
                "status": status,
                "status_detail": detail,
                "exit_code": exit_code,
                "finished_at": record.finished_at or _now(),
                "current_stage": pipeline["active_stage"] or pipeline["next_stage"],
                "completed_stages": pipeline["completed_count"],
                "total_stages": pipeline["total_stages"],
                "progress": pipeline["active_progress"],
                "completion": pipeline["completion"],
            }
        )

    def _with_runtime(self, record: RunRecord, *, limit: int = 24000) -> RunRecord:
        log_tail = ""
        if record.log_path:
            path = Path(record.log_path)
            if path.exists():
                data = path.read_bytes()[-limit:]
                log_tail = data.decode("utf-8", errors="replace")
        thread_id = record.thread_id or _detect_thread_id(log_tail)
        events = self._read_events(record)
        auto_approve = record.auto_approve or _automation_enabled_from_events(events)
        checkpoints = checkpoint_events_from_events(events)
        interaction = _pending_structured_interaction(events)
        if interaction is None:
            interaction = _detect_interaction(log_tail, status=record.status)

        status = record.status
        status_detail = record.status_detail
        alive = self._is_live(record)
        if interaction is not None and status not in {"stopped", "failed", "stale", "pipeline_completed"}:
            status = "waiting_for_input" if alive else "checkpoint_reached"
            status_detail = (
                "The research process is waiting for a structured browser response."
                if alive
                else "The original process is gone; resume this checkpoint explicitly."
            )
        elif status in {"queued", "running", "waiting_for_input"} and alive:
            status = "running"
            status_detail = "Research process is running."

        include_audit = status in {
            "completed",
            "pipeline_completed",
            "failed",
            "stopped",
            "stale",
            "checkpoint_reached",
        }
        pipeline = inspect_pipeline(
            record.workspace,
            run_status=status,
            checkpoint_events=checkpoints,
            include_audit=include_audit,
            audit_parameters=pipeline_parameters_for_record(record),
        )

        if status == "completed":
            if pipeline["completion"]["verified"]:
                status = "pipeline_completed"
            elif interaction is not None:
                status = "checkpoint_reached"
            else:
                status = "failed"
        elif status == "pipeline_completed" and not pipeline["completion"]["verified"]:
            # Persisted metadata can predate stricter completion checks.  Never
            # keep advertising completion when the current proof says otherwise.
            status = "failed"
            if pipeline["completion"].get("missing_gate_proofs"):
                status_detail = (
                    "Recorded completion is unverified because the Stage 12 "
                    "gate proof is missing."
                )
            else:
                status_detail = "Recorded completion no longer passes verification."

        return RunRecord(
            **{
                **asdict(record),
                "status": status,
                "status_detail": status_detail,
                "thread_id": thread_id,
                "auto_approve": auto_approve,
                "log_tail": log_tail,
                "interaction": interaction,
                "current_stage": pipeline["active_stage"] or pipeline["next_stage"],
                "completed_stages": pipeline["completed_count"],
                "total_stages": pipeline["total_stages"],
                "progress": pipeline["active_progress"],
                "completion": pipeline["completion"],
            }
        )

    # Backwards-compatible helper name used by older integrations.
    def _with_log_tail(self, record: RunRecord, *, limit: int = 24000) -> RunRecord:
        return self._with_runtime(record, limit=limit)

    def _finalized_record(self, record: RunRecord, exit_code: int) -> RunRecord:
        runtime = self._with_runtime(record)
        events = self._read_events(runtime)
        checkpoints = checkpoint_events_from_events(events)
        pipeline = inspect_pipeline(
            runtime.workspace,
            run_status="completed" if exit_code == 0 else "failed",
            checkpoint_events=checkpoints,
            include_audit=True,
            audit_parameters=pipeline_parameters_for_record(runtime),
        )
        if exit_code == 0 and pipeline["completion"]["verified"]:
            status = "pipeline_completed"
            detail = "Stage 12, all required checkpoints, final_report.md, and artifact audit are complete."
        elif runtime.interaction is not None:
            status = "checkpoint_reached"
            detail = "The process exited at a recoverable checkpoint. Resume explicitly to continue."
        else:
            status = "failed"
            if exit_code == 0:
                missing = pipeline["completion"].get("missing_checkpoints") or []
                if missing:
                    detail = "Process exited before all structured human checkpoints were recorded."
                elif not pipeline["completion"].get("final_report_present"):
                    detail = "Process exited before Stage 12 produced final_report.md."
                elif pipeline["completion"].get("missing_gate_proofs"):
                    detail = "Process exited without a successful Stage 12 gate proof."
                elif not pipeline["completion"].get("reflections_ready"):
                    detail = "Process exited before all stage reflections were verified."
                else:
                    detail = "Process exited without a passing artifact audit."
            else:
                detail = f"Research process exited with code {exit_code}."
        return RunRecord(
            **{
                **asdict(runtime),
                "status": status,
                "status_detail": detail,
                "exit_code": exit_code,
                "finished_at": runtime.finished_at or _now(),
                "current_stage": pipeline["active_stage"] or pipeline["next_stage"],
                "completed_stages": pipeline["completed_count"],
                "total_stages": pipeline["total_stages"],
                "progress": pipeline["active_progress"],
                "completion": pipeline["completion"],
            }
        )

    def _update_record(self, updated: RunRecord) -> None:
        with self._lock:
            records = self._read_metadata()
            replaced = False
            for index, item in enumerate(records):
                if item.get("run_id") == updated.run_id:
                    records[index] = self._persisted_dict(updated)
                    replaced = True
                    break
            if not replaced:
                records.insert(0, self._persisted_dict(updated))
            self._write_metadata(records)

    @staticmethod
    def _persisted_dict(record: RunRecord) -> dict[str, Any]:
        value = asdict(record)
        value["log_tail"] = ""
        value["interaction"] = None
        return value

    def _read_metadata(self) -> list[dict[str, Any]]:
        if not self.runs_file.exists():
            return []
        try:
            data = json.loads(self.runs_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return []
        return data if isinstance(data, list) else []

    def _write_metadata(self, records: list[dict[str, Any]]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _record_from_dict(data: dict[str, Any]) -> RunRecord:
        fields = RunRecord.__dataclass_fields__
        record = RunRecord(**{key: data[key] for key in fields if key in data})
        if not record.pipeline_parameters:
            inferred = _infer_pipeline_parameters(record.workspace)
            if inferred:
                record = RunRecord(
                    **{
                        **asdict(record),
                        "pipeline_parameters": inferred,
                    }
                )
        return record

    def _read_events(self, record: RunRecord) -> list[dict[str, Any]]:
        if not record.events_path:
            return []
        path = Path(record.events_path)
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
        except OSError:
            return []
        return events

    def _inherit_approved_checkpoints(
        self,
        parent_run_id: str,
        events_path: Path,
        *,
        workspace: Path,
        thread_id: str,
    ) -> None:
        """Seed a resumed run with explicit approvals from its parent chain.

        A model may legitimately end its turn immediately after a human
        checkpoint.  The next browser-managed resume must retain that decision
        so completion proof can span multiple turns on the same LangGraph
        thread.  Only explicit ``approved: true`` responses are inherited;
        rejected decisions and older ambiguous events are intentionally left
        behind.
        """
        parent = self.get_run(parent_run_id)
        if parent is None:
            raise ValueError(f"Parent run not found: {parent_run_id}")
        if Path(parent.workspace).expanduser().resolve() != workspace.resolve():
            raise ValueError("A resumed run must use the parent run's workspace.")
        if parent.thread_id and parent.thread_id != thread_id:
            raise ValueError("A resumed run must use the parent run's thread ID.")

        approved = _explicitly_approved_checkpoint_events(self._read_events(parent))
        if not approved:
            return

        events_path.parent.mkdir(parents=True, exist_ok=True)
        inherited_events: list[dict[str, Any]] = []
        for stage, source_interaction_id in approved:
            interaction_id = f"inherited:{parent_run_id}:{stage}"
            provenance = {
                "inherited_from_run_id": parent_run_id,
                "source_interaction_id": source_interaction_id,
            }
            inherited_events.extend(
                [
                    {
                        "type": "interaction_requested",
                        "at": _now(),
                        "interaction_id": interaction_id,
                        "interaction": {
                            "kind": "checkpoint",
                            "questions": [],
                            "allows_cancel": False,
                        },
                        "checkpoint_stage": stage,
                        **provenance,
                    },
                    {
                        "type": "interaction_resolved",
                        "at": _now(),
                        "interaction_id": interaction_id,
                        "checkpoint_stage": stage,
                        "response": {
                            "status": "answered",
                            "answer_count": 1,
                            "decision": "approve",
                            "approved": True,
                            "inherited": True,
                        },
                        **provenance,
                    },
                ]
            )
        with events_path.open("a", encoding="utf-8") as handle:
            for event in inherited_events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _is_live(self, record: RunRecord) -> bool:
        process = self._processes.get(record.run_id)
        if process is not None:
            return process.poll() is None
        if record.status not in {"queued", "running", "waiting_for_input"}:
            return False
        return self._pid_matches(record)

    @staticmethod
    def _pid_matches(record: RunRecord) -> bool:
        if not record.pid:
            return False
        try:
            os.kill(record.pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        try:
            result = subprocess.run(
                ["ps", "-p", str(record.pid), "-o", "command="],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        command = result.stdout
        if record.run_id and record.run_id in command:
            return True
        return bool(record.workspace and record.workspace in command)

    def _reconcile_persisted_runs(self) -> None:
        if not self.runs_file.is_file():
            return
        with self._lock:
            raw_records = self._read_metadata()
        changed = False
        reconciled: list[dict[str, Any]] = []
        for item in raw_records:
            record = self._record_from_dict(item)
            if record.status not in {"queued", "running", "waiting_for_input", "completed"}:
                persisted = self._persisted_dict(record)
                changed = changed or persisted != item
                reconciled.append(persisted)
                continue
            runtime = self._with_runtime(record)
            alive = self._pid_matches(runtime)
            if runtime.status in {"running", "waiting_for_input"} and alive:
                updated = runtime
            else:
                updated = self._reconcile_dead_record(runtime)
            persisted = self._persisted_dict(updated)
            changed = changed or persisted != item
            reconciled.append(persisted)
        if changed:
            with self._lock:
                self._write_metadata(reconciled)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_pipeline_parameters(values: dict[str, Any] | None) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key in _PIPELINE_PARAMETER_DEFAULTS:
        candidate = (values or {}).get(key)
        try:
            parsed = int(candidate)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized[key] = parsed
    return normalized


def _current_pipeline_parameters() -> dict[str, int]:
    """Snapshot non-secret pipeline limits for reproducible run auditing."""
    try:
        from autoidea.config import get_effective_config

        config = get_effective_config()
    except Exception:  # pragma: no cover - defensive startup fallback
        return dict(_PIPELINE_PARAMETER_DEFAULTS)
    values = {
        key: getattr(config, key, default)
        for key, default in _PIPELINE_PARAMETER_DEFAULTS.items()
    }
    return {
        **_PIPELINE_PARAMETER_DEFAULTS,
        **_normalize_pipeline_parameters(values),
    }


def _infer_pipeline_parameters(workspace: str | Path) -> dict[str, int]:
    """Recover limits for Web runs created before parameter snapshots existed."""
    root = Path(workspace).expanduser().resolve()
    inferred: dict[str, int] = {}
    for name in ("research_brief.md", "final_report.md"):
        path = root / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for key in _PIPELINE_PARAMETER_DEFAULTS:
            if key in inferred:
                continue
            patterns = (
                rf"`?{re.escape(key)}`?\s*(?::|=)\s*`?(\d+)",
                rf"\b{re.escape(key)}\b\s+`?(\d+)",
            )
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and int(match.group(1)) > 0:
                    inferred[key] = int(match.group(1))
                    break
    return inferred


def pipeline_parameters_for_record(record: RunRecord) -> dict[str, int]:
    """Return the immutable run limits, with a legacy artifact fallback."""
    saved = _normalize_pipeline_parameters(record.pipeline_parameters)
    inferred = _infer_pipeline_parameters(record.workspace)
    return {**inferred, **saved}


def _explicitly_approved_checkpoint_events(
    events: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Return one explicit approval per mandatory checkpoint stage."""
    requested: dict[str, str] = {}
    approved: dict[str, str] = {}
    for event in events:
        interaction_id = str(event.get("interaction_id") or "")
        if event.get("type") == "interaction_requested" and interaction_id:
            stage = str(event.get("checkpoint_stage") or "")
            if stage in {"stage_7", "stage_9", "stage_10"}:
                requested[interaction_id] = stage
            continue
        if event.get("type") != "interaction_resolved" or not interaction_id:
            continue
        stage = requested.get(interaction_id)
        response = event.get("response")
        if stage and isinstance(response, dict) and response.get("approved") is True:
            approved[stage] = interaction_id
    return sorted(approved.items())


def _automation_enabled_from_events(events: list[dict[str, Any]]) -> bool:
    """Return whether a manual checkpoint switched the run to automatic mode."""
    return any(
        event.get("type") == "interaction_resolved"
        and isinstance(event.get("response"), dict)
        and event["response"].get("automation_enabled") is True
        for event in events
    )


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return cleaned.strip("-_") or uuid.uuid4().hex[:10]


def _unique_workspace(path: Path) -> Path:
    """Return a non-existing workspace path without reusing prior artifacts."""
    candidate = path.expanduser().resolve()
    if not candidate.exists():
        return candidate
    if candidate.is_dir():
        try:
            if not any(candidate.iterdir()):
                return candidate
        except OSError:
            pass
    counter = 2
    while True:
        alternate = candidate.with_name(f"{candidate.name}-{counter}")
        if not alternate.exists():
            return alternate
        counter += 1


def _pending_structured_interaction(
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    pending: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        interaction_id = str(event.get("interaction_id") or "")
        if not interaction_id:
            continue
        if event.get("type") == "interaction_requested":
            raw = event.get("interaction")
            interaction = dict(raw) if isinstance(raw, dict) else {}
            interaction["interaction_id"] = interaction_id
            checkpoint_stage = str(event.get("checkpoint_stage") or "")
            if checkpoint_stage:
                interaction["checkpoint_stage"] = checkpoint_stage
            pending[interaction_id] = interaction
            order.append(interaction_id)
        elif event.get("type") == "interaction_resolved":
            pending.pop(interaction_id, None)
    for interaction_id in reversed(order):
        if interaction_id in pending:
            return pending[interaction_id]
    return None


_QUESTION_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")
_OPTION_RE = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")
_CHOICE_RE = re.compile(r"Choice \[([A-Z](?:/[A-Z])*)\]:\s*$")
_TEXT_PROMPT_RE = re.compile(r">\s+(Answer|Your answer):\s*$")
_THREAD_RE = re.compile(r"^Thread:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)
_PROGRESS_AFTER_PROMPT_RE = re.compile(
    r"^\s*(?:"
    r"Warning:|"
    r"Error:|"
    r"Traceback|"
    r"\[Error\]|"
    r"[✓●▶⠋]\s|"
    r"Stage\s+\d|"
    r".*Stage gate|"
    r".*multi_source_search|"
    r".*Model call failed|"
    r".*rate limited|"
    r".*timed out|"
    r".*completed"
    r")",
    re.IGNORECASE,
)
_CHECKPOINT_MARKERS = ("HITL", "检查点", "checkpoint", "请审阅", "Would you like")
_CHECKPOINT_QUESTION_MARKERS = ("您希望", "Would you like", "would you like")
_CHECKPOINT_OPTION_ORDER = ("approve", "auto_continue", "modify", "regenerate")
_CHECKPOINT_DEFAULT_OPTIONS = {
    "approve": {"key": "approve", "label": "批准 — 继续进入下一阶段"},
    "auto_continue": {
        "key": "auto_continue",
        "label": "不回答，后续全部自动",
    },
    "modify": {"key": "modify", "label": "修改 — 根据反馈调整当前阶段结果"},
    "regenerate": {"key": "regenerate", "label": "重新运行 — 重新执行当前阶段"},
}


def _detect_thread_id(log_tail: str) -> str:
    match = _THREAD_RE.search(log_tail or "")
    return match.group(1) if match else ""


def _detect_interaction(log_tail: str, *, status: str = "running") -> dict[str, Any] | None:
    """Detect the latest CLI prompt that needs browser input."""
    if not log_tail:
        return None

    lines = log_tail.splitlines()
    if status in {"completed", "checkpoint_reached"}:
        checkpoint = _detect_checkpoint_review(log_tail)
        if checkpoint is not None:
            return checkpoint

    for index in range(len(lines) - 1, -1, -1):
        if _TEXT_PROMPT_RE.search(lines[index]):
            return {
                "kind": "text",
                "prompt": "Answer:",
                "question": _nearest_question(lines, index),
                "options": [],
                "allows_other": True,
            }

        match = _CHOICE_RE.search(lines[index])
        if not match:
            continue
        if _has_progress_after_prompt(lines, index):
            return None
        question_index, question = _nearest_question_with_index(lines, index)
        option_lines = lines[question_index + 1:index] if question_index >= 0 else lines[:index]
        options, allows_other = _choice_options(option_lines)
        return {
            "kind": "multiple_choice",
            "prompt": lines[index].strip(),
            "question": question,
            "options": options,
            "allows_other": allows_other,
        }
    return None


def _has_progress_after_prompt(lines: list[str], prompt_index: int) -> bool:
    for line in lines[prompt_index + 1:]:
        text = line.strip()
        if not text:
            continue
        if _QUESTION_RE.match(line) or _CHOICE_RE.search(line) or _TEXT_PROMPT_RE.search(line):
            return False
        return bool(_PROGRESS_AFTER_PROMPT_RE.search(line)) or not _looks_like_prompt_continuation(line)
    return False


def _looks_like_prompt_continuation(line: str) -> bool:
    stripped = line.strip()
    return bool(_OPTION_RE.match(line) or stripped.startswith(("A.", "B.", "C.", "D.", "E.", "F.")))


def _detect_checkpoint_review(log_tail: str) -> dict[str, Any] | None:
    if not any(marker in log_tail for marker in _CHECKPOINT_MARKERS):
        return None

    lines = log_tail.splitlines()
    question_index, question = _latest_checkpoint_question(lines)
    if question_index < 0:
        return None

    options = _checkpoint_options(lines[question_index + 1: question_index + 16])
    if len(options) < 3:
        return None

    return {
        "kind": "checkpoint_review",
        "prompt": "HITL checkpoint review",
        "question": question,
        "options": options,
        "allows_other": True,
    }


def _followup_prompt(action: str, feedback: str = "") -> str:
    labels = {
        "approve": "批准",
        "auto_continue": "批准当前检查点，后续全部自动",
        "modify": "修改",
        "regenerate": "重新生成",
    }
    label = labels.get(action, action or "继续")
    detail = feedback.strip() or "无补充说明。"
    return (
        "继续上一轮 AutoIdea HITL 检查点。\n"
        f"用户选择：{label}\n"
        f"用户反馈：{detail}\n"
        "请基于该反馈继续执行后续研究流程。"
    )


def _latest_checkpoint_question(lines: list[str]) -> tuple[int, str]:
    for index in range(len(lines) - 1, -1, -1):
        text = lines[index].strip()
        if not text:
            continue
        if any(marker in text for marker in _CHECKPOINT_QUESTION_MARKERS):
            return index, text
    return -1, ""


def _checkpoint_options(lines: list[str]) -> list[dict[str, str]]:
    by_key: dict[str, dict[str, str]] = {}
    for line in lines:
        key = _checkpoint_option_key(line)
        if key is None or key in by_key:
            continue
        by_key[key] = {"key": key, "label": _checkpoint_option_label(line, key)}
    by_key.setdefault("auto_continue", _CHECKPOINT_DEFAULT_OPTIONS["auto_continue"])
    return [by_key[key] for key in _CHECKPOINT_OPTION_ORDER if key in by_key]


def _checkpoint_option_key(line: str) -> str | None:
    normalized = line.casefold()
    if "后续全自动" in line or "continue automatically" in normalized:
        return "auto_continue"
    if "批准" in line or "approve" in normalized:
        return "approve"
    if "修改" in line or "modify" in normalized:
        return "modify"
    if (
        "重新" in line
        or "重跑" in line
        or "rerun" in normalized
        or "re-run" in normalized
        or "regenerate" in normalized
    ):
        return "regenerate"
    return None


def _checkpoint_option_label(line: str, key: str) -> str:
    text = line.strip()
    for token in _checkpoint_label_tokens(key):
        index = text.find(token)
        if index >= 0:
            return text[index:].strip()
    return _CHECKPOINT_DEFAULT_OPTIONS[key]["label"]


def _checkpoint_label_tokens(key: str) -> tuple[str, ...]:
    if key == "approve":
        return ("批准", "Approve", "approve")
    if key == "modify":
        return ("修改", "Modify", "modify")
    if key == "auto_continue":
        return ("不回答", "后续全自动", "Continue automatically")
    return ("重新生成", "重新辩论", "重新运行", "重跑", "重新", "Regenerate", "regenerate", "Re-run", "re-run", "Rerun", "rerun")


def _nearest_question(lines: list[str], before_index: int) -> str:
    return _nearest_question_with_index(lines, before_index)[1]


def _nearest_question_with_index(lines: list[str], before_index: int) -> tuple[int, str]:
    for index in range(before_index - 1, -1, -1):
        match = _QUESTION_RE.match(lines[index])
        if match:
            return index, match.group(1).strip()
    return -1, ""


def _choice_options(lines: list[str]) -> tuple[list[dict[str, str]], bool]:
    options: list[dict[str, str]] = []
    allows_other = False
    for line in lines:
        match = _OPTION_RE.match(line)
        if not match:
            continue
        key = match.group(1)
        label = match.group(2).strip()
        if label.lower().startswith("other"):
            allows_other = True
            continue
        options.append({"key": key, "label": label})
    return options, allows_other
