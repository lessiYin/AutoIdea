"""Structured subprocess bridge between the Web workbench and AutoIdea.

The regular CLI is optimized for a human terminal.  This module runs the same
agent and checkpointer, but persists interaction events and waits for response
files so a browser can resume the graph reliably—even after the Web server is
restarted.  It is intentionally an internal module, launched by
``WebRunManager`` rather than exposed as a public CLI command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any

from .pipeline import infer_checkpoint_stage


class EventBridge:
    """Persist structured events and synchronously await browser responses."""

    def __init__(
        self,
        events_path: Path,
        response_dir: Path,
        workspace: Path,
        *,
        auto_approve: bool,
    ) -> None:
        self.events_path = events_path
        self.response_dir = response_dir
        self.workspace = workspace
        self.auto_approve = auto_approve
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.response_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {
            "type": event_type,
            "at": _now(),
            **_redact(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def ask_user(self, data: dict[str, Any]) -> dict[str, Any]:
        interaction_id = uuid.uuid4().hex
        tool_call_id = str(data.get("tool_call_id") or "")
        checkpoint_stage = _checkpoint_stage_from_tool_call_id(tool_call_id)
        if not checkpoint_stage:
            checkpoint_stage = infer_checkpoint_stage(self.workspace)
        questions = data.get("questions") if isinstance(data, dict) else []
        if not isinstance(questions, list):
            questions = []
        self.emit(
            "interaction_requested",
            interaction_id=interaction_id,
            interaction={
                "kind": "checkpoint" if checkpoint_stage else "ask_user",
                "questions": questions,
                "allows_cancel": not bool(checkpoint_stage),
            },
            checkpoint_stage=checkpoint_stage,
        )
        automatic = self.auto_approve
        if automatic:
            answers: list[str] = []
            for index, question in enumerate(questions):
                choices = question.get("choices", []) if isinstance(question, dict) else []
                values = [
                    str(choice.get("value") or "")
                    for choice in choices
                    if isinstance(choice, dict)
                ]
                approved = next(
                    (value for value in values if value.casefold() == "approve"),
                    "",
                )
                if checkpoint_stage and index == 0:
                    answers.append("approve")
                elif isinstance(question, dict) and question.get("type") == "multiple_choice":
                    answers.append(approved or "Proceed automatically using your best judgment.")
                elif isinstance(question, dict) and question.get("required") is False:
                    answers.append("")
                else:
                    answers.append("Proceed automatically using your best judgment.")
            response = {
                "status": "answered",
                "answers": answers
                or ["approve" if checkpoint_stage else "Proceed automatically using your best judgment."],
            }
        else:
            response = self._wait_for_response(interaction_id)
        status = str(response.get("status") or "answered")
        answers = response.get("answers")
        if not isinstance(answers, list):
            value = response.get("value", "")
            answers = [str(value)] if questions else []
        normalized = {"status": status, "answers": [str(value) for value in answers]}
        decision = _checkpoint_decision(normalized["answers"])
        automation_enabled = bool(checkpoint_stage and decision == "auto_continue")
        if automation_enabled:
            self.auto_approve = True
        approved = bool(
            checkpoint_stage and decision in {"approve", "auto_continue"}
        )
        self.emit(
            "interaction_resolved",
            interaction_id=interaction_id,
            checkpoint_stage=checkpoint_stage,
            response={
                "status": status,
                "answer_count": len(normalized["answers"]),
                "decision": decision,
                "approved": approved,
                "mode": "automatic" if automatic or automation_enabled else "manual",
                "automation_enabled": automation_enabled,
            },
        )
        return normalized

    def approve_actions(self, action_requests: list[Any]) -> list[dict[str, Any]] | None:
        safe_actions = [_plain_action(action) for action in action_requests]
        if self.auto_approve:
            self.emit("tool_approval_resolved", mode="automatic", actions=safe_actions)
            return [{"type": "approve"} for _ in action_requests]

        interaction_id = uuid.uuid4().hex
        self.emit(
            "interaction_requested",
            interaction_id=interaction_id,
            interaction={
                "kind": "tool_approval",
                "actions": safe_actions,
                "questions": [],
                "allows_cancel": True,
            },
            checkpoint_stage="",
        )
        response = self._wait_for_response(interaction_id)
        decision = str(response.get("decision") or response.get("value") or "reject")
        approved = decision.casefold() in {"approve", "approved", "yes", "y", "1"}
        self.emit(
            "interaction_resolved",
            interaction_id=interaction_id,
            checkpoint_stage="",
            response={"decision": "approve" if approved else "reject"},
        )
        if not approved:
            return None
        return [{"type": "approve"} for _ in action_requests]

    def _wait_for_response(self, interaction_id: str) -> dict[str, Any]:
        response_path = self.response_dir / f"{interaction_id}.json"
        while True:
            try:
                if response_path.is_file():
                    value = json.loads(response_path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        return value
                    return {"status": "error", "value": "Invalid response payload."}
            except (OSError, json.JSONDecodeError, UnicodeError):
                # The writer uses atomic rename, but tolerate transient filesystem
                # errors without losing a long research run.
                pass
            time.sleep(0.15)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoIdea structured Web runner")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--events-file", required=True)
    parser.add_argument("--responses-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--seed-papers", default="")
    parser.add_argument("--seed-ideas", default="")
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = Path(args.workdir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    bridge = EventBridge(
        Path(args.events_file).expanduser().resolve(),
        Path(args.responses_dir).expanduser().resolve(),
        workspace,
        auto_approve=bool(args.auto_approve),
    )
    bridge.emit(
        "runner_started",
        run_id=args.run_id,
        thread_id=args.thread_id,
        workspace=str(workspace),
    )

    try:
        _run_agent(args, workspace, bridge)
    except KeyboardInterrupt:
        bridge.emit("runner_stopped", reason="interrupt")
        return 130
    except Exception as exc:  # noqa: BLE001 - persist any runner failure for the UI
        bridge.emit(
            "runner_failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        traceback.print_exc()
        return 1

    bridge.emit("runner_finished", run_id=args.run_id)
    return 0


def _run_agent(args: argparse.Namespace, workspace: Path, bridge: EventBridge) -> None:
    import asyncio

    import nest_asyncio  # type: ignore[import-untyped]

    from autoidea.cli.agent import _load_agent
    from autoidea.config import (
        apply_config_to_env,
        get_effective_config,
        validate_runtime_config,
    )
    from autoidea.paths import ensure_dirs, set_workspace_root
    from autoidea.sessions import get_checkpointer
    from autoidea.stream.display import (
        _inspect_autonomous_progress,
        _run_streaming_to_pipeline_completion,
    )
    from autoidea.tools.stage_gate import configure_web_checkpoint_events
    from autoidea.web.pipeline import checkpoint_events_from_events

    config = get_effective_config()
    if args.model:
        config.model = args.model
    if args.provider:
        config.provider = args.provider
    if args.seed_papers:
        config.seed_papers_file = str(Path(args.seed_papers).expanduser().resolve())
    if args.seed_ideas:
        config.seed_ideas_file = str(Path(args.seed_ideas).expanduser().resolve())
    config.auto_approve = bridge.auto_approve
    config.show_thinking = not bool(args.no_thinking)
    config_errors = validate_runtime_config(config)
    if config_errors:
        raise RuntimeError("Invalid runtime configuration: " + " ".join(config_errors))
    apply_config_to_env(config)
    configure_web_checkpoint_events(bridge.events_path)
    set_workspace_root(str(workspace))
    ensure_dirs()
    nest_asyncio.apply()

    async def execute() -> None:
        async with get_checkpointer() as checkpointer:
            agent = _load_agent(
                workspace_dir=str(workspace),
                checkpointer=checkpointer,
                config=config,
            )
            def progress_probe() -> dict[str, Any]:
                events: list[dict[str, Any]] = []
                try:
                    lines = bridge.events_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()
                except OSError:
                    lines = []
                for line in lines:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        events.append(event)
                return _inspect_autonomous_progress(
                    workspace,
                    checkpoint_events=checkpoint_events_from_events(events),
                    require_checkpoint_events=True,
                    audit_parameters={
                        "deep_reading_top_k": config.deep_reading_top_k,
                        "target_paper_count": config.target_paper_count,
                    },
                )

            _run_streaming_to_pipeline_completion(
                agent=agent,
                message=args.prompt,
                thread_id=args.thread_id,
                show_thinking=config.show_thinking,
                interactive=True,
                metadata={
                    "workspace_dir": str(workspace),
                    "model": config.model,
                    "pipeline_parameters": {
                        "target_paper_count": config.target_paper_count,
                        "deep_reading_top_k": config.deep_reading_top_k,
                        "max_ideas_to_generate": config.max_ideas_to_generate,
                        "top_k_ranked": config.top_k_ranked,
                        "max_debate_rounds": config.max_debate_rounds,
                    },
                },
                hitl_prompt_fn=bridge.approve_actions,
                ask_user_prompt_fn=bridge.ask_user,
                progress_probe=progress_probe,
                on_continuation=lambda attempt, progress: bridge.emit(
                    "runner_continuing",
                    continuation=attempt,
                    next_stage=progress.get("next_stage", ""),
                    reasons=progress.get("reasons", []),
                ),
            )

    asyncio.run(execute())


def _plain_action(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        name = str(action.get("name") or "action")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
    else:
        name = str(getattr(action, "name", "action"))
        candidate = getattr(action, "args", {})
        args = candidate if isinstance(candidate, dict) else {}
    return {"name": name, "args": _redact(args)}


def _checkpoint_stage_from_tool_call_id(tool_call_id: str) -> str:
    prefix = "autoidea-checkpoint:"
    if not tool_call_id.startswith(prefix):
        return ""
    stage = tool_call_id[len(prefix):]
    return stage if stage in {"stage_7", "stage_9", "stage_10"} else ""


def _checkpoint_decision(answers: list[str]) -> str:
    if not answers:
        return ""
    choice = answers[0].strip().casefold()
    candidates = {choice}
    candidates.update(part.strip() for part in choice.split("/") if part.strip())
    if candidates & {
        "approve",
        "approved",
        "accept",
        "continue",
        "yes",
        "批准",
        "通过",
        "继续",
    }:
        return "approve"
    if candidates & {
        "auto_continue",
        "auto-continue",
        "continue automatically",
        "后续全自动",
        "不回答，后续全自动",
    }:
        return "auto_continue"
    if candidates & {"revise", "modify", "修改", "修订"}:
        return "revise"
    if candidates & {"rerun", "re-run", "regenerate", "重新运行", "重新生成"}:
        return "rerun"
    return choice


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in ("key", "token", "secret", "password")):
                redacted[str(key)] = "••••••••"
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main(sys.argv[1:]))
