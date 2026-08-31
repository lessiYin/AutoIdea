from __future__ import annotations

import asyncio

import pytest

from autoidea.cli import interactive
from autoidea.stream import display
from autoidea.stream.display import _format_tool_compact
from autoidea.stream.events import stream_agent_events
from autoidea.tools.pipeline_state import STAGES


class _CaptureAgent:
    def __init__(self) -> None:
        self.config = None

    async def astream(self, _input, *, config, **_kwargs):
        self.config = config
        if False:
            yield None


def test_stream_runtime_serializes_tool_execution_without_dropping_calls() -> None:
    agent = _CaptureAgent()

    async def consume() -> None:
        async for _event in stream_agent_events(agent, "test", "thread-1"):
            pass

    asyncio.run(consume())

    assert agent.config["configurable"]["thread_id"] == "thread-1"
    assert agent.config["max_concurrency"] == 1


def test_tournament_display_counts_serialized_ideas() -> None:
    ideas_json = '[{"id":"IDEA-001"},{"id":"IDEA-002"}]'

    assert (
        _format_tool_compact(
            "rank_ideas_tournament",
            {"ideas_json": ideas_json, "comparisons_json": "[]"},
        )
        == "rank_ideas_tournament(2 ideas)"
    )


def test_cli_auto_continue_makes_later_questions_automatic(monkeypatch) -> None:
    answers = iter(["B", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    display.set_session_auto_approve(False)
    try:
        first = display._resolve_ask_user_prompt(
            {
                "questions": [
                    {
                        "question": "Review this checkpoint?",
                        "type": "multiple_choice",
                        "choices": [
                            {"value": "approve"},
                            {"value": "auto_continue"},
                        ],
                    },
                    {
                        "question": "Optional feedback",
                        "type": "text",
                        "required": False,
                    },
                ]
            }
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda _prompt: pytest.fail("automatic mode must not prompt"),
        )
        second = display._resolve_ask_user_prompt(
            {
                "questions": [
                    {
                        "question": "Review the next checkpoint?",
                        "type": "multiple_choice",
                        "choices": [{"value": "approve"}],
                    }
                ]
            }
        )
    finally:
        display.set_session_auto_approve(False)

    assert first == {
        "status": "answered",
        "answers": ["auto_continue", ""],
    }
    assert second == {"status": "answered", "answers": ["approve"]}


def test_autonomous_runtime_continues_in_same_agent_and_thread(
    monkeypatch,
    tmp_path,
) -> None:
    agent = object()
    calls: list[dict] = []
    continuations: list[tuple[int, str]] = []
    states = iter(
        [
            {"complete": False, "next_stage": "stage_1", "fingerprint": "before"},
            {
                "complete": False,
                "next_stage": "stage_2",
                "reasons": ["missing artifact(s): task_formalization.md"],
                "fingerprint": "stage-1-done",
            },
            {"complete": True, "next_stage": "complete", "fingerprint": "done"},
        ]
    )

    def fake_run_streaming(**kwargs):
        calls.append(kwargs)
        return f"response-{len(calls)}"

    monkeypatch.setattr(display, "_run_streaming", fake_run_streaming)
    result = display._run_streaming_to_pipeline_completion(
        agent=agent,
        message="start",
        thread_id="thread-one",
        show_thinking=False,
        interactive=False,
        metadata={"workspace_dir": str(tmp_path)},
        progress_probe=lambda: next(states),
        on_continuation=lambda attempt, progress: continuations.append(
            (attempt, progress["next_stage"])
        ),
    )

    assert result == "response-2"
    assert len(calls) == 2
    assert {id(call["agent"]) for call in calls} == {id(agent)}
    assert {call["thread_id"] for call in calls} == {"thread-one"}
    assert [call["progress_tracker"].stage for call in calls] == ["stage_1", "stage_2"]
    assert calls[0]["message"].startswith("start\n\n[AutoIdea runtime execution boundary]")
    assert "stage_1" in calls[0]["message"]
    assert "stage_2" in calls[1]["message"]
    assert continuations == [(1, "stage_2")]


def test_cli_manual_checkpoints_still_use_autonomous_pipeline_loop(monkeypatch) -> None:
    captured: dict = {}

    monkeypatch.setattr(
        interactive,
        "_astream_to_console",
        lambda **kwargs: captured.update(kwargs) or "done",
    )
    monkeypatch.setattr(display, "set_session_auto_approve", lambda _enabled: None)

    interactive.cmd_run(
        agent=object(),
        prompt="research this",
        thread_id="thread-manual",
        show_thinking=False,
        workspace_dir="/tmp/manual-progress",
        auto_approve=False,
    )

    assert captured["auto_complete"] is True
    assert captured["interactive"] is False


def test_autonomous_runtime_stops_after_consecutive_no_progress(monkeypatch) -> None:
    calls = 0

    def fake_run_streaming(**_kwargs):
        nonlocal calls
        calls += 1
        return "still working"

    monkeypatch.setattr(display, "_run_streaming", fake_run_streaming)
    stalled = {
        "complete": False,
        "next_stage": "stage_3",
        "reasons": ["missing artifact(s): paper_registry.json"],
        "fingerprint": "unchanged",
    }

    with pytest.raises(RuntimeError, match="no material progress.*stage_3"):
        display._run_streaming_to_pipeline_completion(
            agent=object(),
            message="start",
            thread_id="same-thread",
            show_thinking=False,
            interactive=False,
            progress_probe=lambda: stalled,
            max_continuations=10,
            max_stalled_turns=3,
        )

    assert calls == 3


def test_web_completion_probe_requires_reflections_checkpoints_and_audit(
    monkeypatch,
    tmp_path,
) -> None:
    core_state = {
        "stages": {
            spec.stage: {
                "status": "complete",
                "missing_artifacts": [],
                "has_reflection": True,
                "validation_issues": [],
            }
            for spec in STAGES
        }
    }
    pipeline = {
        "completed_count": len(STAGES),
        "total_stages": len(STAGES),
            "completion": {
                "required_artifacts_ready": True,
                "final_report_present": True,
                "reflections_ready": True,
                "audit_passed": True,
                "audit_issues": [],
            },
    }
    monkeypatch.setattr(
        "autoidea.tools.pipeline_state._build_state",
        lambda _workspace, **_kwargs: core_state,
    )
    monkeypatch.setattr(
        "autoidea.web.pipeline.inspect_pipeline",
        lambda *_args, **_kwargs: pipeline,
    )

    missing_checkpoint = display._inspect_autonomous_progress(
        tmp_path,
        checkpoint_events=["stage_9", "stage_10"],
        require_checkpoint_events=True,
    )
    assert missing_checkpoint["complete"] is False
    assert missing_checkpoint["next_stage"] == "stage_7"

    complete = display._inspect_autonomous_progress(
        tmp_path,
        checkpoint_events=["stage_7", "stage_9", "stage_10"],
        require_checkpoint_events=True,
    )
    assert complete["complete"] is True
    assert complete["next_stage"] == "complete"

    core_state["stages"]["stage_4"]["has_reflection"] = False
    missing_reflection = display._inspect_autonomous_progress(
        tmp_path,
        checkpoint_events=["stage_7", "stage_9", "stage_10"],
        require_checkpoint_events=True,
    )
    assert missing_reflection["complete"] is False
    assert missing_reflection["next_stage"] == "stage_4"
