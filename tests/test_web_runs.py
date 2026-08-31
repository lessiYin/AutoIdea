from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from autoidea.web.pipeline import (
    PIPELINE_STAGES,
    checkpoint_events_from_events,
    inspect_pipeline,
)
from autoidea.web.runner import (
    EventBridge,
    _checkpoint_decision,
    _checkpoint_stage_from_tool_call_id,
)
from autoidea.web.models import RunRecord
from autoidea.web.runs import (
    RunRequest,
    WebRunManager,
    _explicitly_approved_checkpoint_events,
    _resolve_run_request_defaults,
    pipeline_parameters_for_record,
)

ASK_USER_LOG = """
╭──────────────────────────────────────────────────────────────────────────────╮
│ Quick check-in from AutoIdea                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯

  1. 你希望聚焦长视频理解的哪个核心方向？
     A. 长视频问答 (Long Video QA) — 对长视频内容进行推理问答
     B. 时序定位 (Temporal Grounding) — 在长视频中定位特定事件/片段
     C. 视频摘要/描述 (Video Summarization/Captioning) — 生成视频的文字描述
     D. 视频-文本检索 (Video-Text Retrieval) — 跨模态检索与匹配
     E. 通用长视频理解模型 — 构建能处理长视频的统一多模态大模型
     F. Other (type your answer)
  Choice [A/B/C/D/E/F]:
"""

CHECKPOINT_LOG = """
Stage 9 完成 — 研究想法生成 [HITL 检查点]

请审阅以上研究想法。您希望：

 • ✅ 批准 — 继续进入 Stage 9.5（Elo锦标赛排名）
 • ✏️ 修改 — 对某些想法提出修改意见
 • 🔄 重新生成 — 调整方向后重新生成
                                                [Usage: 424,812 in · 10,969 out]
"""

DEBATE_CHECKPOINT_LOG = """
Stage 10 完成 — 对抗性辩论 [HITL 检查点]

所有 5 个研究想法均通过 3 轮对抗性辩论。

请审阅辩论结果。您希望：

 • ✅ 批准 — 继续进入 Stage 11（可行性评估）
 • ✏️ 修改 — 对某些想法的辩论结论提出修改意见
 • 🔄 重新辩论 — 对某些想法进行额外的辩论轮次
                                                [Usage: 604,722 in · 18,380 out]
Task exception was never retrieved
"""

THREAD_LOG = """
Loading agent...
Thread: a469c6f6
Workspace: sample_workspace/runs/lvu
"""


def _wait_for_run(
    manager: WebRunManager,
    run_id: str,
    predicate,
    *,
    timeout: float = 8.0,
):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = manager.get_run(run_id)
        if last is not None and predicate(last):
            return last
        time.sleep(0.03)
    raise AssertionError(f"run did not reach expected state; last={last!r}")


def _write_fake_structured_runner(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import time
            from datetime import datetime, timezone
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--prompt")
            parser.add_argument("--workdir")
            parser.add_argument("--thread-id")
            parser.add_argument("--run-id")
            parser.add_argument("--events-file")
            parser.add_argument("--responses-dir")
            parser.add_argument("--model", default="")
            parser.add_argument("--provider", default="")
            parser.add_argument("--seed-papers", default="")
            parser.add_argument("--seed-ideas", default="")
            parser.add_argument("--auto-approve", action="store_true")
            parser.add_argument("--no-thinking", action="store_true")
            args = parser.parse_args()
            workspace = Path(args.workdir)
            events = Path(args.events_file)
            responses = Path(args.responses_dir)
            workspace.mkdir(parents=True, exist_ok=True)
            events.parent.mkdir(parents=True, exist_ok=True)
            responses.mkdir(parents=True, exist_ok=True)

            def emit(kind, **payload):
                event = {"type": kind, "at": datetime.now(timezone.utc).isoformat(), **payload}
                with events.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\\n")
                    handle.flush()

            def markdown(name, title):
                body = f"# {title}\\n\\n" + ("Grounded research evidence and analysis. " * 24)
                (workspace / name).write_text(body, encoding="utf-8")

            def data(name, value):
                (workspace / name).write_text(json.dumps(value), encoding="utf-8")

            def checkpoint(stage):
                interaction_id = f"{stage}-decision"
                emit(
                    "interaction_requested",
                    interaction_id=interaction_id,
                    checkpoint_stage=stage,
                    interaction={
                        "kind": "checkpoint",
                        "questions": [{
                            "question": f"Approve {stage}?",
                            "type": "multiple_choice",
                            "choices": [{"label": "Approve", "value": "approve"}],
                            "required": True,
                        }],
                        "allows_cancel": False,
                    },
                )
                response = responses / f"{interaction_id}.json"
                if args.auto_approve:
                    value = {"status": "answered", "answers": ["approve"]}
                    mode = "automatic"
                else:
                    deadline = time.monotonic() + 10
                    while not response.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    if not response.exists():
                        raise RuntimeError(f"missing response for {stage}")
                    value = json.loads(response.read_text(encoding="utf-8"))
                    mode = "manual"
                answers = value.get("answers", [])
                decision = str(answers[0]).strip().lower() if answers else ""
                emit(
                    "interaction_resolved",
                    interaction_id=interaction_id,
                    checkpoint_stage=stage,
                    response={
                        "status": value.get("status", "answered"),
                        "decision": decision,
                        "approved": decision == "approve",
                        "mode": mode,
                    },
                )

            emit("runner_started", run_id=args.run_id, thread_id=args.thread_id, workspace=str(workspace))
            markdown("research_brief.md", "Research brief")
            markdown("task_formalization.md", "Task formalization")
            markdown("literature_survey.md", "Literature survey")
            data("paper_registry.json", [{"paper_id": "P1", "title": "A grounded paper"}])
            markdown("paper_deep_reading.md", "Paper deep reading")
            data("paper_positions.json", [{"paper_id": "P1", "summary": "Position"}])
            markdown("expanded_literature.md", "Expanded literature")
            data("evidence_db.json", {"claims": [{"citation_id": "C1", "claim": "Evidence", "source_paper_id": "P1"}]})
            (workspace / "knowledge_synthesis.md").write_text(
                "# Knowledge synthesis\\n\\nG1, G2, and G3 are grounded in C1.\\n\\n"
                + ("Structured gap analysis. " * 24),
                encoding="utf-8",
            )
            data("research_gaps.json", {
                "schema_version": "1.0",
                "generated_from": "evidence_db.json",
                "gaps": [
                    {
                        "gap_id": f"G{index}",
                        "title": f"Gap {index}",
                        "description": f"Unresolved problem {index}.",
                        "gap_type": "methodology_gap",
                        "demand": 4,
                        "coverage": 2,
                        "gap_score": 2,
                        "evidence_links": [{
                            "citation_id": "C1",
                            "relationship": "supports",
                            "rationale": "The canonical Claim establishes this unresolved problem.",
                        }],
                        "why_it_matters": "It blocks a reliable result.",
                        "potential_direction": "Evaluate a bounded intervention.",
                    }
                    for index in range(1, 4)
                ],
            })
            checkpoint("stage_7")
            data("design_space.json", {"axes": [{"name": "Mechanism"}]})
            data("raw_ideas.json", {"ideas": [{"idea_id": "I1", "title": "Idea"}]})
            checkpoint("stage_9")
            data("tournament_rankings.json", {"rankings": [{"idea_id": "I1", "rank": 1}]})
            markdown("debate_log.md", "Adversarial debate")
            data("idea_reviews.json", {"reviews": [{"idea_id": "I1", "verdict": "pass"}]})
            checkpoint("stage_10")
            data("feasibility_assessments.json", {"assessments": [{"idea_id": "I1", "feasible": True}]})
            data("run_status.json", {"stage": "stage_12", "status": "running"})
            markdown("final_report.md", "Final report")
            reflections = workspace / "reflections"
            reflections.mkdir()
            for stage_id in (
                "stage_1", "stage_2", "stage_3", "stage_3.5", "stage_4",
                "stage_5", "stage_6", "stage_7", "stage_8", "stage_9",
                "stage_9.5", "stage_10", "stage_11", "stage_12",
            ):
                (reflections / f"{stage_id}_reflection.json").write_text(
                    json.dumps({
                        "stage": stage_id,
                        "reflection": "complete",
                        "gate_passed": True,
                    }),
                    encoding="utf-8",
                )
            emit("runner_finished", run_id=args.run_id)
            print("deterministic web run complete", flush=True)
            """
        ),
        encoding="utf-8",
    )


def test_checkpoint_tool_call_ids_and_decisions_are_normalized() -> None:
    assert _checkpoint_stage_from_tool_call_id(
        "autoidea-checkpoint:stage_7"
    ) == "stage_7"
    assert _checkpoint_stage_from_tool_call_id(
        "autoidea-checkpoint:stage_10"
    ) == "stage_10"
    assert _checkpoint_stage_from_tool_call_id(
        "autoidea-checkpoint:stage_8"
    ) == ""
    assert _checkpoint_stage_from_tool_call_id("ordinary-tool-call") == ""

    assert _checkpoint_decision(["Approve / 批准"]) == "approve"
    assert _checkpoint_decision(["批准"]) == "approve"
    assert _checkpoint_decision(["RE-RUN"]) == "rerun"
    assert _checkpoint_decision(["修改"]) == "revise"
    assert _checkpoint_decision(["auto_continue"]) == "auto_continue"
    assert _checkpoint_decision([]) == ""


def test_event_bridge_auto_approves_checkpoint_without_response_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = EventBridge(
        tmp_path / "events.jsonl",
        tmp_path / "responses",
        tmp_path / "workspace",
        auto_approve=True,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_response",
        lambda _interaction_id: pytest.fail("automatic mode must not wait"),
    )

    response = bridge.ask_user(
        {
            "tool_call_id": "autoidea-checkpoint:stage_7",
            "questions": [
                {
                    "question": "Approve Stage 7?",
                    "type": "multiple_choice",
                    "choices": [{"value": "approve", "label": "Approve"}],
                    "required": True,
                }
            ],
        }
    )

    assert response == {"status": "answered", "answers": ["approve"]}
    assert list((tmp_path / "responses").glob("*.json")) == []
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events] == [
        "interaction_requested",
        "interaction_resolved",
    ]
    assert events[1]["response"] == {
        "status": "answered",
        "answer_count": 1,
        "decision": "approve",
        "approved": True,
        "mode": "automatic",
        "automation_enabled": False,
    }


def test_event_bridge_automatic_mode_never_waits_for_generic_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = EventBridge(
        tmp_path / "events.jsonl",
        tmp_path / "responses",
        tmp_path / "workspace",
        auto_approve=True,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_response",
        lambda _interaction_id: pytest.fail("automatic mode must not wait"),
    )

    response = bridge.ask_user(
        {
            "questions": [
                {"question": "Choose a scope", "type": "text", "required": True},
                {"question": "Extra detail", "type": "text", "required": False},
            ]
        }
    )

    assert response == {
        "status": "answered",
        "answers": ["Proceed automatically using your best judgment.", ""],
    }
    assert list((tmp_path / "responses").glob("*.json")) == []


def test_event_bridge_manual_auto_continue_switches_later_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = EventBridge(
        tmp_path / "events.jsonl",
        tmp_path / "responses",
        tmp_path / "workspace",
        auto_approve=False,
    )
    waits: list[str] = []

    def answer_once(interaction_id: str) -> dict:
        waits.append(interaction_id)
        return {"status": "answered", "answers": ["auto_continue", ""]}

    monkeypatch.setattr(bridge, "_wait_for_response", answer_once)
    first = bridge.ask_user(
        {
            "tool_call_id": "autoidea-checkpoint:stage_7",
            "questions": [{"type": "multiple_choice"}, {"type": "text"}],
        }
    )
    second = bridge.ask_user(
        {
            "tool_call_id": "autoidea-checkpoint:stage_9",
            "questions": [{"type": "multiple_choice"}],
        }
    )

    assert first["answers"][0] == "auto_continue"
    assert second["answers"] == ["approve"]
    assert bridge.auto_approve is True
    assert len(waits) == 1
    resolved = [
        event
        for event in (
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if event["type"] == "interaction_resolved"
    ]
    assert resolved[0]["response"]["automation_enabled"] is True
    assert resolved[0]["response"]["approved"] is True
    assert resolved[1]["response"]["mode"] == "automatic"


def test_checkpoint_completion_requires_approval_but_keeps_legacy_events() -> None:
    events = [
        {
            "type": "interaction_requested",
            "interaction_id": "rejected",
            "checkpoint_stage": "stage_7",
        },
        {
            "type": "interaction_resolved",
            "interaction_id": "rejected",
            "response": {"approved": False, "decision": "revise"},
        },
        {
            "type": "interaction_requested",
            "interaction_id": "approved",
            "checkpoint_stage": "stage_9",
        },
        {
            "type": "interaction_resolved",
            "interaction_id": "approved",
            "response": {"approved": True, "decision": "approve"},
        },
        {
            "type": "interaction_requested",
            "interaction_id": "legacy",
            "checkpoint_stage": "stage_10",
        },
        {
            "type": "interaction_resolved",
            "interaction_id": "legacy",
        },
    ]

    assert checkpoint_events_from_events(events) == ["stage_10", "stage_9"]


def test_new_web_run_snapshots_non_secret_pipeline_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTOIDEA_DEEP_READING_TOP_K", "5")
    script = tmp_path / "exit.py"
    script.write_text("print('done')\n", encoding="utf-8")
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)

    started = manager.start_run(
        RunRequest(prompt="Snapshot limits", run_name="parameter-snapshot"),
        extra_args=[str(script)],
    )
    finished = _wait_for_run(
        manager,
        started.run_id,
        lambda item: item.exit_code == 0,
    )

    assert finished.pipeline_parameters["deep_reading_top_k"] == 5
    assert set(finished.pipeline_parameters) == {
        "max_search_queries",
        "target_paper_count",
        "deep_reading_top_k",
        "max_ideas_to_generate",
        "top_k_ranked",
        "max_debate_rounds",
        "elo_initial_score",
        "elo_k_factor",
    }


def test_legacy_web_run_recovers_parameters_from_its_own_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTOIDEA_DEEP_READING_TOP_K", "35")
    workspace = tmp_path / "legacy-run"
    workspace.mkdir()
    (workspace / "research_brief.md").write_text(
        "# Research brief\n\n- `deep_reading_top_k: 5`\n- `max_debate_rounds: 2`\n",
        encoding="utf-8",
    )
    record = RunRecord(
        run_id="legacy",
        status="pipeline_completed",
        prompt="research",
        workspace=str(workspace),
    )

    parameters = pipeline_parameters_for_record(record)

    assert parameters["deep_reading_top_k"] == 5
    assert parameters["max_debate_rounds"] == 2


def test_persisted_completed_status_is_revoked_without_stage12_gate_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    record = RunRecord(
        run_id="legacy-unverified",
        status="pipeline_completed",
        prompt="research",
        workspace=str(tmp_path / "legacy-unverified"),
    )
    monkeypatch.setattr(
        "autoidea.web.runs.inspect_pipeline",
        lambda *_args, **_kwargs: {
            "active_stage": "stage_12",
            "next_stage": "stage_12",
            "completed_count": 13,
            "total_stages": 14,
            "active_progress": {},
            "completion": {
                "verified": False,
                "missing_gate_proofs": ["stage_12"],
            },
        },
    )

    refreshed = manager._with_runtime(record)

    assert refreshed.status == "failed"
    assert refreshed.current_stage == "stage_12"
    assert refreshed.completed_stages == 13
    assert "Stage 12 gate proof is missing" in refreshed.status_detail


def test_resume_inherits_only_explicit_parent_checkpoint_approvals(
    tmp_path: Path,
) -> None:
    manager = WebRunManager(
        tmp_path,
        runner_command=[sys.executable, "-c", "import time; time.sleep(0.3)"],
    )
    workspace = tmp_path / "runs" / "shared"
    workspace.mkdir(parents=True)
    manager.events_dir.mkdir(parents=True)
    parent_events_path = manager.events_dir / "parent.jsonl"
    parent_events = [
        {
            "type": "interaction_requested",
            "interaction_id": "approved-stage-7",
            "checkpoint_stage": "stage_7",
        },
        {
            "type": "interaction_resolved",
            "interaction_id": "approved-stage-7",
            "response": {"decision": "approve", "approved": True},
        },
        {
            "type": "interaction_requested",
            "interaction_id": "rejected-stage-9",
            "checkpoint_stage": "stage_9",
        },
        {
            "type": "interaction_resolved",
            "interaction_id": "rejected-stage-9",
            "response": {"decision": "revise", "approved": False},
        },
        {
            "type": "interaction_requested",
            "interaction_id": "legacy-stage-10",
            "checkpoint_stage": "stage_10",
        },
        {
            "type": "interaction_resolved",
            "interaction_id": "legacy-stage-10",
        },
    ]
    parent_events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in parent_events),
        encoding="utf-8",
    )
    manager._write_metadata(
        [
            {
                "run_id": "parent",
                "status": "failed",
                "prompt": "research",
                "workspace": str(workspace),
                "run_name": "shared",
                "thread_id": "same-thread",
                "events_path": str(parent_events_path),
            }
        ]
    )

    assert _explicitly_approved_checkpoint_events(parent_events) == [
        ("stage_7", "approved-stage-7")
    ]

    resumed = manager.start_run(
        RunRequest(
            prompt="continue",
            workspace=str(workspace),
            thread_id="same-thread",
            mode="resume",
            parent_run_id="parent",
        )
    )
    try:
        inherited = manager.list_events(resumed.run_id)
        assert checkpoint_events_from_events(inherited) == ["stage_7"]
        assert len(inherited) == 2
        assert all(event["inherited_from_run_id"] == "parent" for event in inherited)
        assert inherited[1]["response"] == {
            "status": "answered",
            "answer_count": 1,
            "decision": "approve",
            "approved": True,
            "inherited": True,
        }
    finally:
        current = manager.get_run(resumed.run_id)
        if current is not None and current.status in {
            "queued",
            "running",
            "waiting_for_input",
        }:
            manager.stop_run(resumed.run_id)


def test_run_request_builds_autoidea_command(tmp_path: Path) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable="/bin/autoidea")
    request = RunRequest(
        prompt="Generate ideas",
        run_name="demo",
        model="gpt-4o",
        provider="openai",
        thread_id="abc123",
        seed_papers="/tmp/papers.json",
        seed_ideas="/tmp/ideas.md",
        auto_approve=True,
        show_thinking=False,
    )

    command = manager.build_command(request, tmp_path / "runs" / "demo")

    assert command[:2] == ["/bin/autoidea", "--prompt"]
    assert "Generate ideas" in command
    assert "--name" not in command
    assert ["--model", "gpt-4o"] == command[command.index("--model"): command.index("--model") + 2]
    assert ["--provider", "openai"] == command[command.index("--provider"): command.index("--provider") + 2]
    assert ["--thread-id", "abc123"] == command[command.index("--thread-id"): command.index("--thread-id") + 2]
    assert ["--seed-papers", "/tmp/papers.json"] == command[command.index("--seed-papers"): command.index("--seed-papers") + 2]
    assert ["--seed-ideas", "/tmp/ideas.md"] == command[command.index("--seed-ideas"): command.index("--seed-ideas") + 2]
    assert "--auto-approve" in command
    assert "--no-thinking" in command


def test_blank_web_run_model_and_provider_resolve_to_effective_defaults(
    monkeypatch,
) -> None:
    from autoidea.config import AutoIdeaConfig

    monkeypatch.setattr(
        "autoidea.config.get_effective_config",
        lambda: AutoIdeaConfig(provider="custom-openai", model="saved-model"),
    )

    resolved = _resolve_run_request_defaults(RunRequest(prompt="Generate ideas"))
    explicit = _resolve_run_request_defaults(
        RunRequest(
            prompt="Generate ideas",
            provider="openai",
            model="run-model",
        )
    )

    assert resolved.provider == "custom-openai"
    assert resolved.model == "saved-model"
    assert explicit.provider == "openai"
    assert explicit.model == "run-model"


def test_run_request_defaults_to_automatic_and_can_build_manual_command(
    tmp_path: Path,
) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable="/bin/autoidea")

    assert RunRequest(prompt="Automatic by default").auto_approve is True
    assert RunRecord(
        run_id="default",
        status="queued",
        prompt="Automatic by default",
        workspace=str(tmp_path),
    ).auto_approve is True

    command = manager.build_command(
        RunRequest(prompt="Review checkpoints", auto_approve=False),
        tmp_path / "runs" / "manual",
    )
    assert "--manual-checkpoints" in command
    assert "--auto-approve" not in command


def test_run_effective_mode_is_restored_from_auto_continue_event(
    tmp_path: Path,
) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable="/bin/autoidea")
    manager.events_dir.mkdir(parents=True)
    workspace = tmp_path / "runs" / "manual-to-auto"
    workspace.mkdir(parents=True)
    events_path = manager.events_dir / "manual-to-auto.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "type": "interaction_resolved",
                "interaction_id": "stage-7",
                "checkpoint_stage": "stage_7",
                "response": {
                    "decision": "auto_continue",
                    "approved": True,
                    "automation_enabled": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manager._write_metadata(
        [
            {
                "run_id": "manual-to-auto",
                "status": "failed",
                "prompt": "research",
                "workspace": str(workspace),
                "auto_approve": False,
                "events_path": str(events_path),
            }
        ]
    )

    restored = manager.get_run("manual-to-auto")

    assert restored is not None
    assert restored.auto_approve is True


def test_run_manager_uses_current_python_when_autoidea_is_not_on_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("autoidea.web.runs.shutil.which", lambda name: None)
    manager = WebRunManager(tmp_path)

    command = manager.build_command(RunRequest(prompt="Hello"), tmp_path / "run")

    assert command[:4] == [sys.executable, "-m", "autoidea", "--prompt"]
    assert "Hello" in command


def test_run_manager_rejects_exit_zero_without_pipeline_artifacts(tmp_path: Path) -> None:
    script = tmp_path / "fake_autoidea.py"
    script.write_text(
        "import sys\n"
        "print('fake run started')\n"
        "print('args=' + ' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)

    record = manager.start_run(
        RunRequest(prompt="Hello", run_name="demo"),
        extra_args=[str(script)],
    )

    for _ in range(30):
        current = manager.get_run(record.run_id)
        if current and current.status == "failed":
            break
        time.sleep(0.05)

    current = manager.get_run(record.run_id)
    assert current is not None
    assert current.status == "failed"
    assert current.exit_code == 0
    assert current.completion["verified"] is False
    assert "before" in current.status_detail.lower()
    assert "fake run started" in current.log_tail
    assert "Hello" in current.log_tail

    metadata = json.loads((tmp_path / ".autoidea_web" / "runs.json").read_text(encoding="utf-8"))
    assert metadata[0]["run_id"] == record.run_id
    assert metadata[0]["status"] == "failed"


def test_run_manager_keeps_stdin_open_and_accepts_web_input(tmp_path: Path) -> None:
    script = tmp_path / "ask.py"
    answer_path = tmp_path / "answer.txt"
    script.write_text(
        "import pathlib, sys\n"
        "print('Choice [A/B]: ', end='', flush=True)\n"
        "answer = sys.stdin.readline().strip()\n"
        f"pathlib.Path({str(answer_path)!r}).write_text(answer, encoding='utf-8')\n",
        encoding="utf-8",
    )
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)

    record = manager.start_run(
        RunRequest(prompt="Needs answer"),
        extra_args=[str(script)],
    )
    manager.send_input(record.run_id, "B")

    for _ in range(30):
        current = manager.get_run(record.run_id)
        if current and current.status == "checkpoint_reached":
            break
        time.sleep(0.05)

    assert answer_path.read_text(encoding="utf-8") == "B"
    current = manager.get_run(record.run_id)
    assert current is not None
    assert current.status == "checkpoint_reached"


def test_run_record_detects_pending_multiple_choice_prompt(tmp_path: Path) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    manager.state_dir.mkdir(parents=True)
    manager.log_dir.mkdir(parents=True)
    log_path = manager.log_dir / "ask.log"
    log_path.write_text(ASK_USER_LOG, encoding="utf-8")
    manager._write_metadata(
        [
            {
                "run_id": "ask123",
                "status": "running",
                "prompt": "长视频理解",
                "workspace": str(tmp_path),
                "run_name": "lvu",
                "pid": 123,
                "started_at": "2026-07-03T00:00:00+00:00",
                "log_path": str(log_path),
                "command": [sys.executable],
            }
        ]
    )

    record = manager.get_run("ask123")

    assert record is not None
    assert record.interaction is not None
    assert record.interaction["kind"] == "multiple_choice"
    assert record.interaction["question"] == "你希望聚焦长视频理解的哪个核心方向？"
    assert record.interaction["options"][:2] == [
        {"key": "A", "label": "长视频问答 (Long Video QA) — 对长视频内容进行推理问答"},
        {"key": "B", "label": "时序定位 (Temporal Grounding) — 在长视频中定位特定事件/片段"},
    ]
    assert record.interaction["allows_other"] is True


def test_run_record_detects_latest_text_prompt_after_previous_choices_and_warnings(
    tmp_path: Path,
) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    manager.state_dir.mkdir(parents=True)
    manager.log_dir.mkdir(parents=True)
    log_path = manager.log_dir / "ask.log"
    log_path.write_text(
        "  1. Pick a direction?\n"
        "     A. Temporal reasoning\n"
        "     B. Efficient modeling\n"
        "  Choice [A/B]: B\n"
        "  2. Any specific papers?\n"
        "  > Answer: \n"
        "Warning: Stage gate stage_2 FAIL (attempt 1/5)\n",
        encoding="utf-8",
    )
    manager._write_metadata(
        [
            {
                "run_id": "ask-text",
                "status": "running",
                "prompt": "long video",
                "workspace": str(tmp_path),
                "run_name": "lvu",
                "pid": 123,
                "started_at": "2026-07-03T00:00:00+00:00",
                "log_path": str(log_path),
                "command": [sys.executable],
            }
        ]
    )

    record = manager.get_run("ask-text")

    assert record is not None
    assert record.interaction == {
        "kind": "text",
        "prompt": "Answer:",
        "question": "Any specific papers?",
        "options": [],
        "allows_other": True,
    }


def test_run_record_ignores_choice_prompt_after_log_progress(tmp_path: Path) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    manager.state_dir.mkdir(parents=True)
    manager.log_dir.mkdir(parents=True)
    log_path = manager.log_dir / "ask-progress.log"
    log_path.write_text(
        "  1. Pick a direction?\n"
        "     A. Temporal reasoning\n"
        "     B. Efficient modeling\n"
        "  Choice [A/B]: \n"
        "  2. Pick constraints?\n"
        "     A. Lab scale\n"
        "     B. Industry scale\n"
        "  Choice [A/B]: \n"
        "Warning: Stage gate stage_2 FAIL (attempt 1/5)\n"
        "Warning: Model call failed with transient RateLimitError; retrying 1/2 in 2.0s\n"
        "Warning: multi_source_search: arxiv timed out after 60s\n",
        encoding="utf-8",
    )
    manager._write_metadata(
        [
            {
                "run_id": "ask-progress",
                "status": "running",
                "prompt": "long video",
                "workspace": str(tmp_path),
                "run_name": "lvu",
                "pid": 123,
                "started_at": "2026-07-03T00:00:00+00:00",
                "log_path": str(log_path),
                "command": [sys.executable],
            }
        ]
    )

    record = manager.get_run("ask-progress")

    assert record is not None
    assert record.interaction is None


def test_run_record_uses_options_for_latest_question_only(tmp_path: Path) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    manager.state_dir.mkdir(parents=True)
    manager.log_dir.mkdir(parents=True)
    log_path = manager.log_dir / "ask.log"
    log_path.write_text(
        "  1. Pick a direction?\n"
        "     A. Temporal reasoning\n"
        "     B. Efficient modeling\n"
        "  Choice [A/B]: B\n"
        "  2. Pick constraints?\n"
        "     A. Lab scale\n"
        "     B. Industry scale\n"
        "     C. Minimal compute\n"
        "  Choice [A/B/C]: \n",
        encoding="utf-8",
    )
    manager._write_metadata(
        [
            {
                "run_id": "ask-choice",
                "status": "running",
                "prompt": "long video",
                "workspace": str(tmp_path),
                "run_name": "lvu",
                "pid": 123,
                "started_at": "2026-07-03T00:00:00+00:00",
                "log_path": str(log_path),
                "command": [sys.executable],
            }
        ]
    )

    record = manager.get_run("ask-choice")

    assert record is not None
    assert record.interaction is not None
    assert record.interaction["question"] == "Pick constraints?"
    assert record.interaction["options"] == [
        {"key": "A", "label": "Lab scale"},
        {"key": "B", "label": "Industry scale"},
        {"key": "C", "label": "Minimal compute"},
    ]


def test_completed_run_detects_hitl_checkpoint_review(tmp_path: Path) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    manager.state_dir.mkdir(parents=True)
    manager.log_dir.mkdir(parents=True)
    log_path = manager.log_dir / "checkpoint.log"
    log_path.write_text(CHECKPOINT_LOG, encoding="utf-8")
    manager._write_metadata(
        [
            {
                "run_id": "checkpoint",
                "status": "completed",
                "prompt": "长视频理解",
                "workspace": str(tmp_path / "runs" / "lvu"),
                "run_name": "lvu",
                "thread_id": "abc123",
                "pid": 123,
                "exit_code": 0,
                "started_at": "2026-07-03T00:00:00+00:00",
                "finished_at": "2026-07-03T00:01:00+00:00",
                "log_path": str(log_path),
                "command": [sys.executable],
            }
        ]
    )

    record = manager.get_run("checkpoint")

    assert record is not None
    assert record.interaction == {
        "kind": "checkpoint_review",
        "prompt": "HITL checkpoint review",
        "question": "请审阅以上研究想法。您希望：",
        "options": [
            {"key": "approve", "label": "批准 — 继续进入 Stage 9.5（Elo锦标赛排名）"},
            {"key": "auto_continue", "label": "不回答，后续全部自动"},
            {"key": "modify", "label": "修改 — 对某些想法提出修改意见"},
            {"key": "regenerate", "label": "重新生成 — 调整方向后重新生成"},
        ],
        "allows_other": True,
    }


def test_completed_run_detects_debate_checkpoint_review(tmp_path: Path) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    manager.state_dir.mkdir(parents=True)
    manager.log_dir.mkdir(parents=True)
    log_path = manager.log_dir / "debate-checkpoint.log"
    log_path.write_text(DEBATE_CHECKPOINT_LOG, encoding="utf-8")
    manager._write_metadata(
        [
            {
                "run_id": "debate-checkpoint",
                "status": "completed",
                "prompt": "长视频理解",
                "workspace": str(tmp_path / "runs" / "lvu"),
                "run_name": "lvu",
                "thread_id": "abc123",
                "pid": 123,
                "exit_code": 0,
                "started_at": "2026-07-03T00:00:00+00:00",
                "finished_at": "2026-07-03T00:01:00+00:00",
                "log_path": str(log_path),
                "command": [sys.executable],
            }
        ]
    )

    record = manager.get_run("debate-checkpoint")

    assert record is not None
    assert record.interaction == {
        "kind": "checkpoint_review",
        "prompt": "HITL checkpoint review",
        "question": "请审阅辩论结果。您希望：",
        "options": [
            {"key": "approve", "label": "批准 — 继续进入 Stage 11（可行性评估）"},
            {"key": "auto_continue", "label": "不回答，后续全部自动"},
            {"key": "modify", "label": "修改 — 对某些想法的辩论结论提出修改意见"},
            {"key": "regenerate", "label": "重新辩论 — 对某些想法进行额外的辩论轮次"},
        ],
        "allows_other": True,
    }


def test_run_record_reads_thread_id_from_log_tail_when_metadata_is_empty(
    tmp_path: Path,
) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    manager.state_dir.mkdir(parents=True)
    manager.log_dir.mkdir(parents=True)
    log_path = manager.log_dir / "thread.log"
    log_path.write_text(THREAD_LOG, encoding="utf-8")
    manager._write_metadata(
        [
            {
                "run_id": "thread-run",
                "status": "completed",
                "prompt": "长视频理解",
                "workspace": str(tmp_path / "runs" / "lvu"),
                "run_name": "lvu",
                "thread_id": "",
                "pid": 123,
                "exit_code": 0,
                "started_at": "2026-07-03T00:00:00+00:00",
                "finished_at": "2026-07-03T00:01:00+00:00",
                "log_path": str(log_path),
                "command": [sys.executable],
            }
        ]
    )

    record = manager.get_run("thread-run")

    assert record is not None
    assert record.thread_id == "a469c6f6"


def test_run_manager_can_stop_running_process(tmp_path: Path) -> None:
    script = tmp_path / "slow_autoidea.py"
    script.write_text(
        "import time\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)

    record = manager.start_run(
        RunRequest(prompt="Slow"),
        extra_args=[str(script)],
    )

    for _ in range(30):
        current = manager.get_run(record.run_id)
        if current and "ready" in current.log_tail:
            break
        time.sleep(0.05)

    stopped = manager.stop_run(record.run_id)

    assert stopped.status == "stopped"
    assert stopped.exit_code is not None


def test_run_request_requires_prompt(tmp_path: Path) -> None:
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)

    with pytest.raises(ValueError):
        manager.start_run(RunRequest(prompt=""))


def test_new_runs_never_reuse_a_nonempty_workspace_and_have_thread_id(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "runs" / "demo"
    occupied.mkdir(parents=True)
    (occupied / "existing.txt").write_text("preserve me", encoding="utf-8")
    script = tmp_path / "exit.py"
    script.write_text("print('done')\n", encoding="utf-8")
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)

    record = manager.start_run(
        RunRequest(prompt="Hello", run_name="demo"),
        extra_args=[str(script)],
    )

    assert Path(record.workspace).name == "demo-2"
    assert Path(record.workspace) != occupied
    assert (occupied / "existing.txt").read_text(encoding="utf-8") == "preserve me"
    assert record.thread_id
    persisted = json.loads(manager.runs_file.read_text(encoding="utf-8"))[0]
    assert persisted["thread_id"] == record.thread_id


def test_generated_thread_id_survives_log_tail_truncation(tmp_path: Path) -> None:
    script = tmp_path / "long_log.py"
    script.write_text("print('x' * 50000)\n", encoding="utf-8")
    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)

    started = manager.start_run(
        RunRequest(prompt="Long log", run_name="long"),
        extra_args=[str(script)],
    )
    finished = _wait_for_run(
        manager,
        started.run_id,
        lambda item: item.status == "failed" and item.exit_code == 0,
    )

    assert finished.thread_id == started.thread_id
    assert len(finished.log_tail) <= 24000


def test_structured_web_runner_completes_three_checkpoints_and_stage_12(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "fake_structured_runner.py"
    _write_fake_structured_runner(script)

    class PassingAudit:
        def __init__(self) -> None:
            self.has_errors = False
            self.issues: list = []

    monkeypatch.setattr(
        "autoidea.tools.artifact_audit.audit_workspace",
        lambda *_args, **_kwargs: PassingAudit(),
    )
    manager = WebRunManager(tmp_path, runner_command=[sys.executable, str(script)])
    started = manager.start_run(
        RunRequest(
            prompt="Deterministic end-to-end research",
            run_name="verified",
            model="test-model",
            provider="custom-openai",
            seed_papers="/tmp/seed-papers.json",
            seed_ideas="/tmp/seed-ideas.md",
            auto_approve=False,
            show_thinking=False,
        )
    )

    assert started.status == "running"
    assert started.thread_id
    checkpoint_ids = []
    for stage in ("stage_7", "stage_9", "stage_10"):
        waiting = _wait_for_run(
            manager,
            started.run_id,
            lambda item, expected=stage: (
                item.status == "waiting_for_input"
                and item.interaction is not None
                and item.interaction.get("checkpoint_stage") == expected
            ),
        )
        checkpoint_ids.append(waiting.interaction["interaction_id"])
        response = manager.send_input(
            started.run_id,
            {"status": "answered", "answers": ["approve"]},
        )
        assert response["status"] == "accepted"

    finished = _wait_for_run(
        manager,
        started.run_id,
        lambda item: item.status == "pipeline_completed",
        timeout=10,
    )

    assert finished.exit_code == 0
    assert finished.completed_stages == 14
    assert finished.total_stages == 14
    assert finished.current_stage == "complete"
    assert finished.completion["verified"] is True
    assert finished.completion["required_artifacts_ready"] is True
    assert finished.completion["final_report_present"] is True
    assert finished.completion["checkpoint_events"] == ["stage_10", "stage_7", "stage_9"]
    assert finished.completion["audit_passed"] is True
    assert (Path(finished.workspace) / "final_report.md").stat().st_size > 200
    assert len(list(Path(finished.response_dir).glob("*.json"))) == 3
    completed_pipeline = inspect_pipeline(
        finished.workspace,
        run_status=finished.status,
        checkpoint_events=finished.completion["checkpoint_events"],
        include_audit=True,
    )
    assert completed_pipeline["active_stage"] == ""
    assert completed_pipeline["active_detail"] == ""

    events = manager.list_events(started.run_id)
    assert [event["type"] for event in events].count("interaction_requested") == 3
    assert [event["type"] for event in events].count("interaction_resolved") == 3
    assert {event.get("checkpoint_stage") for event in events if event["type"] == "interaction_resolved"} == {
        "stage_7",
        "stage_9",
        "stage_10",
    }
    persisted = json.loads(manager.runs_file.read_text(encoding="utf-8"))[0]
    assert persisted["status"] == "pipeline_completed"
    assert persisted["log_tail"] == ""

    restarted = WebRunManager(tmp_path, runner_command=[sys.executable, str(script)])
    recovered = restarted.get_run(started.run_id)
    assert recovered is not None
    assert recovered.status == "pipeline_completed"
    assert recovered.thread_id == started.thread_id


def test_structured_web_runner_completes_automatically_without_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "fake_structured_runner.py"
    _write_fake_structured_runner(script)

    class PassingAudit:
        def __init__(self) -> None:
            self.has_errors = False
            self.issues: list = []

    monkeypatch.setattr(
        "autoidea.tools.artifact_audit.audit_workspace",
        lambda *_args, **_kwargs: PassingAudit(),
    )
    manager = WebRunManager(tmp_path, runner_command=[sys.executable, str(script)])
    started = manager.start_run(
        RunRequest(
            prompt="Deterministic automatic research",
            run_name="automatic",
            model="test-model",
            provider="custom-openai",
        )
    )

    finished = _wait_for_run(
        manager,
        started.run_id,
        lambda item: item.status == "pipeline_completed",
        timeout=10,
    )

    assert finished.auto_approve is True
    assert finished.completion["checkpoint_events"] == [
        "stage_10",
        "stage_7",
        "stage_9",
    ]
    assert list(Path(finished.response_dir).glob("*.json")) == []
    events = manager.list_events(started.run_id)
    assert [event["type"] for event in events].count("interaction_requested") == 3
    resolved = [event for event in events if event["type"] == "interaction_resolved"]
    assert len(resolved) == 3
    assert all(event["response"]["approved"] is True for event in resolved)
    assert all(event["response"]["mode"] == "automatic" for event in resolved)


def test_detached_run_is_finalized_after_service_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restarted Web service must notice when an inherited runner exits."""

    class PassingAudit:
        def __init__(self) -> None:
            self.has_errors = False
            self.issues: list = []

    monkeypatch.setattr(
        "autoidea.tools.artifact_audit.audit_workspace",
        lambda *_args, **_kwargs: PassingAudit(),
    )
    run_id = "detached-run"
    run_workspace = tmp_path / "runs" / "detached"
    state_dir = tmp_path / ".autoidea_web"
    events_dir = state_dir / "events"
    logs_dir = state_dir / "logs"
    run_workspace.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    for stage in PIPELINE_STAGES:
        for name in stage["artifacts"]:
            target = run_workspace / name
            if target.suffix == ".json":
                target.write_text('[{"verified": true}]', encoding="utf-8")
            else:
                target.write_text(
                    f"# {stage['name']}\n\n" + "Grounded evidence. " * 40,
                    encoding="utf-8",
                )
    reflections = run_workspace / "reflections"
    reflections.mkdir()
    for stage in PIPELINE_STAGES:
        stage_id = str(stage["id"])
        (reflections / f"{stage_id}_reflection.json").write_text(
            json.dumps(
                {
                    "stage": stage_id,
                    "reflection": "complete",
                    "gate_passed": True,
                }
            ),
            encoding="utf-8",
        )
    (run_workspace / "evidence_db.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "citation_id": "C1",
                        "claim": "A bounded unresolved problem is documented.",
                        "source_paper_id": "P1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_workspace / "knowledge_synthesis.md").write_text(
        "# Knowledge synthesis\n\n"
        "G1, G2, and G3 are explicit evidence-grounded research gaps.\n\n"
        + "Grounded evidence. " * 40,
        encoding="utf-8",
    )
    (run_workspace / "research_gaps.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_from": "evidence_db.json",
                "gaps": [
                    {
                        "gap_id": f"G{index}",
                        "title": f"Gap {index}",
                        "description": f"A bounded unresolved problem {index}.",
                        "gap_type": "methodology_gap",
                        "demand": 4,
                        "coverage": 2,
                        "gap_score": 2,
                        "evidence_links": [
                            {
                                "citation_id": "C1",
                                "relationship": "supports",
                                "rationale": (
                                    "The Claim establishes this specific unresolved "
                                    "problem boundary."
                                ),
                            }
                        ],
                        "why_it_matters": "It blocks a reliable result.",
                        "potential_direction": "Evaluate a bounded intervention.",
                    }
                    for index in range(1, 4)
                ],
            }
        ),
        encoding="utf-8",
    )
    events_path = events_dir / f"{run_id}.jsonl"
    events: list[dict[str, str]] = []
    for stage_id in ("stage_7", "stage_9", "stage_10"):
        interaction_id = f"{stage_id}-decision"
        events.extend(
            [
                {
                    "type": "interaction_requested",
                    "interaction_id": interaction_id,
                    "checkpoint_stage": stage_id,
                },
                {
                    "type": "interaction_resolved",
                    "interaction_id": interaction_id,
                    "checkpoint_stage": stage_id,
                },
            ]
        )
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    sleeper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.6)",
            run_id,
            str(run_workspace),
        ]
    )
    try:
        (state_dir / "runs.json").write_text(
            json.dumps(
                [
                    {
                        "run_id": run_id,
                        "status": "running",
                        "prompt": "research",
                        "workspace": str(run_workspace),
                        "run_name": "detached",
                        "pid": sleeper.pid,
                        "started_at": "2026-08-26T00:00:00+00:00",
                        "events_path": str(events_path),
                        "log_path": str(logs_dir / f"{run_id}.log"),
                        "command": [sys.executable, run_id, str(run_workspace)],
                    }
                ]
            ),
            encoding="utf-8",
        )
        manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
        live = manager.get_run(run_id)
        assert live is not None
        assert live.status == "running"

        sleeper.wait(timeout=3)
        recovered = manager.get_run(run_id)
        assert recovered is not None
        assert recovered.status == "pipeline_completed"
        assert recovered.completed_stages == 14
        assert recovered.completion["verified"] is True
        assert "restarted" in recovered.status_detail
    finally:
        if sleeper.poll() is None:
            sleeper.terminate()
            sleeper.wait(timeout=3)


def test_reconciliation_marks_mismatched_live_pid_as_stale(tmp_path: Path) -> None:
    state_dir = tmp_path / ".autoidea_web"
    state_dir.mkdir()
    runs_file = state_dir / "runs.json"
    runs_file.write_text(
        json.dumps(
            [
                {
                    "run_id": "stale-run",
                    "status": "running",
                    "prompt": "research",
                    "workspace": str(tmp_path / "runs" / "stale"),
                    "run_name": "stale",
                    "pid": os.getpid(),
                    "started_at": "2026-08-26T00:00:00+00:00",
                    "command": ["definitely-not-this-process"],
                }
            ]
        ),
        encoding="utf-8",
    )

    manager = WebRunManager(tmp_path, autoidea_executable=sys.executable)
    record = manager.get_run("stale-run")

    assert record is not None
    assert record.status == "stale"
    assert "no longer running" in record.status_detail
