from __future__ import annotations

import json
from pathlib import Path

from autoidea.paths import get_active_workspace, set_active_workspace
from autoidea.tools.stage_gate import (
    check_stage_gate,
    configure_web_checkpoint_events,
    reset_gate_counters,
    save_stage_reflection,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data) -> None:
    _write(path, json.dumps(data, indent=2))


def _write_stage7_artifacts(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "paper_registry.json",
        [
            {"paper_id": "P1", "title": "Paper One"},
            {"paper_id": "P2", "title": "Paper Two"},
        ],
    )
    _write(
        tmp_path / "literature_survey.md",
        """
# Literature Survey
| ID | Paper | Year | Source | Relevance |
|---|---|---:|---|---|
| [P1] | **Paper One** | 2025 | source | relevant |
| [P2] | **Paper Two** | 2025 | source | relevant |
""".strip(),
    )
    _write_json(
        tmp_path / "evidence_db.json",
        {
            "claims": [
                {"citation_id": "C1", "claim": "A limitation exists.", "source_paper_id": "P1"},
                {"citation_id": "C2", "claim": "A partial solution exists.", "source_paper_id": "P2"},
            ]
        },
    )
    _write(
        tmp_path / "knowledge_synthesis.md",
        "# Knowledge Synthesis\n\n" + "G1, G2, and G3 are evidence-grounded gaps. " * 8,
    )
    _write_json(
        tmp_path / "research_gaps.json",
        {
            "schema_version": "1.0",
            "generated_from": "evidence_db.json",
            "gaps": [
                {
                    "gap_id": f"G{index}",
                    "title": f"Gap {index}",
                    "description": f"A precise unresolved problem {index}.",
                    "gap_type": "methodology_gap",
                    "demand": 5,
                    "coverage": 2,
                    "gap_score": 3,
                    "evidence_links": [
                        {
                            "citation_id": "C1",
                            "relationship": "supports",
                            "rationale": "This Claim establishes the unresolved limitation.",
                        },
                        {
                            "citation_id": "C2",
                            "relationship": "partial_coverage",
                            "rationale": "This Claim records only a partial existing solution.",
                        },
                    ],
                    "why_it_matters": "The gap blocks a reliable outcome.",
                    "potential_direction": "Evaluate a bounded intervention.",
                    "supporting_papers": ["P1", "P2"],
                }
                for index in range(1, 4)
            ],
        },
    )


def test_stage_gate_fails_when_actual_artifacts_are_inconsistent(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_json(
            tmp_path / "evidence_db.json",
            {
                "metadata": {"citation_id_policy": "Local deterministic citation IDs C1-C32."},
                "summary": {"citations_count": 2},
                "claims": [
                    {
                        "citation_id": "C1",
                        "source_title": "Paper",
                        "source_url": "https://example.org/paper",
                    }
                ],
            },
        )
        _write_json(
            tmp_path / "reflections" / "stage_6_reflection.json",
            {"stage": "stage_6", "artifacts": {"citations_count": 2}},
        )

        result = check_stage_gate.invoke({"stage": "stage_6", "evidence_json": '{"citations_count": 6}'})

        assert "Status**: FAIL" in result
        assert "Artifact Integrity Failures" in result
        assert "LOCAL_CITATION_IDS" in result
        assert "CITATION_COUNT_MISMATCH" in result
    finally:
        set_active_workspace(old_workspace)


def test_stage3_gate_rejects_self_report_that_exceeds_canonical_registry(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        reset_gate_counters()
        _write_json(tmp_path / "paper_registry.json", [])
        _write(tmp_path / "literature_survey.md", "# Literature Survey\n\nNo papers merged.")

        result = check_stage_gate.invoke(
            {
                "stage": "stage_3",
                "evidence_json": '{"papers_found":10,"sources_used":["arxiv"]}',
            }
        )

        assert "Status**: FAIL" in result
        assert "0 paper(s) on disk" in result
        assert "reports 10" in result
    finally:
        reset_gate_counters()
        set_active_workspace(old_workspace)


def test_stage35_gate_uses_configured_deep_reading_top_k(monkeypatch, tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        monkeypatch.setenv("AUTOIDEA_DEEP_READING_TOP_K", "35")

        result = check_stage_gate.invoke(
            {"stage": "stage_3.5", "evidence_json": '{"papers_read":10,"fulltext_count":10}'}
        )

        assert "Status**: FAIL" in result
        assert "Insufficient papers read: 10 (minimum: 35)" in result
    finally:
        set_active_workspace(old_workspace)


def test_stage9_gate_respects_configured_idea_limit(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MAX_IDEAS_TO_GENERATE", "3")
    monkeypatch.setenv("AUTOIDEA_AUTO_APPROVE", "true")
    configure_web_checkpoint_events(None)
    reset_gate_counters()
    try:
        accepted = check_stage_gate.invoke(
            {
                "stage": "stage_9",
                "evidence_json": '{"ideas":["i1","i2","i3"]}',
            }
        )
        rejected = check_stage_gate.invoke(
            {
                "stage": "stage_9",
                "evidence_json": '{"ideas":["i1","i2","i3","i4"]}',
            }
        )

        assert "Status**: PASS" in accepted
        assert "Status**: FAIL" in rejected
        assert "configured maximum: 3" in rejected
    finally:
        reset_gate_counters()


def test_stage7_gate_cross_checks_structured_gap_link_count(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_stage7_artifacts(tmp_path)

        result = check_stage_gate.invoke(
            {
                "stage": "stage_7",
                "evidence_json": (
                    '{"gaps":["G1","G2","G3"],'
                    '"evidence_gap_links":5}'
                ),
            }
        )

        assert "Status**: FAIL" in result
        assert "research_gaps.json has 6" in result
    finally:
        set_active_workspace(old_workspace)


def test_web_checkpoint_interrupt_accepts_continue_automatically(tmp_path: Path) -> None:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph
    from langgraph.types import Command
    from typing_extensions import TypedDict

    class State(TypedDict, total=False):
        result: str

    def gate_node(_state: State) -> State:
        return {
            "result": check_stage_gate.invoke(
                {
                    "stage": "stage_7",
                    "evidence_json": (
                        '{"gaps":["G1","G2","G3"],'
                        '"evidence_gap_links":6}'
                    ),
                }
            )
        }

    events_path = tmp_path / "checkpoint-events.jsonl"
    _write_stage7_artifacts(tmp_path)
    configure_web_checkpoint_events(events_path)
    reset_gate_counters()
    try:
        builder = StateGraph(State)
        builder.add_node("gate", gate_node)
        builder.set_entry_point("gate")
        builder.set_finish_point("gate")
        graph = builder.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "web-checkpoint-approve"}}

        paused = graph.invoke({}, config)
        assert "__interrupt__" in paused
        interrupt_value = paused["__interrupt__"][0].value
        assert interrupt_value["type"] == "ask_user"
        assert interrupt_value["tool_call_id"] == "autoidea-checkpoint:stage_7"
        assert interrupt_value["questions"][0]["choices"][0]["value"] == "approve"
        assert interrupt_value["questions"][0]["choices"][1]["value"] == "auto_continue"

        resumed = graph.invoke(
            Command(
                resume={"status": "answered", "answers": ["auto_continue", ""]}
            ),
            config,
        )
        assert "Status**: PASS" in resumed["result"]
        assert "Mandatory Web checkpoint: APPROVED" in resumed["result"]
        assert "__interrupt__" not in resumed
    finally:
        configure_web_checkpoint_events(None)
        reset_gate_counters()


def test_web_checkpoint_revision_does_not_unlock_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph
    from langgraph.types import Command
    from typing_extensions import TypedDict

    class State(TypedDict, total=False):
        result: str

    def gate_node(_state: State) -> State:
        return {
            "result": check_stage_gate.invoke(
                {
                    "stage": "stage_9",
                    "evidence_json": '{"ideas":["i1","i2","i3","i4","i5"]}',
                }
            )
        }

    monkeypatch.setenv("AUTOIDEA_MAX_IDEAS_TO_GENERATE", "5")
    configure_web_checkpoint_events(tmp_path / "checkpoint-events.jsonl")
    reset_gate_counters()
    try:
        builder = StateGraph(State)
        builder.add_node("gate", gate_node)
        builder.set_entry_point("gate")
        builder.set_finish_point("gate")
        graph = builder.compile(checkpointer=MemorySaver())
        first_config = {"configurable": {"thread_id": "web-checkpoint-revise"}}

        paused = graph.invoke({}, first_config)
        assert "__interrupt__" in paused
        revised = graph.invoke(
            Command(
                resume={
                    "status": "answered",
                    "answers": ["revise", "Narrow the scope."],
                }
            ),
            first_config,
        )
        assert "Status**: FAIL" in revised["result"]
        assert "Narrow the scope." in revised["result"]

        second_config = {"configurable": {"thread_id": "web-checkpoint-retry"}}
        paused_again = graph.invoke({}, second_config)
        assert "__interrupt__" in paused_again
        assert (
            paused_again["__interrupt__"][0].value["tool_call_id"]
            == "autoidea-checkpoint:stage_9"
        )
    finally:
        configure_web_checkpoint_events(None)
        reset_gate_counters()


def test_web_checkpoint_approval_is_restored_from_events(tmp_path: Path) -> None:
    events_path = tmp_path / "checkpoint-events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "interaction_requested",
                        "interaction_id": "stage-10-review",
                        "checkpoint_stage": "stage_10",
                    }
                ),
                json.dumps(
                    {
                        "type": "interaction_resolved",
                        "interaction_id": "stage-10-review",
                        "response": {"approved": True, "decision": "approve"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    configure_web_checkpoint_events(events_path)
    reset_gate_counters()
    try:
        result = check_stage_gate.invoke(
            {"stage": "stage_10", "evidence_json": '{"debate_rounds":1}'}
        )
        assert "Status**: PASS" in result
        assert "Mandatory Web checkpoint: APPROVED" in result
    finally:
        configure_web_checkpoint_events(None)
        reset_gate_counters()


def test_save_stage35_reflection_rejects_count_mismatch(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write(
            tmp_path / "paper_deep_reading.md",
            """
# Paper Deep Reading Summary

## [P1] Paper A
- **Full-text status**: FULL-TEXT

## [P2] Paper B
- **Full-text status**: ABSTRACT-ONLY
""".strip(),
        )

        result = save_stage_reflection.invoke(
            {
                "stage": "stage_3.5",
                "reflection": "Read two papers.",
                "artifacts_json": '{"papers_read": 2, "fulltext_count": 0}',
            }
        )

        assert "Error:" in result
        assert "fulltext_count" in result
        assert not (tmp_path / "reflections" / "stage_3.5_reflection.json").exists()
    finally:
        set_active_workspace(old_workspace)


def test_failed_gate_cannot_save_stage_reflection(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        reset_gate_counters()

        gate_result = check_stage_gate.invoke(
            {"stage": "stage_11", "evidence_json": "{}"}
        )
        save_result = save_stage_reflection.invoke(
            {
                "stage": "stage_11",
                "reflection": "This must not be recorded as complete.",
                "artifacts_json": "{}",
            }
        )

        assert "Status**: FAIL" in gate_result
        assert "stage gate has not passed" in save_result
        assert not (tmp_path / "reflections" / "stage_11_reflection.json").exists()
    finally:
        reset_gate_counters()
        set_active_workspace(old_workspace)


def test_passed_gate_saves_single_use_persistent_proof(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        reset_gate_counters()

        gate_result = check_stage_gate.invoke(
            {
                "stage": "stage_11",
                "evidence_json": '{"assessment":"Feasible with bounded resources."}',
            }
        )
        save_result = save_stage_reflection.invoke(
            {
                "stage": "stage_11",
                "reflection": "Feasibility was assessed against resources and risks.",
                "artifacts_json": "{}",
            }
        )
        repeated_save = save_stage_reflection.invoke(
            {
                "stage": "stage_11",
                "reflection": "Attempt to reuse a stale pass.",
                "artifacts_json": "{}",
            }
        )

        record = json.loads(
            (tmp_path / "reflections" / "stage_11_reflection.json").read_text(
                encoding="utf-8"
            )
        )
        assert "Status**: PASS" in gate_result
        assert "Stage reflection saved" in save_result
        assert record["gate_passed"] is True
        assert "stage gate has not passed" in repeated_save
        assert record["reflection"] == (
            "Feasibility was assessed against resources and risks."
        )
    finally:
        reset_gate_counters()
        set_active_workspace(old_workspace)


def test_later_failed_check_revokes_an_unconsumed_gate_pass(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        reset_gate_counters()

        passed = check_stage_gate.invoke(
            {
                "stage": "stage_11",
                "evidence_json": '{"assessment":"Initially valid."}',
            }
        )
        failed = check_stage_gate.invoke(
            {"stage": "stage_11", "evidence_json": "{}"}
        )
        save_result = save_stage_reflection.invoke(
            {
                "stage": "stage_11",
                "reflection": "A stale PASS must not authorize this record.",
                "artifacts_json": "{}",
            }
        )

        assert "Status**: PASS" in passed
        assert "Status**: FAIL" in failed
        assert "stage gate has not passed" in save_result
        assert not (tmp_path / "reflections" / "stage_11_reflection.json").exists()
    finally:
        reset_gate_counters()
        set_active_workspace(old_workspace)


def test_artifact_audit_failure_cannot_authorize_reflection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from autoidea.tools.artifact_audit import (
        AuditIssue,
        AuditReport,
        AuditSeverity,
    )

    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        reset_gate_counters()
        failed_audit = AuditReport(
            workspace=str(tmp_path),
            issues=[
                AuditIssue(
                    severity=AuditSeverity.ERROR,
                    code="TEST_INTEGRITY_FAILURE",
                    message="The final artifacts are inconsistent.",
                )
            ],
        )
        monkeypatch.setattr(
            "autoidea.tools.artifact_audit.audit_workspace",
            lambda *_args, **_kwargs: failed_audit,
        )

        gate_result = check_stage_gate.invoke(
            {"stage": "stage_12", "evidence_json": "{}"}
        )
        save_result = save_stage_reflection.invoke(
            {
                "stage": "stage_12",
                "reflection": "A failed audit cannot count as completion.",
                "artifacts_json": "{}",
            }
        )

        assert "Status**: FAIL" in gate_result
        assert "TEST_INTEGRITY_FAILURE" in gate_result
        assert "All gate criteria met" not in gate_result
        assert "stage gate has not passed" in save_result
        assert not (tmp_path / "reflections" / "stage_12_reflection.json").exists()
    finally:
        reset_gate_counters()
        set_active_workspace(old_workspace)
