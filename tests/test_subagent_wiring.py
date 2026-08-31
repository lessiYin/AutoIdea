from __future__ import annotations

from autoidea.autoidea import _build_tool_registry
from autoidea.utils import build_subagent_definitions, load_subagents_yaml


def _subagent_tool_names() -> dict[str, list[str]]:
    raw = load_subagents_yaml("autoidea/subagent.yaml")
    registry, _ = _build_tool_registry()
    subs = build_subagent_definitions(raw, registry)
    return {
        sub["name"]: [getattr(tool, "name", "") for tool in sub["tools"]]
        for sub in subs
    }


def test_evidence_agent_has_dedicated_evidence_writer() -> None:
    tools = _subagent_tool_names()

    assert "write_evidence_db" in tools["evidence-agent"]


def test_synthesis_agent_has_dedicated_gap_writer() -> None:
    tools = _subagent_tool_names()

    assert "write_research_gaps" in tools["synthesis-agent"]


def test_main_agent_can_register_citations_required_by_stage_6() -> None:
    _, base_tools = _build_tool_registry()

    assert "cite_source" in [getattr(tool, "name", "") for tool in base_tools]


def test_positioning_agent_does_not_get_evidence_writer() -> None:
    tools = _subagent_tool_names()

    assert "write_evidence_db" not in tools["positioning-agent"]


def test_batch_tools_are_available_to_relevant_subagents() -> None:
    tools = _subagent_tool_names()

    assert "create_search_batches" in tools["survey-agent"]
    assert "record_batch_result" in tools["survey-agent"]
    assert "read_batch_manifest" in tools["survey-agent"]
    assert "merge_search_batches" in tools["survey-agent"]

    assert "create_reading_batches" in tools["reader-agent"]
    assert "record_batch_result" in tools["reader-agent"]
    assert "merge_reading_batches" in tools["reader-agent"]

    assert "create_evidence_batches" in tools["evidence-agent"]
    assert "record_batch_result" in tools["evidence-agent"]
    assert "merge_evidence_batches" in tools["evidence-agent"]
