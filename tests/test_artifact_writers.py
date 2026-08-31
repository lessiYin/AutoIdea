from __future__ import annotations

import json
from pathlib import Path

from autoidea.paths import get_active_workspace, set_active_workspace
from autoidea.tools.artifact_writers import write_raw_ideas, write_research_gaps
from autoidea.tools.think import write_workspace_file


def test_write_raw_ideas_writes_validated_canonical_artifact(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)

        content = json.dumps(
            {
                "generated_count": 2,
                "kept_top_5": 2,
                "ideas": [
                    {
                        "idea_id": "IDEA-001",
                        "title": "Grounded Temporal Notebook",
                        "one_liner": "A training-free long-video agent records verified temporal evidence before answering.",
                        "description": "Use a sparse notebook to bind events to timestamps and sources.",
                        "key_mechanism": "Iterative segment retrieval plus evidence consolidation.",
                        "supporting_evidence": ["C1"],
                        "target_gaps": ["G1"],
                        "self_assessment": {"novelty": 4, "feasibility": 4, "impact": 4},
                        "composite_score": 4.0,
                    },
                    {
                        "idea_id": "IDEA-002",
                        "title": "Budgeted Multi-hop Video Search",
                        "one_liner": "A training-free planner allocates fixed visual search budget across long-video hypotheses.",
                        "description": "Plan, search, verify, and stop under explicit budget.",
                        "key_mechanism": "Evidence-gated query refinement over video chunks.",
                        "supporting_evidence": ["C2"],
                        "target_gaps": ["G2"],
                        "self_assessment": {"novelty": 3, "feasibility": 5, "impact": 4},
                        "composite_score": 3.9,
                    },
                ],
            }
        )

        result = write_raw_ideas.invoke({"content": content})

        assert "File written successfully: raw_ideas.json" in result
        written = json.loads((tmp_path / "raw_ideas.json").read_text(encoding="utf-8"))
        assert written["generated_count"] == 2
        assert written["kept_top_5"] == 2
        assert written["ideas"][0]["idea_id"] == "IDEA-001"
    finally:
        set_active_workspace(old_workspace)


def test_write_raw_ideas_rejects_malformed_json_without_writing(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)

        result = write_raw_ideas.invoke({"content": '{"generated_count": 1'})

        assert "Error writing raw_ideas.json" in result
        assert "invalid JSON" in result
        assert not (tmp_path / "raw_ideas.json").exists()
    finally:
        set_active_workspace(old_workspace)


def test_write_raw_ideas_rejects_count_above_configured_maximum(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        monkeypatch.setenv("AUTOIDEA_MAX_IDEAS_TO_GENERATE", "3")
        content = json.dumps(
            {
                "generated_count": 4,
                "kept_top_k": ["I1", "I2", "I3", "I4"],
                "ideas": [{"idea_id": f"I{index}"} for index in range(1, 5)],
            }
        )

        result = write_raw_ideas.invoke({"content": content})

        assert "Error writing raw_ideas.json" in result
        assert "max_ideas_to_generate=3" in result
        assert not (tmp_path / "raw_ideas.json").exists()
    finally:
        set_active_workspace(old_workspace)


def _research_gap_catalog(*, citation_id: str = "C1", gap_score: int = 3) -> dict:
    return {
        "schema_version": "1.0",
        "generated_from": "evidence_db.json",
        "gaps": [
            {
                "gap_id": f"G{index}",
                "title": f"Gap {index}",
                "description": f"A precise unresolved problem for gap {index}.",
                "gap_type": "methodology_gap",
                "demand": 5,
                "coverage": 2,
                "gap_score": gap_score,
                "evidence_links": [
                    {
                        "citation_id": citation_id,
                        "relationship": "supports",
                        "rationale": "The cited Claim directly establishes this unresolved limitation.",
                    }
                ],
                "why_it_matters": "It blocks a reliable research outcome.",
                "potential_direction": "Test a bounded evidence-grounded intervention.",
            }
            for index in range(1, 4)
        ],
    }


def test_write_research_gaps_validates_claim_links_and_writes_catalog(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        (tmp_path / "evidence_db.json").write_text(
            json.dumps({"claims": [{"citation_id": "C1"}]}),
            encoding="utf-8",
        )

        result = write_research_gaps.invoke(
            {"content": json.dumps(_research_gap_catalog())}
        )

        assert "File written successfully: research_gaps.json" in result
        written = json.loads(
            (tmp_path / "research_gaps.json").read_text(encoding="utf-8")
        )
        assert [gap["gap_id"] for gap in written["gaps"]] == ["G1", "G2", "G3"]
        assert written["gaps"][0]["evidence_links"][0]["citation_id"] == "C1"
    finally:
        set_active_workspace(old_workspace)


def test_write_research_gaps_rejects_unknown_claim_without_overwriting(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        (tmp_path / "evidence_db.json").write_text(
            json.dumps({"claims": [{"citation_id": "C1"}]}),
            encoding="utf-8",
        )
        existing = _research_gap_catalog()
        (tmp_path / "research_gaps.json").write_text(
            json.dumps(existing),
            encoding="utf-8",
        )

        result = write_research_gaps.invoke(
            {
                "content": json.dumps(
                    _research_gap_catalog(citation_id="C404")
                )
            }
        )

        assert "Error writing research_gaps.json" in result
        assert "C404" in result
        assert json.loads(
            (tmp_path / "research_gaps.json").read_text(encoding="utf-8")
        ) == existing
    finally:
        set_active_workspace(old_workspace)


def test_write_research_gaps_rejects_incorrect_gap_score(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        (tmp_path / "evidence_db.json").write_text(
            json.dumps({"claims": [{"citation_id": "C1"}]}),
            encoding="utf-8",
        )

        result = write_research_gaps.invoke(
            {"content": json.dumps(_research_gap_catalog(gap_score=4))}
        )

        assert "Error writing research_gaps.json" in result
        assert "gap_score must equal demand - coverage" in result
        assert not (tmp_path / "research_gaps.json").exists()
    finally:
        set_active_workspace(old_workspace)


def test_write_research_gaps_rejects_blank_relationship_rationale(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        (tmp_path / "evidence_db.json").write_text(
            json.dumps({"claims": [{"citation_id": "C1"}]}),
            encoding="utf-8",
        )
        catalog = _research_gap_catalog()
        catalog["gaps"][0]["evidence_links"][0]["rationale"] = " " * 12

        result = write_research_gaps.invoke({"content": json.dumps(catalog)})

        assert "Error writing research_gaps.json" in result
        assert "at least 12 characters" in result
        assert not (tmp_path / "research_gaps.json").exists()
    finally:
        set_active_workspace(old_workspace)


def test_write_workspace_file_rejects_aggregated_paper_position_ids(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        existing = [{"paper_id": "P1", "title": "Valid Paper"}]
        (tmp_path / "paper_positions.json").write_text(
            json.dumps(existing, indent=2),
            encoding="utf-8",
        )

        result = write_workspace_file.invoke(
            {
                "file_path": "paper_positions.json",
                "content": json.dumps(
                    [
                        {"paper_id": "P1", "title": "Valid Paper"},
                        {"paper_id": "P36-P93", "title": "Aggregated papers"},
                    ]
                ),
            }
        )

        assert "Error writing paper_positions.json" in result
        assert "invalid paper_id" in result
        written = json.loads((tmp_path / "paper_positions.json").read_text(encoding="utf-8"))
        assert written == existing
    finally:
        set_active_workspace(old_workspace)


def test_write_workspace_file_rejects_placeholder_canonical_artifact(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)

        result = write_workspace_file.invoke(
            {
                "file_path": "expanded_literature.md",
                "content": "test",
            }
        )

        assert "validation failed" in result
        assert not (tmp_path / "expanded_literature.md").exists()
    finally:
        set_active_workspace(old_workspace)


def test_write_workspace_file_can_append_to_markdown_artifact(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        initial = "# Final Report\n\n" + ("Grounded evidence. " * 12) + "Stage 12"
        (tmp_path / "final_report.md").write_text(initial, encoding="utf-8")

        result = write_workspace_file.invoke(
            {
                "file_path": "final_report.md",
                "content": " is complete.\n\n## Conclusion\n\nThe report is auditable.",
                "mode": "append",
            }
        )

        assert "File appended successfully: final_report.md" in result
        assert (tmp_path / "final_report.md").read_text(encoding="utf-8") == (
            initial + " is complete.\n\n## Conclusion\n\nThe report is auditable."
        )
    finally:
        set_active_workspace(old_workspace)


def test_write_workspace_file_rejects_append_to_json(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        path = tmp_path / "paper_registry.json"
        path.write_text("[]", encoding="utf-8")

        result = write_workspace_file.invoke(
            {
                "file_path": "paper_registry.json",
                "content": "{}",
                "mode": "append",
            }
        )

        assert "append and replace modes are supported only for Markdown" in result
        assert path.read_text(encoding="utf-8") == "[]"
    finally:
        set_active_workspace(old_workspace)


def test_write_workspace_file_can_replace_one_markdown_fragment(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        path = tmp_path / "final_report.md"
        initial = "# Final Report\n\n" + ("Grounded evidence. " * 12) + "unfinished tail"
        path.write_text(initial, encoding="utf-8")

        result = write_workspace_file.invoke(
            {
                "file_path": "final_report.md",
                "old_text": "unfinished tail",
                "content": "The report is complete.",
                "mode": "replace",
            }
        )

        assert "File updated successfully: final_report.md" in result
        assert path.read_text(encoding="utf-8") == initial.replace(
            "unfinished tail",
            "The report is complete.",
        )
    finally:
        set_active_workspace(old_workspace)


def test_write_workspace_file_rejects_ambiguous_replacement(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        path = tmp_path / "notes.md"
        path.write_text("duplicate duplicate", encoding="utf-8")

        result = write_workspace_file.invoke(
            {
                "file_path": "notes.md",
                "old_text": "duplicate",
                "content": "replacement",
                "mode": "replace",
            }
        )

        assert "exactly once; found 2 matches" in result
        assert path.read_text(encoding="utf-8") == "duplicate duplicate"
    finally:
        set_active_workspace(old_workspace)
