from __future__ import annotations

import json
from pathlib import Path

from autoidea.paths import get_active_workspace, set_active_workspace
from autoidea.tools.pipeline_state import _build_state, inspect_pipeline_state


def _write(path: Path, text: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data) -> None:
    _write(path, json.dumps(data, indent=2))


def _write_valid_stage3(tmp_path: Path) -> None:
    _write(
        tmp_path / "literature_survey.md",
        """
# Literature Survey

| ID | Paper | Year | Source | Relevance |
|---|---|---:|---|---|
| [P1] | **Paper** | 2025 | arXiv | relevant |
""".strip(),
    )
    _write_json(
        tmp_path / "paper_registry.json",
        [{"paper_id": "P1", "title": "Paper", "url": "https://example.com/paper"}],
    )


def _write_deep_reading(tmp_path: Path, count: int) -> None:
    lines = [
        "# Paper Deep Reading Summary",
        f"- **Total papers selected**: {count}",
        f"- **Full-text extracted**: {count}",
        "- **Abstract-only fallback**: 0",
        "",
    ]
    for idx in range(1, count + 1):
        lines.extend(
            [
                f"## [P{idx}] Paper {idx}",
                "- **Full-text status**: FULL-TEXT",
                "",
                "summary",
                "",
            ]
        )
    _write(tmp_path / "paper_deep_reading.md", "\n".join(lines))


def _write_valid_stage7(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "evidence_db.json",
        {"claims": [{"citation_id": "C1", "source_paper_id": "P1"}]},
    )
    _write(tmp_path / "knowledge_synthesis.md", "G1 G2 G3")
    _write_json(
        tmp_path / "research_gaps.json",
        {
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
                    "evidence_links": [
                        {
                            "citation_id": "C1",
                            "relationship": "supports",
                            "rationale": "This canonical Claim establishes the unresolved problem.",
                        }
                    ],
                    "why_it_matters": "It blocks reliable progress.",
                    "potential_direction": "Evaluate a bounded intervention.",
                    "supporting_papers": ["P1"],
                }
                for index in range(1, 4)
            ],
        },
    )


def test_inspect_pipeline_state_finds_next_stage_from_workspace_artifacts(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        for file_name in [
            "research_brief.md",
            "task_formalization.md",
            "paper_positions.json",
            "expanded_literature.md",
            "evidence_db.json",
            "knowledge_synthesis.md",
            "design_space.json",
        ]:
            _write(tmp_path / file_name, "{}" if file_name.endswith(".json") else "content")
        _write_valid_stage3(tmp_path)
        _write_deep_reading(tmp_path, 35)
        _write_valid_stage7(tmp_path)
        for stage in [
            "stage_1",
            "stage_2",
            "stage_3",
            "stage_3.5",
            "stage_4",
            "stage_5",
            "stage_6",
            "stage_7",
            "stage_8",
        ]:
            _write_json(
                tmp_path / "reflections" / f"{stage}_reflection.json",
                {"stage": stage, "reflection": "done", "artifacts": {}},
            )

        response = inspect_pipeline_state.invoke({})

        state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
        assert state["last_completed_stage"] == "stage_8"
        assert state["next_stage"] == "stage_9"
        assert state["stages"]["stage_9"]["status"] == "pending"
        assert "Next stage: stage_9" in response
        assert "raw_ideas.json" in response
    finally:
        set_active_workspace(old_workspace)


def test_inspect_pipeline_state_marks_missing_required_artifacts(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write(tmp_path / "research_brief.md", "brief")

        from autoidea.tools.pipeline_state import _build_state

        state = _build_state(tmp_path, target_paper_count=2)
        assert state["stages"]["stage_1"]["status"] == "complete"
        assert state["stages"]["stage_2"]["status"] == "pending"
        assert "task_formalization.md" in state["stages"]["stage_2"]["missing_artifacts"]
    finally:
        set_active_workspace(old_workspace)


def test_inspect_pipeline_state_requires_structured_stage7_gap_registry(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write(tmp_path / "knowledge_synthesis.md", "G1 G2 G3")

        inspect_pipeline_state.invoke({})

        state = json.loads(
            (tmp_path / "pipeline_state.json").read_text(encoding="utf-8")
        )
        assert state["stages"]["stage_7"]["status"] == "pending"
        assert state["stages"]["stage_7"]["missing_artifacts"] == [
            "research_gaps.json"
        ]
    finally:
        set_active_workspace(old_workspace)


def test_inspect_pipeline_state_rejects_partial_deep_reading_top_k(
    monkeypatch,
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        monkeypatch.setenv("AUTOIDEA_DEEP_READING_TOP_K", "35")
        _write(tmp_path / "research_brief.md", "brief")
        _write(tmp_path / "task_formalization.md", "task")
        _write_valid_stage3(tmp_path)
        _write_deep_reading(tmp_path, 10)
        for stage in ["stage_1", "stage_2", "stage_3", "stage_3.5"]:
            _write_json(
                tmp_path / "reflections" / f"{stage}_reflection.json",
                {"stage": stage, "reflection": "done", "artifacts": {"papers_read": 10}},
            )

        response = inspect_pipeline_state.invoke({})

        state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
        assert state["stages"]["stage_3.5"]["status"] == "invalid"
        assert state["next_stage"] == "stage_3.5"
        assert "expected at least 35" in response
    finally:
        set_active_workspace(old_workspace)


def test_inspect_pipeline_state_rejects_invalid_stage3_outputs(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write(tmp_path / "research_brief.md", "brief")
        _write(tmp_path / "task_formalization.md", "task")
        _write(
            tmp_path / "literature_survey.md",
            """
# Literature Survey

| ID | Paper | Year | Source | Relevance |
|---|---|---:|---|---|
""".strip(),
        )
        _write_json(
            tmp_path / "paper_registry.json",
            [
                {"paper_id": "P1", "title": "MovieChat+"},
                {"paper_id": "P2", "title": "VideoAgent"},
            ],
        )

        response = inspect_pipeline_state.invoke({})

        state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
        assert state["last_completed_stage"] == "stage_2"
        assert state["next_stage"] == "stage_3"
        assert state["stages"]["stage_3"]["status"] == "invalid"
        assert "Stage 3 artifacts are structurally invalid" in response
        assert "Resume from stage_3" in response
    finally:
        set_active_workspace(old_workspace)


def test_inspect_pipeline_state_enforces_configured_paper_target(
    tmp_path: Path,
) -> None:
    _write_valid_stage3(tmp_path)

    state = _build_state(tmp_path, target_paper_count=2)

    assert state["stages"]["stage_3"]["status"] == "invalid"
    assert state["next_stage"] == "stage_1"
    assert any(
        issue["code"] == "PAPER_REGISTRY_BELOW_TARGET"
        for issue in state["stages"]["stage_3"]["validation_issues"]
    )


def test_inspect_pipeline_state_last_completed_is_contiguous(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_valid_stage3(tmp_path)

        inspect_pipeline_state.invoke({})

        state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
        assert state["stages"]["stage_3"]["status"] == "complete"
        assert state["last_completed_stage"] == ""
        assert state["next_stage"] == "stage_1"
    finally:
        set_active_workspace(old_workspace)
