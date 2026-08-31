from __future__ import annotations

import json
from pathlib import Path

from autoidea.web.workspace import load_workspace_snapshot

FIXTURE = Path(__file__).parent / "fixtures" / "sample_workspace"


def test_load_workspace_snapshot_normalizes_research_artifacts() -> None:
    snapshot = load_workspace_snapshot(FIXTURE)

    assert snapshot.workspace.name == "sample_workspace"
    assert snapshot.counts["papers"] == 2
    assert snapshot.counts["claims"] == 2
    assert snapshot.counts["gaps"] == 3
    assert snapshot.counts["ideas"] == 1
    assert snapshot.papers[0].paper_id == "P1"
    assert snapshot.papers[0].position is not None
    assert snapshot.papers[0].position.weakest_link.startswith("Evidence selection")
    assert snapshot.claims[0].citation_id == "C1"
    assert snapshot.claims[0].source_paper_id == "P1"
    assert snapshot.gaps[0].gap_id == "G1"
    assert snapshot.gaps[0].evidence_links[0].citation_id == "C1"
    assert snapshot.ideas[0].supporting_evidence == ["C1", "C2"]
    # Observed files, not a possibly stale pipeline_state.json, are authoritative.
    assert snapshot.pipeline["next_stage"] == "stage_2"
    assert snapshot.pipeline["persisted_next_stage"] == "stage_9.5"
    assert snapshot.pipeline["persisted_state_stale"] is True
    assert "literature_survey.md" in {artifact.path for artifact in snapshot.artifacts}
    assert snapshot.warnings == []


def test_load_workspace_snapshot_builds_traceability_graph() -> None:
    snapshot = load_workspace_snapshot(FIXTURE)

    node_ids = {node.id for node in snapshot.graph.nodes}
    edge_keys = {(edge.source, edge.target, edge.kind) for edge in snapshot.graph.edges}

    assert {"paper:P1", "paper:P2", "claim:C1", "claim:C2", "idea:I1", "gap:G1"} <= node_ids
    assert ("paper:P1", "claim:C1", "supports") in edge_keys
    assert ("claim:C1", "gap:G1", "partially_covers_gap") in edge_keys
    assert ("claim:C2", "gap:G1", "supports_gap") in edge_keys
    assert ("claim:C1", "idea:I1", "evidence_for") in edge_keys
    assert ("gap:G1", "idea:I1", "targets") in edge_keys


def test_load_workspace_snapshot_records_invalid_json_warning(tmp_path: Path) -> None:
    (tmp_path / "paper_registry.json").write_text("{not valid", encoding="utf-8")

    snapshot = load_workspace_snapshot(tmp_path)

    assert snapshot.papers == []
    assert snapshot.warnings
    assert snapshot.warnings[0].code == "JSON_INVALID"
    assert "paper_registry.json" in snapshot.warnings[0].path


def test_load_workspace_snapshot_supports_legacy_evidence_provenance(tmp_path: Path) -> None:
    (tmp_path / "paper_registry.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": "P9",
                    "title": "A Legacy Evidence Paper",
                    "url": "http://example.org/paper/9/",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "evidence_db.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "C42",
                        "claim_text": "A legacy claim remains traceable.",
                        "source_paper": "P9",
                        "source_url": "https://example.org/paper/9",
                        "confidence": 0.93,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    claim = snapshot.claims[0]
    assert claim.citation_id == "C42"
    assert claim.source_paper_id == "P9"
    assert claim.source_title == "A Legacy Evidence Paper"
    assert claim.source_match == "id"
    assert claim.confidence == "HIGH"
    assert ("paper:P9", "claim:C42", "supports") in {
        (edge.source, edge.target, edge.kind) for edge in snapshot.graph.edges
    }
    assert snapshot.warnings == []


def test_load_workspace_snapshot_resolves_unique_evidence_url(tmp_path: Path) -> None:
    (tmp_path / "paper_registry.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": "P3",
                    "title": "URL Matched Paper",
                    "url": "http://example.org/paper/3/",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "evidence_db.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "citation_id": "C3",
                        "claim": "The URL identifies this source.",
                        "source_url": "https://example.org/paper/3",
                        "confidence": "MEDIUM",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    assert snapshot.claims[0].source_paper_id == "P3"
    assert snapshot.claims[0].source_match == "url"
    assert ("paper:P3", "claim:C3", "supports") in {
        (edge.source, edge.target, edge.kind) for edge in snapshot.graph.edges
    }
    assert snapshot.warnings == []


def test_load_workspace_snapshot_warns_without_inventing_source_edge(tmp_path: Path) -> None:
    (tmp_path / "paper_registry.json").write_text(
        json.dumps([{"paper_id": "P1", "title": "Known Paper"}]),
        encoding="utf-8",
    )
    (tmp_path / "evidence_db.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "citation_id": "C1",
                        "claim": "This source is not in the registry.",
                        "source_paper_id": "P404",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    assert not any(edge.kind == "supports" for edge in snapshot.graph.edges)
    assert "paper:P404" not in {node.id for node in snapshot.graph.nodes}
    assert [warning.code for warning in snapshot.warnings] == [
        "UNMAPPED_EVIDENCE_SOURCE"
    ]


def test_load_workspace_snapshot_supports_mapping_dimensions(tmp_path: Path) -> None:
    (tmp_path / "paper_registry.json").write_text(
        json.dumps([{"paper_id": "P1", "title": "Positioned Paper"}]),
        encoding="utf-8",
    )
    (tmp_path / "paper_positions.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": "P1",
                    "dimensions": {
                        "methodology": "STRONG",
                        "evaluation": {
                            "strength": "MODERATE",
                            "evidence": "Multiple benchmarks are reported.",
                            "concern": "No stress test is included.",
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    dimensions = snapshot.papers[0].position.dimensions
    assert [(item.dimension, item.strength) for item in dimensions] == [
        ("methodology", "STRONG"),
        ("evaluation", "MODERATE"),
    ]
    assert dimensions[1].evidence == "Multiple benchmarks are reported."
    assert dimensions[1].concern == "No stress test is included."


def test_load_workspace_snapshot_supports_canonical_idea_fields(tmp_path: Path) -> None:
    (tmp_path / "raw_ideas.json").write_text(
        json.dumps(
            {
                "ideas": [
                    {
                        "idea_id": "IDEA-001",
                        "title": "Canonical Idea",
                        "gap_addressed": "G1, G2",
                        "evidence_grounding": [
                            {"citation_id": "C1"},
                            {"citation_id": "C2"},
                        ],
                        "self_assessment": {"composite_score": 0.74},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    idea = snapshot.ideas[0]
    assert idea.target_gaps == ["G1", "G2"]
    assert idea.supporting_evidence == ["C1", "C2"]
    assert idea.composite_score == 0.74
    edge_keys = {(edge.source, edge.target, edge.kind) for edge in snapshot.graph.edges}
    assert ("claim:C1", "idea:IDEA-001", "evidence_for") in edge_keys
    assert ("claim:C2", "idea:IDEA-001", "evidence_for") in edge_keys
    assert ("gap:G1", "idea:IDEA-001", "targets") in edge_keys
    assert ("gap:G2", "idea:IDEA-001", "targets") in edge_keys


def test_load_workspace_snapshot_skips_unknown_gap_evidence_without_guessing(
    tmp_path: Path,
) -> None:
    (tmp_path / "paper_registry.json").write_text(
        json.dumps([{"paper_id": "P1", "title": "Known source paper"}]),
        encoding="utf-8",
    )
    (tmp_path / "evidence_db.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "citation_id": "C1",
                        "claim": "Known evidence.",
                        "source_paper_id": "P1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "research_gaps.json").write_text(
        json.dumps(
            {
                "gaps": [
                    {
                        "gap_id": "G1",
                        "title": "Grounded gap",
                        "evidence_links": [
                            {
                                "citation_id": "C404",
                                "relationship": "supports",
                                "rationale": "This ID is intentionally invalid for the test.",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    assert not any(
        edge.source == "claim:C404" and edge.target == "gap:G1"
        for edge in snapshot.graph.edges
    )
    assert "claim:C404" not in {node.id for node in snapshot.graph.nodes}
    assert [warning.code for warning in snapshot.warnings] == [
        "UNMAPPED_GAP_EVIDENCE"
    ]


def test_load_workspace_snapshot_preserves_gap_edge_role_and_rationale(
    tmp_path: Path,
) -> None:
    (tmp_path / "evidence_db.json").write_text(
        json.dumps({"claims": [{"citation_id": "C1", "claim": "Known evidence."}]}),
        encoding="utf-8",
    )
    rationale = "The method addresses only a restricted evaluation setting."
    (tmp_path / "research_gaps.json").write_text(
        json.dumps(
            {
                "gaps": [
                    {
                        "gap_id": "G1",
                        "title": "Evaluation coverage",
                        "gap_type": "evaluation_gap",
                        "evidence_links": [
                            {
                                "citation_id": "C1",
                                "relationship": "partial_coverage",
                                "rationale": rationale,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    edge = next(edge for edge in snapshot.graph.edges if edge.target == "gap:G1")
    assert edge.source == "claim:C1"
    assert edge.kind == "partially_covers_gap"
    assert edge.detail == rationale
    assert next(node for node in snapshot.graph.nodes if node.id == "gap:G1").label == "Evaluation coverage"


def test_load_workspace_snapshot_rejects_incomplete_gap_relationships(
    tmp_path: Path,
) -> None:
    (tmp_path / "paper_registry.json").write_text(
        json.dumps([{"paper_id": "P1", "title": "Known source paper"}]),
        encoding="utf-8",
    )
    (tmp_path / "evidence_db.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "citation_id": "C1",
                        "claim": "Known evidence.",
                        "source_paper_id": "P1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "research_gaps.json").write_text(
        json.dumps(
            {
                "gaps": [
                    {
                        "gap_id": "G1",
                        "title": "Unsupported relationship",
                        "evidence_links": [
                            {
                                "citation_id": "C1",
                                "relationship": "mentions",
                                "rationale": "This relation is intentionally unsupported.",
                            }
                        ],
                    },
                    {
                        "gap_id": "G2",
                        "title": "Missing rationale",
                        "evidence_links": [
                            {
                                "citation_id": "C1",
                                "relationship": "supports",
                                "rationale": "",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_workspace_snapshot(tmp_path)

    assert not any(
        edge.target in {"gap:G1", "gap:G2"}
        and edge.kind in {"supports_gap", "partially_covers_gap", "challenges_gap"}
        for edge in snapshot.graph.edges
    )
    assert [warning.code for warning in snapshot.warnings] == [
        "INVALID_GAP_EVIDENCE_RELATIONSHIP",
        "INVALID_GAP_EVIDENCE_RATIONALE",
    ]
