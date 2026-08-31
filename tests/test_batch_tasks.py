from __future__ import annotations

import json
from pathlib import Path

from autoidea.paths import get_active_workspace, set_active_workspace
from autoidea.tools.batch_tasks import (
    create_evidence_batches,
    create_reading_batches,
    create_search_batches,
    merge_evidence_batches,
    merge_reading_batches,
    merge_search_batches,
    read_batch_manifest,
    record_batch_result,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_fulltext_audit(path: Path) -> None:
    _write_json(
        path / "fulltext_audit.json",
        {
            "records": [
                {
                    "identifier": "Paper A",
                    "status": "full_text",
                    "pdf_url": "https://example.com/a.pdf",
                    "chars_extracted": 12000,
                    "text_path": "paper_texts/paper_a.txt",
                    "reason": "",
                },
                {
                    "identifier": "Paper B",
                    "status": "failed",
                    "pdf_url": "https://example.com/b.pdf",
                    "chars_extracted": 0,
                    "text_path": "",
                    "reason": "download_failed: 403",
                },
            ]
        },
    )
    text_path = path / "paper_texts" / "paper_a.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("full text", encoding="utf-8")


def test_create_search_batches_writes_file_inputs_and_manifest(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)

        result = create_search_batches.invoke(
            {
                "queries_json": json.dumps(
                    [
                        "training-free long video understanding agent",
                        "video rag long video question answering",
                        "agentic video retrieval",
                    ]
                ),
                "batch_size": 2,
                "sources_json": json.dumps(["arxiv", "openalex"]),
                "max_results_per_query": 5,
            }
        )

        assert "Created 2 stage_3_search batch(es)" in result
        first_input = _read_json(
            tmp_path / "batches" / "stage_3_search" / "batch_001" / "input.json"
        )
        assert first_input["queries"] == [
            "training-free long video understanding agent",
            "video rag long video question answering",
        ]
        assert first_input["sources"] == ["arxiv", "openalex"]

        manifest = _read_json(tmp_path / "batch_manifest.json")
        assert manifest["summary"]["total_batches"] == 2
        assert manifest["summary"]["pending"] == 2
        assert manifest["batches"][0]["status"] == "pending"
        assert manifest["batches"][0]["input_file"].endswith("input.json")
    finally:
        set_active_workspace(old_workspace)


def test_create_batches_reuses_identical_input_and_appends_different_input(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        request = {
            "queries_json": json.dumps(["query one"]),
            "batch_size": 1,
            "sources_json": json.dumps(["arxiv"]),
            "max_results_per_query": 5,
        }
        create_search_batches.invoke(request)
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_search_batch_001",
                "status": "passed",
                "result_json": json.dumps({"papers": []}),
                "summary": "No matching papers.",
            }
        )

        repeated = create_search_batches.invoke(request)
        appended = create_search_batches.invoke(
            {
                **request,
                "queries_json": json.dumps(["query two"]),
            }
        )

        assert "1 reused without resetting status" in repeated
        assert "stage_3_search_batch_002" in appended
        first_status = _read_json(
            tmp_path / "batches" / "stage_3_search" / "batch_001" / "status.json"
        )
        second_input = _read_json(
            tmp_path / "batches" / "stage_3_search" / "batch_002" / "input.json"
        )
        manifest = _read_json(tmp_path / "batch_manifest.json")
        assert first_status["status"] == "passed"
        assert second_input["queries"] == ["query two"]
        assert manifest["summary"] == {
            "total_batches": 2,
            "pending": 1,
            "running": 0,
            "passed": 1,
            "failed": 0,
        }
    finally:
        set_active_workspace(old_workspace)


def test_record_batch_result_writes_result_summary_and_returns_short_status(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_fulltext_audit(tmp_path)
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1", "P2"]),
                "batch_size": 2,
            }
        )

        long_detail = "full text detail " * 1000
        result_payload = {
            "papers_processed": 2,
            "fulltext_count": 1,
            "abstract_only_count": 1,
            "failed_count": 0,
            "readings": [
                {
                    "paper_id": "P1",
                    "title": "Paper A",
                    "fulltext_status": "FULL-TEXT",
                    "summary": "Detailed verified summary.",
                },
                {
                    "paper_id": "P2",
                    "title": "Paper B",
                    "fulltext_status": "ABSTRACT-ONLY",
                    "summary": "Abstract-based summary.",
                },
            ],
            "details": long_detail,
        }
        response = record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(result_payload),
                "summary": "Processed P1 as full text and P2 as abstract-only.",
            }
        )

        assert "stage_3_5_reading_batch_001" in response
        assert "result.json" in response
        assert long_detail[:80] not in response
        result_file = _read_json(
            tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "result.json"
        )
        assert result_file["papers_processed"] == 2
        assert (
            tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "summary.md"
        ).read_text(encoding="utf-8").startswith("Processed P1")

        status_file = _read_json(
            tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "status.json"
        )
        assert status_file["status"] == "passed"
        manifest = _read_json(tmp_path / "batch_manifest.json")
        assert manifest["summary"]["passed"] == 1
    finally:
        set_active_workspace(old_workspace)


def test_record_reading_batch_matches_arxiv_audit_from_traceability_note(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_json(
            tmp_path / "fulltext_audit.json",
            {
                "records": [
                    {
                        "identifier": "2403.10517v1",
                        "status": "full_text",
                        "pdf_url": "https://arxiv.org/pdf/2403.10517v1",
                        "chars_extracted": 65553,
                        "text_path": "paper_texts/2403.10517.txt",
                        "reason": "",
                    }
                ]
            },
        )
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P2"]),
                "batch_size": 1,
            }
        )

        response = record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "readings": [
                            {
                                "paper_id": "P2",
                                "title": "VideoAgent: Long-form Video Understanding with Large Language Model as Agent",
                                "fulltext_status": "FULL-TEXT",
                                "summary": "Detailed verified summary from the fetched full text.",
                                "audit_traceability_note": "Fetched with fetch_paper_fulltext using arXiv https://arxiv.org/abs/2403.10517.",
                            }
                        ]
                    }
                ),
                "summary": "Read P2 from arXiv full text.",
            }
        )

        assert "Recorded stage_3_5_reading_batch_001 as passed" in response
        assert (
            tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "result.json"
        ).exists()
    finally:
        set_active_workspace(old_workspace)


def test_read_batch_manifest_returns_concise_summary_not_raw_results(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_fulltext_audit(tmp_path)
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "papers_processed": 1,
                        "readings": [
                            {
                                "paper_id": "P1",
                                "title": "Paper A",
                                "fulltext_status": "FULL-TEXT",
                                "summary": "Short summary.",
                            }
                        ],
                        "details": "RAW_DETAIL_SHOULD_NOT_APPEAR" * 100,
                    }
                ),
                "summary": "Short reading summary.",
            }
        )

        response = read_batch_manifest.invoke({"stage": "stage_3_5_reading"})

        assert "stage_3_5_reading_batch_001" in response
        assert "Short reading summary." in response
        assert "RAW_DETAIL_SHOULD_NOT_APPEAR" not in response
        assert len(response) < 4000
    finally:
        set_active_workspace(old_workspace)


def test_record_reading_batch_rejects_abstract_only_without_fulltext_audit(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_fulltext_audit(tmp_path)
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
            }
        )

        response = record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "readings": [
                            {
                                "paper_id": "P1",
                                "title": "Paper A",
                                "fulltext_status": "ABSTRACT-ONLY",
                                "summary": "ABSTRACT-ONLY conservative summary based on registry metadata.",
                            }
                        ]
                    }
                ),
                "summary": "Recorded abstract-only summary without fetching full text.",
            }
        )

        assert "Error recording batch result" in response
        assert "fulltext_audit.json" in response
        assert not (
            tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "result.json"
        ).exists()
        status_file = _read_json(
            tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "status.json"
        )
        assert status_file["status"] == "pending"
    finally:
        set_active_workspace(old_workspace)


def test_merge_search_batches_skips_failed_and_writes_canonical_outputs(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_search_batches.invoke(
            {
                "queries_json": json.dumps(["q1", "q2"]),
                "batch_size": 1,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_search_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "papers": [
                            {
                                "title": "Paper A",
                                "authors": ["Alice"],
                                "year": 2025,
                                "venue": "arXiv",
                                "url": "https://arxiv.org/abs/2501.00001",
                                "source": "arxiv",
                                "relevance": "anchor",
                            }
                        ]
                    }
                ),
                "summary": "Found Paper A.",
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_search_batch_002",
                "status": "failed",
                "result_json": json.dumps(
                    {
                        "papers": [
                            {
                                "title": "Failed Paper",
                                "url": "https://example.com/failed",
                            }
                        ]
                    }
                ),
                "summary": "Failed batch.",
                "error": "rate limited",
            }
        )

        response = merge_search_batches.invoke({})

        assert "Merged 1 search paper(s)" in response
        registry = _read_json(tmp_path / "paper_registry.json")
        assert [paper["title"] for paper in registry] == ["Paper A"]
        survey = (tmp_path / "literature_survey.md").read_text(encoding="utf-8")
        assert "Paper A" in survey
        assert "Failed Paper" not in survey
    finally:
        set_active_workspace(old_workspace)


def test_merge_search_batches_rejects_pending_batches_without_overwrite(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_search_batches.invoke(
            {
                "queries_json": json.dumps(["q1", "q2"]),
                "batch_size": 1,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_search_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "papers": [
                            {
                                "title": "Paper A",
                                "authors": ["Alice"],
                                "year": 2025,
                                "venue": "arXiv",
                                "url": "https://arxiv.org/abs/2501.00001",
                                "source": "arxiv",
                                "relevance": "anchor",
                            }
                        ]
                    }
                ),
                "summary": "Found Paper A.",
            }
        )

        response = merge_search_batches.invoke({})

        assert "Error merging search batches" in response
        assert "unfinished stage_3_search batch(es)" in response
        assert not (tmp_path / "paper_registry.json").exists()
        assert not (tmp_path / "literature_survey.md").exists()
    finally:
        set_active_workspace(old_workspace)


def test_record_search_batch_result_rejects_notable_titles_without_papers(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_search_batches.invoke(
            {
                "queries_json": json.dumps(["training-free long video understanding agent"]),
                "batch_size": 1,
            }
        )

        response = record_batch_result.invoke(
            {
                "batch_id": "stage_3_search_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "queries": ["training-free long video understanding agent"],
                        "sources": ["arxiv"],
                        "notable_titles": [
                            "VideoAgent: Long-form Video Understanding with Large Language Model as Agent"
                        ],
                    }
                ),
                "summary": "Found notable titles but did not write structured papers.",
            }
        )

        assert "Error recording batch result" in response
        assert "papers" in response
        assert "notable_titles" in response
        assert not (
            tmp_path / "batches" / "stage_3_search" / "batch_001" / "result.json"
        ).exists()
        status_file = _read_json(
            tmp_path / "batches" / "stage_3_search" / "batch_001" / "status.json"
        )
        assert status_file["status"] == "pending"
        manifest = _read_json(tmp_path / "batch_manifest.json")
        assert manifest["summary"]["pending"] == 1
        assert manifest["summary"]["passed"] == 0
    finally:
        set_active_workspace(old_workspace)


def test_record_reading_batch_matches_fulltext_audit_via_registry_url(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_json(
            tmp_path / "paper_registry.json",
            [
                {
                    "paper_id": "P1",
                    "title": "MovieChat+: Question-aware Sparse Memory for Long Video Question Answering",
                    "url": "https://arxiv.org/abs/2404.17176",
                }
            ],
        )
        _write_json(
            tmp_path / "fulltext_audit.json",
            {
                "records": [
                    {
                        "identifier": "2404.17176",
                        "status": "full_text",
                        "pdf_url": "https://arxiv.org/pdf/2404.17176",
                        "chars_extracted": 89971,
                        "text_path": "paper_texts/2404.17176.txt",
                        "reason": "",
                    }
                ]
            },
        )
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
            }
        )

        response = record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "readings": [
                            {
                                "paper_id": "P1",
                                "title": "MovieChat+: Question-aware Sparse Memory for Long Video Question Answering",
                                "fulltext_status": "FULL-TEXT",
                                "summary": "Detailed verified summary from fetched full text.",
                            }
                        ]
                    }
                ),
                "summary": "Read P1 from arXiv full text.",
            }
        )

        assert "Recorded stage_3_5_reading_batch_001 as passed" in response
    finally:
        set_active_workspace(old_workspace)


def test_record_batch_result_rejects_passed_reading_and_evidence_missing_schema(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_fulltext_audit(tmp_path)
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
            }
        )
        reading_response = record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps({"papers_processed": 1}),
                "summary": "Processed one paper but omitted readings.",
            }
        )

        assert "Error recording batch result" in reading_response
        assert "readings" in reading_response
        reading_status = _read_json(
            tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "status.json"
        )
        assert reading_status["status"] == "pending"

        create_evidence_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
                "claim_budget_per_paper": 2,
            }
        )
        evidence_response = record_batch_result.invoke(
            {
                "batch_id": "stage_6_evidence_batch_001",
                "status": "passed",
                "result_json": json.dumps({"papers_processed": 1}),
                "summary": "Processed one paper but omitted claims.",
            }
        )

        assert "Error recording batch result" in evidence_response
        assert "claims" in evidence_response
        evidence_status = _read_json(
            tmp_path / "batches" / "stage_6_evidence" / "batch_001" / "status.json"
        )
        assert evidence_status["status"] == "pending"
    finally:
        set_active_workspace(old_workspace)


def test_record_batch_result_rejects_passed_evidence_claim_without_source(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_evidence_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
                "claim_budget_per_paper": 2,
            }
        )

        response = record_batch_result.invoke(
            {
                "batch_id": "stage_6_evidence_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "claims": [
                            {
                                "claim_text": "A source-less claim is not auditable.",
                                "paper_id": "P1",
                                "confidence": 0.8,
                            }
                        ]
                    }
                ),
                "summary": "Malformed evidence claim.",
            }
        )

        assert "Error recording batch result" in response
        assert "source_title" in response
        status = _read_json(
            tmp_path / "batches" / "stage_6_evidence" / "batch_001" / "status.json"
        )
        assert status["status"] == "pending"
    finally:
        set_active_workspace(old_workspace)


def test_merge_search_batches_rejects_legacy_malformed_passed_batches_without_overwrite(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_search_batches.invoke(
            {
                "queries_json": json.dumps(["training-free long video understanding agent"]),
                "batch_size": 1,
            }
        )
        result_path = tmp_path / "batches" / "stage_3_search" / "batch_001" / "result.json"
        summary_path = tmp_path / "batches" / "stage_3_search" / "batch_001" / "summary.md"
        status_path = tmp_path / "batches" / "stage_3_search" / "batch_001" / "status.json"
        _write_json(result_path, {"notable_titles": ["Paper A"]})
        summary_path.write_text("Malformed legacy batch output.", encoding="utf-8")
        status = _read_json(status_path)
        status.update(
            {
                "status": "passed",
                "result_file": "batches/stage_3_search/batch_001/result.json",
                "summary_file": "batches/stage_3_search/batch_001/summary.md",
            }
        )
        _write_json(status_path, status)
        manifest = _read_json(tmp_path / "batch_manifest.json")
        manifest["batches"][0].update(
            {
                "status": "passed",
                "result_file": "batches/stage_3_search/batch_001/result.json",
                "summary_file": "batches/stage_3_search/batch_001/summary.md",
            }
        )
        manifest["summary"] = {"total_batches": 1, "pending": 0, "running": 0, "passed": 1, "failed": 0}
        _write_json(tmp_path / "batch_manifest.json", manifest)

        existing_registry = [{"paper_id": "P1", "title": "Existing Paper"}]
        existing_survey = "# Literature Survey\n\nExisting Paper\n"
        _write_json(tmp_path / "paper_registry.json", existing_registry)
        (tmp_path / "literature_survey.md").write_text(existing_survey, encoding="utf-8")

        response = merge_search_batches.invoke({})

        assert "Error merging search batches" in response
        assert "missing required `papers`" in response
        assert _read_json(tmp_path / "paper_registry.json") == existing_registry
        assert (tmp_path / "literature_survey.md").read_text(encoding="utf-8") == existing_survey
    finally:
        set_active_workspace(old_workspace)


def test_merge_search_batches_rejects_zero_structured_papers_without_overwrite(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_search_batches.invoke(
            {
                "queries_json": json.dumps(["q1"]),
                "batch_size": 1,
            }
        )
        record_response = record_batch_result.invoke(
            {
                "batch_id": "stage_3_search_batch_001",
                "status": "passed",
                "result_json": json.dumps({"papers": []}),
                "summary": "No structured papers found.",
            }
        )
        assert "Recorded stage_3_search_batch_001 as passed" in record_response

        existing_registry = [{"paper_id": "P1", "title": "Existing Paper"}]
        existing_survey = "# Literature Survey\n\nExisting Paper\n"
        _write_json(tmp_path / "paper_registry.json", existing_registry)
        (tmp_path / "literature_survey.md").write_text(existing_survey, encoding="utf-8")

        response = merge_search_batches.invoke({})

        assert "Error merging search batches" in response
        assert "0 structured paper" in response
        assert _read_json(tmp_path / "paper_registry.json") == existing_registry
        assert (tmp_path / "literature_survey.md").read_text(encoding="utf-8") == existing_survey
    finally:
        set_active_workspace(old_workspace)


def test_merge_reading_and_evidence_batches_write_canonical_outputs(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_fulltext_audit(tmp_path)
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "readings": [
                            {
                                "paper_id": "P1",
                                "title": "Paper A",
                                "fulltext_status": "FULL-TEXT",
                                "summary": "Detailed verified summary.",
                            }
                        ]
                    }
                ),
                "summary": "Read Paper A.",
            }
        )

        reading_response = merge_reading_batches.invoke({})

        assert "Merged 1 reading item(s)" in reading_response
        deep_reading = (tmp_path / "paper_deep_reading.md").read_text(encoding="utf-8")
        assert "Paper A" in deep_reading
        assert "FULL-TEXT" in deep_reading

        create_evidence_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
                "claim_budget_per_paper": 2,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_6_evidence_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "claims": [
                            {
                                "claim_id": "C1",
                                "claim_text": "Paper A supports training-free retrieval.",
                                "source_paper_id": "P1",
                                "source_title": "Paper A",
                                "source_url": "https://arxiv.org/abs/2501.00001",
                                "confidence": 0.8,
                            }
                        ]
                    }
                ),
                "summary": "Extracted one claim.",
            }
        )

        evidence_response = merge_evidence_batches.invoke({})

        assert "Merged 1 evidence claim(s)" in evidence_response
        evidence = _read_json(tmp_path / "evidence_db.json")
        assert evidence["summary"]["total_claims"] == 1
        assert evidence["claims"][0]["claim_id"] == "C1"
    finally:
        set_active_workspace(old_workspace)


def test_merge_evidence_batches_assigns_citation_ids_for_claims_without_ids(
    tmp_path: Path,
) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_evidence_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
                "claim_budget_per_paper": 2,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_6_evidence_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "claims": [
                            {
                                "claim_text": "Paper A supports grounded retrieval.",
                                "paper_id": "P1",
                                "source_title": "Paper A",
                                "source_url": "https://arxiv.org/abs/2501.00001",
                                "evidence_type": "method_description",
                                "confidence": 0.82,
                                "section": "Method",
                            }
                        ]
                    }
                ),
                "summary": "Extracted one claim without a preassigned citation ID.",
            }
        )

        response = merge_evidence_batches.invoke({})

        assert "Merged 1 evidence claim(s)" in response
        evidence = _read_json(tmp_path / "evidence_db.json")
        assert evidence["claims"][0]["citation_id"] == "C1"
        assert evidence["claims"][0]["claim"] == "Paper A supports grounded retrieval."
        assert evidence["summary"]["citations_count"] == 1
        citations = _read_json(tmp_path / "citations.json")
        assert citations[0]["citation_id"] == "C1"
        assert citations[0]["claim"] == "Paper A supports grounded retrieval."
        assert citations[0]["paper_id"] == "P1"
    finally:
        set_active_workspace(old_workspace)


def test_record_evidence_batch_accepts_claim_alias(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        create_evidence_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
                "claim_budget_per_paper": 2,
            }
        )

        response = record_batch_result.invoke(
            {
                "batch_id": "stage_6_evidence_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "claims": [
                            {
                                "claim": "Paper A supports calibrated evidence search.",
                                "source_paper_id": "P1",
                                "source_title": "Paper A",
                                "source_url": "https://arxiv.org/abs/2501.00001",
                                "confidence": 0.8,
                            }
                        ]
                    }
                ),
                "summary": "Extracted one evidence claim with claim alias.",
            }
        )

        assert "Recorded stage_6_evidence_batch_001 as passed" in response
    finally:
        set_active_workspace(old_workspace)


def test_merge_reading_batches_strips_nested_heading_and_status_from_summary(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_fulltext_audit(tmp_path)
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "readings": [
                            {
                                "paper_id": "P1",
                                "title": "Paper A",
                                "fulltext_status": "FULL-TEXT",
                                "summary": "\n".join(
                                    [
                                        "## [P1] Paper A (2026)",
                                        "- **Full-text status**: FULL-TEXT",
                                        "- **Core Problem**: Handles nested heading output.",
                                        "- **Method**: Uses a cleaner during merge.",
                                    ]
                                ),
                            }
                        ]
                    }
                ),
                "summary": "Read Paper A.",
            }
        )

        response = merge_reading_batches.invoke({})

        assert "Merged 1 reading item(s)" in response
        deep_reading = (tmp_path / "paper_deep_reading.md").read_text(encoding="utf-8")
        assert deep_reading.count("## [P1]") == 1
        assert deep_reading.count("Full-text status**: FULL-TEXT") == 1
        assert "- **Core Problem**: Handles nested heading output." in deep_reading
    finally:
        set_active_workspace(old_workspace)


def test_merge_reading_batches_formats_structured_summary_fields(tmp_path: Path) -> None:
    old_workspace = get_active_workspace()
    try:
        set_active_workspace(tmp_path)
        _write_fulltext_audit(tmp_path)
        create_reading_batches.invoke(
            {
                "paper_ids_json": json.dumps(["P1"]),
                "batch_size": 1,
            }
        )
        record_batch_result.invoke(
            {
                "batch_id": "stage_3_5_reading_batch_001",
                "status": "passed",
                "result_json": json.dumps(
                    {
                        "readings": [
                            {
                                "paper_id": "P1",
                                "title": "Paper A",
                                "fulltext_status": "FULL-TEXT",
                                "summary": {
                                    "Core Problem": "Understands long videos.",
                                    "Method / Architecture": "Uses grounded temporal memory.",
                                    "Main Results": "Improves long-video QA.",
                                    "Limitations": "Depends on retrieval quality.",
                                    "Relevance to LVU Agent Idea Search": "Defines a baseline gap.",
                                },
                            }
                        ]
                    }
                ),
                "summary": "Read Paper A.",
            }
        )

        response = merge_reading_batches.invoke({})

        assert "Merged 1 reading item(s)" in response
        deep_reading = (tmp_path / "paper_deep_reading.md").read_text(encoding="utf-8")
        assert "- **Core Problem**: Understands long videos." in deep_reading
        assert "- **Method / Architecture**: Uses grounded temporal memory." in deep_reading
        assert "- **Main Results**: Improves long-video QA." in deep_reading
        assert "- **Limitations**: Depends on retrieval quality." in deep_reading
        assert "- **Relevance to LVU Agent Idea Search**: Defines a baseline gap." in deep_reading
        assert "{'Core Problem'" not in deep_reading
    finally:
        set_active_workspace(old_workspace)
