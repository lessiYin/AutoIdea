from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from autoidea.tools.artifact_audit import (
    _extract_survey_papers,
    _fetch_arxiv_titles,
    audit_workspace,
    extract_arxiv_id,
    extract_title_from_arxiv_html,
    fetch_url_text,
)


def test_extract_survey_papers_accepts_bold_paper_id_before_plain_title() -> None:
    survey = """
- **[P1]** Paper One
2. **[P2]** Paper Two
"""

    assert _extract_survey_papers(survey) == {"P1": "Paper One", "P2": "Paper Two"}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data) -> None:
    _write(path, json.dumps(data, indent=2))


def _valid_gap_catalog(*, citation_ids: tuple[str, str] = ("C1", "C2")) -> dict:
    return {
        "schema_version": "1.0",
        "generated_from": "evidence_db.json",
        "gaps": [
            {
                "gap_id": f"G{index}",
                "title": f"Research gap {index}",
                "description": f"A precise evidence-grounded gap statement {index}.",
                "gap_type": "methodology_gap",
                "demand": 5,
                "coverage": 2,
                "gap_score": 3,
                "evidence_links": [
                    {
                        "citation_id": citation_ids[0],
                        "relationship": "supports",
                        "rationale": "This Claim establishes the unresolved problem boundary.",
                    },
                    {
                        "citation_id": citation_ids[1],
                        "relationship": "partial_coverage",
                        "rationale": "This Claim documents a partial solution and its remaining scope.",
                    },
                ],
                "why_it_matters": "The gap blocks a reliable research outcome.",
                "potential_direction": "Evaluate a bounded mechanism against the documented gap.",
                "supporting_papers": ["P1", "P2"],
            }
            for index in range(1, 4)
        ],
    }


def _write_valid_gap_workspace(tmp_path: Path) -> None:
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
        (
            "# Knowledge Synthesis\n\n"
            "The evidence identifies G1, G2, and G3 as bounded research gaps. "
            "Each gap is linked to canonical Claims rather than inferred from "
            "paper mentions. The structured registry records relationship roles "
            "and rationales so downstream ideas remain independently auditable."
        ),
    )
    _write_json(tmp_path / "research_gaps.json", _valid_gap_catalog())


def test_audit_requires_structured_gap_registry_after_stage7(tmp_path: Path) -> None:
    _write(
        tmp_path / "knowledge_synthesis.md",
        "# Knowledge Synthesis\n\n" + "Narrative-only gap analysis. " * 12,
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    assert "RESEARCH_GAPS_MISSING" in {issue.code for issue in report.issues}


def test_audit_rejects_empty_stage3_registry(tmp_path: Path) -> None:
    _write_json(tmp_path / "paper_registry.json", [])
    _write(tmp_path / "literature_survey.md", "# Literature Survey\n\nNo papers merged.")

    report = audit_workspace(tmp_path, verify_urls=False)

    assert "PAPER_REGISTRY_EMPTY" in {issue.code for issue in report.issues}


def test_audit_rejects_gap_link_to_unknown_claim(tmp_path: Path) -> None:
    _write_valid_gap_workspace(tmp_path)
    catalog = _valid_gap_catalog(citation_ids=("C1", "C404"))
    catalog["gaps"][0]["supporting_papers"] = ["P1"]
    catalog["gaps"][1]["supporting_papers"] = ["P1"]
    catalog["gaps"][2]["supporting_papers"] = ["P1"]
    _write_json(tmp_path / "research_gaps.json", catalog)

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    assert "GAP_EVIDENCE_UNKNOWN_CITATION" in {
        issue.code for issue in report.issues
    }


def test_audit_accepts_structured_claim_to_gap_provenance(tmp_path: Path) -> None:
    _write_valid_gap_workspace(tmp_path)

    report = audit_workspace(tmp_path, verify_urls=False)

    gap_issues = [
        issue
        for issue in report.issues
        if "GAP" in issue.code or "RESEARCH_GAPS" in issue.code
    ]
    assert gap_issues == []


def test_audit_checks_legacy_idea_gap_field_against_canonical_registry(
    tmp_path: Path,
) -> None:
    _write_valid_gap_workspace(tmp_path)
    _write_json(
        tmp_path / "raw_ideas.json",
        {
            "ideas": [
                {
                    "idea_id": "IDEA-001",
                    "title": "Legacy-format idea",
                    "gap_addressed": "G404",
                }
            ]
        },
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert "DOWNSTREAM_GAP_ID_DRIFT" in {
        issue.code for issue in report.issues
    }


def test_artifact_fetch_rejects_non_arxiv_urls() -> None:
    with pytest.raises(ValueError, match="arxiv.org"):
        fetch_url_text("file:///etc/passwd")


def test_artifact_fetch_retries_transient_timeout(monkeypatch) -> None:
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return b"<title>[2403.11481] Example</title>"

    def flaky_urlopen(_request, *, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("temporary timeout")
        return Response()

    monkeypatch.setattr("autoidea.tools.artifact_audit.urlopen", flaky_urlopen)
    monkeypatch.setattr("autoidea.tools.artifact_audit.time.sleep", lambda _delay: None)

    text = fetch_url_text(
        "https://arxiv.org/abs/2403.11481",
        timeout=0.1,
        attempts=2,
        retry_delay=0,
    )

    assert "Example" in text
    assert calls == [0.1, 0.1]


def test_artifact_fetch_does_not_retry_non_transient_http_error(monkeypatch) -> None:
    from urllib.error import HTTPError

    calls = []

    def missing_urlopen(request, *, timeout):
        calls.append(timeout)
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("autoidea.tools.artifact_audit.urlopen", missing_urlopen)
    monkeypatch.setattr("autoidea.tools.artifact_audit.time.sleep", lambda _delay: None)

    with pytest.raises(HTTPError):
        fetch_url_text(
            "https://arxiv.org/abs/2403.11481",
            timeout=0.1,
            attempts=3,
            retry_delay=0,
        )

    assert calls == [0.1]


def test_audit_deduplicates_and_parallelizes_arxiv_title_checks(tmp_path: Path) -> None:
    first_url = "https://arxiv.org/abs/2401.00001"
    second_url = "https://arxiv.org/abs/2401.00002"
    _write_json(
        tmp_path / "paper_registry.json",
        [
            {"paper_id": "P1", "title": "Paper One", "url": first_url},
            {"paper_id": "P2", "title": "Paper Two", "url": second_url},
        ],
    )
    _write_json(
        tmp_path / "citations.json",
        [
            {"citation_id": "C1", "paper_id": "P1"},
            {"citation_id": "C2", "paper_id": "P1"},
            {"citation_id": "C3", "paper_id": "P2"},
        ],
    )
    _write_json(
        tmp_path / "evidence_db.json",
        {
            "summary": {"citations_count": 3},
            "claims": [
                {
                    "citation_id": "C1",
                    "source_paper_id": "P1",
                    "source_title": "Paper One",
                    "source_url": first_url,
                },
                {
                    "citation_id": "C2",
                    "source_paper_id": "P1",
                    "source_title": "Paper One",
                    "source_url": first_url,
                },
                {
                    "citation_id": "C3",
                    "source_paper_id": "P2",
                    "source_title": "Paper Two",
                    "source_url": second_url,
                },
            ],
        },
    )

    barrier = threading.Barrier(2)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def fetcher(url: str) -> str:
        with calls_lock:
            calls.append(url)
        barrier.wait(timeout=2)
        title = "Paper One" if url.endswith("2401.00001") else "Paper Two"
        return f'<meta name="citation_title" content="{title}" />'

    report = audit_workspace(tmp_path, verify_urls=True, fetcher=fetcher)

    arxiv_issues = [issue for issue in report.issues if issue.code.startswith("ARXIV_")]
    assert arxiv_issues == []
    assert sorted(calls) == sorted([first_url, second_url])


def test_audit_treats_unavailable_arxiv_as_warning(tmp_path: Path) -> None:
    source_url = "https://arxiv.org/abs/2401.00001"
    _write_json(
        tmp_path / "paper_registry.json",
        [{"paper_id": "P1", "title": "Paper One", "url": source_url}],
    )
    _write(
        tmp_path / "literature_survey.md",
        "# Literature Survey\n\n| ID | Paper |\n|---|---|\n"
        "| [P1] | **Paper One** |\n\n"
        "Paper One provides the evidence anchor used by this audit fixture.",
    )
    _write_json(
        tmp_path / "citations.json",
        [{"citation_id": "C1", "paper_id": "P1"}],
    )
    _write_json(
        tmp_path / "evidence_db.json",
        {
            "summary": {"citations_count": 1},
            "claims": [
                {
                    "citation_id": "C1",
                    "source_paper_id": "P1",
                    "source_title": "Paper One",
                    "source_url": source_url,
                }
            ],
        },
    )

    def unavailable(_url: str) -> str:
        raise TimeoutError("arXiv is temporarily unreachable")

    report = audit_workspace(tmp_path, verify_urls=True, fetcher=unavailable)

    remote_issues = [
        issue
        for issue in report.issues
        if issue.code == "ARXIV_TITLE_VERIFICATION_UNAVAILABLE"
    ]
    assert not report.has_errors, [
        (issue.code, issue.message) for issue in report.issues
    ]
    assert len(remote_issues) == 1
    assert remote_issues[0].severity.value == "WARNING"


def test_audit_keeps_verified_arxiv_title_mismatch_as_error(tmp_path: Path) -> None:
    source_url = "https://arxiv.org/abs/2401.00001"
    _write_json(
        tmp_path / "paper_registry.json",
        [{"paper_id": "P1", "title": "Expected Paper", "url": source_url}],
    )
    _write_json(
        tmp_path / "citations.json",
        [{"citation_id": "C1", "paper_id": "P1"}],
    )
    _write_json(
        tmp_path / "evidence_db.json",
        {
            "summary": {"citations_count": 1},
            "claims": [
                {
                    "citation_id": "C1",
                    "source_paper_id": "P1",
                    "source_title": "Expected Paper",
                    "source_url": source_url,
                }
            ],
        },
    )

    report = audit_workspace(
        tmp_path,
        verify_urls=True,
        fetcher=lambda _url: (
            '<meta name="citation_title" content="Different Paper" />'
        ),
    )

    assert report.has_errors
    assert "ARXIV_TITLE_MISMATCH" in {issue.code for issue in report.issues}


def test_arxiv_title_checks_obey_overall_deadline(monkeypatch) -> None:
    release = threading.Event()

    def slow_fetcher(_url: str) -> str:
        release.wait(timeout=1)
        return '<meta name="citation_title" content="Paper" />'

    monkeypatch.setattr(
        "autoidea.tools.artifact_audit._ARXIV_VERIFY_TOTAL_TIMEOUT_SECONDS",
        0.01,
    )
    started = time.monotonic()
    try:
        results = _fetch_arxiv_titles(
            ["https://arxiv.org/abs/2401.00001"],
            fetcher=slow_fetcher,
        )
    finally:
        release.set()

    assert time.monotonic() - started < 0.5
    assert results["2401.00001"][0] is None
    assert "overall deadline" in str(results["2401.00001"][1])


@pytest.mark.parametrize(
    "ending",
    [
        "| [P10] | Memory-enhanced RAG | 202",
        "```python\nprint('unfinished')",
        "Stage ",
        "diminishing returns. Stage ",
        "## Limitations",
    ],
)
def test_audit_rejects_truncated_final_report(
    tmp_path: Path,
    ending: str,
) -> None:
    _write(
        tmp_path / "final_report.md",
        "# Final Report\n\n" + ("Evidence-grounded research synthesis. " * 12) + "\n\n" + ending,
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert "FINAL_REPORT_TRUNCATED" in {issue.code for issue in report.issues}


def test_audit_accepts_complete_final_report_tail(tmp_path: Path) -> None:
    _write(
        tmp_path / "final_report.md",
        (
            "# Final Report\n\n"
            + ("Evidence-grounded research synthesis. " * 12)
            + "\n\n## Limitations\n\n"
            "The proposed experiments remain to be executed and independently verified."
        ),
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert "FINAL_REPORT_TRUNCATED" not in {issue.code for issue in report.issues}


def test_audit_rejects_duplicate_numbered_final_report_sections(tmp_path: Path) -> None:
    _write(
        tmp_path / "final_report.md",
        (
            "# Final Report\n\n"
            "## 1. Summary\n\n"
            + ("Evidence-grounded research synthesis. " * 12)
            + "\n\n## 1. Conclusion\n\n"
            "The proposal remains to be evaluated."
        ),
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert "FINAL_REPORT_DUPLICATE_SECTION" in {
        issue.code for issue in report.issues
    }


def test_audit_rejects_regressions_seen_in_workspace0(tmp_path: Path) -> None:
    _write(
        tmp_path / "literature_survey.md",
        """
# Literature Survey
| ID | Paper | Why |
|---|---|---|
| [P1] | **TimeChat** (2024) | anchor |
| [P16] | **LongVU** (2024) | compression |
""".strip(),
    )
    _write_json(
        tmp_path / "paper_positions.json",
        [
            {"paper_id": "P1", "title": "LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding"},
            {"paper_id": "P16", "title": "STORM: Token-Efficient Long Video Understanding for Multimodal LLMs"},
        ],
    )
    _write_json(
        tmp_path / "evidence_db.json",
        {
            "metadata": {"citation_id_policy": "Local deterministic citation IDs C1-C32 for the rerun pipeline."},
            "summary": {"citations_count": 46},
            "claims": [
                {
                    "citation_id": "C20",
                    "claim": "LongVidSearch is a benchmark.",
                    "source_paper_id": "P89",
                    "source_title": "LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos",
                    "source_url": "https://arxiv.org/abs/2604.16965",
                }
            ],
        },
    )
    _write(
        tmp_path / "paper_deep_reading.md",
        """
# Paper Deep Reading Summary
- **Total papers selected**: 1
- **Full-text extracted**: 1
- **Abstract-only fallback**: 0

## [P1] TimeChat (2024)
- **Full-text status**: FULL-TEXT
""".strip(),
    )
    _write_json(
        tmp_path / "reflections" / "stage_6_reflection.json",
        {
            "stage": "stage_6",
            "artifacts": {"artifact": "evidence_db.json", "citations_count": 46},
        },
    )
    _write_json(tmp_path / "workspace" / "evidence_db.json", {"claims": [{"citation_id": "C1"}, {"citation_id": "C2"}]})

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    messages = "\n".join(issue.message for issue in report.issues)
    assert "paper_positions.json maps P1" in messages
    assert "citation count" in messages
    assert "Local deterministic citation IDs" in messages
    assert "FULL-TEXT" in messages and "fulltext_audit.json" in messages
    assert "nested workspace artifact differs" in messages
    assert "paper_registry.json is missing" in messages


def test_audit_rejects_stage3_registry_survey_mismatch_and_malformed_batches(tmp_path: Path) -> None:
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
    _write_json(
        tmp_path / "batch_manifest.json",
        {
            "batches": [
                {
                    "batch_id": "stage_3_search_batch_001",
                    "stage": "stage_3_search",
                    "status": "passed",
                    "result_file": "batches/stage_3_search/batch_001/result.json",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "batches" / "stage_3_search" / "batch_001" / "result.json",
        {"notable_titles": ["VideoAgent"]},
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "LITERATURE_SURVEY_EMPTY" in codes
    assert "REGISTRY_SURVEY_COUNT_MISMATCH" in codes
    assert "BATCH_RESULT_SCHEMA" in codes


def test_audit_rejects_stage3_artifacts_with_pending_search_batches(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "paper_registry.json",
        [
            {"paper_id": "P1", "title": "Paper A", "url": "https://arxiv.org/abs/2501.00001"},
        ],
    )
    _write(
        tmp_path / "literature_survey.md",
        """
# Literature Survey
| ID | Paper | Year | Source | Relevance |
|---|---|---:|---|---|
| [P1] | **Paper A** | 2025 | arxiv | anchor |
""".strip(),
    )
    _write_json(
        tmp_path / "batch_manifest.json",
        {
            "batches": [
                {
                    "batch_id": "stage_3_search_batch_001",
                    "stage": "stage_3_search",
                    "status": "passed",
                    "result_file": "batches/stage_3_search/batch_001/result.json",
                },
                {
                    "batch_id": "stage_3_search_batch_002",
                    "stage": "stage_3_search",
                    "status": "pending",
                    "result_file": "",
                },
            ]
        },
    )
    _write_json(
        tmp_path / "batches" / "stage_3_search" / "batch_001" / "result.json",
        {
            "papers": [
                {
                    "title": "Paper A",
                    "url": "https://arxiv.org/abs/2501.00001",
                    "source": "arxiv",
                }
            ]
        },
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "BATCH_INCOMPLETE" in codes


def test_audit_rejects_placeholder_expanded_literature(tmp_path: Path) -> None:
    _write(tmp_path / "expanded_literature.md", "test")

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "STAGE_ARTIFACT_PLACEHOLDER" in codes


def test_audit_rejects_passed_reading_batches_without_fulltext_audit(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "batch_manifest.json",
        {
            "batches": [
                {
                    "batch_id": "stage_3_5_reading_batch_001",
                    "stage": "stage_3_5_reading",
                    "status": "passed",
                    "result_file": "batches/stage_3_5_reading/batch_001/result.json",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "result.json",
        {
            "readings": [
                {
                    "paper_id": "P1",
                    "title": "Paper A",
                    "fulltext_status": "ABSTRACT-ONLY",
                    "summary": "ABSTRACT-ONLY conservative summary based on registry metadata.",
                }
            ]
        },
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "BATCH_READING_AUDIT_MISSING" in codes


def test_audit_accepts_arxiv_traceability_note_for_reading_batch(tmp_path: Path) -> None:
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
    _write_json(
        tmp_path / "batch_manifest.json",
        {
            "batches": [
                {
                    "batch_id": "stage_3_5_reading_batch_001",
                    "stage": "stage_3_5_reading",
                    "status": "passed",
                    "result_file": "batches/stage_3_5_reading/batch_001/result.json",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "result.json",
        {
            "readings": [
                {
                    "paper_id": "P2",
                    "title": "VideoAgent: Long-form Video Understanding with Large Language Model as Agent",
                    "fulltext_status": "FULL-TEXT",
                    "summary": "Summary grounded in fetched full text.",
                    "audit_traceability_note": "Fetched via https://arxiv.org/abs/2403.10517 before recording.",
                }
            ]
        },
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    codes = {issue.code for issue in report.issues}
    assert "BATCH_READING_AUDIT_MISSING" not in codes


def test_audit_matches_reading_fulltext_via_registry_url(tmp_path: Path) -> None:
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
    _write_json(
        tmp_path / "batch_manifest.json",
        {
            "batches": [
                {
                    "batch_id": "stage_3_5_reading_batch_001",
                    "stage": "stage_3_5_reading",
                    "status": "passed",
                    "result_file": "batches/stage_3_5_reading/batch_001/result.json",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "batches" / "stage_3_5_reading" / "batch_001" / "result.json",
        {
            "readings": [
                {
                    "paper_id": "P1",
                    "title": "MovieChat+: Question-aware Sparse Memory for Long Video Question Answering",
                    "fulltext_status": "FULL-TEXT",
                    "summary": "Summary grounded in fetched full text.",
                }
            ]
        },
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    codes = {issue.code for issue in report.issues}
    assert "BATCH_READING_AUDIT_MISSING" not in codes


def test_audit_rejects_deep_reading_below_configured_top_k(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOIDEA_DEEP_READING_TOP_K", "35")
    _write(
        tmp_path / "paper_deep_reading.md",
        """
# Paper Deep Reading Summary
- **Total papers selected**: 10
- **Full-text extracted**: 10
- **Abstract-only fallback**: 0

## [P1] Paper 1
- **Full-text status**: FULL-TEXT
""".strip(),
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "DEEP_READING_INCOMPLETE" in codes


def test_audit_uses_explicit_run_top_k_instead_of_current_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTOIDEA_DEEP_READING_TOP_K", "35")
    _write(
        tmp_path / "paper_deep_reading.md",
        """
# Paper Deep Reading Summary
- **Total papers selected**: 1
- **Full-text extracted**: 0
- **Abstract-only fallback**: 1

## [P1] Paper 1
- **Full-text status**: ABSTRACT-ONLY
""".strip(),
    )

    report = audit_workspace(
        tmp_path,
        verify_urls=False,
        deep_reading_top_k=1,
    )

    codes = {issue.code for issue in report.issues}
    assert "DEEP_READING_INCOMPLETE" not in codes


def test_audit_accepts_consistent_workspace(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "paper_registry.json",
        [
            {"paper_id": "P1", "title": "TimeChat", "url": "https://arxiv.org/abs/2312.02051"},
            {"paper_id": "P2", "title": "LongVU", "url": "https://arxiv.org/abs/2410.17434"},
        ],
    )
    _write(
        tmp_path / "literature_survey.md",
        """
# Literature Survey
| ID | Paper | Why |
|---|---|---|
| [P1] | **TimeChat** (2024) | anchor |
| [P2] | **LongVU** (2024) | compression |
""".strip(),
    )
    _write_json(
        tmp_path / "paper_positions.json",
        [
            {"paper_id": "P1", "title": "TimeChat"},
            {"paper_id": "P2", "title": "LongVU"},
        ],
    )
    _write_json(
        tmp_path / "citations.json",
        [
            {"citation_id": "C1", "paper_id": "P1"},
            {"citation_id": "C2", "paper_id": "P2"},
        ],
    )
    _write_json(
        tmp_path / "evidence_db.json",
        {
            "metadata": {"citation_id_policy": "cite_source canonical registry"},
            "summary": {"citations_count": 2},
            "claims": [
                {"citation_id": "C1", "source_paper_id": "P1", "source_title": "TimeChat", "source_url": "https://arxiv.org/abs/2312.02051"},
                {"citation_id": "C2", "source_paper_id": "P2", "source_title": "LongVU", "source_url": "https://arxiv.org/abs/2410.17434"},
            ],
        },
    )
    deep_lines = [
        "# Paper Deep Reading Summary",
        "- **Total papers selected**: 35",
        "- **Full-text extracted**: 35",
        "- **Abstract-only fallback**: 0",
        "",
    ]
    records = []
    for idx in range(1, 36):
        title = "TimeChat" if idx == 1 else f"Aux Paper {idx}"
        deep_lines.extend(
            [
                f"## [P{idx}] {title} (2024)",
                "- **Full-text status**: FULL-TEXT",
                "",
            ]
        )
        records.append(
            {
                "identifier": title,
                "status": "full_text",
                "text_path": f"paper_texts/paper_{idx}.txt",
            }
        )
        _write(tmp_path / "paper_texts" / f"paper_{idx}.txt", "full text")
    _write(tmp_path / "paper_deep_reading.md", "\n".join(deep_lines).strip())
    _write_json(
        tmp_path / "fulltext_audit.json",
        {"records": records},
    )
    _write_json(
        tmp_path / "reflections" / "stage_6_reflection.json",
        {"stage": "stage_6", "artifacts": {"artifact": "evidence_db.json", "citations_count": 2}},
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert not report.has_errors, "\n".join(issue.message for issue in report.issues)


def test_audit_rejects_noncanonical_paper_position_ids(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "paper_registry.json",
        [
            {"paper_id": "P1", "title": "TimeChat", "url": "https://arxiv.org/abs/2312.02051"},
        ],
    )
    _write(
        tmp_path / "literature_survey.md",
        """
# Literature Survey
| ID | Paper | Why |
|---|---|---|
| [P1] | **TimeChat** (2024) | anchor |
""".strip(),
    )
    _write_json(
        tmp_path / "paper_positions.json",
        [
            {"paper_id": "[P1]", "title": "TimeChat"},
        ],
    )

    report = audit_workspace(tmp_path, verify_urls=False)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "PAPER_POSITIONS_SCHEMA" in codes


def test_arxiv_helpers_extract_id_and_title() -> None:
    html = '<meta name="citation_title" content="LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos" />'

    assert extract_arxiv_id("https://arxiv.org/abs/2603.14468v1") == "2603.14468"
    assert extract_arxiv_id("https://arxiv.org/pdf/2603.22285") == "2603.22285"
    assert extract_title_from_arxiv_html(html) == "LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos"
