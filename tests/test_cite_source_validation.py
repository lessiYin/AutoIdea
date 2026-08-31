from __future__ import annotations

import pytest

from autoidea.tools.cite import _default_fetcher, validate_source_title


def test_default_fetcher_rejects_non_arxiv_urls() -> None:
    with pytest.raises(ValueError, match="arxiv.org"):
        _default_fetcher("file:///etc/passwd")


def test_validate_source_title_accepts_matching_arxiv_title() -> None:
    def fetcher(url: str) -> str:
        assert url == "https://arxiv.org/abs/2603.14468"
        return '<meta name="citation_title" content="LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos" />'

    result = validate_source_title(
        "LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos",
        "https://arxiv.org/abs/2603.14468",
        fetcher=fetcher,
    )

    assert result.ok
    assert result.actual_title.startswith("LongVidSearch")


def test_validate_source_title_rejects_mismatched_arxiv_title() -> None:
    def fetcher(url: str) -> str:
        assert url == "https://arxiv.org/abs/2604.16965"
        return '<meta name="citation_title" content="Different Perspectives of Memory System Simulation" />'

    result = validate_source_title(
        "LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos",
        "https://arxiv.org/abs/2604.16965",
        fetcher=fetcher,
    )

    assert not result.ok
    assert "Different Perspectives" in result.actual_title
