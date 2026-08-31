"""Source citation registration tool for evidence tracking."""
from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool

# Default path for the citations database — uses active workspace if available.
def _get_citations_path() -> Path:
    """Get the citations file path, preferring the active workspace."""
    try:
        from autoidea.paths import get_active_workspace
        ws = get_active_workspace()
        if ws:
            return Path(ws) / "citations.json"
    except (ImportError, Exception):
        pass
    # Fallback: project root
    return Path(__file__).resolve().parent.parent.parent / "citations.json"


def _get_legacy_citations_path() -> Path | None:
    """Return the old citations path for backward-compatible reads."""
    try:
        from autoidea.paths import get_active_workspace
        ws = get_active_workspace()
        if ws:
            return Path(ws) / "output" / "citations.json"
    except (ImportError, Exception):
        return None
    return None


# Valid evidence types
_EVIDENCE_TYPES = {
    "direct_quote",
    "paraphrase",
    "statistical_result",
    "method_description",
    "gap_identification",
    "limitation",
}

# Known academic URL patterns
_KNOWN_URL_PATTERNS = [
    re.compile(r"arxiv\.org"),
    re.compile(r"doi\.org"),
    re.compile(r"semanticscholar\.org"),
    re.compile(r"openalex\.org"),
    re.compile(r"dblp\.org"),
    re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov"),
    re.compile(r"api\.crossref\.org"),
]

_MOCK_URL_MARKER = "SEARCH_METADATA"  # marker in mock results


def _normalize_title_for_match(title: str) -> str:
    text = title.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _titles_match(expected: str, actual: str) -> bool:
    exp = _normalize_title_for_match(expected)
    act = _normalize_title_for_match(actual)
    if not exp or not act:
        return False
    if exp == act or exp in act or act in exp:
        return True
    exp_words = set(exp.split())
    act_words = set(act.split())
    return bool(exp_words and act_words) and len(exp_words & act_words) / max(len(exp_words), len(act_words)) >= 0.72


def _extract_arxiv_id(source_url: str) -> str | None:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", source_url or "", re.I)
    return match.group(1) if match else None


def _extract_arxiv_title(text: str) -> str:
    import html
    patterns = [
        r'<meta\s+name=["\']citation_title["\']\s+content=["\']([^"\']+)["\']',
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
        r"<title>\[\d{4}\.\d{4,5}\]\s*([^<]+)</title>",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.I)
        if match:
            return html.unescape(match.group(1)).strip()
    return ""


def _load_citations(path: Path) -> list[dict]:
    """Load existing citations from a JSON file, or return an empty list."""
    paths = [path]
    legacy = _get_legacy_citations_path()
    if legacy is not None and legacy != path:
        paths.append(legacy)
    for candidate in paths:
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


@dataclass
class SourceTitleValidation:
    ok: bool
    expected_title: str
    actual_title: str = ""
    reason: str = ""


def _default_fetcher(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "arxiv.org":
        raise ValueError("Citation verification only fetches HTTPS URLs from arxiv.org.")
    req = Request(url, headers={"User-Agent": "AutoIdeaCitationVerifier/1.0"})
    # The URL scheme and exact host are restricted above.
    with urlopen(req, timeout=15.0) as response:  # nosec B310
        return response.read(1_000_000).decode("utf-8", errors="replace")


def validate_source_title(
    source_title: str,
    source_url: str,
    *,
    fetcher: Callable[[str], str] | None = None,
) -> SourceTitleValidation:
    """Validate that an arXiv URL resolves to the supplied source title.

    Non-arXiv URLs are accepted because there is no common metadata format
    across all supported sources.  arXiv URLs are common enough and caused a
    real failure mode, so they are checked directly.
    """
    arxiv_id = _extract_arxiv_id(source_url)
    if not arxiv_id:
        return SourceTitleValidation(ok=True, expected_title=source_title)
    try:
        text = (fetcher or _default_fetcher)(f"https://arxiv.org/abs/{arxiv_id}")
        actual = _extract_arxiv_title(text)
    except Exception as exc:
        return SourceTitleValidation(
            ok=False,
            expected_title=source_title,
            reason=f"could not fetch arXiv metadata: {exc}",
        )
    if not actual:
        return SourceTitleValidation(
            ok=False,
            expected_title=source_title,
            reason="could not extract arXiv title metadata",
        )
    if not _titles_match(source_title, actual):
        return SourceTitleValidation(
            ok=False,
            expected_title=source_title,
            actual_title=actual,
            reason="source_url title does not match source_title",
        )
    return SourceTitleValidation(ok=True, expected_title=source_title, actual_title=actual)


def _save_citations(path: Path, citations: list[dict]) -> None:
    """Write citations list to JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(citations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@tool
def cite_source(
    claim: str,
    source_title: str,
    source_url: str,
    paper_id: str = "",
    evidence_type: Literal[
        "direct_quote",
        "paraphrase",
        "statistical_result",
        "method_description",
        "gap_identification",
        "limitation",
    ] = "paraphrase",
    confidence: float = 0.7,
    section: str = "",
) -> str:
    """Register a citation linking a claim to its source evidence.

    This tool creates a verifiable chain of evidence. Every factual claim
    about external work should be cited using this tool.

    Args:
        claim: The factual claim being made.
        source_title: Title of the source paper or resource.
        source_url: URL of the source (arXiv, Semantic Scholar, etc.).
        paper_id: Internal reference ID for this paper (e.g., "P3").
        evidence_type: Type of evidence.
            - "direct_quote": Exact quote from the paper
            - "paraphrase": Rephrased content from the paper
            - "statistical_result": Reported number/metric from the paper
            - "method_description": Description of a method/approach
            - "gap_identification": A gap or limitation identified in the paper
            - "limitation": A stated limitation of the work
        confidence: Confidence score 0.0-1.0.
            - 0.8-1.0: Full text accessed, claim verified
            - 0.5-0.8: Abstract accessed, partial verification
            - 0.3-0.5: Search snippet only, low confidence
        section: Specific section of the paper (e.g., "Section 3.2").
    Returns:
        Confirmation with citation ID (e.g., [C1], [C2]).
    """
    # Validate evidence type
    if evidence_type not in _EVIDENCE_TYPES:
        return (
            f"Invalid evidence_type '{evidence_type}'. "
            f"Must be one of: {', '.join(sorted(_EVIDENCE_TYPES))}"
        )

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    # --- Lightweight verification ---
    is_mock = _MOCK_URL_MARKER in source_url or not any(
        p.search(source_url) for p in _KNOWN_URL_PATTERNS
    )
    if is_mock:
        confidence = min(confidence, 0.3)
    else:
        title_check = validate_source_title(source_title, source_url)
        if not title_check.ok:
            return (
                "Citation rejected: source URL/title mismatch.\n"
                f"- **Provided title**: {source_title}\n"
                f"- **Source URL**: {source_url}\n"
                f"- **Resolved title**: {title_check.actual_title or 'N/A'}\n"
                f"- **Reason**: {title_check.reason}\n"
                "Use the correct source URL before registering this claim."
            )
        if confidence >= 0.8 and not section:
            confidence = 0.6  # HIGH confidence requires section reference

    # Duplicate detection
    citations_path = _get_citations_path()
    citations = _load_citations(citations_path)
    for existing in citations:
        if (existing.get("claim") == claim
                and existing.get("source_title") == source_title):
            return (
                f"Duplicate claim detected. Reusing existing citation "
                f"[{existing['citation_id']}] with confidence "
                f"{existing['confidence']:.1%}."
            )

    # Assign next citation ID
    citation_num = len(citations) + 1
    citation_id = f"C{citation_num}"

    # Build citation record
    entry = {
        "citation_id": citation_id,
        "claim": claim,
        "source_title": source_title,
        "source_url": source_url,
        "paper_id": paper_id,
        "evidence_type": evidence_type,
        "confidence": round(confidence, 2),
        "section": section,
        "is_mock": is_mock,
        "verified": False,
    }

    citations.append(entry)

    # Persist
    _save_citations(citations_path, citations)

    # Build confidence label
    if confidence >= 0.8:
        conf_label = "HIGH (full text verified)"
    elif confidence >= 0.5:
        conf_label = "MEDIUM (abstract verified)"
    else:
        conf_label = "LOW (search snippet only)"

    return (
        f"Citation [{citation_id}] registered successfully.\n"
        f"- **Source**: {source_title}\n"
        f"- **Evidence type**: {evidence_type}\n"
        f"- **Confidence**: {confidence:.1%} -- {conf_label}\n"
        f"- **Claim**: {claim}"
    )
