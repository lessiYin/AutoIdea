"""Full-text paper content retrieval for AutoIdea.

Fetches paper content from arXiv, Semantic Scholar, and generic URLs.
Provides tools for retrieving full paper text and specific sections.
Enhanced with multi-level fallback: ar5iv HTML → arXiv abstract → S2 TLDR.
"""

from __future__ import annotations

import re
from typing import Optional

from defusedxml import ElementTree as ET
from langchain_core.tools import tool

# Warning prefix for partial content (abstract/TLDR only)
_PARTIAL_CONTENT_WARNING = (
    "⚠️ **Note**: Only abstract/TLDR available. "
    "Full text could not be retrieved.\n\n"
)


async def _fetch_arxiv_html(arxiv_id: str) -> Optional[str]:
    """Try to fetch arXiv HTML5 version of a paper."""
    from autoidea.tools._proxy import get_async_client

    # Try ar5iv (HTML5 version)
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        async with get_async_client(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                from markdownify import markdownify

                content = markdownify(resp.text)
                if len(content) > 30000:
                    # Use section-aware extraction instead of naive truncation
                    from autoidea.tools.scholar import _prioritized_extract

                    return _prioritized_extract(content, 30000)
                return content
    except Exception:
        pass
    return None


async def _fetch_arxiv_abstract(arxiv_id: str) -> Optional[str]:
    """Fetch the abstract of an arXiv paper via the arXiv Atom API.

    This serves as a middle-ground fallback between full HTML and
    the very short S2 TLDR.
    """
    from autoidea.tools._proxy import get_async_client
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        async with get_async_client(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entry = root.find("atom:entry", ns)
                if entry is not None:
                    # Extract title
                    title_el = entry.find("atom:title", ns)
                    title = title_el.text.strip() if title_el is not None else ""
                    # Extract abstract (summary)
                    summary_el = entry.find("atom:summary", ns)
                    abstract = summary_el.text.strip() if summary_el is not None else ""
                    # Extract authors
                    authors = []
                    for author_el in entry.findall("atom:author", ns):
                        name_el = author_el.find("atom:name", ns)
                        if name_el is not None:
                            authors.append(name_el.text.strip())
                    if abstract:
                        parts = []
                        if title:
                            parts.append(f"**{title}**")
                        if authors:
                            parts.append(f"*{', '.join(authors[:5])}*")
                        parts.append(f"\n{abstract}")
                        return "\n".join(parts)
    except Exception:
        pass
    return None


async def _fetch_s2_tldr(paper_id: str) -> Optional[str]:
    """Fetch Semantic Scholar TLDR for a paper."""
    from autoidea.tools._proxy import get_async_client

    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {"fields": "tldr"}
    try:
        async with get_async_client(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                tldr = data.get("tldr")
                if tldr and isinstance(tldr, dict):
                    return tldr.get("text", "")
    except Exception:
        pass
    return None


async def _fetch_url_content(url: str) -> Optional[str]:
    """Fetch and convert webpage content to markdown."""
    from autoidea.tools._proxy import get_async_client

    try:
        async with get_async_client(timeout=20.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            from markdownify import markdownify
            content = markdownify(resp.text)
            if len(content) > 30000:
                from autoidea.tools.scholar import _prioritized_extract

                return _prioritized_extract(content, 30000)
            return content
    except Exception:
        return None


@tool(parse_docstring=True)
async def fetch_paper_content(
    paper_id: str,
    source: str = "auto",
) -> str:
    """Fetch the full text content of a research paper.

    Tries multiple methods to retrieve paper content with graceful
    degradation:
    1. arXiv HTML5 (via ar5iv) if arXiv ID detected — full text
    2. arXiv abstract (via arXiv Atom API) — title + authors + abstract
    3. Semantic Scholar TLDR as final fallback — one-sentence summary
    4. Direct URL fetch if a URL is provided

    Args:
        paper_id: Paper identifier - can be arXiv ID (e.g. "2301.12345"),
            Semantic Scholar ID, DOI, or a direct URL.
        source: Source hint - "arxiv", "s2", "url", or "auto" for automatic detection.

    Returns:
        Paper content in markdown format, or error message listing all
        attempted sources.
    """
    # Detect source from paper_id format
    if source == "auto":
        if re.match(r"\d{4}\.\d{4,5}", paper_id):
            source = "arxiv"
        elif paper_id.startswith("http"):
            source = "url"
        elif re.match(r"10\.\d{4,}/", paper_id):
            source = "doi"
        else:
            source = "s2"

    # Try arXiv fallback chain: ar5iv HTML → arXiv abstract → S2 TLDR
    if source == "arxiv":
        arxiv_id = paper_id.strip()
        tried_sources = []

        # 1) ar5iv full HTML
        tried_sources.append("ar5iv HTML5")
        content = await _fetch_arxiv_html(arxiv_id)
        if content:
            return (
                f"## Paper Content (arXiv: {arxiv_id})\n\n"
                f"*Source: ar5iv HTML5 rendering*\n\n"
                f"{content}"
            )

        # 2) arXiv abstract via Atom API
        tried_sources.append("arXiv abstract API")
        abstract = await _fetch_arxiv_abstract(arxiv_id)
        if abstract:
            return (
                f"{_PARTIAL_CONTENT_WARNING}"
                f"## Paper Abstract (arXiv: {arxiv_id})\n\n"
                f"*Source: arXiv Atom API (abstract only)*\n\n"
                f"{abstract}"
            )

        # 3) S2 TLDR as last resort
        tried_sources.append("Semantic Scholar TLDR")
        tldr = await _fetch_s2_tldr(f"arXiv:{arxiv_id}")
        if tldr:
            return (
                f"{_PARTIAL_CONTENT_WARNING}"
                f"## Paper Summary (arXiv: {arxiv_id})\n\n"
                f"*Source: Semantic Scholar TLDR (one-sentence summary)*\n\n"
                f"{tldr}"
            )

        tried_sources_str = ", ".join(tried_sources)
        return (
            f"Could not fetch content for arXiv paper {arxiv_id}. "
            f"Attempted sources: [{tried_sources_str}]. "
            f"All sources failed or returned empty results."
        )

    elif source == "url":
        content = await _fetch_url_content(paper_id)
        if content:
            return f"## Paper Content\n\n*Source: {paper_id}*\n\n{content}"
        return (
            f"Could not fetch content from URL: {paper_id}. "
            f"Attempted sources: [direct URL fetch]. "
            f"The URL may be inaccessible or require authentication."
        )

    elif source == "doi":
        # Try DOI resolution
        tried_sources = ["DOI resolution via doi.org"]
        doi_url = f"https://doi.org/{paper_id}"
        content = await _fetch_url_content(doi_url)
        if content:
            return f"## Paper Content (DOI: {paper_id})\n\n{content}"

        # Also try S2 with DOI
        tried_sources.append("Semantic Scholar TLDR")
        tldr = await _fetch_s2_tldr(f"DOI:{paper_id}")
        if tldr:
            return (
                f"{_PARTIAL_CONTENT_WARNING}"
                f"## Paper Summary (DOI: {paper_id})\n\n"
                f"*Source: Semantic Scholar TLDR*\n\n"
                f"{tldr}"
            )

        tried_sources_str = ", ".join(tried_sources)
        return (
            f"Could not resolve DOI: {paper_id}. "
            f"Attempted sources: [{tried_sources_str}]."
        )

    else:
        # Semantic Scholar
        tried_sources = ["Semantic Scholar TLDR"]
        tldr = await _fetch_s2_tldr(paper_id)
        if tldr:
            return (
                f"{_PARTIAL_CONTENT_WARNING}"
                f"## Paper Summary (S2: {paper_id})\n\n"
                f"*S2 TLDR summary*\n\n"
                f"{tldr}"
            )
        tried_sources_str = ", ".join(tried_sources)
        return (
            f"Could not fetch content for paper {paper_id}. "
            f"Attempted sources: [{tried_sources_str}]."
        )


@tool(parse_docstring=True)
async def fetch_paper_section(
    paper_id: str,
    section_name: str,
) -> str:
    """Fetch a specific section from a research paper.

    Attempts to retrieve the full paper and extract the requested section.
    Useful for detailed analysis of methodology, experiments, or results.

    Args:
        paper_id: Paper identifier (arXiv ID, S2 ID, DOI, or URL).
        section_name: Name of the section to extract (e.g. "methodology",
            "experiments", "results", "introduction", "conclusion").

    Returns:
        Extracted section content, or error message.
    """
    # First, fetch full content
    full_content = await fetch_paper_content.ainvoke(
        {"paper_id": paper_id, "source": "auto"}
    )

    if full_content.startswith("Could not"):
        return full_content

    # Try to find the section
    section_patterns = [
        rf"#+\s*{re.escape(section_name)}",
        rf"#+\s*\d+\.?\s*{re.escape(section_name)}",
        rf"\*\*{re.escape(section_name)}\*\*",
    ]

    lines = full_content.split("\n")
    section_start = None
    section_end = None

    for pattern in section_patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                section_start = i
                # Find next section header
                for j in range(i + 1, len(lines)):
                    if re.match(r"^#+\s+", lines[j]) and j > i + 2:
                        section_end = j
                        break
                break
        if section_start is not None:
            break

    if section_start is not None:
        if section_end is None:
            section_end = min(section_start + 100, len(lines))
        section_text = "\n".join(lines[section_start:section_end])
        return (
            f"## Section: {section_name} (from {paper_id})\n\n"
            f"{section_text[:10000]}"
        )

    return (
        f"Section '{section_name}' not found in paper {paper_id}. "
        f"The paper content may not have clear section headers."
    )
