"""Web search tools for AutoIdea.

Provides tavily_search, web_search, and paper_lookup for the research agent.
Also includes arXiv-specific search helpers using the Atom XML API.
"""

import asyncio
import re
from html import unescape as _html_unescape
from typing import Literal

import httpx
from defusedxml import ElementTree as ET
from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated

# Lazy initialization
_tavily_client = None


def _get_tavily_client():
    """Get or create the Tavily client (lazy initialization)."""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        _tavily_client = TavilyClient()
    return _tavily_client


@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs and returns search results
    with snippets for research purposes.

    Args:
        query: Search query to execute.

    Returns:
        Formatted search results in markdown.
    """

    def _sync_search() -> dict:
        return _get_tavily_client().search(
            query,
            max_results=max_results,
            topic=topic,
        )

    try:
        search_results = await asyncio.to_thread(_sync_search)
        results = search_results.get("results", [])
        if not results:
            return f"No results found for '{query}'"

        result_texts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("content", "")[:500]
            result_texts.append(
                f"### [{i}] {title}\n"
                f"**URL**: {url}\n\n"
                f"{snippet}\n"
            )

        return (
            f"## Web Search Results\n"
            f"**Query**: `{query}`  |  **Found**: {len(results)} results\n\n"
            + "\n---\n".join(result_texts)
        )
    except Exception as e:
        return f"Search failed: {e}"


@tool(parse_docstring=True)
async def web_search(query: str) -> str:
    """General web search using Tavily.

    A simpler wrapper around tavily_search for quick lookups.

    Args:
        query: Search query string.

    Returns:
        Formatted search results.
    """

    def _sync_search() -> dict:
        return _get_tavily_client().search(query, max_results=3)

    try:
        search_results = await asyncio.to_thread(_sync_search)
        results = search_results.get("results", [])
        if not results:
            return f"No results found for '{query}'"

        parts = [f"## Web Search: `{query}`\n"]
        for i, r in enumerate(results, 1):
            parts.append(
                f"**[{i}]** [{r.get('title', 'Untitled')}]({r.get('url', '')})\n"
                f"{r.get('content', '')[:300]}\n"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"Web search failed: {e}"


@tool(parse_docstring=True)
async def paper_lookup(query: str) -> str:
    """Search for academic papers using Tavily with academic focus.

    Optimized for finding research papers — adds "paper" and "arxiv OR
    semantic scholar" to the query for better academic results.

    Args:
        query: Research topic or paper title to search for.

    Returns:
        Formatted academic search results.
    """
    academic_query = f"{query} paper arxiv OR semantic scholar OR conference"

    def _sync_search() -> dict:
        return _get_tavily_client().search(academic_query, max_results=5)

    try:
        search_results = await asyncio.to_thread(_sync_search)
        results = search_results.get("results", [])
        if not results:
            return f"No academic results found for '{query}'"

        parts = [f"## Academic Paper Search: `{query}`\n"]
        for i, r in enumerate(results, 1):
            url = r.get("url", "")
            title = r.get("title", "Untitled")
            snippet = r.get("content", "")[:400]

            # Try to extract arXiv ID
            arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
            arxiv_id = arxiv_match.group(1) if arxiv_match else ""

            parts.append(
                f"### [{i}] {title}\n"
                f"**URL**: {url}\n"
            )
            if arxiv_id:
                parts.append(f"**arXiv ID**: {arxiv_id}\n")
            parts.append(f"\n{snippet}\n\n---\n")

        return "\n".join(parts)
    except Exception as e:
        return f"Paper lookup failed: {e}"


# ---------------------------------------------------------------------------
# arXiv helpers (used by scholar.py arXiv functions as well)
# ---------------------------------------------------------------------------

_ARXIV_API_BASE = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _format_api_result(result: dict) -> str:
    """Format a single arXiv-style result to markdown."""
    title = result.get("title", "Untitled")
    authors = result.get("authors", "Unknown")
    year = result.get("year", "N/A")
    abstract = result.get("abstract", "")
    arxiv_id = result.get("arxiv_id", "")
    pdf_url = result.get("pdf_url", "")

    parts = [f"### {title}"]
    if arxiv_id:
        parts.append(f"- **arXiv ID**: {arxiv_id}")
    parts.append(f"- **Authors**: {authors}")
    parts.append(f"- **Year**: {year}")
    if pdf_url:
        parts.append(f"- **PDF**: {pdf_url}")
    if abstract:
        parts.append(f"\n{abstract[:500]}")

    return "\n".join(parts)


async def _search_arxiv_api(query: str, max_results: int = 5) -> str:
    """Internal arXiv search via Atom API.

    Returns markdown-formatted results.  Includes retry with exponential
    backoff to handle arXiv rate limits (HTTP 429 / 503).
    """
    import random as _rand

    from autoidea.tools._proxy import get_async_client

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    max_retries = 5
    last_error = None
    for attempt in range(max_retries):
        try:
            async with get_async_client(timeout=20.0) as client:
                resp = await client.get(_ARXIV_API_BASE, params=params)
                resp.raise_for_status()

            root = ET.fromstring(resp.text)
            ns = _ARXIV_NS
            entries = root.findall("atom:entry", ns)

            if not entries:
                return f"No arXiv results for '{query}'"

            results = []
            for entry in entries:
                title_el = entry.find("atom:title", ns)
                arxiv_title = "Untitled"
                if title_el is not None and title_el.text:
                    arxiv_title = re.sub(r"\s+", " ", title_el.text).strip()

                author_els = entry.findall("atom:author/atom:name", ns)
                authors = ", ".join(
                    a.text.strip() for a in author_els if a.text
                )

                abstract_el = entry.find("atom:summary", ns)
                abstract = ""
                if abstract_el is not None and abstract_el.text:
                    abstract = _html_unescape(re.sub(r"\s+", " ", abstract_el.text).strip())

                arxiv_id = ""
                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    if link.get("title") == "pdf":
                        pdf_url = href
                        arxiv_id = href.split("/pdf/")[-1]
                if not arxiv_id:
                    id_el = entry.find("atom:id", ns)
                    if id_el is not None:
                        arxiv_id = id_el.text.split("/abs/")[-1]

                year = None
                published = entry.find("atom:published", ns)
                if published is not None and published.text:
                    year = int(published.text[:4])

                results.append(_format_api_result({
                    "title": arxiv_title,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "arxiv_id": arxiv_id,
                    "pdf_url": pdf_url,
                }))

            return (
                f"## arXiv Search Results\n"
                f"**Query**: `{query}`  |  **Found**: {len(results)} papers\n\n"
                + "\n\n---\n\n".join(results)
            )

        except Exception as e:
            last_error = e
            status = getattr(getattr(e, "response", None), "status_code", 0)
            if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)) or status == 429 or status >= 500:
                wait = 2.0 * (2 ** attempt) * (0.75 + 0.5 * _rand.random())
                await asyncio.sleep(wait)
                continue
            return f"arXiv search failed: {e}"

    return f"arXiv search failed after {max_retries} retries: {last_error}"
