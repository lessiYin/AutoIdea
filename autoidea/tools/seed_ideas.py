"""Seed Ideas -- User-provided research idea documents for AutoIdea.

Allows users to provide brainstorming notes, draft documents, or
preliminary research ideas that the pipeline uses to:

1. Understand the user's research direction more deeply.
2. Extract key concepts, methods, and hypotheses.
3. Generate targeted search keywords for literature discovery.
4. Guide the pipeline's focus throughout all stages.

Supported input formats
-----------------------

**Markdown** (``.md``):  Freeform research notes, drafts, brainstorming.
**Plain text** (``.txt``): Unstructured idea descriptions.
**JSON** (``.json``):  Structured or semi-structured idea objects.

Unlike seed *papers* (which are structured bibliographic references),
seed *ideas* are the user's own thinking -- possibly rough, incomplete,
or speculative.  The system uses LLM-driven analysis to extract
structured research elements from these documents.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────


class SeedIdeaAnalysis(BaseModel):
    """Structured analysis result from a seed idea document.

    Produced by LLM-driven parsing of the user's raw idea text.
    """

    title: str = Field(
        default="",
        description="Extracted core title or topic of the idea.",
    )
    core_concepts: list[str] = Field(
        default_factory=list,
        description="Core research concepts identified (3-7 items).",
    )
    methods: list[str] = Field(
        default_factory=list,
        description="Methods or techniques mentioned or implied.",
    )
    tasks: list[str] = Field(
        default_factory=list,
        description="Research tasks or problem descriptions.",
    )
    hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypotheses or expected outcomes.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Search keywords for literature discovery (10-20 items).",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Recommended search queries for academic APIs (5-10 items).",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Research gaps or open questions identified.",
    )
    raw_text: str = Field(
        default="",
        description="Original text (truncated to 5000 chars).",
    )


class SeedIdea(BaseModel):
    """A single seed idea entry with its analysis."""

    source_file: str = Field(
        default="",
        description="Path to the source file.",
    )
    format: str = Field(
        default="text",
        description="Source format: 'markdown', 'text', or 'json'.",
    )
    analysis: SeedIdeaAnalysis = Field(
        default_factory=SeedIdeaAnalysis,
        description="Structured analysis of the idea.",
    )


# ── Module-level storage ──────────────────────────────────────────────────

_seed_ideas: list[SeedIdea] = []
"""Loaded seed ideas for the current session."""

_RAW_TEXT_LIMIT = 50000
"""Maximum characters of raw text to retain for analysis."""

_PROMPT_PREVIEW_LIMIT = 4000
"""Maximum characters of text preview to inject into system prompt."""


# ── File Loading ──────────────────────────────────────────────────────────


def _detect_format(path: Path) -> str:
    """Detect the file format from extension."""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return "markdown"
    if suffix == ".json":
        return "json"
    return "text"


def _load_text_content(path: Path) -> str:
    """Load text content from a file, handling encoding gracefully."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file {path} with any supported encoding.")


def _extract_document_outline(text: str, max_chars: int = 4000) -> str:
    """Extract a structural outline from a large document.

    For documents that exceed max_chars, extracts all section headings
    with first few content lines under each, giving much better coverage
    than naive head-truncation.
    """
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()

    # Extract section headings
    sections: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped.lstrip("#").strip()
            if heading:
                sections.append((i, level, heading))

    if not sections:
        return text[:max_chars] + "\n[... truncated ...]"

    # Build outline: heading + first few content lines per section
    outline_parts: list[str] = []
    budget_per_section = max(200, max_chars // max(len(sections), 1))
    total_used = 0

    for idx, (line_idx, level, heading) in enumerate(sections):
        if total_used >= max_chars:
            remaining = len(sections) - idx
            outline_parts.append(f"\n[... {remaining} more sections omitted ...]")
            break

        prefix = "#" * level
        outline_parts.append(f"{prefix} {heading}")

        # Collect content lines until next heading
        next_idx = sections[idx + 1][0] if idx + 1 < len(sections) else len(lines)
        content_lines = []
        for j in range(line_idx + 1, min(next_idx, line_idx + 15)):
            cl = lines[j].strip()
            if cl and not cl.startswith("#"):
                content_lines.append(cl)

        content_text = " ".join(content_lines)
        if len(content_text) > budget_per_section:
            content_text = content_text[:budget_per_section] + "..."
        if content_text:
            outline_parts.append(content_text)
        outline_parts.append("")

        total_used += len(prefix) + len(heading) + len(content_text) + 4

    result = "\n".join(outline_parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[... truncated ...]"
    return result


def _parse_json_ideas(data: Any) -> list[str]:
    """Extract idea texts from JSON data.

    Supports:
    - A single string
    - A list of strings
    - A list of objects with 'idea', 'description', 'content', or 'text' keys
    - A dict with 'ideas' key containing any of the above
    """
    texts: list[str] = []

    if isinstance(data, str):
        texts.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                # Try common keys
                for key in ("idea", "description", "content", "text",
                             "title", "summary", "abstract"):
                    val = item.get(key)
                    if val and isinstance(val, str):
                        texts.append(val)
                        break
                else:
                    # Serialize the whole dict as fallback
                    texts.append(json.dumps(item, ensure_ascii=False))
    elif isinstance(data, dict):
        if "ideas" in data:
            texts.extend(_parse_json_ideas(data["ideas"]))
        else:
            # Try to combine all string values
            parts = []
            for key, val in data.items():
                if isinstance(val, str) and val.strip():
                    parts.append(f"**{key}**: {val}")
                elif isinstance(val, list):
                    parts.append(f"**{key}**: {', '.join(str(v) for v in val)}")
            if parts:
                texts.append("\n".join(parts))

    return texts


def load_seed_ideas(
    file_path: str,
    use_llm: bool = True,
    model=None,
) -> list[SeedIdea]:
    """Load seed ideas from a file.

    Supports .md, .txt, and .json formats.

    Args:
        file_path: Path to the seed ideas file.
        use_llm: Whether to use LLM for deep analysis (True) or
            heuristic-only extraction (False). When True, falls back
            to heuristic if LLM is unavailable.
        model: Optional LangChain chat model instance for analysis.

    Returns:
        List of SeedIdea objects with analysis completed.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    global _seed_ideas

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Seed ideas file not found: {path}")

    fmt = _detect_format(path)
    raw_texts: list[str] = []

    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_texts = _parse_json_ideas(data)
    else:
        content = _load_text_content(path)
        if not content.strip():
            raise ValueError(f"Seed ideas file is empty: {path}")
        raw_texts = [content]

    if not raw_texts:
        raise ValueError(
            f"No valid idea content found in {path}. "
            "File must contain text describing research ideas."
        )

    ideas: list[SeedIdea] = []
    for raw in raw_texts:
        truncated = raw[:_RAW_TEXT_LIMIT]
        analysis = analyze_seed_idea(truncated, use_llm=use_llm, model=model)
        idea = SeedIdea(
            source_file=str(path),
            format=fmt,
            analysis=analysis,
        )
        ideas.append(idea)

    _seed_ideas = ideas
    logger.info(
        "Loaded %d seed idea(s) from %s (format: %s)",
        len(ideas), path, fmt,
    )
    return ideas


# ── LLM-Driven Deep Analysis ─────────────────────────────────────────────

_LLM_ANALYSIS_PROMPT = """\
You are analyzing a user's research idea document. Your task is to extract
structured research elements from this text. The document may be rough,
incomplete, or speculative — that's expected.

Analyze the following document and produce a JSON object with these fields:

1. **title** (string): A concise title capturing the core research idea.
   If the document has a clear title, use it. Otherwise, synthesize one.

2. **core_concepts** (list of 3-7 strings): The fundamental research concepts
   or pillars in this idea. These should be specific enough to be useful as
   search decomposition anchors.

3. **methods** (list of strings): Methods, techniques, algorithms, or
   architectural approaches mentioned or implied. Include both explicitly
   stated and implicitly suggested methods.

4. **tasks** (list of strings): Research tasks or problem formulations
   described or implied.

5. **hypotheses** (list of strings): Hypotheses, expectations, predictions,
   or key insights the author proposes. Capture both explicit statements
   and implicit assumptions.

6. **keywords** (list of 10-20 strings): Search keywords optimized for
   academic literature search. Include:
   - Technical terms and their variants
   - Acronyms with expansions
   - Related/adjacent concepts not explicitly stated but clearly relevant
   - Both broad (high-recall) and specific (high-precision) terms

7. **search_queries** (list of 5-10 strings): Complete search queries ready
   for use in academic search APIs (Semantic Scholar, arXiv, etc.). Each
   query should target a different aspect:
   - Core method queries
   - Baseline/comparison queries
   - Application domain queries
   - Related technique queries
   - Evaluation/benchmark queries

8. **gaps** (list of strings): Research gaps, open questions, limitations,
   or challenges identified in the document. Include both explicitly stated
   gaps and those you can infer from the text.

Respond with ONLY the JSON object, no markdown formatting or explanation.

--- DOCUMENT ---
{document}
--- END DOCUMENT ---
"""


def analyze_seed_idea_with_llm(
    text: str,
    model=None,
) -> SeedIdeaAnalysis | None:
    """Analyze a seed idea document using LLM for deep structured extraction.

    This produces significantly richer analysis than the heuristic fallback
    because the LLM can:
    - Understand semantic meaning and implicit hypotheses
    - Generate high-quality search keywords including related concepts
    - Identify research gaps the user may not have explicitly stated
    - Synthesize complete search queries optimized for academic APIs

    Args:
        text: Raw text content of the seed idea document.
        model: Optional LangChain chat model instance. If None, the
            system's configured model is used.

    Returns:
        SeedIdeaAnalysis with LLM-extracted fields, or None if LLM
        analysis fails (caller should fall back to heuristic).
    """
    if not text.strip():
        return None

    try:
        if model is None:
            from autoidea.llm import get_chat_model
            model = get_chat_model()

        prompt_text = _LLM_ANALYSIS_PROMPT.format(
            document=text[:_RAW_TEXT_LIMIT]
        )

        from langchain_core.messages import HumanMessage
        response = model.invoke([HumanMessage(content=prompt_text)])

        # Extract content from response
        content = response.content
        if isinstance(content, list):
            # Handle structured content blocks (e.g., Claude)
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            logger.warning("LLM returned empty response for seed idea analysis")
            return None

        # Parse JSON from response (handle markdown code blocks)
        json_str = content.strip()
        if json_str.startswith("```"):
            # Strip markdown code fence
            lines = json_str.splitlines()
            # Remove first line (```json) and last line (```)
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            json_str = "\n".join(lines[start:end])

        data = json.loads(json_str)

        if not isinstance(data, dict):
            logger.warning("LLM returned non-dict JSON for seed idea analysis")
            return None

        return SeedIdeaAnalysis(
            title=data.get("title", ""),
            core_concepts=_ensure_str_list(data.get("core_concepts", [])),
            methods=_ensure_str_list(data.get("methods", [])),
            tasks=_ensure_str_list(data.get("tasks", [])),
            hypotheses=_ensure_str_list(data.get("hypotheses", [])),
            keywords=_ensure_str_list(data.get("keywords", []))[:20],
            search_queries=_ensure_str_list(data.get("search_queries", []))[:10],
            gaps=_ensure_str_list(data.get("gaps", [])),
            raw_text=text[:_RAW_TEXT_LIMIT],
        )

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM response as JSON: %s", e)
        return None
    except Exception as e:
        logger.warning("LLM seed idea analysis failed: %s", e)
        return None


def _ensure_str_list(val: Any) -> list[str]:
    """Coerce a value to a list of strings."""
    if isinstance(val, list):
        return [str(item) for item in val if item]
    if isinstance(val, str):
        return [val] if val else []
    return []


# ── Combined Analysis (LLM with heuristic fallback) ──────────────────────


def analyze_seed_idea(
    text: str,
    use_llm: bool = True,
    model=None,
) -> SeedIdeaAnalysis:
    """Analyze a seed idea document with LLM deep analysis + heuristic fallback.

    Strategy:
    1. If use_llm=True, attempt LLM-driven deep analysis first.
    2. If LLM fails or is disabled, fall back to heuristic extraction.
    3. If LLM succeeds but some fields are sparse, merge heuristic results
       to fill gaps.

    Args:
        text: Raw text content of the seed idea document.
        use_llm: Whether to attempt LLM analysis. Set False for testing
            or when no LLM is configured.
        model: Optional LangChain chat model instance.

    Returns:
        SeedIdeaAnalysis with the best available extraction.
    """
    heuristic = _extract_analysis_heuristic(text)

    if not use_llm:
        return heuristic

    llm_result = analyze_seed_idea_with_llm(text, model=model)

    if llm_result is None:
        logger.info("LLM analysis unavailable, using heuristic extraction")
        return heuristic

    # Merge: use LLM result as primary, fill empty fields from heuristic
    merged = SeedIdeaAnalysis(
        title=llm_result.title or heuristic.title,
        core_concepts=llm_result.core_concepts or heuristic.core_concepts,
        methods=llm_result.methods or heuristic.methods,
        tasks=llm_result.tasks or heuristic.tasks,
        hypotheses=llm_result.hypotheses or heuristic.hypotheses,
        keywords=llm_result.keywords or heuristic.keywords,
        search_queries=llm_result.search_queries or heuristic.search_queries,
        gaps=llm_result.gaps or heuristic.gaps,
        raw_text=text[:_RAW_TEXT_LIMIT],
    )

    logger.info(
        "Seed idea analyzed with LLM: title=%r, %d concepts, %d keywords, "
        "%d queries, %d gaps",
        merged.title,
        len(merged.core_concepts),
        len(merged.keywords),
        len(merged.search_queries),
        len(merged.gaps),
    )
    return merged


# ── Heuristic Analysis (no LLM required) ─────────────────────────────────


def _extract_analysis_heuristic(text: str) -> SeedIdeaAnalysis:
    """Extract structured elements from raw text using heuristics.

    This provides a baseline analysis without requiring an LLM call.
    The LLM-driven analysis (via the agent) produces richer results
    but this ensures the system works even without LLM access at load time.

    The heuristic approach:
    1. Extracts a title from the first heading or first line
    2. Identifies potential keywords from significant terms
    3. Generates basic search queries from headings and key phrases
    """
    lines = text.strip().splitlines()
    if not lines:
        return SeedIdeaAnalysis(raw_text=text)

    # Extract title from first markdown heading or first non-empty line
    title = ""
    for line in lines:
        stripped = line.strip()
        if stripped:
            heading_match = re.match(r"^#{1,3}\s+(.+)", stripped)
            if heading_match:
                title = heading_match.group(1).strip()
            else:
                title = stripped[:120]
            break

    # Extract headings as potential concept markers
    headings: list[str] = []
    for line in lines:
        m = re.match(r"^#{1,4}\s+(.+)", line.strip())
        if m:
            headings.append(m.group(1).strip())

    # Extract keywords: terms that appear in bold, headings, or are capitalized
    keyword_candidates: set[str] = set()

    # Bold terms (match within single lines to avoid multi-line noise)
    for line in lines:
        for m in re.finditer(r"\*\*(.+?)\*\*", line):
            term = m.group(1).strip()
            if 2 <= len(term) <= 60 and "\n" not in term:
                keyword_candidates.add(term.lower())

    # Heading terms (skip generic section headings)
    _GENERIC_HEADINGS = {
        "introduction", "background", "conclusion", "summary",
        "references", "notes", "abstract", "related work",
        "proposed method", "core idea", "hypotheses", "open questions",
        "potential impact", "open questions / gaps",
    }
    for h in headings:
        if h.lower() in _GENERIC_HEADINGS:
            continue
        keyword_candidates.add(h.lower())

    # Technical terms: capitalized multi-word phrases (per-line to avoid cross-line matches)
    for line in lines:
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", line):
            term = m.group(1)
            if len(term) > 5:
                keyword_candidates.add(term.lower())

    # Acronyms (per-line)
    for line in lines:
        for m in re.finditer(r"\b([A-Z]{2,6})\b", line):
            acronym = m.group(1)
            # Filter out common non-technical acronyms
            if acronym not in ("THE", "AND", "FOR", "BUT", "NOT", "THIS",
                               "THAT", "WITH", "FROM", "ARE", "WAS", "HAS"):
                keyword_candidates.add(acronym)

    # Final cleanup: remove any keywords containing newlines or that are too generic
    keyword_candidates = {
        kw for kw in keyword_candidates
        if "\n" not in kw and len(kw.strip()) >= 2
    }

    keywords = sorted(keyword_candidates)[:20]

    # Generate search queries from title + top keywords
    search_queries: list[str] = []
    if title:
        search_queries.append(title[:100])
    for kw in keywords[:5]:
        if kw != title.lower():
            search_queries.append(kw)

    # Extract potential methods (lines containing method-like terms)
    method_patterns = re.compile(
        r"(?:method|approach|technique|algorithm|architecture|framework|model|"
        r"mechanism|strategy|pipeline|protocol|procedure)",
        re.IGNORECASE,
    )
    methods: list[str] = []
    for line in lines:
        if method_patterns.search(line):
            clean = line.strip().lstrip("-*# ")
            if 10 < len(clean) < 200:
                methods.append(clean)
    methods = methods[:10]

    # Extract hypotheses / expectations
    hyp_patterns = re.compile(
        r"(?:hypothes[ie]s|expect|predict|assume|conjecture|"
        r"we believe|we propose|our idea|key insight)",
        re.IGNORECASE,
    )
    hypotheses: list[str] = []
    for line in lines:
        if hyp_patterns.search(line):
            clean = line.strip().lstrip("-*# ")
            if 10 < len(clean) < 300:
                hypotheses.append(clean)
    hypotheses = hypotheses[:5]

    # Extract gap descriptions
    gap_patterns = re.compile(
        r"(?:gap|limitation|challenge|problem|issue|lack|missing|"
        r"unexplored|under-explored|open question|unresolved)",
        re.IGNORECASE,
    )
    gaps: list[str] = []
    for line in lines:
        if gap_patterns.search(line):
            clean = line.strip().lstrip("-*# ")
            if 10 < len(clean) < 300:
                gaps.append(clean)
    gaps = gaps[:5]

    # Filter headings for core_concepts (skip generic section names)
    meaningful_headings = [
        h for h in headings if h.lower() not in _GENERIC_HEADINGS
    ]

    return SeedIdeaAnalysis(
        title=title,
        core_concepts=meaningful_headings[:7] if meaningful_headings else keywords[:7],
        methods=methods,
        tasks=[],
        hypotheses=hypotheses,
        keywords=keywords,
        search_queries=search_queries[:10],
        gaps=gaps,
        raw_text=text[:_RAW_TEXT_LIMIT],
    )


# ── Public API ────────────────────────────────────────────────────────────


def get_seed_ideas() -> list[SeedIdea]:
    """Return the currently loaded seed ideas."""
    return list(_seed_ideas)


def clear_seed_ideas() -> None:
    """Clear loaded seed ideas (useful for testing)."""
    global _seed_ideas
    _seed_ideas = []


def format_seed_ideas_for_prompt(ideas: list[SeedIdea]) -> str:
    """Format seed ideas into a markdown section for system prompt injection.

    Args:
        ideas: List of SeedIdea objects with completed analysis.

    Returns:
        Markdown-formatted string describing the seed ideas.
    """
    if not ideas:
        return ""

    lines = [
        "",
        "## SEED IDEAS (User-Provided Research Ideas & Notes)",
        "",
        "The user has provided the following research idea document(s) as **input context**.",
        "These seed ideas represent the user's preliminary thinking and MUST be used to:",
        "  1. **Guide scope** in Stage 1 (Requirement Intake) -- align the research brief",
        "     with the user's stated interests, hypotheses, and directions.",
        "  2. **Inform search** in Stage 3 (Literature Survey) -- use extracted keywords",
        "     and search queries as starting points for literature search.",
        "  3. **Anchor ideas** in Stage 9 (Idea Generation) -- the user's seed ideas",
        "     should be refined, validated, or evolved rather than ignored.",
        "  4. **Enrich analysis** throughout -- the user's domain knowledge, identified",
        "     gaps, and methods should inform critique and synthesis.",
        "",
        "These are the user's **own ideas**, not published papers.  Treat them as",
        "valuable hypotheses to investigate, not as established facts to cite.",
        "",
    ]

    for i, idea in enumerate(ideas, 1):
        a = idea.analysis
        lines.append(f"### Seed Idea {i}" + (f": {a.title}" if a.title else ""))
        lines.append(f"- **Source**: `{idea.source_file}` ({idea.format})")
        lines.append("")

        if a.core_concepts:
            lines.append("**Core Concepts**: " + ", ".join(a.core_concepts))
        if a.methods:
            lines.append("**Methods/Techniques Mentioned**:")
            for m in a.methods[:5]:
                lines.append(f"  - {m}")
        if a.hypotheses:
            lines.append("**Hypotheses/Expectations**:")
            for h in a.hypotheses:
                lines.append(f"  - {h}")
        if a.gaps:
            lines.append("**Identified Gaps**:")
            for g in a.gaps:
                lines.append(f"  - {g}")
        if a.keywords:
            lines.append(f"**Search Keywords**: {', '.join(a.keywords[:15])}")
        if a.search_queries:
            lines.append("**Recommended Search Queries**:")
            for q in a.search_queries[:7]:
                lines.append(f"  - `{q}`")
        lines.append("")

        # Include document preview for the agent's reference
        if a.raw_text:
            preview = _extract_document_outline(a.raw_text, _PROMPT_PREVIEW_LIMIT)
            lines.append("<seed_idea_text>")
            lines.append(preview)
            lines.append("</seed_idea_text>")
            lines.append("")

    lines.extend([
        "### Seed Idea Protocol",
        "",
        "- **IMPORTANT**: Begin with **Stage 0.5** before Stage 1.  Call",
        "  `generate_seed_idea_analysis_report` to get a deep LLM analysis",
        "  of the seed ideas, then write `seed_idea_analysis.md`.",
        "- Use `list_seed_ideas` tool at any time to review the full seed idea analysis.",
        "- Use `get_search_keywords_from_seeds` to obtain search keywords extracted",
        "  from seed ideas, ready for use in literature search queries.",
        "- In Stage 1, reference the seed idea analysis when asking clarification questions.",
        "- In Stage 3, incorporate seed idea keywords into your query decomposition.",
        "- In Stage 9, at least one generated idea should build upon or refine the",
        "  user's seed ideas.  If a seed idea is infeasible, explain why explicitly.",
        "- Clearly distinguish between claims from seed ideas (user's hypotheses)",
        "  and claims from published literature (citable facts).",
        "",
    ])

    return "\n".join(lines)


# ── LangChain Tools ──────────────────────────────────────────────────────


@tool(parse_docstring=True)
def list_seed_ideas() -> str:
    """List all user-provided seed ideas with their extracted analysis.

    Returns the complete analysis of seed idea documents that the user
    has provided, including extracted concepts, methods, hypotheses,
    keywords, and search queries.

    Call this tool whenever you need to review the user's original
    research ideas and the system's analysis of them.

    Returns:
        Formatted analysis of seed ideas, or a message indicating
        no seed ideas were provided.
    """
    if not _seed_ideas:
        return (
            "No seed ideas were provided by the user. "
            "The research direction will be determined through conversation."
        )

    lines = [f"## User-Provided Seed Ideas ({len(_seed_ideas)} document(s))\n"]

    for i, idea in enumerate(_seed_ideas, 1):
        a = idea.analysis
        lines.append(
            f"### Seed Idea {i}" + (f": {a.title}" if a.title else "")
        )
        lines.append(f"- Source: `{idea.source_file}` ({idea.format})")

        if a.core_concepts:
            lines.append(f"- Core concepts: {', '.join(a.core_concepts)}")
        if a.methods:
            lines.append("- Methods:")
            for m in a.methods[:5]:
                lines.append(f"  - {m}")
        if a.hypotheses:
            lines.append("- Hypotheses:")
            for h in a.hypotheses:
                lines.append(f"  - {h}")
        if a.gaps:
            lines.append("- Research gaps:")
            for g in a.gaps:
                lines.append(f"  - {g}")
        if a.keywords:
            lines.append(f"- Keywords: {', '.join(a.keywords)}")
        if a.search_queries:
            lines.append("- Search queries:")
            for q in a.search_queries:
                lines.append(f"  - {q}")

        lines.append("")

        # Include raw text excerpt
        if a.raw_text:
            excerpt = _extract_document_outline(a.raw_text, 5000)
            lines.append("**Original Text (outline)**:")
            lines.append(f"```\n{excerpt}\n```")
            lines.append("")

    lines.append(
        "**Reminder**: Seed ideas are the user's own hypotheses. "
        "Validate them against literature rather than treating them as "
        "established facts."
    )

    return "\n".join(lines)


@tool(parse_docstring=True)
def get_search_keywords_from_seeds() -> str:
    """Extract search keywords and queries from user-provided seed ideas.

    Returns a consolidated list of search keywords and recommended queries
    derived from the user's seed idea documents. These are specifically
    designed for use in Stage 3 (Literature Survey) query decomposition.

    Use this tool at the beginning of Stage 3 to incorporate the user's
    research direction into your search strategy.

    Returns:
        Formatted keywords and search queries ready for literature search,
        or a message indicating no seed ideas were provided.
    """
    if not _seed_ideas:
        return (
            "No seed ideas provided. Generate search queries from the "
            "research brief and task formalization instead."
        )

    all_keywords: list[str] = []
    all_queries: list[str] = []
    all_concepts: list[str] = []

    for idea in _seed_ideas:
        a = idea.analysis
        all_keywords.extend(a.keywords)
        all_queries.extend(a.search_queries)
        all_concepts.extend(a.core_concepts)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for kw in all_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in all_queries:
        q_lower = q.lower()
        if q_lower not in seen_q:
            seen_q.add(q_lower)
            unique_queries.append(q)

    seen_c: set[str] = set()
    unique_concepts: list[str] = []
    for c in all_concepts:
        c_lower = c.lower()
        if c_lower not in seen_c:
            seen_c.add(c_lower)
            unique_concepts.append(c)

    lines = [
        "## Search Keywords from Seed Ideas",
        "",
        f"Extracted from {len(_seed_ideas)} seed idea document(s).",
        "",
    ]

    if unique_concepts:
        lines.append("### Core Concepts (for query decomposition)")
        for c in unique_concepts:
            lines.append(f"- {c}")
        lines.append("")

    if unique_keywords:
        lines.append(f"### Keywords ({len(unique_keywords)} total)")
        lines.append(", ".join(unique_keywords))
        lines.append("")

    if unique_queries:
        lines.append("### Recommended Search Queries")
        lines.append("Use these as starting points for your search plan:")
        for i, q in enumerate(unique_queries, 1):
            lines.append(f"{i}. `{q}`")
        lines.append("")

    lines.append(
        "**Usage**: Incorporate these keywords and queries into your "
        "Stage 3 query decomposition plan. They should supplement "
        "(not replace) queries derived from the research brief."
    )

    return "\n".join(lines)


# ── Stage 0.5: Seed Idea Analysis Report ─────────────────────────────────


@tool(parse_docstring=True)
def generate_seed_idea_analysis_report() -> str:
    """Generate a structured seed idea analysis report for Stage 0.5.

    This tool performs **LLM-driven deep analysis** on each seed idea,
    upgrading from the lightweight heuristic extraction done at startup.
    It then produces a formatted markdown report that the agent should
    review, enhance with its own assessments (clarity, novelty, feasibility),
    and write to ``seed_idea_analysis.md`` via ``write_workspace_file``.

    Call this tool at the beginning of Stage 0.5 to get a rich analysis
    foundation before writing the final artifact.

    Returns:
        Markdown-formatted analysis report draft, or a message indicating
        no seed ideas were provided.
    """
    if not _seed_ideas:
        return (
            "No seed ideas were provided by the user. "
            "Skip Stage 0.5 and proceed directly to Stage 1."
        )

    # Attempt LLM deep analysis to upgrade the heuristic results
    upgraded_analyses: list[SeedIdeaAnalysis] = []
    llm_used = False
    for idea in _seed_ideas:
        if idea.analysis.raw_text:
            # Try LLM analysis; check if it actually succeeds
            llm_result = analyze_seed_idea_with_llm(
                idea.analysis.raw_text, model=None,
            )
            if llm_result is not None:
                llm_used = True
                upgraded_analyses.append(llm_result)
            else:
                upgraded_analyses.append(idea.analysis)
        else:
            upgraded_analyses.append(idea.analysis)

    analysis_method = "LLM deep analysis" if llm_used else "heuristic extraction"

    lines = [
        "# Seed Idea Analysis Report",
        "",
        "## Overview",
        f"- **Number of seed ideas**: {len(_seed_ideas)}",
        "- **User's stated topic**: [AGENT: fill in the user's research topic]",
        "- **Primary research direction**: [AGENT: summarize the overall direction]",
        "- **Topic alignment**: [AGENT: ALIGNED / RELATED / UNRELATED — explain how "
        "the seed ideas relate to the user's stated topic and recommend integration strategy]",
        f"- **Analysis method**: {analysis_method}",
        "",
    ]

    for i, (idea, a) in enumerate(zip(_seed_ideas, upgraded_analyses), 1):
        title = a.title or f"Untitled Idea {i}"
        lines.append(f"## Seed Idea {i}: {title}")
        lines.append(f"- **Source**: `{idea.source_file}`")

        if a.core_concepts:
            lines.append(f"- **Core Concepts**: {', '.join(a.core_concepts)}")

        if a.methods:
            lines.append("- **Methods/Techniques**:")
            for m in a.methods:
                lines.append(f"  - {m}")

        if a.hypotheses:
            lines.append("- **Hypotheses**:")
            for h in a.hypotheses:
                lines.append(f"  - {h}")

        if a.gaps:
            lines.append("- **Research Gaps Identified**:")
            for g in a.gaps:
                lines.append(f"  - {g}")

        # Placeholder fields for agent to fill
        lines.extend([
            "- **Clarity Assessment**: [AGENT: HIGH / MEDIUM / LOW — justification]",
            "- **Novelty Potential**: [AGENT: HIGH / MEDIUM / LOW — justification]",
            "- **Feasibility Signals**: [AGENT: HIGH / MEDIUM / LOW — justification]",
            "- **Relevance to User Topic**: [AGENT: how this seed idea connects to the user's stated topic]",
            "- **Key Uncertainties**:",
            "  - [AGENT: what needs literature validation]",
        ])

        if a.search_queries:
            lines.append("- **Recommended Search Directions**:")
            for q in a.search_queries:
                lines.append(f"  - `{q}`")
        elif a.keywords:
            lines.append("- **Recommended Search Directions**:")
            for kw in a.keywords[:7]:
                lines.append(f"  - `{kw}`")

        lines.append("")

        # Include document outline for agent reference
        if a.raw_text:
            outline = _extract_document_outline(a.raw_text, 6000)
            lines.append("<original_text>")
            lines.append(outline)
            lines.append("</original_text>")
            lines.append("")

    # Cross-idea synthesis section
    lines.extend([
        "## Cross-Idea Synthesis",
        "- **Common themes**: [AGENT: identify themes across all seed ideas]",
        "- **Complementary directions**: [AGENT: how ideas might combine]",
        "- **Priority ranking**: [AGENT: rank ideas from most to least promising]",
        "- **Consolidated search strategy**: [AGENT: merged keyword/query list for Stage 3]",
        "",
    ])

    return "\n".join(lines)
