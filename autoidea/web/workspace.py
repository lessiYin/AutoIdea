"""Load and normalize AutoIdea workspace artifacts for the web dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import (
    ArtifactInfo,
    DashboardWarning,
    DesignAxis,
    DimensionPosition,
    EvidenceClaim,
    GapEvidenceLink,
    GraphEdge,
    GraphNode,
    Paper,
    PaperPosition,
    ResearchGraph,
    ResearchGap,
    ResearchIdea,
    WorkspaceInfo,
    WorkspaceSnapshot,
    artifact_title,
)
from .pipeline import inspect_pipeline

KNOWN_ARTIFACTS: tuple[str, ...] = (
    "research_brief.md",
    "task_formalization.md",
    "literature_survey.md",
    "paper_registry.json",
    "paper_deep_reading.md",
    "paper_positions.json",
    "expanded_literature.md",
    "evidence_db.json",
    "knowledge_synthesis.md",
    "research_gaps.json",
    "design_space.json",
    "raw_ideas.json",
    "tournament_rankings.json",
    "debate_log.md",
    "idea_reviews.json",
    "feasibility_assessments.json",
    "final_report.md",
    "pipeline_state.json",
    "run_status.json",
)


def load_workspace_snapshot(workspace: str | Path) -> WorkspaceSnapshot:
    """Read a workspace directory and return normalized dashboard data.

    The loader is intentionally tolerant. Missing artifacts produce empty data;
    invalid JSON is reported as a warning so the dashboard can still render.
    """
    root = Path(workspace).expanduser().resolve()
    warnings: list[DashboardWarning] = []

    def load_json(name: str, default: Any) -> Any:
        path = root / name
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            warnings.append(
                DashboardWarning(
                    code="JSON_INVALID",
                    message=f"Could not parse {name}: {exc}",
                    path=str(path),
                )
            )
            return default

    papers = _normalize_papers(load_json("paper_registry.json", []))
    positions = _normalize_positions(load_json("paper_positions.json", []))
    papers = [
        Paper(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            source=paper.source,
            venue=paper.venue,
            url=paper.url,
            authors=paper.authors,
            relevance=paper.relevance,
            position=positions.get(paper.paper_id),
        )
        for paper in papers
    ]

    claims = _normalize_claims(load_json("evidence_db.json", {}))
    claims = _resolve_claim_sources(
        claims,
        papers,
        warnings,
        evidence_path=root / "evidence_db.json",
    )
    gaps = _normalize_gaps(load_json("research_gaps.json", {}))
    gaps = _resolve_gap_evidence(
        gaps,
        claims,
        warnings,
        gap_path=root / "research_gaps.json",
    )
    ideas = _normalize_ideas(load_json("raw_ideas.json", {}))
    design_axes = _normalize_design_axes(load_json("design_space.json", {}))
    # Derive primary progress from files on disk.  ``pipeline_state.json`` can
    # lag behind the artifacts after a crash, so it is treated as provenance
    # by ``inspect_pipeline`` rather than as the source of truth.
    pipeline = inspect_pipeline(root)
    artifacts = _collect_artifacts(root)
    graph = _build_graph(papers, claims, gaps, ideas)

    counts = {
        "papers": len(papers),
        "claims": len(claims),
        "gaps": len(gaps),
        "ideas": len(ideas),
        "artifacts": len(artifacts),
        "warnings": len(warnings),
    }

    return WorkspaceSnapshot(
        workspace=WorkspaceInfo(path=str(root), name=root.name),
        counts=counts,
        artifacts=artifacts,
        papers=papers,
        claims=claims,
        gaps=gaps,
        ideas=ideas,
        design_axes=design_axes,
        graph=graph,
        pipeline=pipeline,
        warnings=warnings,
    )


def _collect_artifacts(root: Path) -> list[ArtifactInfo]:
    artifacts: list[ArtifactInfo] = []
    for name in KNOWN_ARTIFACTS:
        path = root / name
        if not path.is_file():
            continue
        suffix = path.suffix.lower().lstrip(".") or "file"
        artifacts.append(
            ArtifactInfo(
                path=name,
                kind=suffix,
                size_bytes=path.stat().st_size,
                title=artifact_title(path),
            )
        )
    return artifacts


def _normalize_papers(data: Any) -> list[Paper]:
    if not isinstance(data, list):
        return []
    papers: list[Paper] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        paper_id = _text(item.get("paper_id") or item.get("id") or f"P{index}")
        title = _text(item.get("title")) or paper_id
        authors = item.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        if not isinstance(authors, list):
            authors = []
        papers.append(
            Paper(
                paper_id=paper_id,
                title=title,
                year=_int_or_none(item.get("year")),
                source=_text(item.get("source")),
                venue=_text(item.get("venue")),
                url=_text(item.get("url")),
                authors=[_text(author) for author in authors if _text(author)],
                relevance=_text(item.get("relevance") or item.get("reason")),
            )
        )
    return papers


def _normalize_positions(data: Any) -> dict[str, PaperPosition]:
    if not isinstance(data, list):
        return {}
    positions: dict[str, PaperPosition] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        paper_id = _text(item.get("paper_id") or item.get("id"))
        if not paper_id:
            continue
        dimensions: list[DimensionPosition] = []
        raw_dimensions = item.get("dimensions", [])
        if isinstance(raw_dimensions, list):
            for raw in raw_dimensions:
                if not isinstance(raw, dict):
                    continue
                dimensions.append(
                    DimensionPosition(
                        dimension=_text(raw.get("dimension")),
                        strength=_text(raw.get("strength")),
                        evidence=_text(raw.get("evidence")),
                        concern=_text(raw.get("concern")),
                    )
                )
        elif isinstance(raw_dimensions, dict):
            for name, raw in raw_dimensions.items():
                details = raw if isinstance(raw, dict) else {}
                strength = (
                    details.get("strength")
                    or details.get("rating")
                    or details.get("score")
                    if details
                    else raw
                )
                dimensions.append(
                    DimensionPosition(
                        dimension=_text(name),
                        strength=_text(strength),
                        evidence=_text(details.get("evidence")),
                        concern=_text(details.get("concern")),
                    )
                )
        positions[paper_id] = PaperPosition(
            initial_attack=_text(item.get("initial_attack")),
            weakest_link=_text(item.get("weakest_link")),
            summary=_text(item.get("summary")),
            dimensions=dimensions,
        )
    return positions


def _normalize_claims(data: Any) -> list[EvidenceClaim]:
    raw_claims = data.get("claims", []) if isinstance(data, dict) else []
    if not isinstance(raw_claims, list):
        return []
    claims: list[EvidenceClaim] = []
    for index, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            continue
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            tags = []
        source_paper_id = _text(
            item.get("source_paper_id")
            or item.get("paper_id")
            or item.get("paper")
            or item.get("source_paper")
        )
        claims.append(
            EvidenceClaim(
                citation_id=_text(
                    item.get("citation_id") or item.get("claim_id") or f"C{index}"
                ),
                claim=_text(item.get("claim") or item.get("claim_text")),
                source_paper_id=source_paper_id,
                source_title=_text(item.get("source_title")),
                source_url=_text(item.get("source_url") or item.get("url")),
                confidence=_confidence_label(item.get("confidence")),
                evidence_type=_text(item.get("evidence_type")),
                section=_text(item.get("section")),
                tags=[_text(tag) for tag in tags if _text(tag)],
            )
        )
    return claims


def _resolve_claim_sources(
    claims: list[EvidenceClaim],
    papers: list[Paper],
    warnings: list[DashboardWarning],
    *,
    evidence_path: Path,
) -> list[EvidenceClaim]:
    """Resolve evidence provenance without guessing or creating phantom papers.

    AutoIdea workspaces in the wild use both the current ``source_paper_id``
    field and the older ``source_paper`` alias.  An explicit paper ID is the
    primary key; unique URL and title matches are conservative fallbacks for
    older artifacts that omitted it.
    """

    id_index = _paper_index(papers, lambda paper: _lookup_text(paper.paper_id))
    url_index = _paper_index(papers, lambda paper: _lookup_url(paper.url))
    title_index = _paper_index(papers, lambda paper: _lookup_title(paper.title))
    resolved: list[EvidenceClaim] = []

    for claim in claims:
        paper: Paper | None = None
        match_method = ""
        ambiguous = False
        raw_source = claim.source_paper_id

        candidates = id_index.get(_lookup_text(raw_source), [])
        if len(candidates) == 1:
            paper = candidates[0]
            match_method = "id"
        elif len(candidates) > 1:
            ambiguous = True

        if paper is None and claim.source_url:
            candidates = url_index.get(_lookup_url(claim.source_url), [])
            if len(candidates) == 1:
                paper = candidates[0]
                match_method = "url"
            elif len(candidates) > 1:
                ambiguous = True

        title_values = [claim.source_title]
        if raw_source and not re.fullmatch(r"P\d+", raw_source, flags=re.IGNORECASE):
            title_values.append(raw_source)
        if paper is None:
            for title in title_values:
                candidates = title_index.get(_lookup_title(title), [])
                if len(candidates) == 1:
                    paper = candidates[0]
                    match_method = "title"
                    break
                if len(candidates) > 1:
                    ambiguous = True

        if paper is not None:
            resolved.append(
                replace(
                    claim,
                    source_paper_id=paper.paper_id,
                    source_title=claim.source_title or paper.title,
                    source_url=claim.source_url or paper.url,
                    source_match=match_method,
                )
            )
            continue

        has_source = bool(raw_source or claim.source_title or claim.source_url)
        warnings.append(
            DashboardWarning(
                code=(
                    "AMBIGUOUS_EVIDENCE_SOURCE"
                    if ambiguous
                    else "UNMAPPED_EVIDENCE_SOURCE"
                ),
                message=(
                    f"Evidence claim {claim.citation_id} has an ambiguous paper source."
                    if ambiguous
                    else (
                        f"Evidence claim {claim.citation_id} could not be matched to "
                        "paper_registry.json."
                        if has_source
                        else f"Evidence claim {claim.citation_id} has no paper source."
                    )
                ),
                path=str(evidence_path),
            )
        )
        resolved.append(claim)

    return resolved


def _paper_index(papers: list[Paper], key) -> dict[str, list[Paper]]:
    index: dict[str, list[Paper]] = {}
    for paper in papers:
        value = key(paper)
        if value:
            index.setdefault(value, []).append(paper)
    return index


def _lookup_text(value: Any) -> str:
    return " ".join(_text(value).casefold().split())


def _lookup_url(value: Any) -> str:
    url = _lookup_text(value).rstrip("/")
    if url.startswith("http://"):
        url = f"https://{url[7:]}"
    return url


def _lookup_title(value: Any) -> str:
    return "".join(character for character in _lookup_text(value) if character.isalnum())


def _confidence_label(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    normalized = text.upper()
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
        return normalized
    try:
        score = float(text)
    except ValueError:
        return normalized
    if score > 1:
        score /= 100
    if score >= 0.9:
        return "HIGH"
    if score >= 0.75:
        return "MEDIUM"
    return "LOW"


def _normalize_gaps(data: Any) -> list[ResearchGap]:
    raw_gaps = data.get("gaps", []) if isinstance(data, dict) else []
    if not isinstance(raw_gaps, list):
        return []
    gaps: list[ResearchGap] = []
    for index, item in enumerate(raw_gaps, start=1):
        if not isinstance(item, dict):
            continue
        links: list[GapEvidenceLink] = []
        raw_links = item.get("evidence_links", [])
        if isinstance(raw_links, list):
            for raw_link in raw_links:
                if isinstance(raw_link, str):
                    citation_id = _text(raw_link)
                    relationship = ""
                    rationale = ""
                elif isinstance(raw_link, dict):
                    citation_id = _text(
                        raw_link.get("citation_id") or raw_link.get("claim_id")
                    )
                    relationship = _gap_relationship(
                        raw_link.get("relationship") or raw_link.get("role")
                    )
                    rationale = _text(raw_link.get("rationale"))
                else:
                    continue
                if citation_id:
                    links.append(
                        GapEvidenceLink(
                            citation_id=citation_id,
                            relationship=relationship,
                            rationale=rationale,
                        )
                    )
        gaps.append(
            ResearchGap(
                gap_id=_text(item.get("gap_id") or item.get("id") or f"G{index}"),
                title=_text(item.get("title"))
                or _text(item.get("description"))
                or f"Gap {index}",
                description=_text(item.get("description")),
                gap_type=_text(item.get("gap_type") or item.get("type")),
                demand=_int_or_none(item.get("demand")),
                coverage=_int_or_none(item.get("coverage")),
                gap_score=_float_or_none(item.get("gap_score")),
                evidence_links=_deduplicate_gap_links(links),
                why_it_matters=_text(item.get("why_it_matters")),
                potential_direction=_text(item.get("potential_direction")),
            )
        )
    return gaps


def _resolve_gap_evidence(
    gaps: list[ResearchGap],
    claims: list[EvidenceClaim],
    warnings: list[DashboardWarning],
    *,
    gap_path: Path,
) -> list[ResearchGap]:
    claim_ids = {claim.citation_id for claim in claims}
    resolved: list[ResearchGap] = []
    valid_relationships = {"supports", "partial_coverage", "challenges"}
    for gap in gaps:
        valid_links: list[GapEvidenceLink] = []
        for link in gap.evidence_links:
            if link.relationship not in valid_relationships:
                warnings.append(
                    DashboardWarning(
                        code="INVALID_GAP_EVIDENCE_RELATIONSHIP",
                        message=(
                            f"Research gap {gap.gap_id} uses unsupported evidence "
                            f"relationship {link.relationship!r}; no graph edge was drawn."
                        ),
                        path=str(gap_path),
                    )
                )
                continue
            if len(link.rationale.strip()) < 12:
                warnings.append(
                    DashboardWarning(
                        code="INVALID_GAP_EVIDENCE_RATIONALE",
                        message=(
                            f"Research gap {gap.gap_id} has no substantive rationale "
                            f"for evidence claim {link.citation_id}; no graph edge was drawn."
                        ),
                        path=str(gap_path),
                    )
                )
                continue
            if link.citation_id in claim_ids:
                valid_links.append(link)
                continue
            warnings.append(
                DashboardWarning(
                    code="UNMAPPED_GAP_EVIDENCE",
                    message=(
                        f"Research gap {gap.gap_id} references unknown evidence "
                        f"claim {link.citation_id}; no graph edge was drawn."
                    ),
                    path=str(gap_path),
                )
            )
        resolved.append(replace(gap, evidence_links=valid_links))
    return resolved


def _gap_relationship(value: Any) -> str:
    return _text(value)


def _deduplicate_gap_links(links: list[GapEvidenceLink]) -> list[GapEvidenceLink]:
    result: list[GapEvidenceLink] = []
    seen: set[str] = set()
    for link in links:
        if link.citation_id in seen:
            continue
        seen.add(link.citation_id)
        result.append(link)
    return result


def _normalize_ideas(data: Any) -> list[ResearchIdea]:
    raw_ideas = data.get("ideas", []) if isinstance(data, dict) else []
    if not isinstance(raw_ideas, list):
        return []
    ideas: list[ResearchIdea] = []
    for index, item in enumerate(raw_ideas, start=1):
        if not isinstance(item, dict):
            continue
        self_assessment = (
            item.get("self_assessment")
            if isinstance(item.get("self_assessment"), dict)
            else {}
        )
        target_gaps = _gap_ids(
            item.get("target_gaps") or item.get("gap_addressed")
        )
        supporting_evidence = _string_list(item.get("supporting_evidence"))
        evidence_grounding = item.get("evidence_grounding")
        if isinstance(evidence_grounding, list):
            supporting_evidence.extend(
                _text(entry.get("citation_id") or entry.get("claim_id"))
                for entry in evidence_grounding
                if isinstance(entry, dict)
                and _text(entry.get("citation_id") or entry.get("claim_id"))
            )
        ideas.append(
            ResearchIdea(
                idea_id=_text(item.get("idea_id") or item.get("id") or f"I{index}"),
                title=_text(item.get("title")) or f"Idea {index}",
                one_liner=_text(item.get("one_liner")),
                description=_text(item.get("description")),
                target_gaps=target_gaps,
                supporting_evidence=_deduplicate(supporting_evidence),
                composite_score=_float_or_none(
                    item.get("composite_score")
                    if item.get("composite_score") is not None
                    else self_assessment.get("composite_score")
                ),
                self_assessment=self_assessment,
            )
        )
    return ideas


def _normalize_design_axes(data: Any) -> list[DesignAxis]:
    raw_axes = data.get("axes", []) if isinstance(data, dict) else []
    if not isinstance(raw_axes, list):
        return []
    axes: list[DesignAxis] = []
    for item in raw_axes:
        if not isinstance(item, dict):
            continue
        axes.append(
            DesignAxis(
                name=_text(item.get("name")),
                description=_text(item.get("description")),
                values=_string_list(item.get("values")),
                explored=_string_list(item.get("explored")),
                unexplored=_string_list(item.get("unexplored")),
            )
        )
    return axes


def _build_graph(
    papers: list[Paper],
    claims: list[EvidenceClaim],
    gaps: list[ResearchGap],
    ideas: list[ResearchIdea],
) -> ResearchGraph:
    nodes: dict[str, GraphNode] = {}
    edges: set[tuple[str, str, str, str]] = set()

    def add_node(node: GraphNode) -> None:
        nodes.setdefault(node.id, node)

    paper_ids = {paper.paper_id for paper in papers}

    for paper in papers:
        add_node(
            GraphNode(
                id=f"paper:{paper.paper_id}",
                label=paper.title,
                kind="paper",
                group=paper.source or "paper",
            )
        )

    for claim in claims:
        claim_node = f"claim:{claim.citation_id}"
        add_node(GraphNode(id=claim_node, label=claim.citation_id, kind="claim", group=claim.confidence))
        if claim.source_paper_id in paper_ids:
            paper_node = f"paper:{claim.source_paper_id}"
            edges.add((paper_node, claim_node, "supports", ""))

    gap_edge_kinds = {
        "supports": "supports_gap",
        "partial_coverage": "partially_covers_gap",
        "challenges": "challenges_gap",
    }
    for gap in gaps:
        gap_node = f"gap:{gap.gap_id}"
        add_node(
            GraphNode(
                id=gap_node,
                label=gap.title or gap.gap_id,
                kind="gap",
                group=gap.gap_type or "gap",
            )
        )
        for link in gap.evidence_links:
            claim_node = f"claim:{link.citation_id}"
            if claim_node not in nodes:
                continue
            edge_kind = gap_edge_kinds[link.relationship]
            edges.add((claim_node, gap_node, edge_kind, link.rationale))

    for idea in ideas:
        idea_node = f"idea:{idea.idea_id}"
        add_node(GraphNode(id=idea_node, label=idea.title, kind="idea", group="idea"))
        for citation_id in idea.supporting_evidence:
            claim_node = f"claim:{citation_id}"
            add_node(GraphNode(id=claim_node, label=citation_id, kind="claim", group="referenced"))
            edges.add((claim_node, idea_node, "evidence_for", ""))
        for gap in idea.target_gaps:
            gap_node = f"gap:{gap}"
            add_node(GraphNode(id=gap_node, label=gap, kind="gap", group="gap"))
            edges.add((gap_node, idea_node, "targets", ""))

    return ResearchGraph(
        nodes=sorted(nodes.values(), key=lambda node: (node.kind, node.id)),
        edges=[
            GraphEdge(source=source, target=target, kind=kind, detail=detail)
            for source, target, kind, detail in sorted(edges)
        ],
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _gap_ids(value: Any) -> list[str]:
    values = _string_list(value)
    gaps: list[str] = []
    for item in values:
        matches = re.findall(r"\bG\d+\b", item, flags=re.IGNORECASE)
        if matches:
            gaps.extend(match.upper() for match in matches)
        else:
            gaps.append(item)
    return _deduplicate(gaps)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
