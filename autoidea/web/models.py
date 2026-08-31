"""Typed dashboard models for normalized AutoIdea workspace artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardWarning:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class WorkspaceInfo:
    path: str
    name: str


@dataclass(frozen=True)
class ArtifactInfo:
    path: str
    kind: str
    size_bytes: int
    title: str


@dataclass(frozen=True)
class ArtifactContent:
    path: str
    kind: str
    size_bytes: int
    title: str
    text: str
    html: str = ""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: str
    prompt: str
    workspace: str
    run_name: str = ""
    mode: str = "new"
    parent_run_id: str = ""
    model: str = ""
    provider: str = ""
    thread_id: str = ""
    seed_papers: str = ""
    seed_ideas: str = ""
    auto_approve: bool = True
    show_thinking: bool = True
    pipeline_parameters: dict[str, Any] = field(default_factory=dict)
    pid: int | None = None
    exit_code: int | None = None
    started_at: str = ""
    finished_at: str = ""
    log_path: str = ""
    events_path: str = ""
    response_dir: str = ""
    log_tail: str = ""
    command: list[str] = field(default_factory=list)
    interaction: dict[str, Any] | None = None
    status_detail: str = ""
    current_stage: str = ""
    completed_stages: int = 0
    total_stages: int = 14
    progress: dict[str, Any] = field(default_factory=dict)
    completion: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensionPosition:
    dimension: str
    strength: str
    evidence: str = ""
    concern: str = ""


@dataclass(frozen=True)
class PaperPosition:
    initial_attack: str = ""
    weakest_link: str = ""
    summary: str = ""
    dimensions: list[DimensionPosition] = field(default_factory=list)


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    year: int | None = None
    source: str = ""
    venue: str = ""
    url: str = ""
    authors: list[str] = field(default_factory=list)
    relevance: str = ""
    position: PaperPosition | None = None


@dataclass(frozen=True)
class EvidenceClaim:
    citation_id: str
    claim: str
    source_paper_id: str = ""
    source_title: str = ""
    source_url: str = ""
    source_match: str = ""
    confidence: str = ""
    evidence_type: str = ""
    section: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GapEvidenceLink:
    citation_id: str
    relationship: str = "supports"
    rationale: str = ""


@dataclass(frozen=True)
class ResearchGap:
    gap_id: str
    title: str
    description: str = ""
    gap_type: str = ""
    demand: int | None = None
    coverage: int | None = None
    gap_score: float | None = None
    evidence_links: list[GapEvidenceLink] = field(default_factory=list)
    why_it_matters: str = ""
    potential_direction: str = ""


@dataclass(frozen=True)
class ResearchIdea:
    idea_id: str
    title: str
    one_liner: str = ""
    description: str = ""
    target_gaps: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    composite_score: float | None = None
    self_assessment: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignAxis:
    name: str
    description: str = ""
    values: list[str] = field(default_factory=list)
    explored: list[str] = field(default_factory=list)
    unexplored: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    kind: str
    group: str = ""


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class ResearchGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass(frozen=True)
class WorkspaceSnapshot:
    workspace: WorkspaceInfo
    counts: dict[str, int]
    artifacts: list[ArtifactInfo]
    papers: list[Paper]
    claims: list[EvidenceClaim]
    gaps: list[ResearchGap]
    ideas: list[ResearchIdea]
    design_axes: list[DesignAxis]
    graph: ResearchGraph
    pipeline: dict[str, Any]
    warnings: list[DashboardWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def artifact_title(path: Path) -> str:
    """Convert an artifact filename to a compact display title."""
    return path.name.replace("_", " ").replace(".json", "").replace(".md", "").title()
