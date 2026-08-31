"""Pydantic v2 data models for AutoIdea.

Defines structured schemas for the 12-stage research pipeline:
- Research brief, task formalization
- Paper analysis (Critique-First 6-dimension review)
- Evidence binding, gap identification
- Idea generation (with self-assessment)
- Debate protocol (ATTACK/DEFEND/RE-EVAL)
- Feasibility assessment
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Stage 0.5: Seed Idea Analysis
# =============================================================================


class SeedIdeaAssessment(BaseModel):
    """Assessment of a single seed idea in Stage 0.5."""

    model_config = {"extra": "forbid"}

    title: str = Field(description="Title or topic of the seed idea.")
    source_file: str = Field(default="", description="Path to the source file.")
    core_concepts: list[str] = Field(
        default_factory=list,
        description="Core research concepts identified.",
    )
    methods: list[str] = Field(
        default_factory=list,
        description="Methods or techniques mentioned.",
    )
    hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypotheses or expected outcomes.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Research gaps or open questions.",
    )
    clarity: str = Field(
        default="MEDIUM",
        description="Clarity assessment: HIGH / MEDIUM / LOW.",
    )
    novelty_potential: str = Field(
        default="MEDIUM",
        description="Novelty potential: HIGH / MEDIUM / LOW.",
    )
    feasibility: str = Field(
        default="MEDIUM",
        description="Feasibility signals: HIGH / MEDIUM / LOW.",
    )
    key_uncertainties: list[str] = Field(
        default_factory=list,
        description="Key uncertainties needing literature validation.",
    )
    search_directions: list[str] = Field(
        default_factory=list,
        description="Recommended search queries for literature.",
    )


class SeedIdeaAnalysisReport(BaseModel):
    """Stage 0.5 output: structured analysis of user-provided seed ideas.

    Written as seed_idea_analysis.md.
    """

    model_config = {"extra": "forbid"}

    ideas_analyzed: int = Field(
        default=0,
        description="Number of seed ideas analyzed.",
    )
    primary_direction: str = Field(
        default="",
        description="Primary research direction summary.",
    )
    assessments: list[SeedIdeaAssessment] = Field(
        default_factory=list,
        description="Per-idea assessments.",
    )
    common_themes: list[str] = Field(
        default_factory=list,
        description="Common themes across seed ideas.",
    )
    complementary_directions: list[str] = Field(
        default_factory=list,
        description="How ideas might combine or complement each other.",
    )
    priority_ranking: list[str] = Field(
        default_factory=list,
        description="Ranked ideas from most to least promising.",
    )
    consolidated_search_strategy: list[str] = Field(
        default_factory=list,
        description="Merged keyword/query list for Stage 3.",
    )


# =============================================================================
# Stage 1 & 2: Research Brief and Task Formalization
# =============================================================================


class ResearchBrief(BaseModel):
    """Structured research brief from user requirements.

    Produced in Stage 1 after clarification loop.
    """

    model_config = {"extra": "forbid"}

    topic: str = Field(description="Main research topic or question.")
    domain: str = Field(default="", description="Research domain (e.g., NLP, CV, RL).")
    sub_field: str = Field(default="", description="Specific sub-field.")
    constraints: list[str] = Field(
        default_factory=list,
        description="Any constraints (compute budget, timeline, etc.).",
    )
    goals: list[str] = Field(
        default_factory=list,
        description="Research goals or desired outcomes.",
    )
    context: str = Field(
        default="",
        description="Additional context or motivation.",
    )


class TaskFormalization(BaseModel):
    """Structured task formalization from Stage 2.

    Converts the research brief into a formal task specification.
    """

    model_config = {"extra": "forbid"}

    problem_statement: str = Field(description="Precise problem statement.")
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Measurable success criteria.",
    )
    key_metrics: list[str] = Field(
        default_factory=list,
        description="Key metrics to evaluate solutions.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Technical or practical constraints.",
    )
    scope: str = Field(
        default="",
        description="Scope of the research (what is in/out).",
    )


# =============================================================================
# Stage 4: Critique-First Paper Analysis
# =============================================================================


class InitialAttack(BaseModel):
    """Critique-first analysis applied before any positive summary.

    Attributes:
        most_likely_wrong: What is most likely wrong with this work.
        rejection_reason: If reviewing, what would be the rejection reason.
        weakest_link: The weakest link in the evidence chain.
    """

    model_config = {"extra": "forbid"}

    most_likely_wrong: str = Field(
        description="What is most likely wrong with this work.",
    )
    rejection_reason: str = Field(
        default="",
        description="If reviewing, what would be the rejection reason.",
    )
    weakest_link: str = Field(
        default="",
        description="The weakest link in the evidence chain (core hook for idea generation).",
    )


class DimensionReview(BaseModel):
    """Single dimension of the 6-dimension critical review.

    Attributes:
        dimension: Name of the dimension (e.g. 'Methodology').
        strength: STRONG | MODERATE | SPECULATIVE.
        evidence: Specific evidence supporting the assessment.
        concern: Primary concern in this dimension.
    """

    model_config = {"extra": "forbid"}

    dimension: str = Field(description="Review dimension name.")
    strength: str = Field(
        description="STRONG | MODERATE | SPECULATIVE.",
    )
    evidence: str = Field(default="", description="Specific evidence.")
    concern: str = Field(default="", description="Primary concern.")


class ComparableWork(BaseModel):
    """Comparison to the most closely related prior work.

    Attributes:
        most_similar: Description of the most similar prior work.
        difference: How this paper differs from that work.
    """

    model_config = {"extra": "forbid"}

    most_similar: str = Field(
        default="",
        description="Description of the most similar prior work.",
    )
    difference: str = Field(
        default="",
        description="How this paper differs from that work.",
    )


class HookQuery(BaseModel):
    """A search query derived from a paper's weakest link.

    Hook queries are used to find additional evidence that can strengthen
    or refute the identified vulnerability.

    Attributes:
        source: Description of where this query originated, e.g.
            'weakest_link: temporal consistency not validated'.
        queries: Search queries to find more evidence.
    """

    model_config = {"extra": "forbid"}

    source: str = Field(
        description=(
            "Description of where this query originated, "
            "e.g. 'weakest_link: temporal consistency not validated'."
        ),
    )
    queries: list[str] = Field(
        default_factory=list,
        description="Search queries to find more evidence.",
    )


class PaperPosition(BaseModel):
    """Complete position-first analysis for a single paper.

    This is the output of the positioning-agent for each paper analyzed.
    """

    model_config = {"extra": "forbid"}

    paper_id: str = Field(description="Unique identifier for the paper.")
    paper_title: str = Field(description="Title of the paper.")
    paper_url: str = Field(default="", description="URL to the paper.")
    paper_year: Optional[str] = Field(default=None, description="Publication year.")
    paper_venue: Optional[str] = Field(
        default=None, description="Publication venue (journal/conference)."
    )
    abstract: Optional[str] = Field(default=None, description="Paper abstract.")

    # Critique-First (before any positive summary)
    initial_attack: InitialAttack = Field(
        description="Critique-first analysis applied before any positive summary."
    )

    # Structured analysis
    core_problem_solved: str = Field(
        description="What the paper actually solves (not what authors claim).",
    )
    non_problems: list[str] = Field(
        default_factory=list,
        description="Problems the paper does NOT solve.",
    )
    key_assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions the method relies on.",
    )
    strength: str = Field(
        default="",
        description="What actually works (with evidence).",
    )
    weakness: str = Field(
        default="",
        description="Concrete weaknesses.",
    )
    comparable_work: ComparableWork = Field(
        default_factory=ComparableWork,
        description="Comparison to the most similar prior work.",
    )

    # Hook-driven queries
    hook_queries: list[HookQuery] = Field(
        default_factory=list,
        description="Search queries driven by the weakest link.",
    )


# =============================================================================
# Stage 7: Knowledge Synthesis — OSMOSIS Gap Identification
# =============================================================================


class GapEvidenceLink(BaseModel):
    """A typed Claim-to-Gap relationship recorded during Stage 7."""

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    citation_id: str = Field(
        pattern=r"^C\d+$",
        description="Canonical evidence_db.json citation ID (for example C7).",
    )
    relationship: Literal["supports", "partial_coverage", "challenges"] = Field(
        description=(
            "How this evidence bears on the gap: supports its existence, "
            "partially covers it, or challenges the gap claim."
        ),
    )
    rationale: str = Field(
        min_length=12,
        description="Gap-specific explanation of why this Claim has that relationship.",
    )


class ResearchGap(BaseModel):
    """A research gap identified through OSMOSIS gap analysis.

    Gap_Score = Demand (1-5 importance) - Coverage (1-5 literature coverage).
    """

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    gap_id: str = Field(description="Unique gap identifier (e.g. G1, G2).")
    title: str = Field(min_length=1, description="Short human-readable gap title.")
    description: str = Field(min_length=1, description="Description of the gap.")
    gap_type: Literal[
        "methodology_gap",
        "evaluation_gap",
        "assumption_gap",
        "scope_gap",
    ] = Field(
        description="methodology_gap | evaluation_gap | assumption_gap | scope_gap",
    )
    demand: int = Field(
        ge=1,
        le=5,
        description="Importance of solving the gap on a 1-5 scale.",
    )
    coverage: int = Field(
        ge=1,
        le=5,
        description="How well existing literature covers the gap on a 1-5 scale.",
    )
    gap_score: float = Field(
        description="Demand - Coverage. Higher = more valuable.",
    )
    evidence_links: list[GapEvidenceLink] = Field(
        min_length=1,
        description="Explicit, typed links from canonical evidence Claims to this gap.",
    )
    why_it_matters: str = Field(
        min_length=1,
        description="Research importance of closing the gap.",
    )
    potential_direction: str = Field(
        min_length=1,
        description="A bounded direction suggested by the evidence, not a claimed result.",
    )
    supporting_papers: list[str] = Field(
        default_factory=list,
        description=(
            "Optional convenience list of paper IDs; Claim links remain the "
            "canonical provenance source."
        ),
    )

    @model_validator(mode="after")
    def validate_gap_consistency(self) -> "ResearchGap":
        if not re.fullmatch(r"G\d+", self.gap_id):
            raise ValueError("gap_id must use canonical form G<number>")
        expected = self.demand - self.coverage
        if abs(self.gap_score - expected) > 1e-9:
            raise ValueError(
                f"gap_score must equal demand - coverage ({expected})"
            )
        citation_ids = [link.citation_id for link in self.evidence_links]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("evidence_links must not repeat a citation_id within one gap")
        return self


class ResearchGapCatalog(BaseModel):
    """Canonical Stage 7 artifact written to ``research_gaps.json``."""

    model_config = {"extra": "forbid"}

    schema_version: Literal["1.0"] = "1.0"
    generated_from: Literal["evidence_db.json"] = "evidence_db.json"
    gaps: list[ResearchGap] = Field(
        min_length=3,
        description="Ranked research gaps with explicit evidence provenance.",
    )

    @model_validator(mode="after")
    def validate_unique_gap_ids(self) -> "ResearchGapCatalog":
        gap_ids = [gap.gap_id for gap in self.gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("gaps must use unique gap_id values")
        return self


# =============================================================================
# Stage 9: Idea Generation with Self-Assessment
# =============================================================================


class IdeaSelfAssessment(BaseModel):
    """Self-assessment scores for an idea.

    Every idea must include immediate self-assessment with justification.
    """

    model_config = {"extra": "forbid"}

    novelty: int = Field(
        description="Novelty score 1-5.",
        ge=1,
        le=5,
    )
    novelty_justification: str = Field(
        default="",
        description="Why this is novel.",
    )
    feasibility: int = Field(
        description="Feasibility score 1-5.",
        ge=1,
        le=5,
    )
    feasibility_justification: str = Field(
        default="",
        description="Why this is feasible.",
    )
    impact: int = Field(
        description="Impact score 1-5.",
        ge=1,
        le=5,
    )
    impact_justification: str = Field(
        default="",
        description="Why this matters.",
    )
    composite_score: float = Field(
        default=0.0,
        description="Weighted average of novelty, feasibility, impact.",
    )
    known_limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of this idea.",
    )


class ResearchIdea(BaseModel):
    """A structured research idea with self-assessment.

    Produced in Stage 9 by ideator-agent.
    """

    model_config = {"extra": "forbid"}

    idea_id: str = Field(description="Unique idea identifier (e.g. I1, I2).")
    title: str = Field(description="Concise title for the idea.")
    one_liner: str = Field(
        default="",
        description="One-sentence summary of the idea.",
    )
    description: str = Field(
        default="",
        description="Detailed description of the idea.",
    )
    methodology: str = Field(
        default="",
        description="Proposed methodology or approach.",
    )
    expected_outcome: str = Field(
        default="",
        description="Expected outcome or contribution.",
    )
    gap_trace: list[str] = Field(
        default_factory=list,
        description="Gap IDs this idea addresses (e.g. ['G1', 'G3']).",
    )
    self_assessment: Optional[IdeaSelfAssessment] = Field(
        default=None,
        description="Self-assessment scores.",
    )


# =============================================================================
# Stage 10: Adversarial Debate Protocol
# =============================================================================


class IdeaReview(BaseModel):
    """Critic's review of an idea (Stage 10 ATTACK).

    4-dimension scoring with verdict.
    """

    model_config = {"extra": "forbid"}

    idea_id: str = Field(description="ID of the idea being reviewed.")
    round_number: int = Field(
        default=1,
        description="Debate round number (1=ATTACK, 2=DEFEND, 3=RE-EVAL).",
    )
    role: str = Field(
        default="critic",
        description="Role in this round (critic | proposer).",
    )

    # 4-dimension scoring
    novelty_score: float = Field(
        default=0.0,
        description="Novelty assessment 1-5.",
    )
    feasibility_score: float = Field(
        default=0.0,
        description="Feasibility assessment 1-5.",
    )
    soundness_score: float = Field(
        default=0.0,
        description="Technical soundness 1-5.",
    )
    evaluation_score: float = Field(
        default=0.0,
        description="Evaluation plan quality 1-5.",
    )

    # Verdict
    verdict: str = Field(
        description="ACCEPT | REVISE | REJECT",
    )
    required_revisions: list[str] = Field(
        default_factory=list,
        description="Required revisions (if REVISE).",
    )
    rejection_reason: str = Field(
        default="",
        description="Reason for rejection (if REJECT).",
    )
    comments: str = Field(
        default="",
        description="Detailed review comments.",
    )


# =============================================================================
# Stage 11: Feasibility Assessment
# =============================================================================


class FeasibilityAssessment(BaseModel):
    """Quantified feasibility assessment for a research idea.

    Produced in Stage 11.
    """

    model_config = {"extra": "forbid"}

    idea_id: str = Field(description="ID of the idea being assessed.")
    gpu_hours_estimate: str = Field(
        default="",
        description="Estimated GPU-hours required.",
    )
    latency_estimate: str = Field(
        default="",
        description="Estimated inference latency.",
    )
    hardware_requirements: str = Field(
        default="",
        description="Hardware requirements (GPUs, memory, etc.).",
    )
    data_availability: str = Field(
        default="",
        description="Data availability assessment.",
    )
    timeline: str = Field(
        default="",
        description="Estimated implementation timeline.",
    )
    risk_severity: str = Field(
        default="",
        description="Risk severity (LOW | MEDIUM | HIGH | CRITICAL).",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Identified risk factors.",
    )
    mitigation_strategies: list[str] = Field(
        default_factory=list,
        description="Proposed risk mitigation strategies.",
    )


# =============================================================================
# Design Space (Stage 8)
# =============================================================================


class DesignAxis(BaseModel):
    """A single axis in the design space."""

    model_config = {"extra": "forbid"}

    name: str = Field(description="Axis name (e.g. 'architecture', 'training_method').")
    values: list[str] = Field(
        default_factory=list,
        description="Possible values along this axis.",
    )
    explored: list[str] = Field(
        default_factory=list,
        description="Values already explored in literature.",
    )
    unexplored: list[str] = Field(
        default_factory=list,
        description="Values not yet explored (research opportunities).",
    )


class PromisingCombination(BaseModel):
    """A promising combination in the design space."""

    model_config = {"extra": "forbid"}

    combination: dict[str, str] = Field(
        default_factory=dict,
        description="Axis -> value mapping for this combination.",
    )
    rationale: str = Field(
        default="",
        description="Why this combination is promising.",
    )
    gap_trace: list[str] = Field(
        default_factory=list,
        description="Related gap IDs.",
    )
