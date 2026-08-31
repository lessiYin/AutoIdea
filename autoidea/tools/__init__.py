"""Tools package — re-exports all public tool symbols."""

from .search import tavily_search, web_search, paper_lookup
from .think import think as think_tool, read_workspace_file, write_workspace_file
from .cite import cite_source
from .scholar import (
    semantic_scholar_search,
    semantic_scholar_get_paper,
    arxiv_search,
    arxiv_get_paper,
    openalex_search,
    dblp_search,
    crossref_search,
    crossref_resolve_doi,
    pubmed_search,
    cvf_search,
    list_found_papers,
    multi_source_search,
)
from .reranker import merge_and_rank_search_results
from .paper_content import fetch_paper_content, fetch_paper_section
from .idea_tournament import rank_ideas_tournament, generate_tournament_matchups
from .stage_gate import check_stage_gate, save_stage_reflection, list_stage_reflections
from .evo_memory import recall_ideation_memory, update_ideation_memory, get_memory_stats
from .seed_papers import list_seed_papers
from .seed_ideas import list_seed_ideas, get_search_keywords_from_seeds, generate_seed_idea_analysis_report
from .artifact_writers import (
    write_design_space,
    write_evidence_db,
    write_research_gaps,
    write_raw_ideas,
    write_tournament_rankings,
    write_idea_reviews,
)
from .batch_tasks import (
    create_search_batches,
    create_reading_batches,
    create_evidence_batches,
    record_batch_result,
    read_batch_manifest,
    merge_search_batches,
    merge_reading_batches,
    merge_evidence_batches,
)
from .pipeline_state import inspect_pipeline_state
from .heartbeat import write_run_status, read_run_status


def __getattr__(name: str):
    if name == "audit_workspace_artifacts":
        from .artifact_audit import audit_workspace_artifacts
        return audit_workspace_artifacts
    raise AttributeError(name)

__all__ = [
    "tavily_search",
    "web_search",
    "paper_lookup",
    "think_tool",
    "read_workspace_file",
    "write_workspace_file",
    "cite_source",
    "semantic_scholar_search",
    "semantic_scholar_get_paper",
    "arxiv_search",
    "arxiv_get_paper",
    "openalex_search",
    "dblp_search",
    "crossref_search",
    "crossref_resolve_doi",
    "pubmed_search",
    "cvf_search",
    "list_found_papers",
    "multi_source_search",
    "merge_and_rank_search_results",
    "fetch_paper_content",
    "fetch_paper_section",
    "rank_ideas_tournament",
    "generate_tournament_matchups",
    "check_stage_gate",
    "save_stage_reflection",
    "list_stage_reflections",
    "audit_workspace_artifacts",
    "recall_ideation_memory",
    "update_ideation_memory",
    "get_memory_stats",
    "list_seed_papers",
    "list_seed_ideas",
    "get_search_keywords_from_seeds",
    "generate_seed_idea_analysis_report",
    "write_design_space",
    "write_evidence_db",
    "write_research_gaps",
    "write_raw_ideas",
    "write_tournament_rankings",
    "write_idea_reviews",
    "create_search_batches",
    "create_reading_batches",
    "create_evidence_batches",
    "record_batch_result",
    "read_batch_manifest",
    "merge_search_batches",
    "merge_reading_batches",
    "merge_evidence_batches",
    "inspect_pipeline_state",
    "write_run_status",
    "read_run_status",
]
