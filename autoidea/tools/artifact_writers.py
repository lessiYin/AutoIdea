"""Dedicated writers for canonical AutoIdea pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import ValidationError

from autoidea.schema import ResearchGapCatalog

from .think import write_workspace_file


def _parse_json_object(content: str, artifact_name: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON for {artifact_name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{artifact_name} must be a JSON object")
    return data


def _require_keys(data: dict[str, Any], artifact_name: str, required: tuple[str, ...]) -> None:
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{artifact_name} missing required key(s): {', '.join(missing)}")


def _write_json_artifact(file_path: str, content: str, required: tuple[str, ...]) -> str:
    try:
        data = _parse_json_object(content, file_path)
        _require_keys(data, file_path, required)
        normalized = json.dumps(data, ensure_ascii=False, indent=2)
        return write_workspace_file.invoke(
            {"file_path": file_path, "content": normalized}
        )
    except Exception as exc:
        return f"Error writing {file_path}: {exc}"


@tool(parse_docstring=True)
def write_design_space(content: str) -> str:
    """Validate and write the canonical ``design_space.json`` artifact.

    Use this instead of ``write_workspace_file`` when saving Stage 8 design
    space output. The file path is fixed, so the model cannot omit or mistype it.

    Args:
        content: JSON object content for design_space.json.

    Returns:
        Confirmation message or validation error.
    """
    return _write_json_artifact(
        "design_space.json",
        content,
        ("axes", "promising_combinations"),
    )


@tool(parse_docstring=True)
def write_evidence_db(content: str) -> str:
    """Validate and write the canonical ``evidence_db.json`` artifact.

    Use this instead of ``write_workspace_file`` when saving Stage 6 evidence.
    The file path is fixed, so the model cannot omit or mistype it.

    Args:
        content: JSON object content for evidence_db.json.

    Returns:
        Confirmation message or validation error.
    """
    return _write_json_artifact("evidence_db.json", content, ("claims", "summary"))


@tool(parse_docstring=True)
def write_research_gaps(content: str) -> str:
    """Validate and write the canonical ``research_gaps.json`` artifact.

    Every Stage 7 gap must contain typed links to citation IDs that already
    exist in ``evidence_db.json``. The fixed schema makes Claim-to-Gap
    provenance machine-readable instead of inferring it from Markdown.

    Args:
        content: JSON object content for research_gaps.json.

    Returns:
        Confirmation message or validation error.
    """
    artifact_name = "research_gaps.json"
    try:
        data = _parse_json_object(content, artifact_name)
        catalog = ResearchGapCatalog.model_validate(data)

        from autoidea.paths import get_active_workspace

        evidence_path = Path(get_active_workspace()) / "evidence_db.json"
        if not evidence_path.is_file():
            raise ValueError(
                "evidence_db.json must exist before research_gaps.json is written"
            )
        evidence = _parse_json_object(
            evidence_path.read_text(encoding="utf-8"),
            "evidence_db.json",
        )
        claims = evidence.get("claims")
        if not isinstance(claims, list):
            raise ValueError("evidence_db.json must contain a claims list")
        citation_ids = {
            str(claim.get("citation_id") or "").strip()
            for claim in claims
            if isinstance(claim, dict) and claim.get("citation_id")
        }
        referenced_ids = {
            link.citation_id
            for gap in catalog.gaps
            for link in gap.evidence_links
        }
        missing = sorted(referenced_ids - citation_ids)
        if missing:
            raise ValueError(
                "evidence_links reference citation IDs missing from "
                f"evidence_db.json: {', '.join(missing)}"
            )

        normalized = json.dumps(
            catalog.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        return write_workspace_file.invoke(
            {"file_path": artifact_name, "content": normalized}
        )
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        return f"Error writing {artifact_name}: {exc}"


@tool(parse_docstring=True)
def write_raw_ideas(content: str) -> str:
    """Validate and write the canonical ``raw_ideas.json`` artifact.

    Use this instead of ``write_workspace_file`` when saving Stage 9 ideas.
    The file path is fixed, so the model cannot omit or mistype it.

    Args:
        content: JSON object content for raw_ideas.json.

    Returns:
        Confirmation message or validation error.
    """
    artifact_name = "raw_ideas.json"
    try:
        data = _parse_json_object(content, artifact_name)
        _require_keys(data, artifact_name, ("generated_count", "ideas"))
        if "kept_top_k" not in data and "kept_top_5" not in data:
            raise ValueError(
                "raw_ideas.json missing required key: kept_top_k "
                "(legacy kept_top_5 is also accepted)"
            )

        ideas = data.get("ideas")
        if not isinstance(ideas, list):
            raise ValueError("raw_ideas.json ideas must be a list")

        from autoidea.config import get_effective_config

        configured_max = max(
            1,
            int(getattr(get_effective_config(), "max_ideas_to_generate", 10) or 10),
        )
        try:
            generated_count = int(data.get("generated_count"))
        except (TypeError, ValueError) as exc:
            raise ValueError("raw_ideas.json generated_count must be an integer") from exc
        actual_count = len(ideas)
        if generated_count > configured_max or actual_count > configured_max:
            raise ValueError(
                "idea count exceeds configured max_ideas_to_generate="
                f"{configured_max} (generated_count={generated_count}, ideas={actual_count})"
            )

        normalized = json.dumps(data, ensure_ascii=False, indent=2)
        return write_workspace_file.invoke(
            {"file_path": artifact_name, "content": normalized}
        )
    except (TypeError, ValueError) as exc:
        return f"Error writing {artifact_name}: {exc}"


@tool(parse_docstring=True)
def write_tournament_rankings(content: str) -> str:
    """Validate and write the canonical ``tournament_rankings.json`` artifact.

    Use this instead of ``write_workspace_file`` when saving Stage 9.5
    tournament output. The file path is fixed, so the model cannot omit or
    mistype it.

    Args:
        content: JSON object content for tournament_rankings.json.

    Returns:
        Confirmation message or validation error.
    """
    return _write_json_artifact(
        "tournament_rankings.json",
        content,
        ("rankings",),
    )


@tool(parse_docstring=True)
def write_idea_reviews(content: str) -> str:
    """Validate and write the canonical ``idea_reviews.json`` artifact.

    Use this instead of ``write_workspace_file`` when saving Stage 10 critic
    reviews. The file path is fixed, so the model cannot omit or mistype it.

    Args:
        content: JSON object content for idea_reviews.json.

    Returns:
        Confirmation message or validation error.
    """
    return _write_json_artifact("idea_reviews.json", content, ("reviews",))
