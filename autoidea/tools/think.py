"""Reflection and workspace file tools for AutoIdea."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


_CANONICAL_PIPELINE_ARTIFACTS = {
    "seed_idea_analysis.md",
    "research_brief.md",
    "task_formalization.md",
    "paper_registry.json",
    "literature_survey.md",
    "paper_deep_reading.md",
    "paper_positions.json",
    "expanded_literature.md",
    "evidence_db.json",
    "knowledge_synthesis.md",
    "research_gaps.json",
    "design_space.json",
    "raw_ideas.json",
    "tournament_rankings.json",
    "elo_rankings.json",
    "debate_log.md",
    "idea_reviews.json",
    "feasibility_assessments.json",
    "final_report.md",
}

_PLACEHOLDER_ARTIFACT_CONTENT = {
    "dummy",
    "ok",
    "placeholder",
    "tbd",
    "test",
    "todo",
}


def _canonical_artifact_write_errors(clean_path: str, content: str) -> list[str]:
    name = clean_path.strip().lstrip("/")
    if name not in _CANONICAL_PIPELINE_ARTIFACTS:
        return []
    stripped = content.strip()
    if not stripped:
        return [f"{name} cannot be empty."]
    if stripped.lower() in _PLACEHOLDER_ARTIFACT_CONTENT:
        return [f"{name} contains placeholder/test content, not a real artifact."]
    return []


def _paper_positions_write_errors(content: str, workspace: Path) -> list[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]
    if not isinstance(data, list):
        return ["paper_positions.json must be a JSON list."]

    registry_ids: set[str] = set()
    registry_path = workspace / "paper_registry.json"
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = []
        if isinstance(registry, list):
            registry_ids = {
                str(item.get("paper_id"))
                for item in registry
                if isinstance(item, dict) and item.get("paper_id")
            }

    errors: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"entry {idx} must be an object.")
            continue
        pid = str(item.get("paper_id") or "")
        if not re.fullmatch(r"P\d+", pid):
            errors.append(
                f"entry {idx} has invalid paper_id `{pid}`; use one canonical Pn such as `P36`, never ranges like `P36-P93`."
            )
            continue
        if pid in seen:
            errors.append(f"duplicate paper_id `{pid}`.")
        seen.add(pid)
        if registry_ids and pid not in registry_ids:
            errors.append(f"paper_id `{pid}` is not present in paper_registry.json.")
    return errors


@tool(parse_docstring=True)
def think(reflection: str) -> str:
    """Tool for structured reflection and strategic decision-making.

    Use this tool to pause and reason carefully at any decision point. This creates
    a deliberate checkpoint for quality thinking.

    When to use:
    - Before starting work: What do I know? What prior knowledge is available?
    - After obtaining results: What did I learn? Does this change the approach?
    - When choosing between options: What are the trade-offs?
    - When stuck or failing: What went wrong? Should I try something different?
    - Before concluding: Is the evidence sufficient?

    Your reflection should address relevant dimensions:
    1. Progress - What has been accomplished? What concrete steps remain?
    2. Evidence quality - Is the current evidence sufficient?
    3. Prior knowledge - Have I checked memory/history?
    4. Strategy - Should I continue, adjust, or try something different?
    5. Handoff - Is this phase complete? What does the next phase need?

    Args:
        reflection: Your structured reflection text. Address the relevant
            dimensions above for your current situation.

    Returns:
        Acknowledgment that the reflection was recorded.
    """
    return f"Reflection recorded ({len(reflection)} chars). Continue with your plan."


@tool(parse_docstring=True)
def read_workspace_file(file_path: str, offset: int = 0, limit: int = 0) -> str:
    """Read a file from the workspace directory.

    Use this to read previously generated artifacts like research_brief.md,
    paper_positions.json, knowledge_synthesis.md, research_gaps.json,
    raw_ideas.json, etc.

    Args:
        file_path: Path to the file relative to workspace root, or absolute path.
        offset: Line offset (1-based) to start reading from. 0 means start from beginning.
        limit: Maximum number of lines to return. 0 means return all lines.

    Returns:
        File contents as string, or error message if file not found.
    """
    try:
        from autoidea.paths import get_active_workspace

        ws = get_active_workspace()
        ws_str = str(ws)

        # Normalize the file_path:
        # 1. If it's an absolute path that starts with the workspace dir,
        #    extract just the relative portion to avoid nested paths.
        # 2. Strip leading '/' to prevent treating workspace-relative paths
        #    as absolute paths (e.g. "/task_formalization.md" should resolve
        #    to workspace/task_formalization.md, not /task_formalization.md)
        if file_path.startswith(ws_str):
            # Absolute path containing workspace prefix -> extract relative part
            clean_path = file_path[len(ws_str):].lstrip("/")
            if not clean_path:
                clean_path = file_path.lstrip("/")
        else:
            clean_path = file_path.lstrip("/")

        # Search order:
        # 1. Relative to workspace root
        # 2. In output subdirectory
        # 3. As absolute path (only if original path was absolute)
        candidates = [
            Path(ws) / clean_path,
            Path(ws) / "output" / clean_path,
        ]
        # Only try absolute path if the original file_path was absolute
        if file_path.startswith("/"):
            candidates.append(Path(file_path))

        path = None
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

        if path is None:
            return (
                f"File not found: {file_path}\n"
                f"Searched locations:\n"
                + "\n".join(f"  - {c}" for c in candidates)
                + f"\n\nWorkspace root: {ws}"
            )

        content = path.read_text(encoding="utf-8")

        # Handle line offset and limit
        if offset > 0 or limit > 0:
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)

            if offset > 0:
                # Validate offset (1-based)
                if offset > total_lines:
                    return (
                        f"Error: Line offset {offset} exceeds file length "
                        f"({total_lines} lines). Use offset=0 to read from "
                        f"the beginning, or a value <= {total_lines}."
                    )
                start_idx = offset - 1  # Convert to 0-based
            else:
                start_idx = 0

            if limit > 0:
                end_idx = min(start_idx + limit, total_lines)
            else:
                end_idx = total_lines

            content = "".join(lines[start_idx:end_idx])
            if end_idx < total_lines:
                content += f"\n\n... [{total_lines - end_idx} more lines, total: {total_lines}]"

        if len(content) > 50000:
            content = content[:50000] + "\n\n... [truncated, file too large]"
        return content
    except Exception as e:
        return f"Error reading file {file_path}: {e}"


@tool(parse_docstring=True)
def write_workspace_file(
    file_path: str,
    content: str,
    mode: Literal["overwrite", "append", "replace"] = "overwrite",
    old_text: str = "",
) -> str:
    """Write, append, or precisely replace content in a workspace file.

    Use this to create or overwrite workspace artifacts like research_brief.md,
    task_formalization.md, paper_positions.json, raw_ideas.json, etc.

    IMPORTANT: The content will be post-processed to convert literal '\\n'
    sequences into actual newlines, since LLM outputs sometimes escape
    newline characters as the two-character sequence backslash-n.

    Args:
        file_path: Path to the file relative to workspace root.
        content: The content to write to the file.
        mode: ``overwrite`` replaces the file; ``append`` adds content exactly
            as supplied; ``replace`` substitutes one exact ``old_text`` match.
            Append and replace are available only for Markdown files.
        old_text: Existing text to replace when mode is ``replace``. It must
            occur exactly once so ambiguous edits cannot corrupt an artifact.

    Returns:
        Confirmation message with file path and size, or error message.
    """
    try:
        from autoidea.paths import get_active_workspace

        ws = get_active_workspace()
        ws_str = str(ws)

        # Normalize the file_path:
        # 1. If it's an absolute path that starts with the workspace dir,
        #    extract just the relative portion to avoid nested paths
        #    (e.g. "/path/to/workspace/research_brief.md" -> "research_brief.md")
        # 2. Strip leading '/' to prevent treating workspace-relative paths
        #    as absolute paths
        if file_path.startswith(ws_str):
            clean_path = file_path[len(ws_str):].lstrip("/")
            if not clean_path:
                clean_path = file_path.lstrip("/")
        else:
            clean_path = file_path.lstrip("/")
        target = Path(ws) / clean_path

        if mode in {"append", "replace"} and target.suffix.lower() != ".md":
            return (
                "Error writing file: append and replace modes are supported "
                "only for Markdown files."
            )

        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        # ── Content post-processing ──────────────────────────────────
        # LLMs sometimes emit literal two-character sequences '\\n' instead
        # of actual newline characters.  Detect and fix this pattern.
        # Heuristic: if the content has very few real newlines but many
        # literal '\\n' sequences, it's almost certainly an escaping issue.
        real_newlines = content.count("\n")
        escaped_newlines = content.count("\\n")
        if escaped_newlines > 0 and (
            real_newlines == 0
            or escaped_newlines > real_newlines * 3
        ):
            logger.info(
                "write_workspace_file: fixing escaped newlines "
                "(%d literal '\\\\n' vs %d real newlines) in %s",
                escaped_newlines, real_newlines, file_path,
            )
            content = content.replace("\\n", "\n")

        final_content = content
        if mode == "append" and target.exists():
            final_content = target.read_text(encoding="utf-8") + content
        elif mode == "replace":
            if not target.exists():
                return f"Error writing file: cannot replace text in missing file {clean_path}."
            if not old_text:
                return "Error writing file: old_text is required in replace mode."
            existing_content = target.read_text(encoding="utf-8")
            match_count = existing_content.count(old_text)
            if match_count != 1:
                return (
                    "Error writing file: replace mode requires old_text to occur "
                    f"exactly once; found {match_count} matches in {clean_path}."
                )
            final_content = existing_content.replace(old_text, content, 1)

        if clean_path == "paper_positions.json":
            validation_errors = _paper_positions_write_errors(final_content, Path(ws))
            if validation_errors:
                details = "\n".join(f"- {error}" for error in validation_errors[:10])
                return (
                    "Error writing paper_positions.json: validation failed.\n"
                    f"{details}"
                )

        validation_errors = _canonical_artifact_write_errors(clean_path, final_content)
        if validation_errors:
            details = "\n".join(f"- {error}" for error in validation_errors[:10])
            return (
                f"Error writing {clean_path}: validation failed.\n"
                f"{details}"
            )

        target.write_text(final_content, encoding="utf-8")

        size = target.stat().st_size
        line_count = final_content.count("\n") + 1
        action = {
            "append": "appended",
            "replace": "updated",
        }.get(mode, "written")
        return (
            f"File {action} successfully: {clean_path}\n"
            f"Size: {size} bytes, {line_count} lines\n"
            f"Full path: {target}"
        )
    except Exception as e:
        return f"Error writing file {file_path}: {e}"
