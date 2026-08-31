"""File-in/file-out batch task tools for long AutoIdea stages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


STAGE_SEARCH = "stage_3_search"
STAGE_READING = "stage_3_5_reading"
STAGE_EVIDENCE = "stage_6_evidence"

SUMMARY_CHAR_LIMIT = 1500
TERMINAL_BATCH_STATUSES = {"passed", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace() -> Path:
    from autoidea.paths import get_active_workspace

    return Path(get_active_workspace())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_workspace()))
    except ValueError:
        return str(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_json_array(text: str, name: str) -> list[Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a JSON array")
    return data


def _parse_json_object(text: str, name: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    return data


def _manifest_path() -> Path:
    return _workspace() / "batch_manifest.json"


def _load_manifest() -> dict[str, Any]:
    manifest = _read_json(_manifest_path(), {})
    if not isinstance(manifest, dict):
        manifest = {}
    manifest.setdefault("version", 1)
    manifest.setdefault("updated_at", _now())
    manifest.setdefault("batches", [])
    manifest["summary"] = _compute_summary(manifest["batches"])
    return manifest


def _save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    manifest["summary"] = _compute_summary(manifest.get("batches", []))
    _write_json(_manifest_path(), manifest)


def _compute_summary(batches: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total_batches": len(batches),
        "pending": 0,
        "running": 0,
        "passed": 0,
        "failed": 0,
    }
    for batch in batches:
        status = str(batch.get("status") or "pending")
        if status in counts:
            counts[status] += 1
    return counts


def _batch_dir(stage: str, index: int) -> Path:
    return _workspace() / "batches" / stage / f"batch_{index:03d}"


def _batch_dir_from_id(batch_id: str) -> tuple[str, int, Path]:
    for stage in (STAGE_READING, STAGE_EVIDENCE, STAGE_SEARCH):
        prefix = f"{stage}_batch_"
        if batch_id.startswith(prefix):
            index_text = batch_id[len(prefix):]
            try:
                index = int(index_text)
            except ValueError as exc:
                raise ValueError(f"invalid batch id index: {batch_id}") from exc
            return stage, index, _batch_dir(stage, index)
    raise ValueError(f"unknown batch id: {batch_id}")


def _batch_id_label(batch: dict[str, Any]) -> str:
    return str(batch.get("batch_id") or "unknown")


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]


def _create_batches(stage: str, chunks: list[list[Any]], payload_key: str, extra: dict[str, Any]) -> str:
    manifest = _load_manifest()
    selected: list[str] = []
    reused = 0
    existing_batches = [
        batch
        for batch in manifest.get("batches", [])
        if isinstance(batch, dict) and str(batch.get("stage")) == stage
    ]
    used_indices: set[int] = set()
    for batch in existing_batches:
        batch_id = str(batch.get("batch_id") or "")
        try:
            _, index, _ = _batch_dir_from_id(batch_id)
        except ValueError:
            continue
        used_indices.add(index)

    next_index = max(used_indices, default=0) + 1
    for items in chunks:
        matching_batch_id = ""
        for batch in existing_batches:
            candidate_id = str(batch.get("batch_id") or "")
            try:
                _, _, candidate_dir = _batch_dir_from_id(candidate_id)
            except ValueError:
                continue
            candidate_input = _read_json(candidate_dir / "input.json", {})
            if not isinstance(candidate_input, dict):
                continue
            if (
                candidate_input.get("stage") == stage
                and candidate_input.get(payload_key) == items
                and all(candidate_input.get(key) == value for key, value in extra.items())
            ):
                matching_batch_id = candidate_id
                break

        if matching_batch_id:
            selected.append(matching_batch_id)
            reused += 1
            continue

        while next_index in used_indices:
            next_index += 1
        index = next_index
        next_index += 1
        used_indices.add(index)
        batch_id = f"{stage}_batch_{index:03d}"
        directory = _batch_dir(stage, index)
        directory.mkdir(parents=True, exist_ok=True)
        input_data = {
            "batch_id": batch_id,
            "stage": stage,
            payload_key: items,
            **extra,
        }
        status_data = {
            "batch_id": batch_id,
            "stage": stage,
            "status": "pending",
            "created_at": _now(),
            "started_at": "",
            "finished_at": "",
            "error": "",
            "output_files": [],
        }
        input_path = directory / "input.json"
        status_path = directory / "status.json"
        _write_json(input_path, input_data)
        _write_json(status_path, status_data)

        manifest_entry = {
            "batch_id": batch_id,
            "stage": stage,
            "status": "pending",
            "input_file": _rel(input_path),
            "status_file": _rel(status_path),
            "result_file": "",
            "summary_file": "",
            "summary_preview": "",
            "error": "",
            "item_count": len(items),
        }
        manifest.setdefault("batches", []).append(manifest_entry)
        existing_batches.append(manifest_entry)
        selected.append(batch_id)

    _save_manifest(manifest)
    if reused:
        return (
            f"Prepared {len(selected)} {stage} batch(es): {', '.join(selected)} "
            f"({len(selected) - reused} new, {reused} reused without resetting status). "
            "Process only pending batches and write outputs with record_batch_result."
        )
    return (
        f"Created {len(selected)} {stage} batch(es): {', '.join(selected)}. "
        "Inputs are stored on disk; process one batch at a time and write "
        "outputs with record_batch_result."
    )


@tool(parse_docstring=True)
def create_search_batches(
    queries_json: str,
    batch_size: int = 3,
    sources_json: str = '["semantic_scholar", "arxiv", "openalex"]',
    max_results_per_query: int = 8,
) -> str:
    """Create file-backed Stage 3 literature search batches.

    Args:
        queries_json: JSON array of search query strings.
        batch_size: Number of queries per batch.
        sources_json: JSON array of source names to search.
        max_results_per_query: Maximum desired results for each query.

    Returns:
        Short status listing created batch IDs.
    """
    try:
        queries = [str(item) for item in _parse_json_array(queries_json, "queries_json")]
        sources = [str(item) for item in _parse_json_array(sources_json, "sources_json")]
        return _create_batches(
            STAGE_SEARCH,
            _chunk(queries, batch_size),
            "queries",
            {
                "sources": sources,
                "max_results_per_query": max_results_per_query,
            },
        )
    except Exception as exc:
        return f"Error creating search batches: {exc}"


@tool(parse_docstring=True)
def create_reading_batches(paper_ids_json: str, batch_size: int = 5) -> str:
    """Create file-backed Stage 3.5 reading batches.

    Args:
        paper_ids_json: JSON array of canonical paper IDs, e.g. ["P1", "P2"].
        batch_size: Number of papers per batch.

    Returns:
        Short status listing created batch IDs.
    """
    try:
        paper_ids = [str(item) for item in _parse_json_array(paper_ids_json, "paper_ids_json")]
        return _create_batches(STAGE_READING, _chunk(paper_ids, batch_size), "paper_ids", {})
    except Exception as exc:
        return f"Error creating reading batches: {exc}"


@tool(parse_docstring=True)
def create_evidence_batches(
    paper_ids_json: str,
    batch_size: int = 5,
    claim_budget_per_paper: int = 3,
) -> str:
    """Create file-backed Stage 6 evidence extraction batches.

    Args:
        paper_ids_json: JSON array of canonical paper IDs, e.g. ["P1", "P2"].
        batch_size: Number of papers per batch.
        claim_budget_per_paper: Maximum claims to extract per paper.

    Returns:
        Short status listing created batch IDs.
    """
    try:
        paper_ids = [str(item) for item in _parse_json_array(paper_ids_json, "paper_ids_json")]
        return _create_batches(
            STAGE_EVIDENCE,
            _chunk(paper_ids, batch_size),
            "paper_ids",
            {"claim_budget_per_paper": claim_budget_per_paper},
        )
    except Exception as exc:
        return f"Error creating evidence batches: {exc}"


def _validate_result_payload(stage: str, status: str, result_data: dict[str, Any]) -> None:
    if status != "passed":
        return

    if stage == STAGE_SEARCH:
        papers = result_data.get("papers")
        if not isinstance(papers, list):
            if "notable_titles" in result_data:
                raise ValueError(
                    "passed stage_3_search results must include a structured "
                    "`papers` list. `notable_titles` is only a progress note and "
                    "cannot be merged into paper_registry.json."
                )
            raise ValueError("passed stage_3_search results must include a `papers` list")
        for index, paper in enumerate(papers, start=1):
            if not isinstance(paper, dict):
                raise ValueError(f"stage_3_search papers[{index}] must be an object")
            if not str(paper.get("title") or "").strip():
                raise ValueError(f"stage_3_search papers[{index}] is missing `title`")
        return

    if stage == STAGE_READING:
        readings = result_data.get("readings")
        if not isinstance(readings, list):
            raise ValueError("passed stage_3_5_reading results must include a `readings` list")
        audit_records = _load_fulltext_audit_records()
        for index, reading in enumerate(readings, start=1):
            if not isinstance(reading, dict):
                raise ValueError(f"stage_3_5_reading readings[{index}] must be an object")
            if not str(reading.get("paper_id") or "").strip():
                raise ValueError(f"stage_3_5_reading readings[{index}] is missing `paper_id`")
            if not str(reading.get("summary") or "").strip():
                raise ValueError(f"stage_3_5_reading readings[{index}] is missing `summary`")
            _validate_reading_has_fulltext_attempt(index, reading, audit_records)
        return

    if stage == STAGE_EVIDENCE:
        claims = result_data.get("claims")
        if not isinstance(claims, list):
            raise ValueError("passed stage_6_evidence results must include a `claims` list")
        for index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                raise ValueError(f"stage_6_evidence claims[{index}] must be an object")
            if not str(claim.get("claim_text") or claim.get("claim") or "").strip():
                raise ValueError(f"stage_6_evidence claims[{index}] is missing `claim_text` or `claim`")
            if not str(claim.get("source_title") or "").strip():
                raise ValueError(f"stage_6_evidence claims[{index}] is missing `source_title`")
            if not str(claim.get("source_url") or "").strip():
                raise ValueError(f"stage_6_evidence claims[{index}] is missing `source_url`")
            if not str(claim.get("source_paper_id") or claim.get("paper_id") or "").strip():
                raise ValueError(f"stage_6_evidence claims[{index}] is missing `paper_id` or `source_paper_id`")
            if claim.get("confidence") is None:
                raise ValueError(f"stage_6_evidence claims[{index}] is missing `confidence`")


def _load_fulltext_audit_records() -> list[dict[str, Any]]:
    audit_path = _workspace() / "fulltext_audit.json"
    data = _read_json(audit_path, {})
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _normalize_for_match(value: str) -> str:
    import re

    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_arxiv_ids(value: str) -> set[str]:
    import re

    text = value or ""
    ids: set[str] = set()
    for match in re.finditer(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", text, re.I):
        ids.add(match.group(1).lower())
    for match in re.finditer(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", text, re.I):
        ids.add(match.group(1).lower())
    return ids


def _reading_match_candidates(reading: dict[str, Any]) -> list[str]:
    candidate_fields = (
        "title",
        "paper_id",
        "url",
        "pdf_url",
        "source_identifier",
        "source_url",
        "identifier",
        "arxiv_id",
        "doi",
        "audit_traceability_note",
    )
    candidates = [str(reading.get(field) or "") for field in candidate_fields]
    candidates.extend(_registry_match_candidates_for_reading(reading))
    return candidates


def _registry_match_candidates_for_reading(reading: dict[str, Any]) -> list[str]:
    paper_id = str(reading.get("paper_id") or "").strip()
    if not paper_id:
        return []
    data = _read_json(_workspace() / "paper_registry.json", [])
    items = data.get("papers") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    candidates: list[str] = []
    for item in items:
        if not isinstance(item, dict) or str(item.get("paper_id") or "") != paper_id:
            continue
        for field in (
            "title",
            "url",
            "paper_url",
            "pdf_url",
            "source_url",
            "source_identifier",
            "arxiv_id",
            "doi",
        ):
            value = str(item.get(field) or "").strip()
            if value:
                candidates.append(value)
        break
    return candidates


def _record_match_candidates(record: dict[str, Any]) -> list[str]:
    candidate_fields = (
        "identifier",
        "pdf_url",
        "source_url",
        "resolved_url",
        "url",
        "arxiv_id",
        "doi",
        "text_path",
    )
    return [str(record.get(field) or "") for field in candidate_fields]


def _reading_matches_audit_record(reading: dict[str, Any], record: dict[str, Any]) -> bool:
    record_candidates = _record_match_candidates(record)
    record_arxiv_ids = set().union(*(_extract_arxiv_ids(raw) for raw in record_candidates))
    reading_arxiv_ids = set().union(*(_extract_arxiv_ids(raw) for raw in _reading_match_candidates(reading)))
    if record_arxiv_ids and reading_arxiv_ids and record_arxiv_ids & reading_arxiv_ids:
        return True

    identifiers = [_normalize_for_match(raw) for raw in record_candidates]
    identifiers = [identifier for identifier in identifiers if identifier]
    if not identifiers:
        return False
    for candidate_raw in _reading_match_candidates(reading):
        candidate = _normalize_for_match(candidate_raw)
        if not candidate:
            continue
        for identifier in identifiers:
            if identifier == candidate or identifier in candidate or candidate in identifier:
                return True
    return False


def _validate_reading_has_fulltext_attempt(
    index: int,
    reading: dict[str, Any],
    audit_records: list[dict[str, Any]],
) -> None:
    status = str(reading.get("fulltext_status") or "").strip().upper()
    if status not in {"FULL-TEXT", "ABSTRACT-ONLY"}:
        raise ValueError(
            f"stage_3_5_reading readings[{index}] must set fulltext_status "
            'to "FULL-TEXT" or "ABSTRACT-ONLY"'
        )

    matches = [
        record for record in audit_records
        if _reading_matches_audit_record(reading, record)
    ]
    if not matches:
        raise ValueError(
            f"stage_3_5_reading readings[{index}] has no matching "
            "fulltext_audit.json record. Call fetch_paper_fulltext for this "
            "paper before recording the batch as passed."
        )

    if status == "FULL-TEXT":
        if not any(record.get("status") == "full_text" for record in matches):
            raise ValueError(
                f"stage_3_5_reading readings[{index}] claims FULL-TEXT, "
                "but fulltext_audit.json has no successful full_text record."
            )
        return

    if not any(
        record.get("status") != "full_text" and str(record.get("reason") or "").strip()
        for record in matches
    ):
        raise ValueError(
            f"stage_3_5_reading readings[{index}] claims ABSTRACT-ONLY, "
            "but fulltext_audit.json has no failed full-text attempt with a reason."
        )


@tool(parse_docstring=True)
def record_batch_result(
    batch_id: str,
    status: str,
    result_json: str,
    summary: str,
    error: str = "",
) -> str:
    """Record a batch result to disk and update the batch manifest.

    Args:
        batch_id: Batch ID such as stage_3_search_batch_001.
        status: Final status, either "passed" or "failed".
        result_json: JSON object containing detailed batch output.
        summary: Concise human-readable summary. Long summaries are truncated.
        error: Error text for failed batches.

    Returns:
        Short status with output file paths.
    """
    try:
        if status not in {"passed", "failed"}:
            raise ValueError('status must be "passed" or "failed"')
        stage, _, directory = _batch_dir_from_id(batch_id)
        if not directory.exists():
            raise ValueError(f"batch directory does not exist for {batch_id}")
        result_data = _parse_json_object(result_json, "result_json")
        _validate_result_payload(stage, status, result_data)
        short_summary = summary[:SUMMARY_CHAR_LIMIT]
        if len(summary) > SUMMARY_CHAR_LIMIT:
            short_summary += "\n\n... [summary truncated]"

        result_path = directory / "result.json"
        summary_path = directory / "summary.md"
        status_path = directory / "status.json"
        _write_json(result_path, result_data)
        summary_path.write_text(short_summary, encoding="utf-8")

        status_data = _read_json(status_path, {})
        if not isinstance(status_data, dict):
            status_data = {}
        status_data.update(
            {
                "batch_id": batch_id,
                "stage": stage,
                "status": status,
                "finished_at": _now(),
                "error": error,
                "output_files": [_rel(result_path), _rel(summary_path)],
            }
        )
        _write_json(status_path, status_data)

        manifest = _load_manifest()
        for entry in manifest.get("batches", []):
            if entry.get("batch_id") == batch_id:
                entry.update(
                    {
                        "status": status,
                        "result_file": _rel(result_path),
                        "summary_file": _rel(summary_path),
                        "summary_preview": short_summary[:500],
                        "error": error,
                    }
                )
                break
        else:
            manifest.setdefault("batches", []).append(
                {
                    "batch_id": batch_id,
                    "stage": stage,
                    "status": status,
                    "input_file": _rel(directory / "input.json"),
                    "status_file": _rel(status_path),
                    "result_file": _rel(result_path),
                    "summary_file": _rel(summary_path),
                    "summary_preview": short_summary[:500],
                    "error": error,
                    "item_count": 0,
                }
            )
        _save_manifest(manifest)
        return (
            f"Recorded {batch_id} as {status}. "
            f"result_file={_rel(result_path)}, summary_file={_rel(summary_path)}"
        )
    except Exception as exc:
        return f"Error recording batch result: {exc}"


@tool(parse_docstring=True)
def read_batch_manifest(stage: str = "") -> str:
    """Read a concise batch manifest without raw result payloads.

    Args:
        stage: Optional stage filter: stage_3_search, stage_3_5_reading,
            or stage_6_evidence. Empty string returns all stages.

    Returns:
        Compact markdown summary of batch status and output paths.
    """
    manifest = _load_manifest()
    batches = [
        batch for batch in manifest.get("batches", [])
        if isinstance(batch, dict) and (not stage or batch.get("stage") == stage)
    ]
    summary = _compute_summary(batches)
    lines = [
        "# Batch Manifest",
        "",
        f"- total_batches: {summary['total_batches']}",
        f"- pending: {summary['pending']}",
        f"- running: {summary['running']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        "",
        "## Batches",
    ]
    for batch in batches:
        lines.append(
            f"- {batch.get('batch_id')} [{batch.get('status')}] "
            f"items={batch.get('item_count', 0)} "
            f"result={batch.get('result_file') or 'not-written'} "
            f"summary={batch.get('summary_file') or 'not-written'}"
        )
        preview = str(batch.get("summary_preview") or "").strip()
        if preview:
            lines.append(f"  summary_preview: {preview[:500]}")
        error = str(batch.get("error") or "").strip()
        if error:
            lines.append(f"  error: {error[:300]}")
    return "\n".join(lines)


def _passed_batch_results(stage: str) -> list[dict[str, Any]]:
    manifest = _load_manifest()
    results: list[dict[str, Any]] = []
    for batch in manifest.get("batches", []):
        if not isinstance(batch, dict):
            continue
        if batch.get("stage") != stage or batch.get("status") != "passed":
            continue
        result_file = str(batch.get("result_file") or "")
        if not result_file:
            continue
        result = _read_json(_workspace() / result_file, {})
        if isinstance(result, dict):
            results.append({"batch": batch, "result": result})
    return results


def _passed_batch_count(stage: str) -> int:
    manifest = _load_manifest()
    return sum(
        1
        for batch in manifest.get("batches", [])
        if isinstance(batch, dict)
        and batch.get("stage") == stage
        and batch.get("status") == "passed"
    )


def _unfinished_batch_ids(stage: str) -> list[str]:
    manifest = _load_manifest()
    ids: list[str] = []
    for batch in manifest.get("batches", []):
        if not isinstance(batch, dict) or batch.get("stage") != stage:
            continue
        status = str(batch.get("status") or "pending")
        if status not in TERMINAL_BATCH_STATUSES:
            ids.append(_batch_id_label(batch))
    return ids


def _raise_if_unfinished_batches(stage: str) -> None:
    unfinished = _unfinished_batch_ids(stage)
    if unfinished:
        raise ValueError(
            f"unfinished {stage} batch(es) must be recorded as passed or failed "
            "before merging: "
            + ", ".join(unfinished)
        )


def _normalize_title(title: str) -> str:
    import re

    text = title.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_reading_summary(summary: Any) -> str:
    import re

    lines: list[str] = []
    for raw_line in str(summary or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+\[P\d+\](?:\s|$)", stripped):
            continue
        if re.match(r"^-?\s*\*\*Full-text status\*\*\s*:", stripped, re.I):
            continue
        lines.append(line)

    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _format_reading_summary(summary: Any) -> str:
    if not isinstance(summary, dict):
        return _clean_reading_summary(summary)

    ordered_fields = (
        "Core Problem",
        "Method / Architecture",
        "Main Results",
        "Limitations",
        "Relevance to LVU Agent Idea Search",
        "Relevance to Our Research",
    )
    lines: list[str] = []
    seen: set[str] = set()
    for field in ordered_fields:
        value = summary.get(field)
        if value is None:
            continue
        text = _clean_reading_summary(value)
        if not text:
            continue
        lines.append(f"- **{field}**: {text}")
        seen.add(field)

    for field, value in summary.items():
        if field in seen:
            continue
        text = _clean_reading_summary(value)
        if not text:
            continue
        lines.append(f"- **{field}**: {text}")

    return "\n".join(lines).strip()


@tool(parse_docstring=True)
def merge_search_batches() -> str:
    """Merge passed Stage 3 search batch results into canonical artifacts.

    Returns:
        Short merge status. Failed batches are skipped.
    """
    try:
        _raise_if_unfinished_batches(STAGE_SEARCH)
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        passed_results = _passed_batch_results(STAGE_SEARCH)
        invalid_batches: list[str] = []
        for item in passed_results:
            batch = item["batch"]
            papers = item["result"].get("papers")
            if not isinstance(papers, list):
                invalid_batches.append(_batch_id_label(batch))
                continue
            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                title = str(paper.get("title") or "").strip()
                if not title:
                    continue
                key = _normalize_title(title)
                if key in seen:
                    continue
                seen.add(key)
                paper_id = f"P{len(merged) + 1}"
                merged.append(
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "authors": paper.get("authors") or [],
                        "year": paper.get("year"),
                        "venue": paper.get("venue") or paper.get("source") or "",
                        "url": paper.get("url") or paper.get("pdf_url") or "",
                        "source": paper.get("source") or "",
                        "relevance": paper.get("relevance") or "",
                    }
                )

        if invalid_batches:
            raise ValueError(
                "passed search batch(es) missing required `papers` list: "
                + ", ".join(invalid_batches)
                + ". Re-run those batches with structured paper objects before merging."
            )
        if _passed_batch_count(STAGE_SEARCH) > 0 and not merged:
            raise ValueError(
                "0 structured paper(s) found in passed Stage 3 search batches. "
                "Refusing to overwrite paper_registry.json or literature_survey.md."
            )

        _write_json(_workspace() / "paper_registry.json", merged)
        lines = [
            "# Literature Survey",
            "",
            "| ID | Paper | Year | Source | Relevance |",
            "|---|---|---:|---|---|",
        ]
        for paper in merged:
            lines.append(
                f"| [{paper['paper_id']}] | **{paper['title']}** | "
                f"{paper.get('year') or ''} | {paper.get('source') or ''} | "
                f"{paper.get('relevance') or ''} |"
            )
        (_workspace() / "literature_survey.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return f"Merged {len(merged)} search paper(s) into paper_registry.json and literature_survey.md."
    except Exception as exc:
        return f"Error merging search batches: {exc}"


@tool(parse_docstring=True)
def merge_reading_batches() -> str:
    """Merge passed Stage 3.5 reading batch results into paper_deep_reading.md.

    Returns:
        Short merge status. Failed batches are skipped.
    """
    try:
        _raise_if_unfinished_batches(STAGE_READING)
        readings: list[dict[str, Any]] = []
        for item in _passed_batch_results(STAGE_READING):
            batch_readings = item["result"].get("readings", [])
            if isinstance(batch_readings, list):
                readings.extend(entry for entry in batch_readings if isinstance(entry, dict))

        fulltext_count = sum(
            1 for item in readings
            if str(item.get("fulltext_status") or "").upper() == "FULL-TEXT"
        )
        abstract_count = sum(
            1 for item in readings
            if str(item.get("fulltext_status") or "").upper() == "ABSTRACT-ONLY"
        )
        lines = [
            "# Paper Deep Reading Summary",
            f"- **Total papers selected**: {len(readings)}",
            f"- **Full-text extracted**: {fulltext_count}",
            f"- **Abstract-only fallback**: {abstract_count}",
            "",
        ]
        for item in readings:
            paper_id = item.get("paper_id") or "UNKNOWN"
            title = item.get("title") or "Untitled"
            status = item.get("fulltext_status") or "UNKNOWN"
            summary = _format_reading_summary(item.get("summary") or "")
            lines.extend(
                [
                    f"## [{paper_id}] {title}",
                    f"- **Full-text status**: {status}",
                    "",
                    str(summary).strip(),
                    "",
                ]
            )
        (_workspace() / "paper_deep_reading.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        return f"Merged {len(readings)} reading item(s) into paper_deep_reading.md."
    except Exception as exc:
        return f"Error merging reading batches: {exc}"


@tool(parse_docstring=True)
def merge_evidence_batches() -> str:
    """Merge passed Stage 6 evidence batch results into evidence_db.json.

    Returns:
        Short merge status. Failed batches are skipped.
    """
    try:
        _raise_if_unfinished_batches(STAGE_EVIDENCE)
        claims: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in _passed_batch_results(STAGE_EVIDENCE):
            batch_claims = item["result"].get("claims", [])
            if not isinstance(batch_claims, list):
                continue
            for claim in batch_claims:
                if not isinstance(claim, dict):
                    continue
                key = str(claim.get("claim_id") or claim.get("claim_text") or "")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                claims.append(claim)

        normalized_claims: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        for index, claim in enumerate(claims, start=1):
            citation_id = f"C{index}"
            claim_text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
            paper_id = str(claim.get("source_paper_id") or claim.get("paper_id") or "").strip()
            source_title = str(claim.get("source_title") or "").strip()
            source_url = str(claim.get("source_url") or "").strip()
            evidence_type = str(claim.get("evidence_type") or "paraphrase").strip()
            section = str(claim.get("section") or "").strip()
            confidence = float(claim.get("confidence") or 0)

            normalized = dict(claim)
            normalized["citation_id"] = citation_id
            normalized["claim"] = claim_text
            normalized["claim_text"] = claim_text
            normalized["source_paper_id"] = paper_id
            normalized["paper_id"] = paper_id
            normalized["source_title"] = source_title
            normalized["source_url"] = source_url
            normalized["evidence_type"] = evidence_type
            normalized["confidence"] = round(confidence, 2)
            if section:
                normalized["section"] = section
            normalized_claims.append(normalized)

            citations.append(
                {
                    "citation_id": citation_id,
                    "claim": claim_text,
                    "source_title": source_title,
                    "source_url": source_url,
                    "paper_id": paper_id,
                    "evidence_type": evidence_type,
                    "confidence": round(confidence, 2),
                    "section": section,
                    "is_mock": False,
                    "verified": bool(section and confidence >= 0.8),
                }
            )

        claims = normalized_claims
        high = sum(1 for claim in claims if float(claim.get("confidence") or 0) >= 0.8)
        medium = sum(1 for claim in claims if 0.5 <= float(claim.get("confidence") or 0) < 0.8)
        low = sum(1 for claim in claims if float(claim.get("confidence") or 0) < 0.5)
        citation_ids = {
            str(claim.get("citation_id"))
            for claim in claims
            if str(claim.get("citation_id") or "").strip()
        }
        evidence = {
            "claims": claims,
            "summary": {
                "total_claims": len(claims),
                "citations_count": len(citation_ids),
                "high_confidence": high,
                "medium_confidence": medium,
                "low_confidence": low,
            },
            "metadata": {
                "source": "merged stage_6_evidence batch outputs",
                "citation_id_policy": (
                    "merge_evidence_batches canonical Cn IDs; each ID resolves "
                    "to evidence_db.json and citations.json records with "
                    "source_title/source_url."
                ),
                "updated_at": _now(),
            },
        }
        _write_json(_workspace() / "evidence_db.json", evidence)
        _write_json(_workspace() / "citations.json", citations)
        return f"Merged {len(claims)} evidence claim(s) into evidence_db.json."
    except Exception as exc:
        return f"Error merging evidence batches: {exc}"
