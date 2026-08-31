"""Workspace artifact integrity audit for AutoIdea.

The audit checks the files that the research pipeline writes, not the
agent's self-reported counts.  It is intentionally conservative: errors are
conditions that can corrupt a final report, while warnings flag weaker
reproducibility issues.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool
from pydantic import ValidationError

from autoidea.schema import ResearchGapCatalog


class AuditSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class AuditIssue:
    severity: AuditSeverity
    code: str
    message: str
    path: str = ""


@dataclass
class AuditReport:
    workspace: str
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == AuditSeverity.ERROR for issue in self.issues)

    def add(self, severity: AuditSeverity, code: str, message: str, path: str = "") -> None:
        self.issues.append(AuditIssue(severity=severity, code=code, message=message, path=path))

    def to_dict(self) -> dict[str, Any]:
        errors = sum(1 for issue in self.issues if issue.severity == AuditSeverity.ERROR)
        warnings = sum(1 for issue in self.issues if issue.severity == AuditSeverity.WARNING)
        return {
            "workspace": self.workspace,
            "status": "FAIL" if errors else "PASS",
            "errors": errors,
            "warnings": warnings,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "code": issue.code,
                    "message": issue.message,
                    "path": issue.path,
                }
                for issue in self.issues
            ],
        }

    def to_markdown(self) -> str:
        data = self.to_dict()
        lines = [
            "# AutoIdea Artifact Audit",
            f"- **Workspace**: `{self.workspace}`",
            f"- **Status**: {data['status']}",
            f"- **Errors**: {data['errors']}",
            f"- **Warnings**: {data['warnings']}",
            "",
        ]
        if not self.issues:
            lines.append("No artifact integrity issues found.")
            return "\n".join(lines)
        lines.append("## Issues")
        for issue in self.issues:
            suffix = f" (`{issue.path}`)" if issue.path else ""
            lines.append(f"- **{issue.severity.value} {issue.code}**{suffix}: {issue.message}")
        return "\n".join(lines)


def normalize_title(title: str) -> str:
    text = html.unescape(title or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_matches(expected: str, actual: str) -> bool:
    exp = normalize_title(expected)
    act = normalize_title(actual)
    if not exp or not act:
        return False
    if exp == act or exp in act or act in exp:
        return True
    exp_words = set(exp.split())
    act_words = set(act.split())
    if not exp_words or not act_words:
        return False
    overlap = len(exp_words & act_words) / max(len(exp_words), len(act_words))
    return overlap >= 0.72


def extract_arxiv_id(url: str) -> str | None:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", url or "", re.I)
    return match.group(1) if match else None


def extract_title_from_arxiv_html(text: str) -> str | None:
    patterns = [
        r'<meta\s+name=["\']citation_title["\']\s+content=["\']([^"\']+)["\']',
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
        r"<title>\[\d{4}\.\d{4,5}\]\s*([^<]+)</title>",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.I)
        if match:
            return html.unescape(match.group(1)).strip()
    return None


_RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
_ARXIV_VERIFY_WORKERS = 8
_ARXIV_VERIFY_REQUEST_TIMEOUT_SECONDS = 4.0
_ARXIV_VERIFY_TOTAL_TIMEOUT_SECONDS = 12.0


def fetch_url_text(
    url: str,
    timeout: float = 15.0,
    *,
    attempts: int = 3,
    retry_delay: float = 1.0,
) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "arxiv.org":
        raise ValueError("Artifact verification only fetches HTTPS URLs from arxiv.org.")
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")
    req = Request(url, headers={"User-Agent": "AutoIdeaArtifactAudit/1.0"})
    for attempt in range(attempts):
        try:
            # The URL scheme and exact host are restricted above.
            with urlopen(req, timeout=timeout) as response:  # nosec B310
                raw = response.read(1_000_000)
            return raw.decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_STATUS or attempt == attempts - 1:
                raise
        except (TimeoutError, URLError):
            if attempt == attempts - 1:
                raise
        time.sleep(retry_delay * (2**attempt))

    raise RuntimeError("arXiv fetch retry loop exited unexpectedly.")


def fetch_arxiv_title(url: str, fetcher: Callable[[str], str] | None = None) -> str | None:
    arxiv_id = extract_arxiv_id(url)
    if not arxiv_id:
        return None
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    text = (fetcher or fetch_url_text)(abs_url)
    return extract_title_from_arxiv_html(text)


def _fetch_arxiv_titles(
    urls: list[str],
    *,
    fetcher: Callable[[str], str] | None = None,
) -> dict[str, tuple[str | None, str | None]]:
    """Fetch each distinct arXiv paper once using bounded concurrency.

    Evidence databases commonly contain several claims from one paper. Fetching
    the same page once per claim made a valid Stage 6 gate exceed the global
    five-minute tool deadline whenever arXiv was slow. Results are keyed by the
    canonical arXiv identifier so versioned ``/abs`` and ``/pdf`` URLs share a
    request. The entire remote check has a hard deadline; unavailable results
    are returned to the audit as unverifiable rather than blocking completion.
    """
    canonical_urls: dict[str, str] = {}
    for url in urls:
        arxiv_id = extract_arxiv_id(url)
        if arxiv_id:
            canonical_urls.setdefault(arxiv_id, f"https://arxiv.org/abs/{arxiv_id}")

    if not canonical_urls:
        return {}

    def fetch_one(arxiv_id: str, url: str) -> tuple[str, str | None, str | None]:
        try:
            if fetcher is None:
                text = fetch_url_text(
                    url,
                    timeout=_ARXIV_VERIFY_REQUEST_TIMEOUT_SECONDS,
                    attempts=1,
                )
                title = extract_title_from_arxiv_html(text)
            else:
                title = fetch_arxiv_title(url, fetcher=fetcher)
            return arxiv_id, title, None
        except Exception as exc:  # noqa: BLE001 - report each remote verification failure
            return arxiv_id, None, f"{type(exc).__name__}: {exc}"

    tasks: Queue[tuple[str, str]] = Queue()
    responses: Queue[tuple[str, str | None, str | None]] = Queue()
    stop = Event()
    for item in canonical_urls.items():
        tasks.put(item)

    def worker() -> None:
        while not stop.is_set():
            try:
                arxiv_id, url = tasks.get_nowait()
            except Empty:
                return
            responses.put(fetch_one(arxiv_id, url))

    workers = min(_ARXIV_VERIFY_WORKERS, len(canonical_urls))
    for index in range(workers):
        Thread(
            target=worker,
            name=f"autoidea-arxiv-audit-{index + 1}",
            daemon=True,
        ).start()

    results: dict[str, tuple[str | None, str | None]] = {}
    remaining = set(canonical_urls)
    deadline = time.monotonic() + _ARXIV_VERIFY_TOTAL_TIMEOUT_SECONDS
    while remaining:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            break
        try:
            arxiv_id, title, error = responses.get(timeout=timeout)
        except Empty:
            break
        results[arxiv_id] = (title, error)
        remaining.discard(arxiv_id)

    stop.set()
    for arxiv_id in remaining:
        results[arxiv_id] = (
            None,
            "TimeoutError: arXiv title verification exceeded the "
            f"{_ARXIV_VERIFY_TOTAL_TIMEOUT_SECONDS:g}s overall deadline.",
        )
    return results


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(report: AuditReport, path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception as exc:
        report.add(AuditSeverity.ERROR, "JSON_INVALID", f"Could not parse JSON: {exc}", str(path))
        return None


def _extract_survey_papers(text: str) -> dict[str, str]:
    papers: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        title = ""
        pid = ""

        table = re.match(r"^\|\s*\[(P\d+)\]\s*\|\s*(.+?)\s*\|", stripped)
        ranked = re.match(r"^\d+\.\s+\*\*\[(P\d+)\]\s*([^*]+)\*\*", stripped)
        bullet = re.match(r"^[-*]\s+\*\*\[(P\d+)\]\s*([^*]+)\*\*", stripped)
        ranked_bold_id = re.match(r"^\d+\.\s+\*\*\[(P\d+)\]\*\*\s+(.+)$", stripped)
        bullet_bold_id = re.match(r"^[-*]\s+\*\*\[(P\d+)\]\*\*\s+(.+)$", stripped)
        heading = re.match(r"^#{1,6}\s+\[(P\d+)\]\s+(.+)$", stripped)

        if table:
            pid = table.group(1)
            title = table.group(2)
        elif ranked:
            pid = ranked.group(1)
            title = ranked.group(2)
        elif bullet:
            pid = bullet.group(1)
            title = bullet.group(2)
        elif ranked_bold_id:
            pid = ranked_bold_id.group(1)
            title = ranked_bold_id.group(2)
        elif bullet_bold_id:
            pid = bullet_bold_id.group(1)
            title = bullet_bold_id.group(2)
        elif heading:
            pid = heading.group(1)
            title = heading.group(2)
        else:
            continue

        title = re.sub(r"\*\*", "", title)
        title = re.sub(r"<[^>]+>", "", title)
        title = title.split("|")[0]
        title = title.split("—")[0]
        title = title.split(" - ")[0]
        title = re.sub(r"\s*\((?:19|20)\d{2}(?:/\d{4})?\)\s*$", "", title).strip()
        if title and len(title) > 2:
            papers[pid] = title
    return papers


def _validate_stage3_artifacts(workspace: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    survey_path = workspace / "literature_survey.md"
    registry_path = workspace / "paper_registry.json"

    registry_data: Any = None
    registry_items: list[dict[str, Any]] = []
    if registry_path.exists():
        try:
            registry_data = _load_json(registry_path)
        except Exception as exc:
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "JSON_INVALID",
                    f"Could not parse JSON: {exc}",
                    str(registry_path),
                )
            )
        if registry_data is not None:
            if not isinstance(registry_data, list):
                issues.append(
                    AuditIssue(
                        AuditSeverity.ERROR,
                        "PAPER_REGISTRY_SCHEMA",
                        "paper_registry.json must be a list.",
                        str(registry_path),
                    )
                )
            else:
                registry_items = [item for item in registry_data if isinstance(item, dict)]
                if not registry_data:
                    issues.append(
                        AuditIssue(
                            AuditSeverity.ERROR,
                            "PAPER_REGISTRY_EMPTY",
                            "paper_registry.json must contain at least one paper.",
                            str(registry_path),
                        )
                    )
                elif len(registry_items) != len(registry_data):
                    issues.append(
                        AuditIssue(
                            AuditSeverity.ERROR,
                            "PAPER_REGISTRY_SCHEMA",
                            "Every paper_registry.json entry must be an object.",
                            str(registry_path),
                        )
                    )

    survey_papers: dict[str, str] = {}
    if survey_path.exists():
        text = survey_path.read_text(encoding="utf-8", errors="replace")
        survey_papers = _extract_survey_papers(text)
        if registry_items and not survey_papers:
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "LITERATURE_SURVEY_EMPTY",
                    "literature_survey.md does not contain any parseable [Pxx] paper entries while paper_registry.json is non-empty.",
                    str(survey_path),
                )
            )
        if registry_items and len(survey_papers) != len(registry_items):
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "REGISTRY_SURVEY_COUNT_MISMATCH",
                    f"paper_registry.json has {len(registry_items)} paper(s), but literature_survey.md has only {len(survey_papers)} parseable [Pxx] entries.",
                    str(survey_path),
                )
            )
    elif registry_items:
        issues.append(
            AuditIssue(
                AuditSeverity.ERROR,
                "LITERATURE_SURVEY_MISSING",
                "paper_registry.json exists but literature_survey.md is missing.",
                str(survey_path),
            )
        )

    manifest_path = workspace / "batch_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _load_json(manifest_path)
        except Exception as exc:
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "JSON_INVALID",
                    f"Could not parse JSON: {exc}",
                    str(manifest_path),
                )
            )
            manifest = {}
        batches = manifest.get("batches") if isinstance(manifest, dict) else None
        if isinstance(batches, list):
            for batch in batches:
                if not isinstance(batch, dict):
                    continue
                if batch.get("stage") != "stage_3_search" or batch.get("status") != "passed":
                    continue
                batch_id = str(batch.get("batch_id") or "unknown")
                result_file = str(batch.get("result_file") or "")
                if not result_file:
                    issues.append(
                        AuditIssue(
                            AuditSeverity.ERROR,
                            "BATCH_RESULT_SCHEMA",
                            f"Passed Stage 3 search batch {batch_id} has no result_file.",
                            str(manifest_path),
                        )
                    )
                    continue
                result_path = workspace / result_file
                if not result_path.exists():
                    issues.append(
                        AuditIssue(
                            AuditSeverity.ERROR,
                            "BATCH_RESULT_SCHEMA",
                            f"Passed Stage 3 search batch {batch_id} points to missing result_file `{result_file}`.",
                            str(manifest_path),
                        )
                    )
                    continue
                try:
                    result = _load_json(result_path)
                except Exception as exc:
                    issues.append(
                        AuditIssue(
                            AuditSeverity.ERROR,
                            "BATCH_RESULT_SCHEMA",
                            f"Passed Stage 3 search batch {batch_id} has invalid result JSON: {exc}",
                            str(result_path),
                        )
                    )
                    continue
                papers = result.get("papers") if isinstance(result, dict) else None
                if not isinstance(papers, list):
                    issues.append(
                        AuditIssue(
                            AuditSeverity.ERROR,
                            "BATCH_RESULT_SCHEMA",
                            f"Passed Stage 3 search batch {batch_id} is missing required `papers` list.",
                            str(result_path),
                        )
                    )
                elif any(not isinstance(item, dict) or not str(item.get("title") or "").strip() for item in papers):
                    issues.append(
                        AuditIssue(
                            AuditSeverity.ERROR,
                            "BATCH_RESULT_SCHEMA",
                            f"Passed Stage 3 search batch {batch_id} contains paper entries without titles.",
                            str(result_path),
                        )
                    )

    return issues


def _audit_incomplete_batches(workspace: Path, report: AuditReport) -> None:
    manifest_path = workspace / "batch_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _load_json_if_exists(report, manifest_path)
    batches = manifest.get("batches") if isinstance(manifest, dict) else None
    if not isinstance(batches, list):
        return

    stage_artifacts = {
        "stage_3_search": ["paper_registry.json", "literature_survey.md"],
        "stage_3_5_reading": ["paper_deep_reading.md"],
        "stage_6_evidence": ["evidence_db.json"],
    }
    terminal = {"passed", "failed"}
    for stage, artifact_names in stage_artifacts.items():
        if not any((workspace / name).exists() for name in artifact_names):
            continue
        unfinished = [
            str(batch.get("batch_id") or "unknown")
            for batch in batches
            if isinstance(batch, dict)
            and batch.get("stage") == stage
            and str(batch.get("status") or "pending") not in terminal
        ]
        if unfinished:
            report.add(
                AuditSeverity.ERROR,
                "BATCH_INCOMPLETE",
                (
                    f"{stage} canonical artifact(s) exist, but unfinished batch(es) "
                    f"remain: {', '.join(unfinished)}. Record each batch as passed "
                    "or failed before merging/proceeding."
                ),
                str(manifest_path),
            )


def validate_stage3_artifacts(workspace: str | Path) -> list[dict[str, str]]:
    """Return Stage 3 structural issues as plain dictionaries."""
    return [
        {
            "severity": issue.severity.value,
            "code": issue.code,
            "message": issue.message,
            "path": issue.path,
        }
        for issue in _validate_stage3_artifacts(Path(workspace).resolve())
    ]


def _paper_registry(workspace: Path, report: AuditReport) -> dict[str, dict[str, Any]]:
    path = workspace / "paper_registry.json"
    if not path.exists():
        if (workspace / "final_report.md").exists() or (workspace / "evidence_db.json").exists():
            report.add(
                AuditSeverity.ERROR,
                "PAPER_REGISTRY_MISSING",
                "paper_registry.json is missing; cross-stage [Pxx] references cannot be canonicalized.",
                str(path),
            )
        return {}
    data = _load_json_if_exists(report, path)
    if data is None:
        return {}
    if not isinstance(data, list):
        report.add(AuditSeverity.ERROR, "PAPER_REGISTRY_SCHEMA", "paper_registry.json must be a list.", str(path))
        return {}
    registry: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            report.add(AuditSeverity.ERROR, "PAPER_REGISTRY_SCHEMA", f"Entry {idx} is not an object.", str(path))
            continue
        pid = str(item.get("paper_id") or "")
        title = str(item.get("title") or "")
        if not re.fullmatch(r"P\d+", pid):
            report.add(AuditSeverity.ERROR, "PAPER_REGISTRY_SCHEMA", f"Entry {idx} has invalid paper_id `{pid}`.", str(path))
        if not title:
            report.add(AuditSeverity.ERROR, "PAPER_REGISTRY_SCHEMA", f"Entry {idx} is missing title.", str(path))
        if pid in registry:
            report.add(AuditSeverity.ERROR, "PAPER_REGISTRY_DUPLICATE", f"Duplicate paper_id `{pid}`.", str(path))
        if pid:
            registry[pid] = item
    return registry


def _audit_paper_ids(workspace: Path, report: AuditReport, registry: dict[str, dict[str, Any]]) -> None:
    survey_path = workspace / "literature_survey.md"
    survey: dict[str, str] = {}
    if survey_path.exists():
        survey = _extract_survey_papers(survey_path.read_text(encoding="utf-8", errors="replace"))
    if registry and survey:
        for pid, title in survey.items():
            if pid in registry and not title_matches(registry[pid].get("title", ""), title):
                report.add(
                    AuditSeverity.ERROR,
                    "PAPER_ID_DRIFT",
                    f"literature_survey.md maps {pid} to `{title}`, but paper_registry.json maps it to `{registry[pid].get('title')}`.",
                    str(survey_path),
                )

    positions_path = workspace / "paper_positions.json"
    positions = _load_json_if_exists(report, positions_path)
    if positions is not None and not isinstance(positions, list):
        report.add(AuditSeverity.ERROR, "PAPER_POSITIONS_SCHEMA", "paper_positions.json must be a list.", str(positions_path))
        return
    if isinstance(positions, list):
        seen: set[str] = set()
        for item in positions:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("paper_id") or "")
            title = str(item.get("title") or "")
            if not re.fullmatch(r"P\d+", pid):
                report.add(
                    AuditSeverity.ERROR,
                    "PAPER_POSITIONS_SCHEMA",
                    f"paper_positions.json has invalid paper_id `{pid}`; expected canonical form like `P1`, not `[P1]`.",
                    str(positions_path),
                )
            if pid in seen:
                report.add(AuditSeverity.ERROR, "PAPER_POSITIONS_DUPLICATE", f"Duplicate paper_id `{pid}` in paper_positions.json.", str(positions_path))
            seen.add(pid)
            if registry and pid and re.fullmatch(r"P\d+", pid) and pid not in registry:
                report.add(
                    AuditSeverity.ERROR,
                    "PAPER_ID_DRIFT",
                    f"paper_positions.json references `{pid}`, which is not present in paper_registry.json.",
                    str(positions_path),
                )
            if survey and pid in survey and title and not title_matches(survey[pid], title):
                report.add(
                    AuditSeverity.ERROR,
                    "PAPER_ID_DRIFT",
                    f"paper_positions.json maps {pid} to `{title}`, but literature_survey.md maps it to `{survey[pid]}`.",
                    str(positions_path),
                )
            if registry and pid in registry and title and not title_matches(registry[pid].get("title", ""), title):
                report.add(
                    AuditSeverity.ERROR,
                    "PAPER_ID_DRIFT",
                    f"paper_positions.json maps {pid} to `{title}`, but paper_registry.json maps it to `{registry[pid].get('title')}`.",
                    str(positions_path),
                )


def _extract_deep_reading_counts(text: str) -> dict[str, int]:
    labels = {
        "total": r"Total papers selected\*\*:\s*(\d+)",
        "full": r"Full-text extracted\*\*:\s*(\d+)",
        "abstract": r"Abstract-only fallback\*\*:\s*(\d+)",
    }
    counts: dict[str, int] = {}
    for key, pattern in labels.items():
        match = re.search(pattern, text, re.I)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _configured_deep_reading_top_k() -> int:
    import os

    env_value = os.getenv("AUTOIDEA_DEEP_READING_TOP_K")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    try:
        from autoidea.config import load_config

        return int(getattr(load_config(), "deep_reading_top_k", 20) or 20)
    except Exception:
        return 20


def _audit_deep_reading(
    workspace: Path,
    report: AuditReport,
    *,
    expected_top_k: int | None = None,
) -> None:
    path = workspace / "paper_deep_reading.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = _extract_deep_reading_counts(text)
    full_statuses = len(re.findall(r"Full-text status\*\*:\s*FULL-TEXT", text, re.I))
    abstract_statuses = len(re.findall(r"Full-text status\*\*:\s*ABSTRACT-ONLY", text, re.I))
    if counts.get("full") is not None and counts["full"] != full_statuses:
        report.add(AuditSeverity.ERROR, "DEEP_READING_COUNT_MISMATCH", f"Header full-text count {counts['full']} does not match {full_statuses} FULL-TEXT blocks.", str(path))
    if counts.get("abstract") is not None and counts["abstract"] != abstract_statuses:
        report.add(AuditSeverity.ERROR, "DEEP_READING_COUNT_MISMATCH", f"Header abstract-only count {counts['abstract']} does not match {abstract_statuses} ABSTRACT-ONLY blocks.", str(path))
    if counts.get("total") is not None and counts["total"] != full_statuses + abstract_statuses:
        report.add(AuditSeverity.ERROR, "DEEP_READING_COUNT_MISMATCH", f"Header total count {counts['total']} does not match {full_statuses + abstract_statuses} status blocks.", str(path))
    expected = (
        expected_top_k
        if isinstance(expected_top_k, int) and expected_top_k > 0
        else _configured_deep_reading_top_k()
    )
    actual = full_statuses + abstract_statuses
    if actual < expected:
        report.add(
            AuditSeverity.ERROR,
            "DEEP_READING_INCOMPLETE",
            f"paper_deep_reading.md has {actual} reading block(s), expected at least {expected} from deep_reading_top_k.",
            str(path),
        )
    if full_statuses:
        audit_path = workspace / "fulltext_audit.json"
        data = _load_json_if_exists(report, audit_path)
        records = data.get("records", data) if isinstance(data, dict) else data
        if not audit_path.exists() or not isinstance(records, list):
            report.add(AuditSeverity.ERROR, "FULLTEXT_AUDIT_MISSING", "paper_deep_reading.md claims FULL-TEXT reads but fulltext_audit.json is missing or invalid.", str(audit_path))
        else:
            success = [r for r in records if isinstance(r, dict) and r.get("status") == "full_text"]
            if len(success) < full_statuses:
                report.add(AuditSeverity.ERROR, "FULLTEXT_AUDIT_INSUFFICIENT", f"paper_deep_reading.md claims {full_statuses} FULL-TEXT reads but fulltext_audit.json has {len(success)} successful records.", str(audit_path))
            for record in success:
                text_path = record.get("text_path")
                if text_path and not (workspace / str(text_path)).exists():
                    report.add(AuditSeverity.ERROR, "FULLTEXT_TEXT_MISSING", f"Full-text audit record points to missing text_path `{text_path}`.", str(audit_path))


def _normalize_for_audit_match(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_arxiv_ids_for_audit(value: str) -> set[str]:
    text = value or ""
    ids: set[str] = set()
    for match in re.finditer(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", text, re.I):
        ids.add(match.group(1).lower())
    for match in re.finditer(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", text, re.I):
        ids.add(match.group(1).lower())
    return ids


def _reading_audit_match_candidates(reading: dict[str, Any]) -> list[str]:
    return [
        str(reading.get(field) or "")
        for field in (
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
            "registry_title",
            "registry_url",
            "registry_paper_url",
            "registry_pdf_url",
            "registry_source_url",
            "registry_arxiv_id",
            "registry_doi",
        )
    ]


def _fulltext_record_match_candidates(record: dict[str, Any]) -> list[str]:
    return [
        str(record.get(field) or "")
        for field in (
            "identifier",
            "pdf_url",
            "source_url",
            "resolved_url",
            "url",
            "arxiv_id",
            "doi",
            "text_path",
        )
    ]


def _read_audit_records(workspace: Path) -> list[dict[str, Any]]:
    audit_path = workspace / "fulltext_audit.json"
    if not audit_path.exists():
        return []
    data = _load_json(audit_path)
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _reading_matches_fulltext_record(reading: dict[str, Any], record: dict[str, Any]) -> bool:
    record_candidates = _fulltext_record_match_candidates(record)
    record_arxiv_ids = set().union(*(_extract_arxiv_ids_for_audit(raw) for raw in record_candidates))
    reading_arxiv_ids = set().union(*(_extract_arxiv_ids_for_audit(raw) for raw in _reading_audit_match_candidates(reading)))
    if record_arxiv_ids and reading_arxiv_ids and record_arxiv_ids & reading_arxiv_ids:
        return True

    identifiers = [_normalize_for_audit_match(raw) for raw in record_candidates]
    identifiers = [identifier for identifier in identifiers if identifier]
    if not identifiers:
        return False
    for raw in _reading_audit_match_candidates(reading):
        candidate = _normalize_for_audit_match(raw)
        if not candidate:
            continue
        for identifier in identifiers:
            if identifier == candidate or identifier in candidate or candidate in identifier:
                return True
    return False


def _augment_reading_with_registry(
    reading: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pid = str(reading.get("paper_id") or "").strip()
    if not pid or pid not in registry:
        return reading
    entry = registry[pid]
    augmented = dict(reading)
    mapping = {
        "title": "registry_title",
        "url": "registry_url",
        "paper_url": "registry_paper_url",
        "pdf_url": "registry_pdf_url",
        "source_url": "registry_source_url",
        "arxiv_id": "registry_arxiv_id",
        "doi": "registry_doi",
    }
    for source_key, target_key in mapping.items():
        value = str(entry.get(source_key) or "").strip()
        if value and not augmented.get(target_key):
            augmented[target_key] = value
    return augmented


def _audit_reading_batches(
    workspace: Path,
    report: AuditReport,
    registry: dict[str, dict[str, Any]],
) -> None:
    manifest_path = workspace / "batch_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return
    batches = manifest.get("batches") if isinstance(manifest, dict) else None
    if not isinstance(batches, list):
        return

    audit_records = _read_audit_records(workspace)
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        if batch.get("stage") != "stage_3_5_reading" or batch.get("status") != "passed":
            continue
        batch_id = str(batch.get("batch_id") or "unknown")
        result_file = str(batch.get("result_file") or "")
        result_path = workspace / result_file if result_file else manifest_path
        if not result_file or not result_path.exists():
            report.add(
                AuditSeverity.ERROR,
                "BATCH_READING_SCHEMA",
                f"Passed Stage 3.5 reading batch {batch_id} has no readable result_file.",
                str(manifest_path),
            )
            continue
        result = _load_json_if_exists(report, result_path)
        readings = result.get("readings") if isinstance(result, dict) else None
        if not isinstance(readings, list):
            report.add(
                AuditSeverity.ERROR,
                "BATCH_READING_SCHEMA",
                f"Passed Stage 3.5 reading batch {batch_id} is missing required `readings` list.",
                str(result_path),
            )
            continue
        for idx, reading in enumerate(readings, start=1):
            if not isinstance(reading, dict):
                continue
            status = str(reading.get("fulltext_status") or "").strip().upper()
            if status not in {"FULL-TEXT", "ABSTRACT-ONLY"}:
                report.add(
                    AuditSeverity.ERROR,
                    "BATCH_READING_SCHEMA",
                    f"{batch_id} reading {idx} has invalid fulltext_status `{status}`.",
                    str(result_path),
                )
                continue
            augmented = _augment_reading_with_registry(reading, registry)
            matches = [
                record for record in audit_records
                if _reading_matches_fulltext_record(augmented, record)
            ]
            if not matches:
                report.add(
                    AuditSeverity.ERROR,
                    "BATCH_READING_AUDIT_MISSING",
                    f"{batch_id} reading {idx} has no matching fulltext_audit.json record.",
                    str(result_path),
                )
                continue
            if status == "FULL-TEXT":
                if not any(record.get("status") == "full_text" for record in matches):
                    report.add(
                        AuditSeverity.ERROR,
                        "BATCH_READING_AUDIT_MISSING",
                        f"{batch_id} reading {idx} claims FULL-TEXT but has no successful full_text audit record.",
                        str(result_path),
                    )
            elif not any(
                record.get("status") != "full_text" and str(record.get("reason") or "").strip()
                for record in matches
            ):
                report.add(
                    AuditSeverity.ERROR,
                    "BATCH_READING_AUDIT_MISSING",
                    f"{batch_id} reading {idx} claims ABSTRACT-ONLY but has no failed full-text audit record with reason.",
                    str(result_path),
                )


def _audit_evidence(workspace: Path, report: AuditReport, registry: dict[str, dict[str, Any]], verify_urls: bool, fetcher: Callable[[str], str] | None) -> None:
    path = workspace / "evidence_db.json"
    data = _load_json_if_exists(report, path)
    if data is None:
        return
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list):
        report.add(AuditSeverity.ERROR, "EVIDENCE_SCHEMA", "evidence_db.json must contain a claims list.", str(path))
        return
    summary_count = data.get("summary", {}).get("citations_count") if isinstance(data.get("summary"), dict) else None
    unique_ids = {c.get("citation_id") for c in claims if isinstance(c, dict) and c.get("citation_id")}
    if summary_count is not None and int(summary_count) != len(unique_ids):
        report.add(AuditSeverity.ERROR, "CITATION_COUNT_MISMATCH", f"evidence_db.json summary citation count {summary_count} does not match {len(unique_ids)} unique citation IDs.", str(path))
    policy = str(data.get("metadata", {}).get("citation_id_policy", "")) if isinstance(data.get("metadata"), dict) else ""
    if "local deterministic" in policy.lower():
        report.add(AuditSeverity.ERROR, "LOCAL_CITATION_IDS", "Local deterministic citation IDs are not allowed for final evidence; use cite_source/citations.json registry.", str(path))
    citation_registry = _load_json_if_exists(report, workspace / "citations.json")
    if citation_registry is not None and isinstance(citation_registry, list):
        registered = {entry.get("citation_id") for entry in citation_registry if isinstance(entry, dict)}
        missing = sorted(cid for cid in unique_ids if cid not in registered)
        if missing:
            report.add(AuditSeverity.ERROR, "CITATION_REGISTRY_MISMATCH", f"evidence_db.json uses citation IDs not present in citations.json: {', '.join(missing)}.", str(path))
    arxiv_results: dict[str, tuple[str | None, str | None]] = {}
    if verify_urls:
        arxiv_results = _fetch_arxiv_titles(
            [
                str(claim.get("source_url") or "")
                for claim in claims
                if isinstance(claim, dict)
            ],
            fetcher=fetcher,
        )
    reported_fetch_failures: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        pid = str(claim.get("source_paper_id") or claim.get("paper_id") or "")
        title = str(claim.get("source_title") or "")
        url = str(claim.get("source_url") or "")
        if registry and pid in registry and title and not title_matches(registry[pid].get("title", ""), title):
            report.add(AuditSeverity.ERROR, "EVIDENCE_PAPER_ID_DRIFT", f"evidence_db.json maps {pid} to `{title}`, but paper_registry.json maps it to `{registry[pid].get('title')}`.", str(path))
        arxiv_id = extract_arxiv_id(url)
        if verify_urls and url and title and arxiv_id:
            actual, error = arxiv_results.get(
                arxiv_id,
                (None, "No verification result was produced."),
            )
            if error or not actual:
                if arxiv_id not in reported_fetch_failures:
                    reason = error or "The response did not contain a parseable title."
                    report.add(
                        AuditSeverity.WARNING,
                        "ARXIV_TITLE_VERIFICATION_UNAVAILABLE",
                        f"Could not verify arXiv title for {url}: {reason}",
                        str(path),
                    )
                    reported_fetch_failures.add(arxiv_id)
                continue
            if not title_matches(title, actual):
                report.add(AuditSeverity.ERROR, "ARXIV_TITLE_MISMATCH", f"source_url `{url}` resolves to `{actual}`, not `{title}`.", str(path))


def _stage7_is_expected(workspace: Path) -> bool:
    """Return whether the workspace has reached Stage 7 or later."""
    return any(
        (workspace / name).exists()
        for name in (
            "knowledge_synthesis.md",
            "design_space.json",
            "raw_ideas.json",
            "tournament_rankings.json",
            "idea_reviews.json",
            "feasibility_assessments.json",
            "final_report.md",
        )
    )


def _validation_error_summary(exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg") or "invalid value")
        details.append(f"{location}: {message}" if location else message)
    return "; ".join(details[:12])


def _validate_stage7_artifacts(workspace: Path) -> list[AuditIssue]:
    """Validate the canonical Stage 7 gap registry and its cross-file links."""
    issues: list[AuditIssue] = []
    gap_path = workspace / "research_gaps.json"
    if not gap_path.exists():
        if _stage7_is_expected(workspace):
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "RESEARCH_GAPS_MISSING",
                    (
                        "Stage 7 or a downstream stage exists, but "
                        "research_gaps.json is missing. Claim-to-Gap provenance "
                        "cannot be verified from Markdown alone."
                    ),
                    str(gap_path),
                )
            )
        return issues

    try:
        raw_catalog = _load_json(gap_path)
    except Exception as exc:
        issues.append(
            AuditIssue(
                AuditSeverity.ERROR,
                "RESEARCH_GAPS_JSON_INVALID",
                f"Could not parse research_gaps.json: {exc}",
                str(gap_path),
            )
        )
        return issues

    try:
        catalog = ResearchGapCatalog.model_validate(raw_catalog)
    except ValidationError as exc:
        issues.append(
            AuditIssue(
                AuditSeverity.ERROR,
                "RESEARCH_GAPS_SCHEMA",
                _validation_error_summary(exc),
                str(gap_path),
            )
        )
        return issues

    evidence_path = workspace / "evidence_db.json"
    try:
        evidence = _load_json(evidence_path)
    except FileNotFoundError:
        issues.append(
            AuditIssue(
                AuditSeverity.ERROR,
                "GAP_EVIDENCE_DB_MISSING",
                "research_gaps.json exists but evidence_db.json is missing.",
                str(evidence_path),
            )
        )
        return issues
    except Exception as exc:
        issues.append(
            AuditIssue(
                AuditSeverity.ERROR,
                "GAP_EVIDENCE_DB_INVALID",
                f"Could not parse evidence_db.json for Claim-to-Gap validation: {exc}",
                str(evidence_path),
            )
        )
        return issues

    claims = evidence.get("claims") if isinstance(evidence, dict) else None
    if not isinstance(claims, list):
        issues.append(
            AuditIssue(
                AuditSeverity.ERROR,
                "GAP_EVIDENCE_DB_INVALID",
                "evidence_db.json must contain a claims list.",
                str(evidence_path),
            )
        )
        return issues

    claim_by_id = {
        str(claim.get("citation_id") or "").strip(): claim
        for claim in claims
        if isinstance(claim, dict) and str(claim.get("citation_id") or "").strip()
    }
    for gap in catalog.gaps:
        missing = sorted(
            link.citation_id
            for link in gap.evidence_links
            if link.citation_id not in claim_by_id
        )
        if missing:
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "GAP_EVIDENCE_UNKNOWN_CITATION",
                    (
                        f"{gap.gap_id} references citation ID(s) absent from "
                        f"evidence_db.json: {', '.join(missing)}."
                    ),
                    str(gap_path),
                )
            )
            continue

        source_papers = {
            str(
                claim_by_id[link.citation_id].get("source_paper_id")
                or claim_by_id[link.citation_id].get("paper_id")
                or ""
            ).strip()
            for link in gap.evidence_links
        }
        source_papers.discard("")
        if len(source_papers) < 2:
            issues.append(
                AuditIssue(
                    AuditSeverity.WARNING,
                    "GAP_EVIDENCE_SINGLE_SOURCE",
                    (
                        f"{gap.gap_id} is grounded in fewer than two independent "
                        "paper sources; retain it only if the narrow evidence base "
                        "is explicit."
                    ),
                    str(gap_path),
                )
            )
        if gap.supporting_papers:
            declared = set(gap.supporting_papers)
            ungrounded = sorted(declared - source_papers)
            if ungrounded:
                issues.append(
                    AuditIssue(
                        AuditSeverity.ERROR,
                        "GAP_SUPPORTING_PAPER_DRIFT",
                        (
                            f"{gap.gap_id} lists supporting paper(s) without a "
                            "linked Claim: " + ", ".join(ungrounded)
                        ),
                        str(gap_path),
                    )
                )

    synthesis_path = workspace / "knowledge_synthesis.md"
    if synthesis_path.exists():
        synthesis = synthesis_path.read_text(encoding="utf-8", errors="replace")
        markdown_gap_ids = set(re.findall(r"\bG\d+\b", synthesis))
        catalog_gap_ids = {gap.gap_id for gap in catalog.gaps}
        missing_from_markdown = sorted(catalog_gap_ids - markdown_gap_ids)
        unknown_in_markdown = sorted(markdown_gap_ids - catalog_gap_ids)
        if missing_from_markdown or unknown_in_markdown:
            details: list[str] = []
            if missing_from_markdown:
                details.append(
                    "missing from knowledge_synthesis.md: "
                    + ", ".join(missing_from_markdown)
                )
            if unknown_in_markdown:
                details.append(
                    "not registered in research_gaps.json: "
                    + ", ".join(unknown_in_markdown)
                )
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "RESEARCH_GAP_ID_DRIFT",
                    "; ".join(details),
                    str(synthesis_path),
                )
            )

    catalog_gap_ids = {gap.gap_id for gap in catalog.gaps}
    downstream_specs = (
        ("design_space.json", "promising_combinations", ("supporting_gaps",)),
        ("raw_ideas.json", "ideas", ("target_gaps", "gap_addressed")),
    )
    for file_name, collection_key, gap_keys in downstream_specs:
        downstream_path = workspace / file_name
        if not downstream_path.exists():
            continue
        try:
            downstream = _load_json(downstream_path)
        except Exception:
            continue
        entries = downstream.get(collection_key) if isinstance(downstream, dict) else None
        if not isinstance(entries, list):
            continue
        referenced: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for gap_key in gap_keys:
                value = entry.get(gap_key)
                if isinstance(value, str):
                    referenced.update(re.findall(r"G\d+", value))
                elif isinstance(value, list):
                    referenced.update(
                        str(item).strip() for item in value if str(item).strip()
                    )
        unknown = sorted(referenced - catalog_gap_ids)
        if unknown:
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "DOWNSTREAM_GAP_ID_DRIFT",
                    (
                        f"{file_name} references gap ID(s) absent from "
                        f"research_gaps.json: {', '.join(unknown)}."
                    ),
                    str(downstream_path),
                )
            )
    return issues


def validate_stage7_artifacts(workspace: str | Path) -> list[dict[str, str]]:
    """Return Stage 7 structural and cross-file issues as plain dictionaries."""
    return [
        {
            "severity": issue.severity.value,
            "code": issue.code,
            "message": issue.message,
            "path": issue.path,
        }
        for issue in _validate_stage7_artifacts(Path(workspace).resolve())
    ]


def _audit_research_gaps(workspace: Path, report: AuditReport) -> None:
    report.issues.extend(_validate_stage7_artifacts(workspace))


def _audit_reflections(workspace: Path, report: AuditReport) -> None:
    reflections = workspace / "reflections"
    if not reflections.exists():
        return
    evidence = _load_json_if_exists(report, workspace / "evidence_db.json")
    claim_count = None
    if isinstance(evidence, dict) and isinstance(evidence.get("claims"), list):
        claim_count = len({c.get("citation_id") for c in evidence["claims"] if isinstance(c, dict) and c.get("citation_id")})
    positions = _load_json_if_exists(report, workspace / "paper_positions.json")
    positioned_count = len(positions) if isinstance(positions, list) else None
    gap_catalog = _load_json_if_exists(report, workspace / "research_gaps.json")
    gaps = gap_catalog.get("gaps") if isinstance(gap_catalog, dict) else None
    gap_count = len(gaps) if isinstance(gaps, list) else None
    gap_link_count = (
        sum(
            len(gap.get("evidence_links", []))
            for gap in gaps
            if isinstance(gap, dict)
            and isinstance(gap.get("evidence_links"), list)
        )
        if isinstance(gaps, list)
        else None
    )
    deep_text = (workspace / "paper_deep_reading.md").read_text(encoding="utf-8", errors="replace") if (workspace / "paper_deep_reading.md").exists() else ""
    deep_status_count = len(re.findall(r"Full-text status\*\*:\s*(?:FULL-TEXT|ABSTRACT-ONLY)", deep_text, re.I))
    full_status_count = len(re.findall(r"Full-text status\*\*:\s*FULL-TEXT", deep_text, re.I))
    for path in reflections.glob("stage_*_reflection.json"):
        data = _load_json_if_exists(report, path)
        if not isinstance(data, dict):
            continue
        artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
        if claim_count is not None and artifacts.get("citations_count") is not None and int(artifacts["citations_count"]) != claim_count:
            report.add(AuditSeverity.ERROR, "REFLECTION_COUNT_MISMATCH", f"{path.name} reports citations_count {artifacts['citations_count']} but evidence_db.json has {claim_count}.", str(path))
        if positioned_count is not None and artifacts.get("papers_positioned") is not None and int(artifacts["papers_positioned"]) != positioned_count:
            report.add(AuditSeverity.ERROR, "REFLECTION_COUNT_MISMATCH", f"{path.name} reports papers_positioned {artifacts['papers_positioned']} but paper_positions.json has {positioned_count}.", str(path))
        if artifacts.get("papers_read") is not None and int(artifacts["papers_read"]) != deep_status_count:
            report.add(AuditSeverity.ERROR, "REFLECTION_COUNT_MISMATCH", f"{path.name} reports papers_read {artifacts['papers_read']} but paper_deep_reading.md has {deep_status_count} status blocks.", str(path))
        if artifacts.get("fulltext_count") is not None and int(artifacts["fulltext_count"]) != full_status_count:
            report.add(AuditSeverity.ERROR, "REFLECTION_COUNT_MISMATCH", f"{path.name} reports fulltext_count {artifacts['fulltext_count']} but paper_deep_reading.md has {full_status_count} FULL-TEXT blocks.", str(path))
        if gap_count is not None and artifacts.get("gaps_count") is not None and int(artifacts["gaps_count"]) != gap_count:
            report.add(AuditSeverity.ERROR, "REFLECTION_COUNT_MISMATCH", f"{path.name} reports gaps_count {artifacts['gaps_count']} but research_gaps.json has {gap_count} gaps.", str(path))
        if gap_link_count is not None and artifacts.get("evidence_gap_links") is not None and int(artifacts["evidence_gap_links"]) != gap_link_count:
            report.add(AuditSeverity.ERROR, "REFLECTION_COUNT_MISMATCH", f"{path.name} reports evidence_gap_links {artifacts['evidence_gap_links']} but research_gaps.json has {gap_link_count} links.", str(path))


def _audit_nested_workspace(workspace: Path, report: AuditReport) -> None:
    nested = workspace / "workspace"
    if not nested.exists():
        return
    for name in ["evidence_db.json", "paper_positions.json", "final_report.md", "elo_rankings.json", "writer_round2.json"]:
        top = workspace / name
        inner = nested / name
        if top.exists() and inner.exists() and top.read_bytes() != inner.read_bytes():
            report.add(AuditSeverity.ERROR, "NESTED_WORKSPACE_DRIFT", f"nested workspace artifact differs from top-level `{name}`; canonical output is ambiguous.", str(inner))


_CANONICAL_ARTIFACT_MIN_CHARS = {
    "seed_idea_analysis.md": 80,
    "research_brief.md": 80,
    "task_formalization.md": 80,
    "literature_survey.md": 80,
    "paper_deep_reading.md": 200,
    "expanded_literature.md": 200,
    "knowledge_synthesis.md": 200,
    "debate_log.md": 120,
    "final_report.md": 200,
}

_PLACEHOLDER_ARTIFACT_CONTENT = {
    "dummy",
    "ok",
    "placeholder",
    "tbd",
    "test",
    "todo",
}


def _audit_stage_artifact_placeholders(workspace: Path, report: AuditReport) -> None:
    for name, min_chars in _CANONICAL_ARTIFACT_MIN_CHARS.items():
        path = workspace / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        stripped = text.strip()
        if not stripped:
            report.add(
                AuditSeverity.ERROR,
                "STAGE_ARTIFACT_PLACEHOLDER",
                f"{name} exists but is empty.",
                str(path),
            )
        elif stripped.lower() in _PLACEHOLDER_ARTIFACT_CONTENT or len(stripped) < min_chars:
            report.add(
                AuditSeverity.ERROR,
                "STAGE_ARTIFACT_PLACEHOLDER",
                f"{name} is too small or placeholder-like to be a real stage artifact ({len(stripped)} chars).",
                str(path),
            )


def _audit_final_report_integrity(workspace: Path, report: AuditReport) -> None:
    """Reject common signs that a model-generated final report was cut off."""
    path = workspace / "final_report.md"
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8", errors="replace").rstrip()
    if not text:
        return

    nonempty_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return

    reasons: list[str] = []
    last_line = nonempty_lines[-1].strip()

    # A generated reference or result table most often gets cut off halfway
    # through its final row. Markdown permits tables without outer pipes, but a
    # row that starts with one and then loses the closing pipe is unambiguous.
    if last_line.startswith("|") and not last_line.endswith("|"):
        reasons.append("the final Markdown table row has no closing pipe")

    fence_markers = [
        match.group(1)[0]
        for line in text.splitlines()
        if (match := re.match(r"^\s*(`{3,}|~{3,})", line))
    ]
    for marker in ("`", "~"):
        if fence_markers.count(marker) % 2:
            reasons.append(f"an opening {marker * 3} code fence is not closed")

    numbered_sections = re.findall(
        r"^##\s+(\d+(?:\.\d+)*)[.)]?\s+",
        text,
        flags=re.MULTILINE,
    )
    duplicate_sections = sorted(
        {number for number in numbered_sections if numbered_sections.count(number) > 1}
    )
    if duplicate_sections:
        report.add(
            AuditSeverity.ERROR,
            "FINAL_REPORT_DUPLICATE_SECTION",
            "final_report.md repeats numbered level-2 section(s): "
            + ", ".join(duplicate_sections)
            + ".",
            str(path),
        )

    # A trailing heading has no body. Generated final-report prose is required
    # to finish with terminal punctuation; the observed failures ended with
    # fragments such as ``Stage`` and ``for gap provenance``. Complete table
    # rows and HTML completion markers are handled separately above/below.
    if re.match(r"^#{1,6}\s+\S", last_line):
        reasons.append("the report ends with a heading that has no body")
    elif not last_line.startswith(("|", "<")):
        plain_last_line = re.sub(r"[`*_~]", "", last_line).strip()
        if re.search(
            r"\b(?:stage|section|chapter|part|appendix)\s*\d*$",
            plain_last_line,
            re.IGNORECASE,
        ):
            reasons.append("the report ends after an unfinished structural cue")
        elif re.search(r"[\w)]$", plain_last_line) and not re.search(
            r"[.!?。！？:：;；\]\)}>'\"]$",
            plain_last_line,
        ):
            reasons.append(
                "the report ends with a punctuation-free prose fragment"
            )

    if reasons:
        report.add(
            AuditSeverity.ERROR,
            "FINAL_REPORT_TRUNCATED",
            "final_report.md appears truncated: " + "; ".join(reasons) + ".",
            str(path),
        )


def audit_workspace(
    workspace: str | Path,
    *,
    verify_urls: bool = False,
    fetcher: Callable[[str], str] | None = None,
    deep_reading_top_k: int | None = None,
) -> AuditReport:
    ws = Path(workspace).resolve()
    report = AuditReport(workspace=str(ws))
    if not ws.exists():
        report.add(AuditSeverity.ERROR, "WORKSPACE_MISSING", "Workspace does not exist.", str(ws))
        return report
    registry = _paper_registry(ws, report)
    _audit_stage_artifact_placeholders(ws, report)
    _audit_final_report_integrity(ws, report)
    report.issues.extend(_validate_stage3_artifacts(ws))
    _audit_incomplete_batches(ws, report)
    _audit_paper_ids(ws, report, registry)
    _audit_reading_batches(ws, report, registry)
    _audit_deep_reading(ws, report, expected_top_k=deep_reading_top_k)
    _audit_evidence(ws, report, registry, verify_urls=verify_urls, fetcher=fetcher)
    _audit_research_gaps(ws, report)
    _audit_reflections(ws, report)
    _audit_nested_workspace(ws, report)
    return report


@tool(parse_docstring=True)
def audit_workspace_artifacts(
    workspace_path: str = "",
    verify_urls: bool = True,
) -> str:
    """Audit AutoIdea workspace artifacts for cross-file integrity errors.

    Args:
        workspace_path: Workspace directory to audit. Defaults to the active workspace.
        verify_urls: If true, verify arXiv source_url titles by fetching arXiv metadata.

    Returns:
        Markdown audit report. Any ERROR means the pipeline must not proceed
        to final reporting until the artifact is corrected.
    """
    if workspace_path:
        workspace = Path(workspace_path)
    else:
        from autoidea.paths import get_active_workspace
        workspace = get_active_workspace()
    report = audit_workspace(workspace, verify_urls=verify_urls)
    report_path = Path(workspace) / "audit_report.json"
    try:
        report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return report.to_markdown()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    workspace = Path(args[0]) if args else Path.cwd()
    verify_urls = "--verify-urls" in args
    report = audit_workspace(workspace, verify_urls=verify_urls)
    print(report.to_markdown())
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
