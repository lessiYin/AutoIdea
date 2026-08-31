"""Safe artifact loading and lightweight rendering for the web workbench."""

from __future__ import annotations

import html
import re
from pathlib import Path

from .models import ArtifactContent, artifact_title


class ArtifactAccessError(ValueError):
    """Raised when a requested workspace artifact cannot be read safely."""


def read_artifact(workspace: str | Path, artifact_path: str) -> ArtifactContent:
    """Read a single workspace artifact.

    Paths are resolved under *workspace* and path traversal is rejected.
    Markdown is rendered to conservative HTML for in-browser reading.
    """
    root = Path(workspace).expanduser().resolve()
    requested = (root / artifact_path).resolve()
    if requested != root and root not in requested.parents:
        raise ArtifactAccessError("Artifact path escapes the workspace.")
    if not requested.is_file():
        raise ArtifactAccessError("Artifact does not exist.")

    text = requested.read_text(encoding="utf-8", errors="replace")
    suffix = requested.suffix.lower().lstrip(".") or "file"
    relative_path = requested.relative_to(root).as_posix()
    rendered = render_markdown(text) if suffix in {"md", "markdown"} else ""
    return ArtifactContent(
        path=relative_path,
        kind=suffix,
        size_bytes=requested.stat().st_size,
        title=artifact_title(requested),
        text=text,
        html=rendered,
    )


def render_markdown(markdown: str) -> str:
    """Render a practical Markdown subset without adding a runtime dependency."""
    lines = markdown.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append("<ul>" + "".join(list_items) + "</ul>")
            list_items = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        header = table_rows[0]
        body = table_rows[1:]
        head_html = "".join(f"<th>{_inline(cell)}</th>" for cell in header)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>"
            for row in body
        )
        blocks.append(
            "<table><thead><tr>"
            + head_html
            + "</tr></thead><tbody>"
            + body_html
            + "</tbody></table>"
        )
        table_rows = []
        in_table = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            flush_list()
            flush_table()
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue

        if _is_table_separator(stripped):
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            flush_list()
            in_table = True
            table_rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
            continue
        if in_table:
            flush_table()

        list_match = re.match(r"^[-*]\s+(.+)$", stripped)
        if list_match:
            flush_paragraph()
            list_items.append(f"<li>{_inline(list_match.group(1))}</li>")
            continue

        flush_list()
        paragraph.append(stripped)

    flush_paragraph()
    flush_list()
    flush_table()
    return "\n".join(blocks)


def _is_table_separator(line: str) -> bool:
    compact = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return not compact and "|" in line and "-" in line


def _inline(text: str) -> str:
    value = html.escape(text, quote=False)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"`(.+?)`", r"<code>\1</code>", value)
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        _render_link,
        value,
    )
    return value


def _render_link(match: re.Match[str]) -> str:
    """Render only links whose scheme is safe for browser navigation."""
    label = match.group(1)
    href = html.unescape(match.group(2)).strip()
    if not _safe_link_href(href):
        return label
    escaped_href = html.escape(href, quote=True)
    if re.match(r"^https?://", href, flags=re.IGNORECASE):
        return (
            f'<a href="{escaped_href}" target="_blank" '
            f'rel="noopener noreferrer">{label}</a>'
        )
    return f'<a href="{escaped_href}">{label}</a>'


def _safe_link_href(href: str) -> bool:
    if not href or any(ord(character) < 32 for character in href):
        return False
    compact = re.sub(r"\s+", "", html.unescape(href))
    if compact.startswith("//"):
        return False
    scheme = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", compact)
    if scheme:
        return scheme.group(1).casefold() in {"http", "https", "mailto"}
    return True
