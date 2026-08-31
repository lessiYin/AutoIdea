from __future__ import annotations

from pathlib import Path

import pytest

from autoidea.web.artifacts import ArtifactAccessError, read_artifact


def test_read_artifact_renders_markdown_html(tmp_path: Path) -> None:
    (tmp_path / "final_report.md").write_text(
        "# Final Idea\n\n- **Claim**: grounded result\n",
        encoding="utf-8",
    )

    artifact = read_artifact(tmp_path, "final_report.md")

    assert artifact.path == "final_report.md"
    assert artifact.kind == "md"
    assert artifact.title == "Final Report"
    assert artifact.text.startswith("# Final Idea")
    assert "<h1>Final Idea</h1>" in artifact.html
    assert "<strong>Claim</strong>" in artifact.html


def test_read_artifact_returns_json_text_without_markdown_html(tmp_path: Path) -> None:
    (tmp_path / "raw_ideas.json").write_text('{"ideas": []}', encoding="utf-8")

    artifact = read_artifact(tmp_path, "raw_ideas.json")

    assert artifact.kind == "json"
    assert artifact.text == '{"ideas": []}'
    assert artifact.html == ""


def test_read_artifact_rejects_path_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.md"
    outside.write_text("# secret", encoding="utf-8")

    with pytest.raises(ArtifactAccessError):
        read_artifact(tmp_path, "../secret.md")


def test_read_artifact_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(ArtifactAccessError):
        read_artifact(tmp_path, "missing.md")


def test_markdown_renderer_rejects_active_content_link_schemes(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text(
        "[safe](https://example.org) [mail](mailto:team@example.org) "
        "[relative](notes.md) [unsafe](javascript:alert(1)) "
        "[encoded](javascript&#x3a;alert(2))",
        encoding="utf-8",
    )

    artifact = read_artifact(tmp_path, "report.md")

    assert 'href="https://example.org"' in artifact.html
    assert 'href="mailto:team@example.org"' in artifact.html
    assert 'href="notes.md"' in artifact.html
    assert "javascript:" not in artifact.html.lower()
    assert "javascript&amp;#x3a;" not in artifact.html.lower()
    assert "unsafe" in artifact.html
