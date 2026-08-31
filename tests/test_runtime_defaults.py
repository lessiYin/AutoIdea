from __future__ import annotations

import importlib
import os
import tomllib
from pathlib import Path

from autoidea.config import AutoIdeaConfig
from autoidea.llm import DEFAULT_MODEL, list_models
from autoidea.sessions import generate_thread_id
from autoidea.prompts import SYSTEM_PROMPT


def test_default_provider_and_model_use_openai() -> None:
    config = AutoIdeaConfig()

    assert config.provider == "openai"
    assert config.model == "gpt-5.6-sol"
    assert DEFAULT_MODEL == "gpt-5.6-sol"
    assert list_models()[DEFAULT_MODEL] == ("openai", "gpt-5.6-sol")


def test_automatic_prompt_never_waits_for_clarification() -> None:
    assert "never end a turn waiting\nfor clarification" in SYSTEM_PROMPT
    assert "continue\nthrough Stage 12 in the same run" in SYSTEM_PROMPT


def test_stage_three_handoff_carries_runtime_search_limits() -> None:
    assert "include the concrete" in SYSTEM_PROMPT
    assert "`max_search_queries` and `target_paper_count`" in SYSTEM_PROMPT
    assert "Stop launching new searches once" in SYSTEM_PROMPT


def test_stage_nine_treats_configured_idea_count_as_a_hard_cap() -> None:
    assert "Treat `max_ideas_to_generate` from Section 9 as a **hard upper bound**" in SYSTEM_PROMPT
    assert "never generate filler ideas" in SYSTEM_PROMPT
    assert "tournament_rankings.json" in SYSTEM_PROMPT


def test_system_prompt_requires_stage_by_stage_execution() -> None:
    assert "Work on exactly one current stage at a time" in SYSTEM_PROMPT
    assert "Never mark a stage `passed`" in SYSTEM_PROMPT


def test_thread_ids_are_random_eight_character_hex_values() -> None:
    thread_ids = {generate_thread_id() for _ in range(32)}

    assert len(thread_ids) == 32
    assert all(
        len(value) == 8 and all(char in "0123456789abcdef" for char in value)
        for value in thread_ids
    )


def test_project_declares_pymupdf_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = [dep.lower() for dep in data["project"]["dependencies"]]

    assert any(dep.startswith("pymupdf") for dep in dependencies)


def test_project_declares_safe_xml_parser_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = [dep.lower() for dep in data["project"]["dependencies"]]

    assert any(dep.startswith("defusedxml") for dep in dependencies)


def test_openai_stream_chunk_timeout_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", raising=False)

    models = importlib.import_module("autoidea.llm.models")
    models._ensure_openai_stream_chunk_timeout_default()

    assert os.environ["LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S"] == "0"


def test_openai_stream_chunk_timeout_default_does_not_override_user_value(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "300")

    models = importlib.import_module("autoidea.llm.models")
    models._ensure_openai_stream_chunk_timeout_default()

    assert os.environ["LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S"] == "300"


def test_reasoning_effort_env_is_forwarded_to_openai_model_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_REASONING_EFFORT", "xhigh")

    models = importlib.import_module("autoidea.llm.models")

    assert models._get_reasoning_effort_kwargs() == {"reasoning_effort": "xhigh"}
