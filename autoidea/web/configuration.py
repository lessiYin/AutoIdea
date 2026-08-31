"""Browser-facing configuration metadata and persistence helpers."""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from autoidea.config import settings

PROVIDER_OPTIONS = [
    "anthropic",
    "openai",
    "google-genai",
    "ollama",
    "custom-openai",
    "custom-anthropic",
]

SECRET_FIELDS = {
    "anthropic_api_key",
    "openai_api_key",
    "google_api_key",
    "tavily_api_key",
    "semantic_scholar_api_key",
    "custom_openai_api_key",
    "custom_anthropic_api_key",
}

CONFIG_GROUPS = [
    {
        "id": "quick",
        "title": "Quick Setup",
        "title_zh": "快速设置",
        "fields": [
            "provider",
            "model",
            "max_tokens",
            "workspace_dir",
            "auto_approve",
            "show_thinking",
            "enable_ask_user",
            "enable_web_search",
            "target_paper_count",
            "max_ideas_to_generate",
        ],
    },
    {
        "id": "credentials",
        "title": "Credentials",
        "title_zh": "密钥与服务",
        "fields": [
            "anthropic_api_key",
            "openai_api_key",
            "google_api_key",
            "tavily_api_key",
            "semantic_scholar_api_key",
            "custom_openai_api_key",
            "custom_openai_base_url",
            "custom_anthropic_api_key",
            "custom_anthropic_base_url",
            "ollama_base_url",
        ],
    },
    {
        "id": "search",
        "title": "Search & Literature",
        "title_zh": "搜索与文献",
        "fields": [
            "enable_mock_fallback",
            "enable_web_search",
            "web_search_max_uses",
            "web_search_allowed_domains",
            "web_search_blocked_domains",
            "max_search_queries",
            "target_paper_count",
            "deep_reading_top_k",
            "seed_papers_file",
            "seed_ideas_file",
        ],
    },
    {
        "id": "pipeline",
        "title": "Pipeline & Ranking",
        "title_zh": "流程与排序",
        "fields": [
            "max_debate_rounds",
            "max_ideas_to_generate",
            "top_k_ranked",
            "elo_initial_score",
            "elo_k_factor",
            "recursion_limit",
        ],
    },
    {
        "id": "memory",
        "title": "Memory & Context",
        "title_zh": "记忆与上下文",
        "fields": [
            "memory_dir",
            "memory_trigger_messages",
            "extraction_model",
            "enable_auto_compaction",
            "auto_compaction_trigger_messages",
            "auto_compaction_keep_messages",
            "auto_compaction_trim_tokens",
        ],
    },
    {
        "id": "paths",
        "title": "Paths & Network",
        "title_zh": "路径与网络",
        "fields": [
            "workspace_dir",
            "runs_dir",
            "memory_dir",
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "shell_allow_list",
        ],
    },
]

FIELD_LABELS = {
    "http_proxy": "HTTP proxy",
    "https_proxy": "HTTPS proxy",
    "no_proxy": "No proxy",
    "anthropic_api_key": "Anthropic API key",
    "openai_api_key": "OpenAI API key",
    "google_api_key": "Google API key",
    "tavily_api_key": "Tavily API key",
    "semantic_scholar_api_key": "Semantic Scholar API key",
    "custom_openai_api_key": "Custom OpenAI API key",
    "custom_openai_base_url": "Custom OpenAI base URL",
    "custom_anthropic_api_key": "Custom Anthropic API key",
    "custom_anthropic_base_url": "Custom Anthropic base URL",
    "ollama_base_url": "Ollama base URL",
    "provider": "Provider",
    "model": "Model",
    "max_tokens": "Max tokens",
    "workspace_dir": "Workspace directory",
    "runs_dir": "Runs directory",
    "memory_dir": "Memory directory",
    "show_thinking": "Show thinking stream",
    "auto_approve": "Fully automatic checkpoints (default)",
    "shell_allow_list": "Shell allow list",
    "enable_ask_user": "Allow agent questions",
    "recursion_limit": "Recursion limit",
    "seed_papers_file": "Seed papers file",
    "seed_ideas_file": "Seed ideas file",
    "enable_mock_fallback": "Enable mock fallback",
    "enable_web_search": "Enable web search",
    "web_search_max_uses": "Web search max uses",
    "web_search_allowed_domains": "Allowed search domains",
    "web_search_blocked_domains": "Blocked search domains",
    "max_search_queries": "Max search queries",
    "max_debate_rounds": "Max debate rounds",
    "target_paper_count": "Target paper count",
    "deep_reading_top_k": "Deep reading top K",
    "max_ideas_to_generate": "Max ideas to generate",
    "top_k_ranked": "Top K ranked ideas",
    "elo_initial_score": "Elo initial score",
    "elo_k_factor": "Elo K factor",
    "memory_trigger_messages": "Memory trigger messages",
    "extraction_model": "Extraction model",
    "enable_auto_compaction": "Enable auto compaction",
    "auto_compaction_trigger_messages": "Compaction trigger messages",
    "auto_compaction_keep_messages": "Compaction keep messages",
    "auto_compaction_trim_tokens": "Compaction trim tokens",
}


def get_config_payload() -> dict[str, Any]:
    """Return browser-safe configuration values and UI metadata."""
    config = settings.load_config()
    effective = settings.get_effective_config()
    defaults = settings.AutoIdeaConfig()
    field_defs = {field.name: field for field in fields(settings.AutoIdeaConfig)}
    env_by_field = {config_key: env_key for env_key, config_key in settings._ENV_MAPPING.items()}

    payload_fields: dict[str, dict[str, Any]] = {}
    for key, field in field_defs.items():
        value = getattr(config, key)
        effective_value = getattr(effective, key)
        default_value = getattr(defaults, key)
        env_var = env_by_field.get(key, "")
        env_overridden = bool(env_var and os.getenv(env_var))
        secret = key in SECRET_FIELDS
        stored_is_set = value not in ("", None)
        effective_is_set = effective_value not in ("", None)
        field_type = _field_type(key, field.type)
        payload_fields[key] = {
            "key": key,
            "label": FIELD_LABELS.get(key, _humanize(key)),
            "type": field_type,
            "value": "" if secret else value,
            "effective_value": "" if secret else effective_value,
            "default": "" if secret else default_value,
            "secret": secret,
            "is_set": effective_is_set,
            "stored_is_set": stored_is_set,
            "masked_value": _mask_secret(value) if secret and value else "",
            "env_var": env_var,
            "env_overridden": env_overridden,
            "options": PROVIDER_OPTIONS if key == "provider" else [],
        }

    return {
        "path": _display_config_path(settings.get_config_path()),
        "groups": CONFIG_GROUPS,
        "fields": payload_fields,
    }


def _display_config_path(path: Path, *, home: Path | None = None) -> str:
    """Render paths below the current home without exposing the username."""
    resolved_home = (home or Path.home()).expanduser()
    expanded_path = path.expanduser()
    try:
        relative = expanded_path.relative_to(resolved_home)
    except ValueError:
        return str(expanded_path)
    return "~" if str(relative) == "." else f"~/{relative.as_posix()}"


def update_config_values(values: dict[str, Any]) -> dict[str, Any]:
    """Persist a partial config update and return the refreshed payload."""
    if not isinstance(values, dict):
        raise TypeError("values must be an object")

    config = settings.load_config()
    field_defs = {field.name: field for field in fields(settings.AutoIdeaConfig)}
    for key, value in values.items():
        if key not in field_defs:
            raise ValueError(f"Unknown config key: {key}")
        if key in SECRET_FIELDS and value == "":
            continue
        coerced = _coerce_value(key, value, field_defs[key].type)
        setattr(config, key, coerced)
    settings.save_config(config)
    return get_config_payload()


def reset_config_values() -> dict[str, Any]:
    """Reset config.yaml and return the default payload."""
    settings.reset_config()
    return get_config_payload()


def _field_type(key: str, annotation: Any) -> str:
    if key == "provider":
        return "select"
    if annotation == "bool":
        return "bool"
    if annotation == "int":
        return "int"
    if annotation == "float":
        return "float"
    return "str"


def _coerce_value(key: str, value: Any, annotation: Any) -> Any:
    if annotation == "bool":
        return _coerce_bool(key, value)
    if annotation == "int":
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
    if annotation == "float":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
    return "" if value is None else str(value)


def _coerce_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean")


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _humanize(key: str) -> str:
    return key.replace("_", " ").capitalize()
