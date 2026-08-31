"""Configuration loading and persistence for AutoIdea.

Runtime configuration is kept outside the installed package. Values are
resolved in this order: CLI arguments, environment variables, the user config
file, then built-in defaults.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, asdict, fields
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv


# =============================================================================
# Configuration paths
# =============================================================================


def get_config_dir() -> Path:
    """Return the per-user configuration directory."""
    if override := os.getenv("AUTOIDEA_CONFIG_DIR", "").strip():
        return Path(override).expanduser()
    if xdg_home := os.getenv("XDG_CONFIG_HOME", "").strip():
        return Path(xdg_home).expanduser() / "autoidea"
    return Path.home() / ".config" / "autoidea"


def get_config_path() -> Path:
    """Return the user configuration file path."""
    if override := os.getenv("AUTOIDEA_CONFIG_FILE", "").strip():
        return Path(override).expanduser()
    return get_config_dir() / "config.yaml"


def get_state_dir() -> Path:
    """Return the per-user directory for sessions and CLI history."""
    if override := os.getenv("AUTOIDEA_STATE_DIR", "").strip():
        return Path(override).expanduser()
    if xdg_home := os.getenv("XDG_STATE_HOME", "").strip():
        return Path(xdg_home).expanduser() / "autoidea"
    return Path.home() / ".local" / "state" / "autoidea"


# =============================================================================
# Configuration dataclass
# =============================================================================


@dataclass
class AutoIdeaConfig:
    """AutoIdea configuration settings.

    All parameters can be configured via config.yaml file.

    Attributes:
        anthropic_api_key: Anthropic API key for Claude models.
        openai_api_key: OpenAI API key for GPT models.
        google_api_key: Google API key for Gemini models.
        tavily_api_key: Tavily API key for web search.
        provider: Default LLM provider.
        model: Default model name.
        max_tokens: Maximum output tokens for LLM responses.
        show_thinking: Whether to show thinking panels in CLI.
    """

    # =========================================================================
    # Network Proxy Settings
    # =========================================================================
    http_proxy: str = ""      # HTTP proxy URL
    https_proxy: str = ""     # HTTPS proxy URL
    no_proxy: str = ""        # Comma-separated hosts that bypass proxies

    # =========================================================================
    # API Keys (can also be set via environment variables)
    # =========================================================================
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""
    semantic_scholar_api_key: str = ""  # Raises Semantic Scholar rate limits
    custom_openai_api_key: str = ""
    custom_openai_base_url: str = ""
    custom_anthropic_api_key: str = ""
    custom_anthropic_base_url: str = ""
    ollama_base_url: str = ""

    # =========================================================================
    # LLM Settings
    # =========================================================================
    provider: str = "openai"
    model: str = "gpt-5.6-sol"
    max_tokens: int = 16384  # Maximum LLM output tokens

    # =========================================================================
    # Path Settings
    # =========================================================================
    workspace_dir: str = ""      # Workspace directory; empty uses the current directory
    runs_dir: str = ""           # Run directory; empty uses workspace/runs
    memory_dir: str = ""         # Memory directory; empty uses workspace/memory

    # =========================================================================
    # UI Settings
    # =========================================================================
    show_thinking: bool = True

    # =========================================================================
    # HITL (Human-in-the-Loop) Settings
    # =========================================================================
    auto_approve: bool = True
    shell_allow_list: str = ""

    # =========================================================================
    # Agent Settings
    # =========================================================================
    enable_ask_user: bool = True
    recursion_limit: int = 1000  # LangGraph recursion limit

    # =========================================================================
    # Seed Papers
    # =========================================================================
    seed_papers_file: str = ""

    # =========================================================================
    # Seed Ideas (user-provided research idea documents)
    # =========================================================================
    seed_ideas_file: str = ""   # Seed-idea file path (.md/.txt/.json)

    # =========================================================================
    # Search Settings
    # =========================================================================
    enable_mock_fallback: bool = False  # Use mock data after API failures; disabled by default

    # =========================================================================
    # Claude Web Search Settings
    # =========================================================================
    enable_web_search: bool = True           # Enable native Claude Web Search
    web_search_max_uses: int = 10            # Maximum searches per conversation
    web_search_allowed_domains: str = ""     # Comma-separated domain allowlist
    web_search_blocked_domains: str = ""     # Comma-separated domain blocklist

    # =========================================================================
    # Pipeline Parameters
    # =========================================================================
    max_search_queries: int = 50      # Maximum search queries
    max_debate_rounds: int = 5        # Maximum debate rounds
    target_paper_count: int = 20      # Target paper count
    deep_reading_top_k: int = 20      # Top-K papers selected for deep reading
    max_ideas_to_generate: int = 10   # Maximum ideas to generate
    top_k_ranked: int = 20            # Number of top-ranked ideas to retain

    # =========================================================================
    # Idea Tournament Settings (Elo Rating)
    # =========================================================================
    elo_initial_score: int = 1500     # Initial Elo rating
    elo_k_factor: int = 32            # Elo K-factor

    # =========================================================================
    # Memory Settings
    # =========================================================================
    memory_trigger_messages: int = 20  # Extract memory every N messages
    extraction_model: str = ""         # Memory extraction model; empty uses the main model

    # =========================================================================
    # Context Compaction Settings
    # =========================================================================
    enable_auto_compaction: bool = True
    auto_compaction_trigger_messages: int = 60
    auto_compaction_keep_messages: int = 20
    auto_compaction_trim_tokens: int = 4000


# =============================================================================
# Config file operations
# =============================================================================


def load_config() -> AutoIdeaConfig:
    """Load configuration from file.

    Returns:
        AutoIdeaConfig instance with values from file, or defaults if
        file doesn't exist.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return AutoIdeaConfig()

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # Filter to only valid fields
        valid_fields = {f.name for f in fields(AutoIdeaConfig)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return AutoIdeaConfig(**filtered_data)
    except Exception:
        return AutoIdeaConfig()


def save_config(config: AutoIdeaConfig) -> None:
    """Save configuration atomically with user-only file permissions."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            delete=False,
        ) as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
            temporary_path = Path(handle.name)
        temporary_path.chmod(0o600)
        temporary_path.replace(config_path)
        config_path.chmod(0o600)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def reset_config() -> None:
    """Reset configuration to defaults by deleting the config file."""
    config_path = get_config_path()
    if config_path.exists():
        config_path.unlink()


# =============================================================================
# Config value operations
# =============================================================================


def get_config_value(key: str) -> Any:
    """Get a single configuration value."""
    config = load_config()
    valid_fields = {f.name for f in fields(AutoIdeaConfig)}
    if key not in valid_fields:
        raise KeyError(f"Unknown config key: {key}")
    return getattr(config, key)


def set_config_value(key: str, value: Any) -> None:
    """Set a single configuration value and save."""
    config = load_config()
    valid_fields = {f.name for f in fields(AutoIdeaConfig)}
    if key not in valid_fields:
        raise KeyError(f"Unknown config key: {key}")

    # Type conversion
    field_type = {f.name: f.type for f in fields(AutoIdeaConfig)}
    target_type = field_type.get(key, "str")

    if target_type == "bool":
        if isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")
    elif target_type == "int":
        value = int(value)
    elif target_type == "float":
        value = float(value)

    setattr(config, key, value)
    save_config(config)


def list_config() -> dict[str, Any]:
    """List all configuration values."""
    config = load_config()
    result = {}
    for f in fields(AutoIdeaConfig):
        value = getattr(config, f.name)
        # Mask API keys
        if "api_key" in f.name and value:
            value = value[:8] + "..." if len(value) > 8 else "***"
        result[f.name] = value
    return result


# =============================================================================
# Effective config (merged from all sources)
# =============================================================================

_ENV_MAPPING = {
    # Config and state paths are consumed directly by their path helpers.
    # API Keys
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "OPENAI_API_KEY": "openai_api_key",
    "GOOGLE_API_KEY": "google_api_key",
    "TAVILY_API_KEY": "tavily_api_key",
    "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
    "CUSTOM_OPENAI_API_KEY": "custom_openai_api_key",
    "CUSTOM_OPENAI_BASE_URL": "custom_openai_base_url",
    "CUSTOM_ANTHROPIC_API_KEY": "custom_anthropic_api_key",
    "CUSTOM_ANTHROPIC_BASE_URL": "custom_anthropic_base_url",
    "OLLAMA_BASE_URL": "ollama_base_url",
    # LLM Settings
    "AUTOIDEA_PROVIDER": "provider",
    "AUTOIDEA_MODEL": "model",
    "AUTOIDEA_MAX_TOKENS": "max_tokens",
    # Path Settings
    "AUTOIDEA_WORKSPACE_DIR": "workspace_dir",
    "AUTOIDEA_RUNS_DIR": "runs_dir",
    "AUTOIDEA_MEMORY_DIR": "memory_dir",
    # UI and agent settings
    "AUTOIDEA_SHOW_THINKING": "show_thinking",
    "AUTOIDEA_AUTO_APPROVE": "auto_approve",
    "AUTOIDEA_SHELL_ALLOW_LIST": "shell_allow_list",
    "AUTOIDEA_ENABLE_ASK_USER": "enable_ask_user",
    # Agent Settings
    "AUTOIDEA_RECURSION_LIMIT": "recursion_limit",
    # Seed inputs
    "AUTOIDEA_SEED_PAPERS_FILE": "seed_papers_file",
    "AUTOIDEA_SEED_IDEAS_FILE": "seed_ideas_file",
    # Search behavior
    "AUTOIDEA_ENABLE_MOCK_FALLBACK": "enable_mock_fallback",
    # Web Search
    "ENABLE_WEB_SEARCH": "enable_web_search",
    "WEB_SEARCH_MAX_USES": "web_search_max_uses",
    "WEB_SEARCH_ALLOWED_DOMAINS": "web_search_allowed_domains",
    "WEB_SEARCH_BLOCKED_DOMAINS": "web_search_blocked_domains",
    # Pipeline Parameters
    "AUTOIDEA_MAX_SEARCH_QUERIES": "max_search_queries",
    "AUTOIDEA_MAX_DEBATE_ROUNDS": "max_debate_rounds",
    "AUTOIDEA_TARGET_PAPER_COUNT": "target_paper_count",
    "AUTOIDEA_DEEP_READING_TOP_K": "deep_reading_top_k",
    "AUTOIDEA_MAX_IDEAS_TO_GENERATE": "max_ideas_to_generate",
    "AUTOIDEA_TOP_K_RANKED": "top_k_ranked",
    # Elo Settings
    "AUTOIDEA_ELO_INITIAL_SCORE": "elo_initial_score",
    "AUTOIDEA_ELO_K_FACTOR": "elo_k_factor",
    # Memory Settings
    "AUTOIDEA_MEMORY_TRIGGER_MESSAGES": "memory_trigger_messages",
    "AUTOIDEA_EXTRACTION_MODEL": "extraction_model",
    # Context Compaction Settings
    "AUTOIDEA_ENABLE_AUTO_COMPACTION": "enable_auto_compaction",
    "AUTOIDEA_AUTO_COMPACTION_TRIGGER_MESSAGES": "auto_compaction_trigger_messages",
    "AUTOIDEA_AUTO_COMPACTION_KEEP_MESSAGES": "auto_compaction_keep_messages",
    "AUTOIDEA_AUTO_COMPACTION_TRIM_TOKENS": "auto_compaction_trim_tokens",
}


def get_active_env_override(config_key: str) -> str:
    """Return the non-empty environment variable overriding *config_key*.

    An empty string is treated as unset, matching ``get_effective_config``.
    """
    for env_key, mapped_key in _ENV_MAPPING.items():
        if mapped_key == config_key and os.getenv(env_key):
            return env_key
    return ""


def _coerce_env_value(config_key: str, value: str) -> Any:
    """Coerce an environment variable string to the correct field type.

    Uses the type annotation on ``AutoIdeaConfig`` to decide the target
    type.  Falls back to returning the raw string if the field is not
    found or the type is already ``str``.
    """
    field_types = {f.name: f.type for f in fields(AutoIdeaConfig)}
    target = field_types.get(config_key, "str")

    if target == "bool":
        return value.lower() in ("true", "1", "yes")
    if target == "int":
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if target == "float":
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


def get_effective_config() -> AutoIdeaConfig:
    """Get configuration with env vars merged in (env > file > defaults).

    Also loads .env files if present.
    """
    # Load .env files
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file)

    config = load_config()

    # Overlay env vars (with type coercion for non-string fields)
    for env_key, config_key in _ENV_MAPPING.items():
        env_val = os.getenv(env_key)
        if env_val:
            setattr(config, config_key, _coerce_env_value(config_key, env_val))

    return config


def apply_config_to_env(config: AutoIdeaConfig) -> None:
    """Push config values into environment variables.

    All values are converted to strings since os.environ only accepts strings.
    Also applies proxy settings to enable network access.
    """
    reverse_mapping = {v: k for k, v in _ENV_MAPPING.items()}
    for config_key, env_key in reverse_mapping.items():
        value = getattr(config, config_key, "")
        if value is not None and value != "":
            # Convert to string for environment variable
            if isinstance(value, bool):
                os.environ[env_key] = "true" if value else "false"
            else:
                os.environ[env_key] = str(value)

    # Apply proxy settings (critical for network access)
    if config.http_proxy:
        os.environ["http_proxy"] = config.http_proxy
        os.environ["HTTP_PROXY"] = config.http_proxy
    if config.https_proxy:
        os.environ["https_proxy"] = config.https_proxy
        os.environ["HTTPS_PROXY"] = config.https_proxy
    if config.no_proxy:
        os.environ["no_proxy"] = config.no_proxy
        os.environ["NO_PROXY"] = config.no_proxy


SUPPORTED_PROVIDERS = {
    "anthropic",
    "openai",
    "google-genai",
    "ollama",
    "custom-openai",
    "custom-anthropic",
}

_PROVIDER_KEY_FIELDS = {
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "google-genai": ("google_api_key", "GOOGLE_API_KEY"),
    "custom-openai": ("custom_openai_api_key", "CUSTOM_OPENAI_API_KEY"),
    "custom-anthropic": (
        "custom_anthropic_api_key",
        "CUSTOM_ANTHROPIC_API_KEY",
    ),
}

_CUSTOM_BASE_URL_FIELDS = {
    "custom-openai": ("custom_openai_base_url", "CUSTOM_OPENAI_BASE_URL"),
    "custom-anthropic": (
        "custom_anthropic_base_url",
        "CUSTOM_ANTHROPIC_BASE_URL",
    ),
}

_OPTIONAL_PROVIDER_MODULES = {
    "google-genai": ("langchain_google_genai", "google"),
    "ollama": ("langchain_ollama", "ollama"),
}


def validate_runtime_config(config: AutoIdeaConfig) -> list[str]:
    """Return actionable errors that would prevent an agent run."""
    errors: list[str] = []
    provider = str(config.provider).strip().lower()

    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        errors.append(f"Unsupported provider '{config.provider}'. Choose one of: {supported}.")
        return errors

    if not str(config.model).strip():
        errors.append("AUTOIDEA_MODEL must not be empty.")

    key_requirement = _PROVIDER_KEY_FIELDS.get(provider)
    if key_requirement:
        field_name, env_name = key_requirement
        if not str(getattr(config, field_name, "")).strip():
            errors.append(f"Missing {env_name} for provider '{provider}'.")

    base_requirement = _CUSTOM_BASE_URL_FIELDS.get(provider)
    if base_requirement:
        field_name, env_name = base_requirement
        base_url = str(getattr(config, field_name, "")).strip()
        if not base_url:
            errors.append(f"Missing {env_name} for provider '{provider}'.")
        elif not base_url.startswith(("http://", "https://")):
            errors.append(f"{env_name} must start with http:// or https://.")

    optional_dependency = _OPTIONAL_PROVIDER_MODULES.get(provider)
    if optional_dependency:
        module_name, extra_name = optional_dependency
        if find_spec(module_name) is None:
            errors.append(
                f"Provider '{provider}' requires the '{extra_name}' extra: "
                f'python -m pip install -e ".[web,{extra_name}]"'
            )

    for field_name, env_name in (
        ("max_tokens", "AUTOIDEA_MAX_TOKENS"),
        ("recursion_limit", "AUTOIDEA_RECURSION_LIMIT"),
    ):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{env_name} must be a positive integer.")

    return errors
