"""Multi-provider LLM model registry for AutoIdea.

Supports Anthropic, OpenAI, Google GenAI, Ollama, and custom OpenAI-compatible endpoints.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_MAX_TOKENS = 16384


def _ensure_openai_stream_chunk_timeout_default() -> None:
    """Disable LangChain OpenAI stream idle timeout unless user configured it.

    AutoIdea often runs long, tool-heavy research stages where the model may
    stay silent while planning or waiting on tool results. LangChain OpenAI's
    default stream idle timeout can abort those valid runs after 120 seconds.
    """
    os.environ.setdefault("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "0")


def _get_max_tokens() -> int:
    """Get max_tokens from config, with fallback to default.

    Priority: AUTOIDEA_MAX_TOKENS env > config.yaml > default
    """
    # 1. Check environment variable
    env_val = os.getenv("AUTOIDEA_MAX_TOKENS")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass

    # 2. Check config file
    try:
        from ..config import load_config
        config = load_config()
        cfg_val = getattr(config, "max_tokens", None)
        if cfg_val:
            return int(cfg_val)
    except Exception:
        pass

    # 3. Default
    return DEFAULT_MAX_TOKENS


def _get_reasoning_effort_kwargs() -> dict[str, str]:
    """Return OpenAI-compatible reasoning-effort kwargs from the environment."""
    value = os.getenv("AUTOIDEA_REASONING_EFFORT", "").strip()
    if not value:
        return {}
    return {"reasoning_effort": value}

# Model registry: short_name -> (provider, full_model_id)
MODELS: dict[str, tuple[str, str]] = {
    # Anthropic
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "claude-opus-4-6": ("anthropic", "claude-opus-4-6"),
    "claude-sonnet-4-5": ("anthropic", "claude-sonnet-4-5"),
    "claude-haiku-4-5": ("anthropic", "claude-haiku-4-5"),
    # OpenAI
    "gpt-5.6-sol": ("openai", "gpt-5.6-sol"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "o1": ("openai", "o1"),
    "o1-mini": ("openai", "o1-mini"),
    "o3": ("openai", "o3"),
    "o3-mini": ("openai", "o3-mini"),
    # Google GenAI
    "gemini-2.5-pro": ("google-genai", "gemini-2.5-pro"),
    "gemini-2.5-flash": ("google-genai", "gemini-2.5-flash"),
    "gemini-2.0-flash": ("google-genai", "gemini-2.0-flash"),
    # Ollama (local)
    "llama3": ("ollama", "llama3"),
    "llama3:70b": ("ollama", "llama3:70b"),
    "mistral": ("ollama", "mistral"),
    "qwen2.5": ("ollama", "qwen2.5"),
}


def list_models() -> dict[str, tuple[str, str]]:
    """Return the full model registry."""
    return dict(MODELS)


def get_chat_model(
    model: str | None = None,
    provider: str | None = None,
    **kwargs: Any,
):
    """Create a chat model instance.

    Args:
        model: Model short name or full ID.
        provider: Provider override ('anthropic', 'openai', 'google-genai',
                  'ollama', 'custom-openai', 'custom-anthropic').
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        A LangChain chat model instance.
    """
    model = model or DEFAULT_MODEL
    resolved_provider = provider

    # Resolve from registry if short name
    if model in MODELS:
        reg_provider, full_id = MODELS[model]
        if resolved_provider is None:
            resolved_provider = reg_provider
        model = full_id
    elif resolved_provider is None:
        resolved_provider = "openai"

    # Build model based on provider
    if resolved_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = kwargs.pop("api_key", None) or os.getenv("ANTHROPIC_API_KEY", "")
        base_url = kwargs.pop("base_url", None) or os.getenv("ANTHROPIC_BASE_URL")

        model_kwargs = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", _get_max_tokens()),
        }
        if api_key:
            model_kwargs["api_key"] = api_key
        if base_url:
            model_kwargs["base_url"] = base_url

        # Extended thinking support for Claude
        thinking = kwargs.pop("thinking", None)
        if thinking:
            model_kwargs["thinking"] = thinking

        model_kwargs.update(kwargs)
        chat_model = ChatAnthropic(**model_kwargs)
        chat_model._is_claude_model = True  # type: ignore[attr-defined]
        return chat_model

    elif resolved_provider == "openai":
        from langchain_openai import ChatOpenAI

        _ensure_openai_stream_chunk_timeout_default()
        api_key = kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY", "")
        model_kwargs = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", _get_max_tokens()),
        }
        model_kwargs.update(_get_reasoning_effort_kwargs())
        if api_key:
            model_kwargs["api_key"] = api_key
        model_kwargs.update(kwargs)
        return ChatOpenAI(**model_kwargs)

    elif resolved_provider == "google-genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = kwargs.pop("api_key", None) or os.getenv("GOOGLE_API_KEY", "")
        model_kwargs = {
            "model": model,
        }
        if api_key:
            model_kwargs["google_api_key"] = api_key
        model_kwargs.update(kwargs)
        return ChatGoogleGenerativeAI(**model_kwargs)

    elif resolved_provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = kwargs.pop("base_url", None) or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        model_kwargs = {
            "model": model,
            "base_url": base_url,
        }
        model_kwargs.update(kwargs)
        return ChatOllama(**model_kwargs)

    elif resolved_provider == "custom-openai":
        from langchain_openai import ChatOpenAI

        _ensure_openai_stream_chunk_timeout_default()
        api_key = kwargs.pop("api_key", None) or os.getenv("CUSTOM_OPENAI_API_KEY", "")
        base_url = kwargs.pop("base_url", None) or os.getenv("CUSTOM_OPENAI_BASE_URL", "")
        model_kwargs = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", _get_max_tokens()),
        }
        model_kwargs.update(_get_reasoning_effort_kwargs())
        if api_key:
            model_kwargs["api_key"] = api_key
        if base_url:
            model_kwargs["base_url"] = base_url
        model_kwargs.update(kwargs)
        return ChatOpenAI(**model_kwargs)

    elif resolved_provider == "custom-anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = kwargs.pop("api_key", None) or os.getenv("CUSTOM_ANTHROPIC_API_KEY", "")
        base_url = kwargs.pop("base_url", None) or os.getenv("CUSTOM_ANTHROPIC_BASE_URL", "")
        model_kwargs = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", _get_max_tokens()),
        }
        if api_key:
            model_kwargs["api_key"] = api_key
        if base_url:
            model_kwargs["base_url"] = base_url
        model_kwargs.update(kwargs)
        chat_model = ChatAnthropic(**model_kwargs)
        chat_model._is_claude_model = True  # type: ignore[attr-defined]
        return chat_model

    else:
        raise ValueError(
            f"Unknown provider: {resolved_provider!r}. "
            f"Supported: anthropic, openai, google-genai, ollama, "
            f"custom-openai, custom-anthropic"
        )


def bind_native_tools(model: Any, native_tools: list[dict]) -> Any:
    """Bind native tool definitions, such as Claude Web Search, to a model.

    Wrap the model's ``bind_tools`` method so every call appends the native
    tool definitions to the supplied tool list.

    Args:
        model: LangChain chat model instance.
        native_tools: List of Anthropic native tool dicts
            (e.g. ``{"type": "web_search_20250305", ...}``).

    Returns:
        The same model instance with patched ``bind_tools``.
    """
    if not native_tools:
        return model

    original_bind_tools = model.bind_tools

    def patched_bind_tools(tools, **kwargs):  # type: ignore[override]
        # Merge LangChain tools with the native tool definitions.
        merged = list(tools) + native_tools
        return original_bind_tools(merged, **kwargs)

    # Recent LangChain chat models are Pydantic models that reject normal
    # assignment for methods not declared as fields. Bypass the Pydantic setter
    # so the instance-level wrapper remains compatible across versions.
    object.__setattr__(model, "bind_tools", patched_bind_tools)
    logger.info(
        "Patched bind_tools with %d native tool(s): %s",
        len(native_tools),
        [t.get("name", t.get("type", "?")) for t in native_tools],
    )
    return model
