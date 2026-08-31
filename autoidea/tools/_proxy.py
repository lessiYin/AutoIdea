"""HTTP proxy configuration for academic API requests.

Provides a shared httpx.AsyncClient factory that respects proxy
settings from multiple sources with the following priority:
    1. Environment variables (HTTPS_PROXY, HTTP_PROXY)
    2. config.yaml settings
    3. No proxy
"""

from __future__ import annotations

import os
from typing import Optional

import httpx


def _get_proxy_from_config() -> Optional[str]:
    """Get proxy URL from config.yaml file.

    This ensures proxy settings are applied even if environment
    variables are not set (e.g., when autoidea starts from a
    fresh shell).
    """
    try:
        from autoidea.config import load_config
        config = load_config()
        # Check https_proxy first, then http_proxy
        proxy = config.https_proxy or config.http_proxy
        if proxy:
            return proxy
    except Exception:
        pass
    return None


def _get_proxy_url() -> Optional[str]:
    """Get proxy URL from environment variables or config file.

    Priority: env vars > config.yaml
    """
    # 1. Check environment variables first (highest priority)
    env_proxy = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )
    if env_proxy:
        return env_proxy

    # 2. Fall back to config file
    return _get_proxy_from_config()


def _apply_proxy_to_env() -> None:
    """Apply proxy from config to environment variables if not already set.

    This ensures that any subprocess or library that reads environment
    variables for proxy settings will work correctly.
    """
    proxy = _get_proxy_from_config()
    if proxy:
        # Only set if not already set in environment
        if not os.getenv("http_proxy") and not os.getenv("HTTP_PROXY"):
            os.environ["http_proxy"] = proxy
            os.environ["HTTP_PROXY"] = proxy
        if not os.getenv("https_proxy") and not os.getenv("HTTPS_PROXY"):
            os.environ["https_proxy"] = proxy
            os.environ["HTTPS_PROXY"] = proxy


# Apply proxy on module import (early initialization)
_apply_proxy_to_env()


def get_async_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with optional proxy support.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        An httpx.AsyncClient instance.
    """
    proxy_url = _get_proxy_url()
    kwargs: dict = {
        "timeout": timeout,
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
        "follow_redirects": True,
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url

    return httpx.AsyncClient(**kwargs)


def get_sync_client(timeout: float = 30.0) -> httpx.Client:
    """Create an httpx.Client (sync) with optional proxy support.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        An httpx.Client instance.
    """
    proxy_url = _get_proxy_url()
    kwargs: dict = {
        "timeout": timeout,
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
        "follow_redirects": True,
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url

    return httpx.Client(**kwargs)
