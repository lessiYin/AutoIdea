"""Retry transient provider failures around model calls."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:
    try:
        from deepagents_langgraph.middleware import AgentMiddleware
    except ImportError:
        AgentMiddleware = object

logger = logging.getLogger(__name__)

DEFAULT_MODEL_RETRY_ATTEMPTS = 3
DEFAULT_MODEL_RETRY_BACKOFF_S = 2.0
DEFAULT_MODEL_RATE_LIMIT_BACKOFF_S = 30.0


def _get_model_retry_attempts() -> int:
    raw = os.getenv("AUTOIDEA_MODEL_RETRY_ATTEMPTS", str(DEFAULT_MODEL_RETRY_ATTEMPTS))
    try:
        attempts = int(raw)
    except (TypeError, ValueError):
        attempts = DEFAULT_MODEL_RETRY_ATTEMPTS
    return max(1, attempts)


def _get_model_retry_backoff() -> float:
    raw = os.getenv("AUTOIDEA_MODEL_RETRY_BACKOFF_S", str(DEFAULT_MODEL_RETRY_BACKOFF_S))
    try:
        backoff = float(raw)
    except (TypeError, ValueError):
        backoff = DEFAULT_MODEL_RETRY_BACKOFF_S
    return max(0.0, backoff)


def _get_model_rate_limit_backoff() -> float:
    raw = os.getenv(
        "AUTOIDEA_MODEL_RATE_LIMIT_BACKOFF_S",
        str(DEFAULT_MODEL_RATE_LIMIT_BACKOFF_S),
    )
    try:
        backoff = float(raw)
    except (TypeError, ValueError):
        backoff = DEFAULT_MODEL_RATE_LIMIT_BACKOFF_S
    return max(0.0, backoff)


def _model_error_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    return (
        _model_error_status(exc) == 429
        or name.endswith("RateLimitError")
        or "too many requests" in str(exc).casefold()
    )


def _is_transient_model_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    status = _model_error_status(exc)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True

    name = exc.__class__.__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "OpenAIConnectionError",
        "OpenAITimeoutError",
        "RateLimitError",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
    }


def _retry_after_seconds(exc: BaseException) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return 0.0
    try:
        value = headers.get("retry-after", "")
        return max(0.0, float(value)) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


def _retry_delay(attempt_index: int, exc: BaseException) -> float:
    base = (
        _get_model_rate_limit_backoff()
        if _is_rate_limit_error(exc)
        else _get_model_retry_backoff()
    )
    return max(base * (2 ** attempt_index), _retry_after_seconds(exc))


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient model-provider failures before failing the run."""

    name = "model_retry"

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        attempts = _get_model_retry_attempts()
        for attempt_index in range(attempts):
            try:
                return handler(request)
            except Exception as exc:
                if attempt_index >= attempts - 1 or not _is_transient_model_error(exc):
                    raise
                delay = _retry_delay(attempt_index, exc)
                logger.warning(
                    "Model call failed with transient %s; retrying %d/%d in %.1fs",
                    exc.__class__.__name__,
                    attempt_index + 1,
                    attempts - 1,
                    delay,
                )
                if delay:
                    time.sleep(delay)
        raise RuntimeError("unreachable model retry state")

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        attempts = _get_model_retry_attempts()
        for attempt_index in range(attempts):
            try:
                return await handler(request)
            except Exception as exc:
                if attempt_index >= attempts - 1 or not _is_transient_model_error(exc):
                    raise
                delay = _retry_delay(attempt_index, exc)
                logger.warning(
                    "Model call failed with transient %s; retrying %d/%d in %.1fs",
                    exc.__class__.__name__,
                    attempt_index + 1,
                    attempts - 1,
                    delay,
                )
                if delay:
                    await asyncio.sleep(delay)
        raise RuntimeError("unreachable model retry state")
