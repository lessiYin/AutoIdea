from __future__ import annotations

import asyncio

import httpx

from autoidea.middleware.model_retry import ModelRetryMiddleware


class DummyRequest:
    pass


def test_model_retry_retries_transient_sync_read_error(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_BACKOFF_S", "0")
    middleware = ModelRetryMiddleware()
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadError("stream interrupted")
        return "ok"

    assert middleware.wrap_model_call(DummyRequest(), handler) == "ok"
    assert attempts == 2


def test_model_retry_retries_transient_async_read_error(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_BACKOFF_S", "0")
    middleware = ModelRetryMiddleware()
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadError("stream interrupted")
        return "ok"

    async def run_call() -> str:
        return await middleware.awrap_model_call(DummyRequest(), handler)

    assert asyncio.run(run_call()) == "ok"
    assert attempts == 2


def test_model_retry_retries_langchain_openai_connection_error(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_BACKOFF_S", "0")
    middleware = ModelRetryMiddleware()
    attempts = 0

    class OpenAIConnectionError(Exception):
        pass

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OpenAIConnectionError("Connection error.")
        return "ok"

    async def run_call() -> str:
        return await middleware.awrap_model_call(DummyRequest(), handler)

    assert asyncio.run(run_call()) == "ok"
    assert attempts == 2


def test_model_retry_waits_for_rate_limit_window(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("AUTOIDEA_MODEL_RATE_LIMIT_BACKOFF_S", "30")
    middleware = ModelRetryMiddleware()
    attempts = 0
    delays: list[float] = []

    class OpenAIRateLimitError(Exception):
        def __init__(self) -> None:
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            self.response = httpx.Response(
                429,
                request=request,
                headers={"Retry-After": "45"},
            )
            super().__init__("Too many requests")

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OpenAIRateLimitError()
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def run_call() -> str:
        return await middleware.awrap_model_call(DummyRequest(), handler)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    assert asyncio.run(run_call()) == "ok"
    assert attempts == 2
    assert delays == [45.0]


def test_model_retry_does_not_retry_non_transient_error(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MODEL_RETRY_ATTEMPTS", "3")
    middleware = ModelRetryMiddleware()
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        raise ValueError("bad prompt")

    try:
        middleware.wrap_model_call(DummyRequest(), handler)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError was not raised")

    assert attempts == 1
