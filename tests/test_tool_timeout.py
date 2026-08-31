from __future__ import annotations

import asyncio
from dataclasses import dataclass

from autoidea.middleware.tool_error_handler import ToolErrorHandlerMiddleware
from autoidea.tools import scholar


@dataclass
class DummyToolRequest:
    tool_call_id: str = "call_timeout"
    tool_call: dict | None = None


def test_async_tool_calls_timeout_and_return_error_message(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_TOOL_CALL_TIMEOUT_S", "0.01")
    middleware = ToolErrorHandlerMiddleware()

    async def never_returns(_request):
        await asyncio.sleep(60)
        return "unreachable"

    async def run_call():
        return await middleware.awrap_tool_call(DummyToolRequest(), never_returns)

    message = asyncio.run(run_call())

    assert message.status == "error"
    assert message.tool_call_id == "call_timeout"
    assert "timed out" in str(message.content)


def test_composite_subagent_task_is_not_cut_off_by_single_tool_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUTOIDEA_TOOL_CALL_TIMEOUT_S", "0.01")
    middleware = ToolErrorHandlerMiddleware()

    async def completes_after_single_call_deadline(_request):
        await asyncio.sleep(0.02)
        return "complete"

    async def run_call():
        request = DummyToolRequest(tool_call={"name": "task"})
        return await middleware.awrap_tool_call(
            request,
            completes_after_single_call_deadline,
        )

    assert asyncio.run(run_call()) == "complete"


def test_multi_source_search_deadline_returns_partial_results(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MULTI_SOURCE_SEARCH_TIMEOUT_S", "0.01")

    async def slow_source(_query, limit=10):
        await asyncio.sleep(60)
        return []

    async def fast_source(_query, limit=10):
        return [
            {
                "title": "Training-Free Long Video Agent Search",
                "authors": ["A"],
                "year": 2026,
                "venue": "Test",
                "url": "https://example.com/paper",
                "source": "fast",
            }
        ]

    monkeypatch.setattr(
        scholar,
        "_SOURCE_REGISTRY",
        {"slow": slow_source, "fast": fast_source},
    )
    monkeypatch.setattr(scholar, "_DEFAULT_SOURCES", ["slow", "fast"])

    result = asyncio.run(
        scholar.multi_source_search.ainvoke(
            {
                "query": "training-free long video agent",
                "limit": 1,
                "sources": "slow,fast",
            }
        )
    )

    assert "Training-Free Long Video Agent Search" in result
    assert "slow: search timed out" in result


def test_multi_source_search_drains_late_source_exceptions(monkeypatch) -> None:
    monkeypatch.setenv("AUTOIDEA_MULTI_SOURCE_SEARCH_TIMEOUT_S", "0.01")
    loop_exceptions: list[dict] = []

    async def stubborn_source(_query, limit=10):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)
            raise ValueError("late network cleanup failure")
        return []

    async def fast_source(_query, limit=10):
        return [
            {
                "title": "Robust Long Video Search",
                "authors": ["A"],
                "year": 2026,
                "venue": "Test",
                "url": "https://example.com/robust",
                "source": "fast",
            }
        ]

    monkeypatch.setattr(
        scholar,
        "_SOURCE_REGISTRY",
        {"stubborn": stubborn_source, "fast": fast_source},
    )
    monkeypatch.setattr(scholar, "_DEFAULT_SOURCES", ["stubborn", "fast"])

    async def run_search() -> str:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: loop_exceptions.append(context))
        result = await scholar.multi_source_search.ainvoke(
            {
                "query": "robust long video search",
                "limit": 1,
                "sources": "stubborn,fast",
            }
        )
        await asyncio.sleep(0.05)
        return result

    result = asyncio.run(run_search())

    assert "Robust Long Video Search" in result
    assert "stubborn: search timed out" in result
    assert loop_exceptions == []
