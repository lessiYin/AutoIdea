from __future__ import annotations

from dataclasses import dataclass

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from autoidea.middleware.tool_call_serialization import ToolCallSerializationMiddleware


@dataclass
class DummyRequest:
    pass


def test_tool_call_serialization_keeps_only_first_tool_call() -> None:
    middleware = ToolCallSerializationMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "semantic_scholar_search", "args": {"query": "a"}, "id": "call_1", "type": "tool_call"},
                    {"name": "arxiv_search", "args": {"query": "b"}, "id": "call_2", "type": "tool_call"},
                ],
            )
        ]
    )

    def handler(_request):
        return response

    guarded = middleware.wrap_model_call(DummyRequest(), handler)
    message = guarded.result[0]

    assert isinstance(message, AIMessage)
    assert [call["id"] for call in message.tool_calls] == ["call_1"]
    assert "Only the first tool call" in message.content
    assert "arxiv_search" in message.content


def test_tool_call_serialization_discards_tool_bearing_prose_to_prevent_echo_growth() -> None:
    middleware = ToolCallSerializationMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(
                content="verbose planning text that must not accumulate",
                tool_calls=[
                    {"name": "first", "args": {}, "id": "call_1", "type": "tool_call"},
                    {"name": "second", "args": {}, "id": "call_2", "type": "tool_call"},
                ],
            )
        ]
    )

    guarded = middleware.wrap_model_call(DummyRequest(), lambda _request: response)

    assert "verbose planning text" not in guarded.result[0].content
    assert "second" in guarded.result[0].content


def test_tool_call_serialization_leaves_single_tool_call_unchanged() -> None:
    middleware = ToolCallSerializationMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(
                content="single",
                tool_calls=[
                    {"name": "read_workspace_file", "args": {"file_path": "x"}, "id": "call_1", "type": "tool_call"}
                ],
            )
        ]
    )

    guarded = middleware.wrap_model_call(DummyRequest(), lambda _request: response)

    assert guarded.result[0].content == "single"
    assert guarded.result[0].tool_calls == response.result[0].tool_calls
