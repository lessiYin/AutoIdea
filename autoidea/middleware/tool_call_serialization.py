"""Middleware that serializes model-emitted tool calls."""

from __future__ import annotations

from typing import Any

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain.agents.middleware.types import ModelResponse
except ImportError:
    try:
        from deepagents_langgraph.middleware import AgentMiddleware
        from langchain.agents.middleware.types import ModelResponse
    except ImportError:
        AgentMiddleware = object
        ModelResponse = Any  # type: ignore[misc,assignment]

from langchain_core.messages import AIMessage


def _tool_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or call.get("function", {}).get("name") or "unknown")
    return str(getattr(call, "name", "unknown"))


def _serialize_ai_message(message: AIMessage) -> AIMessage:
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if len(tool_calls) <= 1:
        return message

    skipped = ", ".join(_tool_name(call) for call in tool_calls[1:])
    note = (
        "\n\n[AutoIdea concurrency guard] Only the first tool call was allowed "
        "to execute in this step. The following tool calls were intentionally "
        f"deferred to avoid provider concurrency-limit failures: {skipped}. "
        "After the first tool result returns, issue the next required tool "
        "call in a separate step."
    )
    return message.model_copy(
        update={
            # Tool-bearing prose is not needed to execute the retained call.
            # Keeping it is actively harmful for OpenAI-compatible models that
            # echo prior assistant prose on every replanning turn: the echoed
            # text can grow geometrically and crowd the remaining tool calls
            # out of the context window.  Replace it with one bounded reminder.
            "content": note.strip(),
            "tool_calls": tool_calls[:1],
        }
    )


def _serialize_response(response: Any) -> Any:
    if isinstance(response, AIMessage):
        return _serialize_ai_message(response)

    result = getattr(response, "result", None)
    if not isinstance(result, list):
        return response

    changed = False
    new_result = []
    for message in result:
        if isinstance(message, AIMessage):
            new_message = _serialize_ai_message(message)
            changed = changed or new_message is not message
            new_result.append(new_message)
        else:
            new_result.append(message)

    if not changed:
        return response

    if isinstance(response, ModelResponse):
        return ModelResponse(
            result=new_result,
            structured_response=getattr(response, "structured_response", None),
        )
    try:
        return response.model_copy(update={"result": new_result})
    except AttributeError:
        response.result = new_result
        return response


class ToolCallSerializationMiddleware(AgentMiddleware):
    """Allow at most one model-emitted tool call per agent step."""

    name = "tool_call_serialization"

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        response = handler(request)
        return _serialize_response(response)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        response = await handler(request)
        return _serialize_response(response)
