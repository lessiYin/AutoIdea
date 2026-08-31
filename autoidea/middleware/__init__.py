"""Middleware package for AutoIdea.

Re-exports middleware classes and factory functions for the
research agent pipeline.
"""

from .ask_user import (
    AskUserMiddleware,
    AskUserRequest,
    AskUserWidgetResult,
    Choice,
    Question,
)
from .memory import (
    AutoIdeaMemoryMiddleware,
    AutoIdeaMemoryState,
    ExtractedMemory,
    create_memory_middleware,
)
from .source_verify import SourceVerifyMiddleware
from .tool_error_handler import ToolErrorHandlerMiddleware
from .tool_call_serialization import ToolCallSerializationMiddleware
from .model_retry import ModelRetryMiddleware

__all__ = [
    "AskUserMiddleware",
    "AskUserRequest",
    "AskUserWidgetResult",
    "Choice",
    "AutoIdeaMemoryMiddleware",
    "AutoIdeaMemoryState",
    "ExtractedMemory",
    "Question",
    "SourceVerifyMiddleware",
    "ToolErrorHandlerMiddleware",
    "ToolCallSerializationMiddleware",
    "create_memory_middleware",
    "ModelRetryMiddleware",
]
