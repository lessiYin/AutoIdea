"""Ask user middleware for agent-initiated interactive questions (AutoIdea v3.0).

Enables the agent to proactively ask the user for clarification during
research workflows.  The middleware registers an ``ask_user`` tool that
uses LangGraph ``interrupt()`` to pause the graph; the UI layer collects
answers and resumes with ``Command(resume={...})``.

Modeled after EvoScientist's ``AskUserMiddleware`` with a system prompt
tailored for AutoIdea's multi-stage research pipeline and its specific
HITL (Human-in-the-Loop) checkpoints.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    pass

from typing import NotRequired

# ---------------------------------------------------------------------------
# AgentMiddleware import with graceful fallback
# ---------------------------------------------------------------------------

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:
    try:
        from deepagents_langgraph.middleware import AgentMiddleware
    except ImportError:  # pragma: no cover — fallback for standalone usage
        AgentMiddleware = object  # type: ignore[assignment,misc]

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt
from pydantic import BeforeValidator, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Choice(TypedDict):
    """A single choice option for a multiple-choice question."""

    value: Annotated[str, Field(description="The display label for this choice.")]


class Question(TypedDict):
    """A question to ask the user."""

    question: Annotated[str, Field(description="The question text to display.")]

    type: Annotated[
        Literal["text", "multiple_choice"],
        Field(
            description=(
                "Question type. 'text' for free-form input, 'multiple_choice' "
                "for predefined options."
            )
        ),
    ]

    choices: NotRequired[
        Annotated[
            list[Choice],
            Field(
                description=(
                    "Options for multiple_choice questions. An 'Other' "
                    "free-form option is always appended automatically."
                )
            ),
        ]
    ]

    required: NotRequired[
        Annotated[
            bool,
            Field(
                description=(
                    "Whether the user must answer. Defaults to true if omitted."
                )
            ),
        ]
    ]


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def _coerce_questions_list(v: Any) -> Any:
    """Accept a JSON string and parse it to a list.

    LLMs sometimes serialize the ``questions`` argument as a JSON string
    instead of a native list.  This :class:`~pydantic.BeforeValidator`
    transparently handles that case so the tool invocation succeeds on the
    first try.
    """
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return v


# Annotated type for the tool parameter -- schema stays ``list[Question]``
# for the LLM, but runtime accepts JSON strings too.
QuestionsList = Annotated[list[Question], BeforeValidator(_coerce_questions_list)]


# ---------------------------------------------------------------------------
# Request / response data types
# ---------------------------------------------------------------------------


class AskUserRequest(TypedDict):
    """Request payload sent via ``interrupt()`` when asking the user questions."""

    type: Literal["ask_user"]

    questions: list[Question]

    tool_call_id: str


class AskUserAnswered(TypedDict):
    """Widget result when the user submits answers."""

    type: Literal["answered"]
    """Discriminator tag, always ``'answered'``."""

    answers: list[str]
    """User-provided answers, one per question."""


class AskUserCancelled(TypedDict):
    """Widget result when the user cancels the prompt."""

    type: Literal["cancelled"]
    """Discriminator tag, always ``'cancelled'``."""


# Discriminated union for the ask_user widget Future result.
AskUserWidgetResult = AskUserAnswered | AskUserCancelled


# ---------------------------------------------------------------------------
# Prompts (AutoIdea research-context)
# ---------------------------------------------------------------------------


ASK_USER_TOOL_DESCRIPTION = """\
Ask the user one or more questions when you need clarification or specific input
before proceeding with a research task.

Each question can be:
- "text": Free-form text response
- "multiple_choice": User selects from predefined options (an "Other" option is
  always appended)

Use when: dataset/model/framework selection is ambiguous, experiment parameters
are unclear, research scope needs clarification, a significant plan change needs
confirmation, resource estimation before heavy compute, execution failures need
user-guided recovery, or you are at an AutoIdea HITL checkpoint that requires
explicit user approval.

Do NOT use for trivial decisions or questions answerable from context, memory,
web search, or prior conversation."""


ASK_USER_SYSTEM_PROMPT = """\
## `ask_user` -- Interactive Clarification Tool (AutoIdea v3.0)

You have access to the `ask_user` tool to ask the user questions when you need
information that cannot be determined from the conversation, loaded skills,
or available tools.

### When to use `ask_user`:
- **Dataset or benchmark selection**: "Which dataset should I use: CIFAR-10, \
ImageNet, or a custom dataset?"
- **Experiment parameters**: "What GPU memory budget should I target? What \
batch size range is acceptable?"
- **Research scope**: "Should I focus on accuracy improvements or inference \
speed?"
- **Methodology choice**: "For the baseline comparison, should I reimplement \
from the paper or use the official repo?"
- **Paper or report preferences**: "Which venue format should I target: \
NeurIPS, ICML, or ICLR?"
- **Ambiguous instructions**: When the user's request has multiple valid \
interpretations
- **Resource constraints**: When the approach depends on available compute, \
time, or data

### Resource & execution awareness (`ask_user` is especially valuable here):
- **Pre-execution estimation**: Before heavy compute (training, large-scale \
eval), estimate time/memory/cost and confirm. E.g. "Training needs ~2 h and \
~16 GB GPU. Proceed, or reduce model size?"
- **Timeout & failure recovery**: When a command times out (exit code 124) or \
fails with OOM/CUDA errors, present recovery options. E.g. "Training timed \
out. Options: (A) run in background, (B) reduce epochs, (C) switch to smaller \
model"
- **Intermediate checkpoints**: When results diverge from expectations, ask \
before continuing. E.g. "Baseline accuracy 62 % vs expected 80 %. Investigate \
or proceed?"

### AutoIdea HITL Checkpoints:
The following pipeline stages are **mandatory** HITL checkpoints -- you MUST
use `ask_user` at each of these decision points:

1. **Stage 7 -- Gap Identification Review**
   After identifying research gaps from the literature survey, present the
   gaps to the user for review and confirmation before proceeding.  Ask
   whether the gaps are relevant, if any should be added or removed, and
   which gaps to prioritize.

2. **Stage 9 -- Research Idea Approval**
   Before committing to a research idea, present the candidate idea(s) to
   the user.  Include a brief summary of each idea's motivation, novelty,
   and estimated feasibility.  Proceed only after the user explicitly
   approves an idea (or asks for revisions).

3. **Stage 10 -- Debate Verdict Acceptance**
   After the internal idea-debate process produces a verdict (accept,
   revise, or reject), present the debate summary and verdict to the user
   for final acceptance.  The user may override the verdict or request
   additional debate rounds.

At each checkpoint, include an `auto_continue` choice labelled "Continue
automatically / 后续全自动". Selecting it approves the current checkpoint and
means later checkpoints must continue without waiting for another answer.

This middleware is installed only in manual-checkpoint mode. In that mode,
failure to consult the user at these checkpoints is a pipeline error.

### When NOT to use `ask_user`:
- Simple yes/no decisions -- proceed with your best judgment
- Information already provided in the conversation or memory
- Trivial choices that don't meaningfully affect outcomes
- Questions you can answer by searching with `tavily_search` or reading files
- During sub-agent execution (only the main agent should ask the user)

### Guidelines:
- Be concise and specific -- avoid vague, open-ended questions
- Use `multiple_choice` when there are 2-5 clear options
- Use `text` for open-ended input (preferred language, custom parameters, etc.)
- Group related questions into a **single** `ask_user` call (max 5 questions)
- Never ask more than once per decision point -- respect the user's time
- After receiving answers, summarize what you understood before proceeding"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_questions(questions: list[Question]) -> None:
    """Validate ask_user question structure before interrupting.

    Raises:
        ValueError: If the questions list or an individual question is invalid.
    """
    if not questions:
        msg = "ask_user requires at least one question"
        raise ValueError(msg)

    for q in questions:
        question_text = q.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            msg = "ask_user questions must have non-empty 'question' text"
            raise ValueError(msg)

        question_type = q.get("type")
        if question_type not in {"text", "multiple_choice"}:
            msg = f"unsupported ask_user question type: {question_type!r}"
            raise ValueError(msg)

        if question_type == "multiple_choice" and not q.get("choices"):
            msg = (
                f"multiple_choice question "
                f"{q.get('question')!r} requires a "
                f"non-empty 'choices' list"
            )
            raise ValueError(msg)

        if question_type == "text" and q.get("choices"):
            msg = (
                f"text question {q.get('question')!r} must not define "
                f"'choices'"
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _parse_answers(
    response: object,
    questions: list[Question],
    tool_call_id: str,
) -> Command[Any]:
    """Parse an interrupt response into a ``Command`` with a ``ToolMessage``.

    Supports explicit status signaling from the adapter layer:

    - ``answered`` (default): consume provided ``answers``
    - ``cancelled``: synthesize ``(cancelled)`` answers
    - ``error``: synthesize ``(error: ...)`` answers

    Malformed payloads are converted into explicit error answers instead of
    silently defaulting to ``(no answer)``.
    """
    status: str = "answered"
    error_text: str | None = None
    answers: list[str]

    # ---- Validate top-level shape ----
    if not isinstance(response, dict):
        logger.error(
            "ask_user received malformed resume payload "
            "(expected dict, got %s); returning explicit error answers",
            type(response).__name__,
        )
        answers = []
        status = "error"
        error_text = "invalid ask_user response payload"
    else:
        response_dict = cast("dict[str, Any]", response)
        response_status = response_dict.get("status")
        if isinstance(response_status, str):
            status = response_status

        # ---- Extract answers list ----
        if "answers" not in response_dict:
            if status == "answered":
                logger.error(
                    "ask_user received resume payload without 'answers'; "
                    "returning explicit error answers"
                )
                answers = []
                status = "error"
                error_text = "missing ask_user answers payload"
            else:
                answers = []
        else:
            raw_answers = response_dict["answers"]
            if isinstance(raw_answers, list):
                answers = [str(answer) for answer in raw_answers]
            else:
                logger.error(
                    "ask_user received non-list 'answers' payload (%s); "
                    "returning explicit error answers",
                    type(raw_answers).__name__,
                )
                answers = []
                status = "error"
                error_text = "invalid ask_user answers payload"

        # ---- Interpret status ----
        if status == "error":
            response_error = response_dict.get("error")
            if isinstance(response_error, str) and response_error:
                error_text = response_error
        elif status == "cancelled":
            answers = ["(cancelled)" for _ in questions]
        elif status == "answered":
            if len(answers) != len(questions):
                logger.warning(
                    "ask_user answer count mismatch: expected %d, got %d",
                    len(questions),
                    len(answers),
                )
        else:
            logger.error(
                "ask_user received unknown status %r; "
                "returning explicit error answers",
                status,
            )
            answers = []
            status = "error"
            error_text = "invalid ask_user response status"

    # ---- Synthesize error answers ----
    if status == "error":
        detail = error_text or "ask_user interaction failed"
        answers = [f"(error: {detail})" for _ in questions]

    # ---- Format final tool message ----
    formatted_answers: list[str] = []
    for i, q in enumerate(questions):
        answer = answers[i] if i < len(answers) else "(no answer)"
        formatted_answers.append(f"Q: {q['question']}\nA: {answer}")
    result_text = "\n\n".join(formatted_answers)

    return Command(
        update={
            "messages": [
                ToolMessage(result_text, tool_call_id=tool_call_id)
            ],
        }
    )


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class AskUserMiddleware(AgentMiddleware):
    """Middleware that provides an ``ask_user`` tool for interactive questioning.

    Adds an ``ask_user`` tool that allows the main AutoIdea agent to ask the
    user questions during execution.  Questions can be free-form text or
    multiple choice.  The tool uses LangGraph ``interrupt()`` to pause
    execution and wait for user input.

    This is especially critical at AutoIdea's mandatory HITL checkpoints:
    Stage 7 (gap identification review), Stage 9 (research idea approval),
    and Stage 10 (debate verdict acceptance).
    """

    def __init__(
        self,
        *,
        system_prompt: str = ASK_USER_SYSTEM_PROMPT,
        tool_description: str = ASK_USER_TOOL_DESCRIPTION,
    ) -> None:
        # AgentMiddleware may or may not expose __init__; guard accordingly.
        if callable(getattr(super(), "__init__", None)):
            super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        # ----- Build the ``ask_user`` tool as a closure -----
        @tool(description=self.tool_description)
        def _ask_user(
            questions: QuestionsList,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command[Any]:
            """Ask the user one or more questions."""
            _validate_questions(questions)
            ask_request = AskUserRequest(
                type="ask_user",
                questions=questions,
                tool_call_id=tool_call_id,
            )
            response = interrupt(ask_request)
            return _parse_answers(response, questions, tool_call_id)

        _ask_user.name = "ask_user"
        self.tools: list[Any] = [_ask_user]

    # -----------------------------------------------------------------
    # Model-call wrappers — inject the system prompt into every LLM call
    # -----------------------------------------------------------------

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Inject the ``ask_user`` system prompt into a synchronous LLM call.

        Parameters
        ----------
        request:
            The model request object (expected to have ``system_message``
            and an ``override`` helper).
        handler:
            The next handler in the middleware chain.
        """
        if getattr(request, "system_message", None) is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [
                {"type": "text", "text": self.system_prompt}
            ]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Inject the ``ask_user`` system prompt into an async LLM call.

        Parameters
        ----------
        request:
            The model request object (expected to have ``system_message``
            and an ``override`` helper).
        handler:
            The next async handler in the middleware chain.
        """
        if getattr(request, "system_message", None) is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [
                {"type": "text", "text": self.system_prompt}
            ]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(
            request.override(system_message=new_system_message)
        )
