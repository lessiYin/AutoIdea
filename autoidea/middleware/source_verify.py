"""Source verification middleware for AutoIdea.

Validates and verifies source citations in agent responses to ensure
research integrity.  Two mechanisms:

1. **Citation Registry Check** — verifies that [Cn] tags in the response
   correspond to entries in the workspace's ``citations.json``.
2. **Uncited Claim Detection** — scans for 8 common claim patterns
   (``"showed that"``, ``"demonstrated"``, …) and checks a 200-character
   window around each match for nearby ``[Pn]``/``[Cn]`` tags.  Missing
   tags trigger a warning that is **injected** into the next system
   message so the agent can self-correct.
"""

from __future__ import annotations

import re
from typing import Any

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:
    try:
        from deepagents_langgraph.middleware import AgentMiddleware
    except ImportError:
        AgentMiddleware = object

# ── 8 Claim Patterns (v2.0 P0 fix) ────────────────────────────────────

CLAIM_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"showed\s+that",
        r"demonstrated\s+that",
        r"found\s+that",
        r"reported\s+that",
        r"achieved\s+(?:a\s+)?(?:state[- ]of[- ]the[- ]art|sota|significant|superior)",
        r"outperform(?:s|ed|ing)?",
        r"proved?\s+that",
        r"established\s+that",
    ]
]

_CITATION_TAG_RE = re.compile(r"\[(?:P|C)\d+\]")
_WINDOW_CHARS = 200  # characters before/after claim to search for tags


def scan_for_uncited_claims(text: str) -> list[dict[str, Any]]:
    """Scan *text* for claims that lack a nearby ``[Pn]``/``[Cn]`` tag.

    For each match of the 8 claim patterns, a 200-character window
    (both before and after the match position) is checked.  If no
    citation tag is found inside that window the claim is flagged.

    Returns:
        List of dicts, each with keys ``pattern``, ``match``,
        ``position``, and ``context`` (the 200-char excerpt).
    """
    uncited: list[dict[str, Any]] = []

    for pattern in CLAIM_PATTERNS:
        for m in pattern.finditer(text):
            start = max(0, m.start() - _WINDOW_CHARS)
            end = min(len(text), m.end() + _WINDOW_CHARS)
            window = text[start:end]

            if not _CITATION_TAG_RE.search(window):
                uncited.append({
                    "pattern": pattern.pattern,
                    "match": m.group(),
                    "position": m.start(),
                    "context": window.strip()[:120],
                })

    return uncited


class SourceVerifyMiddleware(AgentMiddleware):
    """Middleware that verifies source citations in agent output.

    After each model response the middleware:

    1. Checks that every ``[Cn]`` tag references a registered citation
       in the workspace ``citations.json``.
    2. Scans for 8 common *claim patterns* and verifies that each has a
       ``[Pn]``/``[Cn]`` tag within a 200-character window.  Uncited
       claims produce warnings that are **injected into the next system
       message** so the agent can self-correct (rather than just logging).
    """

    name = "source_verify"

    def __init__(self, strict: bool = False):
        self.strict = strict
        # Pending warnings to inject into the next LLM call
        self._pending_warnings: list[str] = []

    # ── Citation registry check ──────────────────────────────────────

    def _extract_citation_tags(self, text: str) -> tuple[set[str], set[str]]:
        paper_tags = set(re.findall(r"\[P\d+\]", text))
        citation_tags = set(re.findall(r"\[C\d+\]", text))
        return paper_tags, citation_tags

    def _verify_citations(self, text: str) -> list[str]:
        """Verify [Cn] tags against the citation registry file."""
        _, citation_tags = self._extract_citation_tags(text)
        warnings: list[str] = []

        if not citation_tags:
            return warnings

        try:
            import json
            from pathlib import Path
            from autoidea.paths import get_active_workspace

            workspace = get_active_workspace()
            citations_path = Path(workspace) / "citations.json"
            legacy_path = Path(workspace) / "output" / "citations.json"
            if not citations_path.exists() and legacy_path.exists():
                citations_path = legacy_path

            if citations_path.exists():
                with open(citations_path, "r", encoding="utf-8") as f:
                    citations = json.load(f)

                registered_ids = {
                    entry.get("citation_id", "")
                    for entry in citations
                    if entry.get("citation_id")
                }

                for tag in sorted(citation_tags):
                    if tag not in registered_ids:
                        warnings.append(
                            f"Unregistered citation {tag}: "
                            f"not found in citation registry"
                        )
            else:
                warnings.append(
                    f"Citation registry not found but {len(citation_tags)} "
                    f"citation tags used in response"
                )
        except Exception:
            pass

        return warnings

    # ── Uncited claim scanning ───────────────────────────────────────

    def _check_uncited_claims(self, text: str) -> list[str]:
        """Return warning strings for claims missing nearby tags."""
        uncited = scan_for_uncited_claims(text)
        warnings: list[str] = []
        for item in uncited:
            warnings.append(
                f"UNCITED CLAIM near position {item['position']}: "
                f"\"{item['match']}\" — no [Pn]/[Cn] tag within "
                f"{_WINDOW_CHARS}-char window. "
                f"Context: \"{item['context']}…\""
            )
        return warnings

    # ── Middleware hooks ──────────────────────────────────────────────

    def _process_response(self, response):
        """Shared logic for sync/async after_model."""
        if not hasattr(response, "content"):
            return response

        text = response.content if isinstance(response.content, str) else ""
        if not text:
            return response

        all_warnings: list[str] = []
        all_warnings.extend(self._verify_citations(text))
        all_warnings.extend(self._check_uncited_claims(text))

        if all_warnings:
            import logging
            logger = logging.getLogger(__name__)
            for w in all_warnings:
                logger.warning("Source verification: %s", w)
            # Queue for injection into next system message
            self._pending_warnings.extend(all_warnings)

        return response

    def after_model(self, request, response, **kwargs):
        """Check citations after each model response."""
        return self._process_response(response)

    async def aafter_model(self, request, response, **kwargs):
        """Async version of after_model."""
        return self._process_response(response)

    # ── System-message injection of pending warnings ─────────────────

    def _inject_warnings(self, request):
        """Inject accumulated warnings into the system message."""
        if not self._pending_warnings:
            return request

        warning_block = (
            "\n\n<source_verification_warnings>\n"
            "The following claims in your previous response lacked "
            "proper citation tags ([Pn]/[Cn]).  Please ensure every "
            "empirical claim is backed by a cited source.\n\n"
            + "\n".join(f"- {w}" for w in self._pending_warnings)
            + "\n</source_verification_warnings>"
        )

        # Append to existing system message
        if hasattr(request, "system_message") and request.system_message:
            from langchain_core.messages import SystemMessage
            sm = request.system_message
            if hasattr(sm, "content"):
                new_content = (sm.content if isinstance(sm.content, str) else str(sm.content)) + warning_block
            else:
                new_content = warning_block
            request.system_message = SystemMessage(content=new_content)
        elif hasattr(request, "messages") and request.messages:
            from langchain_core.messages import SystemMessage
            # Prepend as a new SystemMessage if none exists
            has_system = any(
                isinstance(m, SystemMessage) for m in request.messages
            )
            if has_system:
                for i, m in enumerate(request.messages):
                    if isinstance(m, SystemMessage):
                        request.messages[i] = SystemMessage(
                            content=m.content + warning_block
                        )
                        break
            else:
                request.messages.insert(
                    0, SystemMessage(content=warning_block.strip())
                )

        # Clear consumed warnings
        self._pending_warnings.clear()
        return request

    def modify_request(self, request):
        """Inject pending warnings before the next LLM call."""
        return self._inject_warnings(request)

    def wrap_model_call(self, request, handler):
        """Sync wrapper — inject, call, then scan."""
        request = self._inject_warnings(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        """Async wrapper — inject, call, then scan."""
        request = self._inject_warnings(request)
        return await handler(request)
