"""Utility functions for AutoIdea.

Helpers for displaying messages and prompts, subagent loading,
and lightweight configuration used by the agent runtime.
"""

import json
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def format_message_content(message):
    """Convert message content to displayable string.

    Args:
        message: A LangChain message object with content attribute.

    Returns:
        Formatted string representation of the message content.
    """
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
                elif block.get("type") == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        parts.append(f"[thinking]\n{thinking}\n[/thinking]")
                elif block.get("type") == "tool_use":
                    tool_calls_processed = True
                    name = block.get("name", "unknown")
                    input_data = block.get("input", {})
                    parts.append(f"[tool_call: {name}({json.dumps(input_data, ensure_ascii=False)[:200]})]")
            elif isinstance(block, str):
                parts.append(block)

    # Handle tool_calls attribute (if present and not already processed)
    if not tool_calls_processed and hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            parts.append(f"[tool_call: {name}({json.dumps(args, ensure_ascii=False)[:200]})]")

    return "\n".join(parts) if parts else str(message.content)


def display_message(role: str, content: str, style: str = "blue"):
    """Display a message in a Rich panel.

    Args:
        role: Message role (e.g. 'user', 'assistant')
        content: Message content to display
        style: Panel border style (Rich color name)
    """
    panel = Panel(
        Text(content[:2000] + ("..." if len(content) > 2000 else "")),
        title=f"[bold]{role}[/bold]",
        border_style=style,
        expand=True,
    )
    console.print(panel)


def load_subagents_yaml(yaml_path: str | Path) -> dict:
    """Load sub-agent definitions from a YAML file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Dictionary of sub-agent definitions.
    """
    path = Path(yaml_path)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("subagents", data)


def build_subagent_definitions(yaml_data: dict, tool_registry: dict) -> list[dict]:
    """Build sub-agent definitions from YAML data and tool registry.

    Maps YAML keys to the dict format expected by ``create_deep_agent``:
      - ``handoff_description`` -> ``description``  (required by deepagents)
      - ``instructions`` + ``strategy`` -> ``system_prompt``  (required)
      - ``tools`` resolved from string names via *tool_registry*

    Args:
        yaml_data: Parsed YAML sub-agent definitions.
        tool_registry: Dictionary mapping tool names to tool objects.

    Returns:
        List of sub-agent definition dicts ready for create_deep_agent.
    """
    subagents = []
    for name, spec in yaml_data.items():
        tool_names = spec.get("tools", [])
        tools = []
        for tn in tool_names:
            if tn in tool_registry:
                tools.append(tool_registry[tn])
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Sub-agent '%s': tool '%s' not found in registry — skipped", name, tn,
                )

        # Build system_prompt from instructions + strategy
        instructions = spec.get("instructions", "")
        strategy = spec.get("strategy", "")
        if strategy:
            system_prompt = f"{instructions}\n\n## Execution Strategy\n\n{strategy}"
        else:
            system_prompt = instructions

        sa_def: dict = {
            "name": name,
            "description": spec.get("handoff_description", spec.get("description", "")),
            "system_prompt": system_prompt,
            "tools": tools,
        }

        subagents.append(sa_def)

    return subagents
