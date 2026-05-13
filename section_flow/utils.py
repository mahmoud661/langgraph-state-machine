"""State helper functions for the section flow middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage

if TYPE_CHECKING:
    from .config import SectionConfig


def get_effective_section(state: dict[str, Any], fallback_section: str) -> str:
    """Return ``state["current_section"]``, or *fallback_section* if unset."""
    return state.get("current_section") or fallback_section


def build_section_prompt_with_transitions(
    section_config: SectionConfig,
    current_section: str,
) -> str:
    """Compose the section prompt with an XML context block appended.

    The context block tells the agent which section it is in and which
    transitions are available.  Formatting the block as XML keeps it
    clearly delimited from the main prompt and is cache-friendly because
    the base prompt text is unchanged between calls.

    Args:
        section_config: Config for the currently active section.
        current_section: Name of the current section (used in the context
            block).

    Returns:
        The full prompt string to inject into the system message.
    """
    section_prompt = section_config.prompt
    allowed_transitions = section_config.allowed_transitions or []

    if allowed_transitions:
        transitions_info = (
            f"\n\n<SectionContext>\n"
            f"<CurrentSection>{current_section}</CurrentSection>\n"
            f"<AvailableTransitions>{', '.join(allowed_transitions)}</AvailableTransitions>\n"
            f"<FlowInstruction>Calling change_section does NOT pause the flow. "
            f"Continue executing next steps immediately in the new section context.</FlowInstruction>\n"
            f"</SectionContext>"
        )
        section_prompt = section_prompt + transitions_info

    return section_prompt


def inject_section_prompt_into_request(
    request: ModelRequest,
    section_prompt: str,
) -> ModelRequest:
    """Return a new request with *section_prompt* prepended as a SystemMessage.

    Prepending rather than appending ensures the section context is the
    first thing the model sees, which typically improves instruction
    following without requiring changes to the base system prompt.

    Args:
        request: Original :class:`ModelRequest`.
        section_prompt: Prompt text built by
            :func:`build_section_prompt_with_transitions`.

    Returns:
        A modified copy of *request* (via ``request.override``).
    """
    section_system_msg = SystemMessage(content=section_prompt)
    modified_messages = [section_system_msg] + request.messages
    return request.override(messages=modified_messages)


__all__ = [
    "get_effective_section",
    "build_section_prompt_with_transitions",
    "inject_section_prompt_into_request",
]
