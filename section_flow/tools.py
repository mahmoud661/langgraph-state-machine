"""The ``change_section`` tool — lets the agent drive its own flow transitions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

if TYPE_CHECKING:
    from .manager import SectionManager

logger = logging.getLogger(__name__)


def create_change_section_tool(section_manager: SectionManager) -> StructuredTool:
    """Build the ``change_section`` tool bound to *section_manager*.

    The returned tool is a :class:`~langchain_core.tools.StructuredTool` with
    both sync and async implementations.  It validates the requested target
    section against the manager's registry before committing the transition.

    Args:
        section_manager: The :class:`~section_flow.manager.SectionManager`
            instance that owns the section registry.

    Returns:
        A ``StructuredTool`` named ``"change_section"``.

    Example::

        tool = create_change_section_tool(manager)
        # Agent calls: change_section(target_section="payment", reason="Cart is ready")
    """

    def change_section(
        target_section: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        reason: str | None = None,
    ) -> Command:
        """Change the current section to a new section.

        Use this tool when you have completed the current section's goals and
        need to advance the workflow.  Always supply a brief *reason* so the
        transition is auditable.

        Args:
            target_section: Name of the section to transition to.
            reason: Short explanation of why the section is changing.
        """
        available_sections = section_manager.get_section_names()

        if target_section not in available_sections:
            error_msg = (
                f"Section '{target_section}' not found. "
                f"Available sections: {', '.join(available_sections)}"
            )
            logger.warning(error_msg)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=f"Error: {error_msg}",
                            name="change_section",
                            status="error",
                        )
                    ]
                }
            )

        current_section = state.get("current_section")
        message_content = (
            f"Successfully changed section from '{current_section}' to '{target_section}'."
        )
        if reason:
            message_content += f" Reason: {reason}"

        logger.info(message_content)

        return Command(
            update={
                "current_section": target_section,
                "messages": [
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=message_content,
                        name="change_section",
                        status="success",
                    )
                ],
            }
        )

    async def achange_section(
        target_section: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        reason: str | None = None,
    ) -> Command:
        """Async counterpart of ``change_section``."""
        return change_section(target_section, state, tool_call_id, reason)

    section_list = section_manager.get_section_names()

    return StructuredTool.from_function(
        name="change_section",
        func=change_section,
        coroutine=achange_section,
        description=(
            "Change to a different section of the workflow. "
            "Use this when you have completed the current section's tasks and need to advance. "
            f"Available sections: {', '.join(section_list)}. "
            "Always provide a clear reason for the transition."
        ),
    )


__all__ = ["create_change_section_tool"]
