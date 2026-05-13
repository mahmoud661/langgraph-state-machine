"""State type definitions for section-based workflow tracking."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _last_value(left: Any, right: Any) -> Any:
    """Reducer: when multiple updates race, keep the most recent value."""
    if isinstance(right, list):
        return right[-1] if right else left
    return right


class SectionFlowState(TypedDict, total=False):
    """State extension for section-based flow control.

    Merge with your existing agent state to add section flow capabilities::

        class MyAgentState(SectionFlowState):
            messages: Annotated[list[BaseMessage], add_messages]
            # ... other fields

    Attributes:
        current_section: Name of the currently active section.
            Uses ``_last_value`` reducer so concurrent updates always converge
            to the most recent transition.
        section_data: Shared key-value store passed across sections within
            a single conversation turn.
        visited_sections: Ordered list of every section activated during the
            current conversation (duplicates excluded).
    """

    current_section: Annotated[str, _last_value]
    section_data: dict[str, Any]
    visited_sections: list[str]


__all__ = ["SectionFlowState"]
