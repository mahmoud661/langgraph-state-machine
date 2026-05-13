"""LangGraph Section Flow Middleware.

Section-based flow control for LangGraph React agents: divide conversational
workflows into discrete phases, each with its own prompt, tools, and
transition rules.

Quick start::

    from section_flow import SectionFlowMiddleware, SectionConfig

    sections = {
        "gather": SectionConfig(
            name="gather",
            prompt="Collect the user's requirements.",
            tools=[my_tool],
            allowed_transitions=["recommend"],
        ),
        "recommend": SectionConfig(
            name="recommend",
            prompt="Suggest the best options.",
            tools=[search_tool],
        ),
    }

    middleware = SectionFlowMiddleware(
        sections=sections,
        initial_section="gather",
    )
"""

from .config import SectionConfig, TransitionCondition
from .manager import SectionManager
from .middleware import SectionFlowMiddleware
from .state import SectionFlowState

__version__ = "0.1.0"

__all__ = [
    "SectionFlowMiddleware",
    "SectionConfig",
    "SectionFlowState",
    "SectionManager",
    "TransitionCondition",
    "__version__",
]
