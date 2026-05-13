"""SectionFlowMiddleware – section-based flow control for LangGraph React agents."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

from .config import SectionConfig
from .manager import SectionManager
from .state import SectionFlowState
from .types import SectionName
from .utils import (
    build_section_prompt_with_transitions,
    get_effective_section,
    inject_section_prompt_into_request,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class SectionFlowMiddleware(AgentMiddleware):
    """Middleware that provides section-based flow control for LangGraph React agents.

    A *section* is a named phase of a conversation, each with its own:

    * **Prompt fragment** – appended to the system message when active.
    * **Tool set** – only the listed tools are visible to the model.
    * **Transition rules** – which sections may follow the current one.
    * **Auto-transition conditions** – predicates evaluated before every model
      call that can advance the flow without agent intervention.
    * **LLM override** – optionally swap to a different model for this phase.

    The middleware hooks into the LangGraph runtime at two points:

    1. ``before_model`` / ``abefore_model`` – initialises section state on the
       first run, resolves fallbacks for removed sections, and fires
       auto-transitions.
    2. ``wrap_model_call`` / ``awrap_model_call`` – injects the section prompt,
       filters the tool list, and optionally swaps the model.

    The agent can also trigger transitions itself via the ``change_section``
    tool, which is registered automatically (``include_transition_tool=True``).

    Example::

        from section_flow import SectionFlowMiddleware, SectionConfig

        sections = {
            "gather": SectionConfig(
                name="gather",
                prompt="Collect the user's name, email and budget.",
                tools=[collect_info_tool],
                allowed_transitions=["recommend"],
            ),
            "recommend": SectionConfig(
                name="recommend",
                prompt="Suggest products matching the user's budget.",
                tools=[search_catalog_tool],
                allowed_transitions=["checkout"],
            ),
            "checkout": SectionConfig(
                name="checkout",
                prompt="Guide the user through the purchase.",
                tools=[process_payment_tool],
            ),
        }

        agent = create_react_agent(
            model="openai:gpt-4o",
            system_prompt="You are a helpful shopping assistant.",
            middleware=[
                SectionFlowMiddleware(
                    sections=sections,
                    initial_section="gather",
                )
            ],
        )
    """

    @property
    def state_schema(self) -> type[SectionFlowState]:
        """Declare the state fields this middleware needs."""
        return SectionFlowState

    def __init__(
        self,
        sections: dict[SectionName, SectionConfig] | None = None,
        initial_section: SectionName | None = None,
        strict_validation: bool = True,
        include_transition_tool: bool = True,
        section_manager: SectionManager | None = None,
        fallback_section: SectionName | None = None,
        global_tools: list[Any] | None = None,
        subagent_graphs: dict[str, Any] | list[Any] | None = None,
        all_middleware: list[Any] | None = None,
    ) -> None:
        """Initialise the middleware.

        Args:
            sections: Dict mapping section names to
                :class:`~section_flow.config.SectionConfig` objects.
                Ignored when *section_manager* is provided.
            initial_section: Name of the starting section.
                Ignored when *section_manager* is provided.
            strict_validation: Whether to reject invalid transitions and
                missing required fields.  Defaults to ``True``.
            include_transition_tool: Register the ``change_section`` tool so
                the agent can trigger transitions itself.  Defaults to ``True``.
            section_manager: Pre-configured :class:`~section_flow.manager.SectionManager`.
                When supplied, *sections*, *initial_section*, *strict_validation*,
                and *fallback_section* are all ignored.
            fallback_section: Section to activate when ``current_section`` in
                the persisted state is no longer registered (e.g. after a
                deployment that removed that section).
            global_tools: Tools available in *every* section.  A global tool
                with the same name as a section-level tool takes precedence.
            subagent_graphs: Registry of subagent graphs; used to filter the
                ``task`` tool description per section.
            all_middleware: All middleware instances in the agent stack.  Used
                to resolve string tool names (e.g. ``"task"``) across
                middleware boundaries.
        """
        super().__init__()

        self.global_tools: list[Any] = global_tools or []
        self.subagent_graphs: dict[str, Any] | list[Any] = subagent_graphs or {}
        self.all_middleware: list[Any] = all_middleware or []
        self._section_tool_cache: dict[str, list[Any]] = {}

        if section_manager is not None:
            self.section_manager = section_manager
        else:
            if sections is None or initial_section is None:
                raise ValueError(
                    "Provide either a section_manager, or both 'sections' and 'initial_section'."
                )
            self.section_manager = SectionManager(
                sections=sections,
                initial_section=initial_section,
                strict_validation=strict_validation,
                fallback_section=fallback_section,
            )

        self.initial_section: SectionName = self.section_manager.initial_section

        if include_transition_tool:
            from .tools import create_change_section_tool

            self.tools: list[Any] = [create_change_section_tool(self.section_manager)]
        else:
            self.tools = []

    # ------------------------------------------------------------------
    # before_model hook
    # ------------------------------------------------------------------

    def before_model(
        self,
        state: SectionFlowState,
        runtime: Runtime | None = None,
    ) -> dict[str, Any] | None:
        """Initialise section state and evaluate auto-transitions.

        Returns a state-update dict when changes are needed, or ``None`` when
        the state is already correct and no update is required.
        """
        current_section = state.get("current_section")

        # First call – bootstrap section state and fire on_enter for the initial section
        if current_section is None:
            initial_config = self.section_manager.get_section(self.initial_section)
            hook_update = (initial_config.execute_on_enter(state) or {}) if initial_config else {}
            return {
                "current_section": self.initial_section,
                "section_data": {},
                "visited_sections": [],
                **hook_update,
            }

        # Handle removed sections (e.g. after a deployment)
        _, used_fallback = self.section_manager.get_section_with_fallback(current_section)
        if used_fallback:
            self._section_tool_cache.clear()
            return {"current_section": self.section_manager.fallback_section}

        visited_sections = list(state.get("visited_sections") or [])
        if current_section not in visited_sections:
            visited_sections.append(current_section)

        # Evaluate auto-transitions before the model call
        target_section = self.section_manager.evaluate_auto_transitions(state)
        if target_section and target_section != current_section:
            if target_section not in visited_sections:
                visited_sections.append(target_section)
            self._section_tool_cache.clear()

            # Fire lifecycle hooks for the transition
            from_config = self.section_manager.get_section(current_section)
            to_config = self.section_manager.get_section(target_section)
            exit_update = (from_config.execute_on_exit(state) or {}) if from_config else {}
            enter_update = (to_config.execute_on_enter(state) or {}) if to_config else {}

            return {
                "current_section": target_section,
                "visited_sections": visited_sections,
                **exit_update,
                **enter_update,
            }

        # Update visited list if it changed
        if visited_sections != list(state.get("visited_sections") or []):
            return {"visited_sections": visited_sections}

        return None

    async def abefore_model(
        self,
        state: SectionFlowState,
        runtime: Runtime | None = None,
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`before_model`."""
        return self.before_model(state, runtime)

    # ------------------------------------------------------------------
    # wrap_model_call hook
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject section prompt and tools, optionally swap the model."""
        state = request.state
        current_section = get_effective_section(state, self.initial_section)
        section_config, _ = self.section_manager.get_section_with_fallback(current_section)

        section_prompt = build_section_prompt_with_transitions(section_config, current_section)
        modified_request = inject_section_prompt_into_request(request, section_prompt)

        section_tools = self._get_section_tools(current_section)
        if section_tools:
            modified_request.tools = section_tools

        if section_config.llm is not None:
            modified_request.model = section_config.llm

        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Async variant of :meth:`wrap_model_call`."""
        state = request.state
        current_section = get_effective_section(state, self.initial_section)
        section_config, _ = self.section_manager.get_section_with_fallback(current_section)

        section_prompt = build_section_prompt_with_transitions(section_config, current_section)
        modified_request = inject_section_prompt_into_request(request, section_prompt)

        section_tools = self._get_section_tools(current_section)
        if section_tools:
            modified_request.tools = section_tools

        if section_config.llm is not None:
            modified_request.model = section_config.llm

        return await handler(modified_request)

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    def _get_section_tools(self, current_section: SectionName) -> list[Any]:
        """Return the resolved, deduplicated tool list for *current_section*.

        Resolution order:
        1. Tools explicitly listed in ``SectionConfig.tools`` (object or string).
        2. The ``change_section`` tool (always appended last).
        3. Global tools override section tools of the same name.

        Results are cached per section; call :meth:`clear_tool_cache` to
        invalidate (e.g. after runtime section modification).
        """
        if current_section in self._section_tool_cache:
            return self._section_tool_cache[current_section]

        section_config, _ = self.section_manager.get_section_with_fallback(current_section)
        section_tools = section_config.tools or []
        tools_dict: dict[str, Any] = {}

        for tool in section_tools:
            if isinstance(tool, str):
                if tool == "task":
                    task_tool = self._get_task_tool_with_filtered_description(section_config)
                    if task_tool:
                        tools_dict["task"] = task_tool
                else:
                    # Resolve named tool from other middleware
                    for middleware in self.all_middleware:
                        for mw_tool in getattr(middleware, "tools", []):
                            mw_name = getattr(mw_tool, "name", getattr(mw_tool, "__name__", str(mw_tool)))
                            if mw_name == tool:
                                tools_dict[tool] = mw_tool
                                break
                        if tool in tools_dict:
                            break
            else:
                tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                global_override = next(
                    (
                        g
                        for g in self.global_tools
                        if getattr(g, "name", getattr(g, "__name__", str(g))) == tool_name
                    ),
                    None,
                )
                tools_dict[tool_name] = global_override if global_override else tool

        # Always include the change_section tool
        for tool in self.tools:
            tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
            tools_dict[tool_name] = tool

        result = list(tools_dict.values())
        self._section_tool_cache[current_section] = result
        return result

    def _get_task_tool_with_filtered_description(
        self,
        section_config: SectionConfig,
    ) -> Any | None:
        """Return the ``task`` tool with its description scoped to *section_config*'s allowed subagents."""
        original_task_tool: Any = None
        for middleware in self.all_middleware:
            for tool in getattr(middleware, "tools", []):
                if getattr(tool, "name", "") == "task":
                    original_task_tool = tool
                    break
            if original_task_tool:
                break

        if not original_task_tool:
            return None

        allowed_subagents = section_config.allowed_subagents or []
        if not allowed_subagents:
            return original_task_tool

        # Normalise subagent_graphs to a dict
        subagent_dict: dict[str, Any]
        if isinstance(self.subagent_graphs, list):
            subagent_dict = {
                s["name"]: s for s in self.subagent_graphs if isinstance(s, dict) and "name" in s
            }
        elif isinstance(self.subagent_graphs, dict):
            subagent_dict = self.subagent_graphs
        else:
            return original_task_tool

        filtered = {k: v for k, v in subagent_dict.items() if k in allowed_subagents}
        if not filtered:
            return original_task_tool

        try:
            filtered_tool = deepcopy(original_task_tool)
            filtered_tool.description = (
                "Launch an ephemeral subagent to handle a complex subtask. "
                f"Available subagents for this section: {', '.join(filtered)}"
            )
            return filtered_tool
        except Exception:
            return original_task_tool

    def clear_tool_cache(self) -> None:
        """Invalidate the per-section tool cache.

        Call this after modifying sections at runtime (e.g. hot-reloading
        section configs without restarting the process).
        """
        self._section_tool_cache.clear()


__all__ = ["SectionFlowMiddleware"]
