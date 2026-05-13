"""Unit tests for SectionFlowMiddleware.before_model and _get_section_tools.

wrap_model_call / awrap_model_call require a live ModelRequest object from
langchain.agents.middleware and are exercised via integration tests only.
The business logic tested here (state initialisation, auto-transitions,
fallback handling, tool filtering, lifecycle hooks) is fully independent of
that boundary.
"""

import pytest

from section_flow import SectionConfig, SectionManager, SectionFlowMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool(name: str):
    """Return a minimal tool-like object with a .name attribute."""
    class _Tool:
        pass
    t = _Tool()
    t.name = name
    return t


def make_middleware(
    section_names: list[str] | None = None,
    initial: str = "a",
    fallback: str | None = None,
    include_transition_tool: bool = False,
    section_tools: dict[str, list] | None = None,
    auto_conditions: dict[str, object] | None = None,
) -> SectionFlowMiddleware:
    names = section_names or ["a", "b", "c"]
    sections = {}
    for i, n in enumerate(names):
        kwargs = dict(
            name=n,
            prompt=f"Prompt for {n}.",
            allowed_transitions=[names[i + 1]] if i < len(names) - 1 else [],
            tools=(section_tools or {}).get(n, []),
        )
        if auto_conditions and n in auto_conditions:
            kwargs["auto_transition_conditions"] = auto_conditions[n]
        sections[n] = SectionConfig(**kwargs)

    manager = SectionManager(
        sections=sections,
        initial_section=initial,
        fallback_section=fallback or initial,
    )
    return SectionFlowMiddleware(
        section_manager=manager,
        include_transition_tool=include_transition_tool,
    )


# ---------------------------------------------------------------------------
# before_model – initialisation
# ---------------------------------------------------------------------------

class TestBeforeModelInit:
    def test_first_call_sets_initial_section(self):
        mw = make_middleware()
        result = mw.before_model({})
        assert result["current_section"] == "a"

    def test_first_call_sets_empty_section_data(self):
        mw = make_middleware()
        result = mw.before_model({})
        assert result["section_data"] == {}

    def test_first_call_sets_empty_visited_sections(self):
        mw = make_middleware()
        result = mw.before_model({})
        assert result["visited_sections"] == []

    def test_subsequent_call_with_valid_section_returns_none(self):
        mw = make_middleware()
        state = {"current_section": "a", "visited_sections": ["a"]}
        assert mw.before_model(state) is None

    def test_on_enter_called_on_first_init(self):
        entered = []
        sections = {
            "start": SectionConfig(
                name="start",
                prompt="p",
                on_enter=lambda _: entered.append("start") or None,
            )
        }
        manager = SectionManager(sections=sections, initial_section="start", fallback_section="start")
        mw = SectionFlowMiddleware(section_manager=manager, include_transition_tool=False)
        mw.before_model({})
        assert "start" in entered


# ---------------------------------------------------------------------------
# before_model – visited sections tracking
# ---------------------------------------------------------------------------

class TestBeforeModelVisited:
    def test_adds_current_section_to_visited(self):
        mw = make_middleware()
        result = mw.before_model({"current_section": "a", "visited_sections": []})
        assert result is not None
        assert "a" in result["visited_sections"]

    def test_does_not_duplicate_visited_sections(self):
        mw = make_middleware()
        result = mw.before_model({"current_section": "a", "visited_sections": ["a"]})
        assert result is None  # nothing changed

    def test_visited_sections_is_not_mutated_in_place(self):
        mw = make_middleware()
        original = ["a"]
        mw.before_model({"current_section": "a", "visited_sections": original})
        assert original == ["a"]


# ---------------------------------------------------------------------------
# before_model – fallback for removed sections
# ---------------------------------------------------------------------------

class TestBeforeModelFallback:
    def test_unknown_section_activates_fallback(self):
        mw = make_middleware(fallback="a")
        result = mw.before_model({"current_section": "removed_section"})
        assert result["current_section"] == "a"

    def test_tool_cache_cleared_on_fallback(self):
        mw = make_middleware(fallback="a")
        mw._section_tool_cache["removed_section"] = []
        mw.before_model({"current_section": "removed_section"})
        assert "removed_section" not in mw._section_tool_cache


# ---------------------------------------------------------------------------
# before_model – auto-transitions
# ---------------------------------------------------------------------------

class TestBeforeModelAutoTransitions:
    def _make_auto_mw(self, condition_fires: bool) -> SectionFlowMiddleware:
        target = "b" if condition_fires else None
        return make_middleware(
            auto_conditions={"a": lambda _: target},
        )

    def test_auto_transition_updates_current_section(self):
        mw = self._make_auto_mw(True)
        result = mw.before_model({"current_section": "a", "visited_sections": []})
        assert result["current_section"] == "b"

    def test_auto_transition_adds_target_to_visited(self):
        mw = self._make_auto_mw(True)
        result = mw.before_model({"current_section": "a", "visited_sections": []})
        assert "b" in result["visited_sections"]

    def test_no_auto_transition_when_condition_false(self):
        mw = self._make_auto_mw(False)
        result = mw.before_model({"current_section": "a", "visited_sections": ["a"]})
        assert result is None

    def test_tool_cache_cleared_on_auto_transition(self):
        mw = self._make_auto_mw(True)
        mw._section_tool_cache["a"] = []
        mw.before_model({"current_section": "a", "visited_sections": []})
        assert "a" not in mw._section_tool_cache

    def test_on_exit_and_on_enter_called_during_auto_transition(self):
        log = []
        sections = {
            "a": SectionConfig(
                name="a",
                prompt="p",
                allowed_transitions=["b"],
                auto_transition_conditions=lambda _: "b",
                on_exit=lambda _: log.append("exit:a") or None,
            ),
            "b": SectionConfig(
                name="b",
                prompt="p",
                on_enter=lambda _: log.append("enter:b") or None,
            ),
        }
        manager = SectionManager(sections=sections, initial_section="a", fallback_section="a")
        mw = SectionFlowMiddleware(section_manager=manager, include_transition_tool=False)
        mw.before_model({"current_section": "a", "visited_sections": []})
        assert "exit:a" in log
        assert "enter:b" in log

    def test_hook_return_value_merged_into_state_update(self):
        sections = {
            "a": SectionConfig(
                name="a",
                prompt="p",
                allowed_transitions=["b"],
                auto_transition_conditions=lambda _: "b",
                on_exit=lambda _: {"section_data": {"exited": True}},
            ),
            "b": SectionConfig(name="b", prompt="p"),
        }
        manager = SectionManager(sections=sections, initial_section="a", fallback_section="a")
        mw = SectionFlowMiddleware(section_manager=manager, include_transition_tool=False)
        result = mw.before_model({"current_section": "a", "visited_sections": []})
        assert result.get("section_data", {}).get("exited") is True


# ---------------------------------------------------------------------------
# _get_section_tools
# ---------------------------------------------------------------------------

class TestGetSectionTools:
    def test_returns_section_specific_tools(self):
        tool_a = make_tool("search")
        mw = make_middleware(section_tools={"a": [tool_a]})
        tools = mw._get_section_tools("a")
        assert any(getattr(t, "name", None) == "search" for t in tools)

    def test_does_not_include_other_section_tools(self):
        tool_b = make_tool("pay")
        mw = make_middleware(section_tools={"b": [tool_b]})
        tools = mw._get_section_tools("a")
        assert not any(getattr(t, "name", None) == "pay" for t in tools)

    def test_results_are_cached(self):
        mw = make_middleware()
        first = mw._get_section_tools("a")
        second = mw._get_section_tools("a")
        assert first is second

    def test_clear_tool_cache_invalidates(self):
        mw = make_middleware()
        first = mw._get_section_tools("a")
        mw.clear_tool_cache()
        second = mw._get_section_tools("a")
        assert first is not second

    def test_global_tool_overrides_section_tool_with_same_name(self):
        section_tool = make_tool("search")
        global_tool = make_tool("search")
        sections = {
            "a": SectionConfig(name="a", prompt="p", tools=[section_tool]),
        }
        manager = SectionManager(sections=sections, initial_section="a", fallback_section="a")
        mw = SectionFlowMiddleware(
            section_manager=manager,
            include_transition_tool=False,
            global_tools=[global_tool],
        )
        tools = mw._get_section_tools("a")
        # Should have exactly one "search" tool and it should be the global one
        search_tools = [t for t in tools if getattr(t, "name", None) == "search"]
        assert len(search_tools) == 1
        assert search_tools[0] is global_tool

    def test_change_section_tool_always_appended(self):
        mw = make_middleware(include_transition_tool=True)
        tools = mw._get_section_tools("a")
        assert any(getattr(t, "name", None) == "change_section" for t in tools)

    def test_no_duplicate_tools_from_cache(self):
        tool_a = make_tool("search")
        mw = make_middleware(section_tools={"a": [tool_a]})
        tools = mw._get_section_tools("a")
        names = [getattr(t, "name", None) for t in tools]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# async before_model delegates to sync
# ---------------------------------------------------------------------------

class TestAsyncBeforeModel:
    @pytest.mark.asyncio
    async def test_abefore_model_matches_before_model(self):
        mw = make_middleware()
        sync_result = mw.before_model({})
        async_result = await mw.abefore_model({})
        assert sync_result == async_result
