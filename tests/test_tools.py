"""Unit tests for create_change_section_tool."""

import pytest
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from section_flow import SectionConfig, SectionManager
from section_flow.tools import create_change_section_tool


def make_manager() -> SectionManager:
    sections = {
        "start": SectionConfig(name="start", prompt="p", allowed_transitions=["end"]),
        "end": SectionConfig(name="end", prompt="p"),
    }
    return SectionManager(sections=sections, initial_section="start", fallback_section="start")


class TestCreateChangeSectionTool:
    def test_returns_structured_tool(self):
        manager = make_manager()
        tool = create_change_section_tool(manager)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "change_section"

    def test_description_lists_sections(self):
        manager = make_manager()
        tool = create_change_section_tool(manager)
        assert "start" in tool.description
        assert "end" in tool.description


class TestChangeSectionSync:
    def _call(self, tool, target: str, reason: str | None = None, current: str = "start") -> Command:
        state = {"current_section": current}
        return tool.func(
            target_section=target,
            state=state,
            tool_call_id="test-id",
            reason=reason,
        )

    def test_valid_transition_returns_command(self):
        tool = create_change_section_tool(make_manager())
        result = self._call(tool, "end")
        assert isinstance(result, Command)

    def test_valid_transition_updates_current_section(self):
        tool = create_change_section_tool(make_manager())
        result = self._call(tool, "end")
        assert result.update["current_section"] == "end"

    def test_valid_transition_includes_success_tool_message(self):
        tool = create_change_section_tool(make_manager())
        result = self._call(tool, "end")
        messages = result.update["messages"]
        assert len(messages) == 1
        assert messages[0].status == "success"

    def test_reason_included_in_message(self):
        tool = create_change_section_tool(make_manager())
        result = self._call(tool, "end", reason="Done with start")
        content = result.update["messages"][0].content
        assert "Done with start" in content

    def test_invalid_target_returns_error_command(self):
        tool = create_change_section_tool(make_manager())
        result = self._call(tool, "nonexistent")
        messages = result.update["messages"]
        assert messages[0].status == "error"
        assert "not found" in messages[0].content

    def test_invalid_target_does_not_update_section(self):
        tool = create_change_section_tool(make_manager())
        result = self._call(tool, "nonexistent")
        assert "current_section" not in result.update


class TestChangeSectionAsync:
    @pytest.mark.asyncio
    async def test_async_produces_same_result_as_sync(self):
        tool = create_change_section_tool(make_manager())
        state = {"current_section": "start"}

        sync_result = tool.func(
            target_section="end", state=state, tool_call_id="id-1"
        )
        async_result = await tool.coroutine(
            target_section="end", state=state, tool_call_id="id-2"
        )

        assert sync_result.update["current_section"] == async_result.update["current_section"]
        assert (
            sync_result.update["messages"][0].status
            == async_result.update["messages"][0].status
        )
