"""Unit tests for SectionManager."""

import pytest

from section_flow import SectionConfig, SectionManager


def make_manager(
    section_names: list[str] | None = None,
    initial: str = "a",
    fallback: str | None = None,
    strict: bool = True,
) -> SectionManager:
    names = section_names or ["a", "b", "c"]
    sections = {
        n: SectionConfig(
            name=n,
            prompt=f"Prompt for {n}.",
            allowed_transitions=[names[i + 1]] if i < len(names) - 1 else [],
        )
        for i, n in enumerate(names)
    }
    return SectionManager(
        sections=sections,
        initial_section=initial,
        strict_validation=strict,
        fallback_section=fallback,
    )


class TestSectionManagerInit:
    def test_empty_sections_raises(self):
        with pytest.raises(ValueError, match="At least one section"):
            SectionManager(sections={}, initial_section="x")

    def test_missing_initial_raises(self):
        cfg = SectionConfig(name="a", prompt="p")
        with pytest.raises(ValueError, match="Initial section"):
            SectionManager(sections={"a": cfg}, initial_section="missing")

    def test_missing_fallback_raises(self):
        cfg = SectionConfig(name="a", prompt="p")
        with pytest.raises(ValueError, match="Fallback section"):
            SectionManager(sections={"a": cfg}, initial_section="a", fallback_section="gone")

    def test_defaults_fallback_to_initial(self):
        manager = make_manager()
        assert manager.fallback_section == manager.initial_section


class TestGetSection:
    def test_returns_config_for_known_section(self):
        manager = make_manager()
        assert manager.get_section("a") is not None

    def test_returns_none_for_unknown_section(self):
        manager = make_manager()
        assert manager.get_section("z") is None


class TestGetSectionWithFallback:
    def test_known_section_no_fallback(self):
        manager = make_manager()
        cfg, used = manager.get_section_with_fallback("a")
        assert cfg.name == "a"
        assert used is False

    def test_unknown_section_uses_fallback(self):
        manager = make_manager(fallback="a")
        cfg, used = manager.get_section_with_fallback("gone")
        assert used is True
        assert cfg.name == "a"

    def test_none_name_uses_fallback_silently(self):
        manager = make_manager(fallback="a")
        cfg, used = manager.get_section_with_fallback(None)
        assert used is True


class TestCanTransition:
    def test_valid_transition_allowed(self):
        manager = make_manager()
        ok, err = manager.can_transition("a", "b", {})
        assert ok is True
        assert err is None

    def test_invalid_transition_rejected_strict(self):
        manager = make_manager(strict=True)
        ok, err = manager.can_transition("a", "c", {})
        assert ok is False
        assert err is not None

    def test_missing_from_section_rejected(self):
        manager = make_manager()
        ok, err = manager.can_transition("z", "a", {})
        assert ok is False

    def test_missing_to_section_rejected(self):
        manager = make_manager()
        ok, err = manager.can_transition("a", "z", {})
        assert ok is False

    def test_required_fields_block_transition(self):
        sections = {
            "a": SectionConfig(name="a", prompt="p", allowed_transitions=["b"]),
            "b": SectionConfig(
                name="b",
                prompt="p",
                required_state_fields={"budget": int},
                strict_validation=True,
            ),
        }
        manager = SectionManager(sections=sections, initial_section="a", fallback_section="a")
        ok, err = manager.can_transition("a", "b", {"section_data": {}})
        assert ok is False
        assert "budget" in err

    def test_required_fields_pass_when_present(self):
        sections = {
            "a": SectionConfig(name="a", prompt="p", allowed_transitions=["b"]),
            "b": SectionConfig(
                name="b",
                prompt="p",
                required_state_fields={"budget": int},
            ),
        }
        manager = SectionManager(sections=sections, initial_section="a", fallback_section="a")
        ok, _ = manager.can_transition("a", "b", {"section_data": {"budget": 500}})
        assert ok is True


class TestEvaluateAutoTransitions:
    def test_no_conditions_returns_none(self):
        manager = make_manager()
        state = {"current_section": "a"}
        assert manager.evaluate_auto_transitions(state) is None

    def test_firing_condition_returns_target(self):
        sections = {
            "start": SectionConfig(
                name="start",
                prompt="p",
                allowed_transitions=["end"],
                auto_transition_conditions=lambda _: "end",
            ),
            "end": SectionConfig(name="end", prompt="p"),
        }
        manager = SectionManager(sections=sections, initial_section="start", fallback_section="start")
        result = manager.evaluate_auto_transitions({"current_section": "start"})
        assert result == "end"

    def test_invalid_auto_transition_is_blocked(self):
        sections = {
            "start": SectionConfig(
                name="start",
                prompt="p",
                allowed_transitions=["other"],  # does NOT include "end"
                auto_transition_conditions=lambda _: "end",
            ),
            "other": SectionConfig(name="other", prompt="p"),
            "end": SectionConfig(name="end", prompt="p"),
        }
        manager = SectionManager(sections=sections, initial_section="start", fallback_section="start")
        result = manager.evaluate_auto_transitions({"current_section": "start"})
        assert result is None

    def test_only_current_section_conditions_evaluated(self):
        fired = []

        def spy_condition(state):
            fired.append("fired")
            return "b"

        sections = {
            "a": SectionConfig(
                name="a",
                prompt="p",
                allowed_transitions=["b"],
                auto_transition_conditions=spy_condition,
            ),
            "b": SectionConfig(name="b", prompt="p"),
        }
        manager = SectionManager(sections=sections, initial_section="a", fallback_section="a")

        # When in section "b", "a"'s conditions must NOT fire
        manager.evaluate_auto_transitions({"current_section": "b"})
        assert fired == []


class TestGetSectionNames:
    def test_returns_all_names(self):
        manager = make_manager(section_names=["x", "y", "z"], initial="x")
        assert set(manager.get_section_names()) == {"x", "y", "z"}
