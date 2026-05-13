"""Unit tests for SectionConfig and TransitionCondition."""

import pytest

from section_flow import SectionConfig, TransitionCondition


def make_section(**kwargs) -> SectionConfig:
    defaults = {"name": "test", "prompt": "Test prompt."}
    return SectionConfig(**{**defaults, **kwargs})


class TestSectionConfigValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            make_section(name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            make_section(name="   ")

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt"):
            make_section(prompt="")

    def test_valid_config(self):
        cfg = make_section(name="gather", prompt="Collect info.")
        assert cfg.name == "gather"
        assert cfg.prompt == "Collect info."


class TestCanTransitionTo:
    def test_empty_allowed_transitions_permits_any(self):
        cfg = make_section(allowed_transitions=[])
        assert cfg.can_transition_to("anywhere") is True

    def test_listed_target_is_permitted(self):
        cfg = make_section(allowed_transitions=["next"])
        assert cfg.can_transition_to("next") is True

    def test_unlisted_target_is_rejected(self):
        cfg = make_section(allowed_transitions=["next"])
        assert cfg.can_transition_to("other") is False


class TestValidateRequiredFields:
    def test_all_fields_present_and_correct_type(self):
        cfg = make_section(required_state_fields={"budget": int})
        state = {"section_data": {"budget": 1000}}
        ok, missing = cfg.validate_required_fields(state)
        assert ok is True
        assert missing == []

    def test_missing_field_reported(self):
        cfg = make_section(required_state_fields={"budget": int})
        state = {"section_data": {}}
        ok, missing = cfg.validate_required_fields(state)
        assert ok is False
        assert any("budget" in m for m in missing)

    def test_wrong_type_reported(self):
        cfg = make_section(required_state_fields={"budget": int})
        state = {"section_data": {"budget": "not-an-int"}}
        ok, missing = cfg.validate_required_fields(state)
        assert ok is False
        assert any("budget" in m for m in missing)


class TestEvaluateAutoTransitions:
    def test_no_conditions_returns_none(self):
        cfg = make_section()
        assert cfg.evaluate_auto_transitions({}) is None

    def test_callable_condition_returning_section(self):
        cfg = make_section(auto_transition_conditions=lambda _: "next")
        assert cfg.evaluate_auto_transitions({}) == "next"

    def test_callable_condition_returning_none(self):
        cfg = make_section(auto_transition_conditions=lambda _: None)
        assert cfg.evaluate_auto_transitions({}) is None

    def test_callable_exception_returns_none(self):
        def bad(_):
            raise RuntimeError("oops")

        cfg = make_section(auto_transition_conditions=bad)
        assert cfg.evaluate_auto_transitions({}) is None

    def test_priority_conditions_highest_wins(self):
        conditions = [
            TransitionCondition(target="low", condition=lambda _: True, priority=0),
            TransitionCondition(target="high", condition=lambda _: True, priority=10),
        ]
        cfg = make_section(auto_transition_conditions=conditions)
        assert cfg.evaluate_auto_transitions({}) == "high"

    def test_only_matching_condition_fires(self):
        conditions = [
            TransitionCondition(target="match", condition=lambda _: True, priority=5),
            TransitionCondition(target="nomatch", condition=lambda _: False, priority=10),
        ]
        cfg = make_section(auto_transition_conditions=conditions)
        assert cfg.evaluate_auto_transitions({}) == "match"


class TestTransitionCondition:
    def test_evaluate_returns_true_on_match(self):
        cond = TransitionCondition(target="x", condition=lambda _: True)
        assert cond.evaluate({}) is True

    def test_evaluate_returns_false_on_exception(self):
        def bad(_):
            raise ValueError("boom")

        cond = TransitionCondition(target="x", condition=bad)
        assert cond.evaluate({}) is False
