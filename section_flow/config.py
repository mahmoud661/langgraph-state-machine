"""Configuration types for section-based workflow control."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .types import (
    AutoTransitionCondition,
    PromptPosition,
    SectionHook,
    SectionName,
    StateDict,
    StateValidator,
)


class TransitionCondition(BaseModel):
    """A condition-guarded transition with an explicit priority.

    When multiple conditions are defined for a section, they are evaluated in
    descending priority order and the first matching one wins.

    Example::

        TransitionCondition(
            target="payment",
            condition=lambda state: bool(state.get("section_data", {}).get("cart")),
            priority=10,
            description="Move to payment once cart is populated",
        )
    """

    target: SectionName
    condition: StateValidator
    priority: int = 0
    description: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def evaluate(self, state: StateDict) -> bool:
        """Return True if the condition is satisfied, False on any exception."""
        try:
            return self.condition(state)
        except Exception:
            return False


class SectionConfig(BaseModel):
    """Full configuration for a single workflow section.

    Every section has at minimum a ``name`` and a ``prompt``. Everything else
    is optional and defaults to the least-restrictive behaviour.

    Args:
        name: Unique identifier for this section.
        prompt: System-prompt fragment injected when the agent is in this section.
        tools: Tools available *only* in this section. Supply tool objects or
            the string name ``"task"`` to include the task-delegation tool.
        allowed_transitions: Sections this one may explicitly transition to.
            Empty list means any transition is permitted.
        required_state_fields: ``{field: type}`` pairs that must be present in
            ``section_data`` before the agent may *enter* this section.
        auto_transition_conditions: Either a callable ``(state) -> section_name | None``
            or a list of :class:`TransitionCondition` objects evaluated before
            each model call.
        strict_validation: Whether to enforce ``required_state_fields`` and
            ``allowed_transitions`` checks. Defaults to ``True``.
        on_enter: Hook called when the section is entered.
        on_exit: Hook called when the section is left.
        prompt_position: ``"append"`` (default) or ``"prepend"`` – where the
            section prompt is placed relative to other system messages.
        allowed_subagents: Names of subagents available via the ``task`` tool
            in this section. ``None`` means all registered subagents.
        metadata: Arbitrary metadata stored with the config (not used by the
            middleware itself).
        context_flags: Key-value flags surfaced to the agent via the section
            prompt context block.
        llm: Optional model override for this section. Replaces the graph's
            default model for every model call while this section is active.
    """

    name: SectionName
    prompt: str
    tools: list[Any] = Field(default_factory=list)
    allowed_transitions: list[SectionName] = Field(default_factory=list)
    required_state_fields: dict[str, type] = Field(default_factory=dict)
    auto_transition_conditions: AutoTransitionCondition | list[TransitionCondition] | None = None
    strict_validation: bool = True
    on_enter: SectionHook | None = None
    on_exit: SectionHook | None = None
    prompt_position: PromptPosition = "append"
    allowed_subagents: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    context_flags: dict[str, Any] = Field(default_factory=dict)
    llm: Any | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("name", "prompt")
    @classmethod
    def validate_not_empty(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_required_fields(self, state: StateDict) -> tuple[bool, list[str]]:
        """Check that all required_state_fields are present in *section_data*.

        Returns:
            ``(ok, missing)`` – ``ok`` is False when any field is absent or
            has the wrong type; ``missing`` lists the offending fields.
        """
        missing: list[str] = []
        section_data = state.get("section_data", {})

        for field, expected_type in self.required_state_fields.items():
            value = section_data.get(field)
            if value is None:
                missing.append(f"{field} (missing)")
            elif not isinstance(value, expected_type):
                missing.append(f"{field} (wrong type: expected {expected_type.__name__})")

        return len(missing) == 0, missing

    def can_transition_to(self, target: SectionName) -> bool:
        """Return True if *target* is in ``allowed_transitions`` (or list is empty)."""
        return not self.allowed_transitions or target in self.allowed_transitions

    # ------------------------------------------------------------------
    # Auto-transition evaluation
    # ------------------------------------------------------------------

    def evaluate_auto_transitions(self, state: StateDict) -> SectionName | None:
        """Return the name of the next section if any auto-transition fires."""
        if not self.auto_transition_conditions:
            return None

        if callable(self.auto_transition_conditions):
            try:
                return self.auto_transition_conditions(state)
            except Exception:
                return None

        for cond in sorted(self.auto_transition_conditions, key=lambda c: c.priority, reverse=True):
            if cond.evaluate(state):
                return cond.target

        return None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def execute_hook(self, hook: SectionHook | None, state: StateDict) -> dict[str, Any] | None:
        """Run *hook* safely, swallowing exceptions and returning None on failure."""
        if hook is None:
            return None
        try:
            return hook(state)
        except Exception:
            return None

    def execute_on_enter(self, state: StateDict) -> dict[str, Any] | None:
        return self.execute_hook(self.on_enter, state)

    def execute_on_exit(self, state: StateDict) -> dict[str, Any] | None:
        return self.execute_hook(self.on_exit, state)


__all__ = ["SectionConfig", "TransitionCondition"]
