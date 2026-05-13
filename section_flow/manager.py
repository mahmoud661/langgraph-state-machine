"""SectionManager – validates and executes section transitions."""

from __future__ import annotations

import logging
from typing import Any

from .config import SectionConfig
from .types import SectionName

logger = logging.getLogger(__name__)


class SectionManager:
    """Central registry and gatekeeper for all section transitions.

    The manager holds the full map of :class:`~section_flow.config.SectionConfig`
    objects, validates requested transitions, and evaluates auto-transition
    conditions.  Instantiate it directly or let
    :class:`~section_flow.middleware.SectionFlowMiddleware` build one for you.

    Args:
        sections: Complete dictionary of ``{section_name: SectionConfig}``.
        initial_section: The section the agent starts in.
        strict_validation: When ``True`` (default), transitions that violate
            ``allowed_transitions`` or missing ``required_state_fields`` are
            rejected.  When ``False`` they produce a warning only.
        fallback_section: Section to activate when ``current_section`` refers
            to a name that no longer exists in *sections* (e.g. after a
            deployment that removed a section).  Defaults to
            ``initial_section``.
    """

    def __init__(
        self,
        sections: dict[SectionName, SectionConfig],
        initial_section: SectionName,
        strict_validation: bool = True,
        fallback_section: SectionName | None = None,
    ) -> None:
        if not sections:
            raise ValueError("At least one section must be defined")
        if initial_section not in sections:
            raise ValueError(f"Initial section '{initial_section}' not found in sections")

        self.sections = sections
        self.initial_section = initial_section
        self.strict_validation = strict_validation

        if fallback_section is None:
            logger.warning(
                "No fallback_section specified. Using initial_section '%s' as fallback. "
                "Consider explicitly setting fallback_section to handle removed sections in production.",
                initial_section,
            )
            self.fallback_section = initial_section
        else:
            self.fallback_section = fallback_section

        if self.fallback_section not in sections:
            raise ValueError(f"Fallback section '{self.fallback_section}' not found in sections")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_section(self, name: SectionName) -> SectionConfig | None:
        """Return config for *name*, or ``None`` if not found."""
        return self.sections.get(name)

    def get_section_with_fallback(self, name: SectionName | None) -> tuple[SectionConfig, bool]:
        """Return ``(config, used_fallback)``.

        If *name* exists in sections the first element is its config and the
        second is ``False``.  If not (e.g. removed after a deployment) the
        fallback config is returned with ``used_fallback=True`` and a warning
        is emitted.  A ``None`` or empty *name* silently returns the fallback.
        """
        if name and (config := self.sections.get(name)):
            return config, False

        if name:
            logger.warning(
                "Section '%s' not found; using fallback '%s'. "
                "This typically means a section was removed while users still have it in persisted state.",
                name,
                self.fallback_section,
            )

        return self.sections[self.fallback_section], True

    def get_current_section_config(self, state: dict[str, Any]) -> SectionConfig | None:
        """Convenience wrapper that reads ``current_section`` from *state*."""
        current = state.get("current_section")
        return self.get_section(current) if current else None

    def get_section_names(self) -> list[SectionName]:
        """Return a stable-ordered list of all registered section names."""
        return list(self.sections.keys())

    # ------------------------------------------------------------------
    # Transition validation
    # ------------------------------------------------------------------

    def can_transition(
        self,
        from_section: SectionName,
        to_section: SectionName,
        state: dict[str, Any],
        strict: bool | None = None,
    ) -> tuple[bool, str | None]:
        """Check whether a transition from *from_section* to *to_section* is allowed.

        Args:
            from_section: Source section name.
            to_section: Target section name.
            state: Current agent state (used to validate required fields).
            strict: Override the manager-level ``strict_validation`` flag for
                this call.

        Returns:
            ``(allowed, error_message)`` – *error_message* is ``None`` when
            the transition is allowed.
        """
        use_strict = strict if strict is not None else self.strict_validation

        from_config = self.get_section(from_section)
        to_config = self.get_section(to_section)

        if not from_config:
            return False, f"Source section '{from_section}' not found"
        if not to_config:
            return False, f"Target section '{to_section}' not found"

        if not from_config.can_transition_to(to_section):
            if use_strict and from_config.strict_validation:
                return False, f"Transition '{from_section}' → '{to_section}' is not in allowed_transitions"
            logger.warning("Non-strict transition allowed: %s → %s", from_section, to_section)

        if use_strict and to_config.strict_validation:
            valid, missing = to_config.validate_required_fields(state)
            if not valid:
                return False, f"Missing required fields for '{to_section}': {missing}"

        return True, None

    # ------------------------------------------------------------------
    # Auto-transition evaluation
    # ------------------------------------------------------------------

    def evaluate_auto_transitions(self, state: dict[str, Any]) -> SectionName | None:
        """Evaluate auto-transition conditions for the *current* section only.

        Only the active section's conditions are checked – this prevents
        earlier sections from re-firing when the agent has already moved on.

        Returns:
            The target section name if an auto-transition should fire, else
            ``None``.
        """
        current = state.get("current_section") or self.initial_section
        config = self.get_current_section_config(state)
        if not config:
            return None

        target = config.evaluate_auto_transitions(state)

        if target:
            can_trans, error = self.can_transition(current, target, state)
            if not can_trans:
                logger.warning("Auto-transition blocked: %s → %s. %s", current, target, error)
                return None

        return target


__all__ = ["SectionManager"]
