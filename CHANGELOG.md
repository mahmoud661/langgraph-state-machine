# Changelog

All notable changes to `langgraph-section-flow` will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] – 2026-05-13

### Added

- `SectionFlowMiddleware` – core middleware class with `before_model`,
  `abefore_model`, `wrap_model_call`, and `awrap_model_call` hooks.
- `SectionConfig` – Pydantic model describing a single section (prompt,
  tools, transitions, hooks, LLM override).
- `TransitionCondition` – priority-ordered condition-guarded transition.
- `SectionManager` – validates transitions and evaluates auto-transition
  conditions.
- `SectionFlowState` – `TypedDict` extension for LangGraph state with
  `current_section`, `section_data`, and `visited_sections`.
- `create_change_section_tool` – `StructuredTool` (sync + async) that lets
  the agent drive its own section transitions.
- Per-section tool filtering with caching.
- Per-section LLM override via `SectionConfig.llm`.
- Auto-transition conditions (callable or list of `TransitionCondition`).
- Fallback section for graceful handling of removed sections in production.
- Global tool overrides that take precedence over section-level tools.
- Three usage examples covering basic flow, auto-transitions, and per-section
  LLM overrides.
- Full unit-test suite for `SectionManager`, `SectionConfig`, and
  `create_change_section_tool`.
