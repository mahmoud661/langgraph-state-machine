# LangGraph Section Flow Middleware

> **Section-based flow control for LangGraph React agents.**
> Divide a conversational agent into discrete, self-contained phases — each with its own prompt, tools, and transition rules — without writing a custom graph.

---

## Why section flow?

A typical React agent sees all tools and the full system prompt on every call.
As workflows grow (qualification → recommendation → booking → payment), this
creates two problems:

1. **Context pollution** — the agent is distracted by tools and instructions
   that are irrelevant to the current step.
2. **Unclear guardrails** — it's hard to restrict what the agent can do at
   each stage without building a bespoke multi-node graph.

`SectionFlowMiddleware` solves both by layering a lightweight state machine
on top of `create_react_agent`.  You describe sections in plain Python; the
middleware handles everything else.

---

## Features

| Feature | Description |
|---|---|
| **Section-scoped tools** | Only the tools listed for the active section are visible to the model |
| **Section-scoped prompts** | Phase instructions are injected as a prepended system message (prompt-cache friendly) |
| **Auto-transitions** | Conditions evaluated before every model call advance the flow automatically |
| **Agent-initiated transitions** | The built-in `change_section` tool lets the model move itself through the workflow |
| **Per-section LLM** | Swap to a different model for a specific phase (e.g. a cheaper model for data-gathering) |
| **Strict validation** | Optionally require specific state fields before entering a section |
| **Fallback sections** | Gracefully recover when persisted state references a section that no longer exists |
| **Global tool overrides** | Certain tools can span all sections and override section-level counterparts |

---

## Installation

```bash
pip install langgraph-state-machine
```

**Requirements:** Python ≥ 3.10, `langchain ≥ 1.0.0`, `langgraph ≥ 0.2.0`

---

## Quick start

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent

from section_flow import SectionFlowMiddleware, SectionConfig

# --- Define your tools (stubs for illustration) ---
def collect_requirements(query: str) -> str: ...
def search_catalog(query: str) -> str: ...
def process_payment(amount: float) -> str: ...

# --- Configure sections ---
sections = {
    "gather": SectionConfig(
        name="gather",
        prompt=(
            "Your goal is to understand the user's needs. "
            "Ask for their name, budget, and product category before moving on."
        ),
        tools=[collect_requirements],
        allowed_transitions=["recommend"],
    ),
    "recommend": SectionConfig(
        name="recommend",
        prompt="Suggest the three best products that match the user's stated budget and category.",
        tools=[search_catalog],
        allowed_transitions=["checkout"],
    ),
    "checkout": SectionConfig(
        name="checkout",
        prompt="Guide the user through payment. Confirm the amount before charging.",
        tools=[process_payment],
    ),
}

# --- Build the agent ---
agent = create_react_agent(
    model="openai:gpt-4o",
    system_prompt="You are a friendly shopping assistant.",
    middleware=[
        SectionFlowMiddleware(
            sections=sections,
            initial_section="gather",
        )
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hi, I need a new laptop."}]})
```

---

## Core concepts

### SectionConfig

Each section is a `pydantic.BaseModel`:

```python
SectionConfig(
    name="gather",                        # unique identifier
    prompt="...",                         # injected system message fragment
    tools=[my_tool],                      # tool objects available in this section
    allowed_transitions=["recommend"],    # sections this one may transition to
    required_state_fields={"budget": int},# fields that must exist in section_data before entering
    auto_transition_conditions=lambda state: (
        "recommend" if state.get("section_data", {}).get("budget") else None
    ),
    strict_validation=True,               # enforce allowed_transitions and required_state_fields
    on_enter=lambda state: None,          # lifecycle hook (called on entry)
    on_exit=lambda state: None,           # lifecycle hook (called on exit)
    llm=ChatOpenAI(model="gpt-4o-mini"), # optional per-section model override
    allowed_subagents=["research_agent"], # limit task-tool subagents for this section
)
```

### SectionFlowState

The middleware extends your agent state with three fields:

```python
class MyState(SectionFlowState):         # merge with your existing state
    messages: Annotated[list, add_messages]

# Fields added by the middleware:
# current_section  – name of the active section
# section_data     – shared dict for cross-section data (e.g. {"budget": 1000})
# visited_sections – ordered list of activated sections
```

### Transitions

There are three ways to advance the flow:

| Method | When to use |
|---|---|
| **Auto-transition** | Data-driven: move when `section_data` satisfies a condition |
| **`change_section` tool** | Agent-driven: model explicitly calls the tool |
| **`before_model` fallback** | Safety net: fallback section for missing/removed sections |

#### Auto-transition (callable)

```python
SectionConfig(
    name="gather",
    auto_transition_conditions=lambda state: (
        "recommend"
        if state.get("section_data", {}).get("budget")
        else None
    ),
    ...
)
```

#### Auto-transition (priority list)

```python
from section_flow import TransitionCondition

SectionConfig(
    name="gather",
    auto_transition_conditions=[
        TransitionCondition(
            target="vip_recommend",
            condition=lambda s: s.get("section_data", {}).get("budget", 0) > 5000,
            priority=10,
        ),
        TransitionCondition(
            target="recommend",
            condition=lambda s: bool(s.get("section_data", {}).get("budget")),
            priority=0,
        ),
    ],
    ...
)
```

### Per-section LLM

```python
from langchain_openai import ChatOpenAI

SectionConfig(
    name="checkout",
    prompt="Process payment carefully.",
    tools=[process_payment],
    llm=ChatOpenAI(model="gpt-4o"),  # override the graph's default model here
)
```

---

## Advanced usage

### Pre-built SectionManager

Reuse the same manager across multiple agents:

```python
from section_flow import SectionManager, SectionFlowMiddleware

manager = SectionManager(
    sections=sections,
    initial_section="gather",
    fallback_section="gather",
)

agent1 = create_react_agent(..., middleware=[SectionFlowMiddleware(section_manager=manager)])
agent2 = create_react_agent(..., middleware=[SectionFlowMiddleware(section_manager=manager)])
```

### Global tools

Tools that should always be available regardless of section:

```python
SectionFlowMiddleware(
    sections=sections,
    initial_section="gather",
    global_tools=[escalate_to_human],   # overrides section tools with same name
)
```

### Disable the transition tool

If you prefer purely automatic or state-driven transitions:

```python
SectionFlowMiddleware(
    sections=sections,
    initial_section="gather",
    include_transition_tool=False,
)
```

### Runtime cache invalidation

If you modify sections at runtime (hot-reload), clear the tool cache:

```python
middleware.clear_tool_cache()
```

---

## API reference

### `SectionFlowMiddleware`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sections` | `dict[str, SectionConfig]` | — | Section registry |
| `initial_section` | `str` | — | Starting section |
| `strict_validation` | `bool` | `True` | Enforce transition and field rules |
| `include_transition_tool` | `bool` | `True` | Register `change_section` tool |
| `section_manager` | `SectionManager` | `None` | Pre-built manager (overrides above) |
| `fallback_section` | `str` | `initial_section` | Fallback for removed sections |
| `global_tools` | `list` | `[]` | Tools available in every section |
| `subagent_graphs` | `dict\|list` | `{}` | Subagent registry for task-tool filtering |
| `all_middleware` | `list` | `[]` | Other middleware for string-name tool resolution |

### `SectionConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | **required** | Unique section identifier |
| `prompt` | `str` | **required** | System message fragment injected when active |
| `tools` | `list` | `[]` | Tool objects (or `"task"`) available here |
| `allowed_transitions` | `list[str]` | `[]` | Reachable sections (empty = all allowed) |
| `required_state_fields` | `dict[str, type]` | `{}` | Fields required in `section_data` to enter |
| `auto_transition_conditions` | callable or list | `None` | Conditions evaluated before each model call |
| `strict_validation` | `bool` | `True` | Enforce rules for this section |
| `on_enter` | callable | `None` | Called when section activates |
| `on_exit` | callable | `None` | Called when section deactivates |
| `llm` | any | `None` | Model override for this section |
| `allowed_subagents` | `list[str]` | `None` | Subagents available via `task` tool |

---

## Examples

See the [`examples/`](examples/) directory:

| File | What it demonstrates |
|---|---|
| [`01_basic_sections.py`](examples/01_basic_sections.py) | Three-section shopping assistant with agent-initiated transitions |
| [`02_auto_transitions.py`](examples/02_auto_transitions.py) | Data-driven auto-transitions using `TransitionCondition` |
| [`03_per_section_llm.py`](examples/03_per_section_llm.py) | Swapping models per section to balance quality and cost |

---

## How it works

```
User message
     │
     ▼
before_model()          ← initialise state, resolve fallbacks, fire auto-transitions
     │
     ▼
wrap_model_call()       ← prepend section prompt, filter tools, swap model if needed
     │
     ▼
LLM call
     │
     ▼
[agent calls change_section tool]  ← updates current_section in state
     │
     ▼
next before_model() ...
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
