"""Example 02 – Auto-transitions with TransitionCondition.

Demonstrates:
- Using a priority-ordered list of TransitionCondition objects
- Data stored in section_data driving automatic flow advancement
- Multiple conditional targets from a single section
"""

import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain_core.tools import tool

from section_flow import SectionFlowMiddleware, SectionConfig, TransitionCondition


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def record_budget(amount: int) -> dict:
    """Record the user's budget.

    Returns a state-update dict so the middleware can persist it.
    """
    # In real usage you'd return a Command that writes to section_data.
    # Here we just confirm the recording for illustration.
    return {"status": "ok", "budget": amount}


@tool
def show_standard_catalog(category: str) -> str:
    """Show the standard product catalog."""
    return f"Standard catalog for '{category}': Model A ($500), Model B ($800), Model C ($1000)"


@tool
def show_premium_catalog(category: str) -> str:
    """Show the premium/enterprise product catalog."""
    return f"Premium catalog for '{category}': Pro X ($3000), Elite Z ($5000), Ultra S ($8000)"


@tool
def finalize_order(item: str, price: float) -> str:
    """Finalize the order."""
    return f"Order confirmed: {item} @ ${price:.2f}. Reference: ORD-{hash(item) % 10000:04d}"


# ---------------------------------------------------------------------------
# Auto-transition conditions
#
# From the "intake" section we branch into two different recommendation paths
# based on the user's budget stored in section_data.
# ---------------------------------------------------------------------------

def _budget_is_set(state: dict) -> bool:
    return state.get("section_data", {}).get("budget") is not None

def _budget_high(state: dict) -> bool:
    return state.get("section_data", {}).get("budget", 0) > 2000

sections = {
    "intake": SectionConfig(
        name="intake",
        prompt=(
            "You are in the INTAKE section.\n"
            "Ask the user for their budget (a number in USD) and record it with record_budget.\n"
            "After recording, do NOT call change_section – the system will advance automatically."
        ),
        tools=[record_budget],
        allowed_transitions=["premium_recommend", "standard_recommend"],
        # Priority 10 fires first: high budget → premium track
        # Priority 0 fires as fallback: any budget → standard track
        auto_transition_conditions=[
            TransitionCondition(
                target="premium_recommend",
                condition=lambda s: _budget_is_set(s) and _budget_high(s),
                priority=10,
                description="Route high-budget users to premium catalog",
            ),
            TransitionCondition(
                target="standard_recommend",
                condition=_budget_is_set,
                priority=0,
                description="Route all other users to standard catalog",
            ),
        ],
    ),
    "premium_recommend": SectionConfig(
        name="premium_recommend",
        prompt=(
            "You are in the PREMIUM RECOMMENDATION section.\n"
            "The user has a high budget. Present the premium catalog and help them choose.\n"
            "Transition to 'finalize' once they have picked an item."
        ),
        tools=[show_premium_catalog],
        allowed_transitions=["finalize"],
    ),
    "standard_recommend": SectionConfig(
        name="standard_recommend",
        prompt=(
            "You are in the STANDARD RECOMMENDATION section.\n"
            "Present the standard catalog and help the user choose within their budget.\n"
            "Transition to 'finalize' once they have picked an item."
        ),
        tools=[show_standard_catalog],
        allowed_transitions=["finalize"],
    ),
    "finalize": SectionConfig(
        name="finalize",
        prompt=(
            "You are in the FINALIZE section.\n"
            "Confirm the selected item and its price, then call finalize_order."
        ),
        tools=[finalize_order],
        allowed_transitions=[],
    ),
}

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

middleware = SectionFlowMiddleware(
    sections=sections,
    initial_section="intake",
    fallback_section="intake",
)

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"]),
    system_prompt="You are a product advisor. Follow your current section's instructions.",
    middleware=[middleware],
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # High-budget user – should auto-route to premium_recommend
    print("=== High-budget user ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I need a laptop. My budget is $6000."}]}
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}\n")

    print("\n=== Standard-budget user ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I need a laptop. My budget is $900."}]}
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}\n")
