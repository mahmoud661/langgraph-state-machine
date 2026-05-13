"""Example 01 – Basic three-section shopping assistant.

Demonstrates:
- Defining sections with different tools and prompts
- Agent-initiated transitions via the change_section tool
- Accessing section_data across sections
"""

import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain_core.tools import tool

from section_flow import SectionFlowMiddleware, SectionConfig


# ---------------------------------------------------------------------------
# Tools – one set per section
# ---------------------------------------------------------------------------

@tool
def save_user_info(name: str, budget: int, category: str) -> str:
    """Save the user's name, budget, and product category."""
    # In a real app you'd write this to state or a database.
    return f"Saved: name={name}, budget={budget}, category={category}"


@tool
def search_catalog(query: str, max_price: int) -> str:
    """Search the product catalog and return matching items."""
    # Stub – replace with a real catalog search
    return (
        f"Top results for '{query}' under ${max_price}:\n"
        "  1. ProBook 450 – $799\n"
        "  2. EliteBook 840 – $949\n"
        "  3. SpectrePad X – $999"
    )


@tool
def process_payment(item_name: str, amount: float) -> str:
    """Process payment for the selected item."""
    # Stub – replace with a real payment processor
    return f"Payment of ${amount:.2f} for '{item_name}' processed successfully. Order #12345."


# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

sections = {
    "gather": SectionConfig(
        name="gather",
        prompt=(
            "You are in the GATHER section.\n"
            "Your only job right now is to collect three pieces of information from the user:\n"
            "  1. Their full name\n"
            "  2. Their budget (in USD)\n"
            "  3. The product category they are interested in\n\n"
            "Once you have all three, call save_user_info and then transition to 'recommend'."
        ),
        tools=[save_user_info],
        allowed_transitions=["recommend"],
    ),
    "recommend": SectionConfig(
        name="recommend",
        prompt=(
            "You are in the RECOMMEND section.\n"
            "Use the information collected in the previous section to search the catalog.\n"
            "Present exactly three options, compare their pros and cons, and ask the user "
            "to pick one before transitioning to 'checkout'."
        ),
        tools=[search_catalog],
        allowed_transitions=["checkout"],
    ),
    "checkout": SectionConfig(
        name="checkout",
        prompt=(
            "You are in the CHECKOUT section.\n"
            "Confirm the item name and total price with the user, then process payment.\n"
            "After a successful payment, thank the user and end the conversation."
        ),
        tools=[process_payment],
        allowed_transitions=[],  # terminal section
    ),
}

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

middleware = SectionFlowMiddleware(
    sections=sections,
    initial_section="gather",
    fallback_section="gather",
)

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"]),
    system_prompt=(
        "You are a friendly and efficient shopping assistant. "
        "Follow the current section's instructions precisely."
    ),
    middleware=[middleware],
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Hi! I'm looking for a new laptop."}
            ]
        }
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}\n")
