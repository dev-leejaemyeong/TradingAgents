"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


# The five tiers are written as position management ("maintain the current
# position", "trim exposure"), which is exactly right when something is already
# held and wrong when it is not: with no position, Hold/Underweight/Sell all
# collapse into "do nothing" and only Buy asks for an action. NEW_ENTRY_SCALE
# restates the same five tiers as entry decisions so the middle of the scale
# stays meaningful (2026-09-16). ``evaluation_mode`` unset keeps the original.
HELD_REVIEW_SCALE = """**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced."""

NEW_ENTRY_SCALE = """**Rating Scale** (use exactly one). No position is held in this name yet, so every tier describes whether and how to START one:
- **Buy**: Strong conviction in the bull thesis; enter at full target weight
- **Overweight**: Constructive view; enter at reduced size
- **Underweight**: Cautious view; do not enter, and name the primary concern
- **Hold**: The evidence on both sides is genuinely balanced; do not enter now, revisit later
- **Sell**: Strong conviction in the bear thesis; avoid this name

Commit to a clear stance whenever the debate's strongest arguments warrant one. Hold is not a safe default here -- it means the debate produced no usable edge in either direction."""


def rating_scale_for(evaluation_mode: str | None) -> str:
    """The rating-scale block matching what the caller is deciding."""
    return NEW_ENTRY_SCALE if evaluation_mode == "new_entry" else HELD_REVIEW_SCALE


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")
        rating_scale = rating_scale_for(state.get("evaluation_mode"))

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

{rating_scale}

---

**Debate History:**
{history}

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
