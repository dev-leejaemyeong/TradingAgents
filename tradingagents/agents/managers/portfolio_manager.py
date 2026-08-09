"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  The parsed object itself is also kept, in
``final_trade_decision_structured`` — a downstream Python hard-clamp layer
needs the typed ``stop_loss``/``take_profit``/``position_size_usd`` fields,
not a re-parse of the markdown.  When a provider does not expose structured
output, the agent falls back gracefully to free-text generation and
``final_trade_decision_structured`` is ``None``.

Sizing context (office-hours design session, 2026-08-08): the prompt tells
the PM ``total_capital_usd``/``max_positions`` — a stable, order-independent
equal-weight baseline (``total_capital_usd / max_positions``) it can scale up
or down from based on conviction. ``available_budget_usd`` (today's actual
remaining cash) is deliberately NOT included — it fluctuates candidate to
candidate within a single run, and telling the PM a shrinking number risks
it self-rationing against hypothetical future candidates it can't see,
recreating the exact "divide what's left" distortion this design avoids.
The real cash ceiling is still enforced, just downstream and silently by
``position_sizer.clamp_to_budget()`` (orchestrator.py) — the PM's number is
never trusted blindly regardless of what it's told.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        total_capital_usd = state.get("total_capital_usd")
        max_positions = state.get("max_positions")
        sizing_context_line = ""
        if total_capital_usd and max_positions:
            baseline_usd = total_capital_usd / max_positions
            sizing_context_line = (
                f"- Portfolio scale: total capital ${total_capital_usd:,.2f} across up to "
                f"{max_positions} positions. Equal-weight baseline for an average-conviction "
                f"Buy: ${baseline_usd:,.2f} — this is a reference point, not a fixed target. "
                f"Size larger than this for stronger conviction, smaller for weaker conviction, "
                f"at your discretion. (This baseline does not account for today's actual "
                f"remaining cash — a downstream system enforces that limit separately, so size "
                f"based on conviction, not on guessing what's left to spend.)\n"
            )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{sizing_context_line}{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.
Set concrete stop_loss/take_profit/position_size_usd values — consider the
Trader's proposal above but decide the final numbers yourself; these are the
numbers that will actually be used to size and manage the position.

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        final_trade_decision, pm_decision = invoke_structured(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "final_trade_decision_structured": pm_decision,
        }

    return portfolio_manager_node
