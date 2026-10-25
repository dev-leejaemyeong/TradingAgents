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

``pm_size_range_low``/``pm_size_range_high`` (2026-09-04): the "typically
within roughly Nx-Mx the baseline" guidance below is rendered from these
state values rather than hardcoded, because orchestrator.py independently
checks the PM's actual proposal against
``PM_SIZE_TYPICAL_RANGE_LOW``/``HIGH`` for its "sizing outside typical
range" alert — two copies of the same number with nothing enforcing they
match used to risk the alert firing against a range the PM was never
actually told to follow. Defaulting to 0.5/2.0 here (below) reproduces the
exact text this prompt had before this change for any caller that doesn't
pass them (e.g. TradingAgents CLI callers outside this project).
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


# Mirrors research_manager.py's two scales (see the comment there): the
# original wording describes managing a position that exists, which is wrong
# for a name nothing is held in. The Research Manager rates first and this
# node saw its rating in 46 of 46 debates -- so both nodes need the matching
# scale, or the one that decides first keeps the position-management framing.
HELD_REVIEW_SCALE = """**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry"""

NEW_ENTRY_SCALE = """**Rating Scale** (use exactly one). Nothing is held in this name yet, so every tier describes whether and how to START a position:
- **Buy**: Strong conviction; enter at full target weight
- **Overweight**: Favorable outlook; enter at reduced size
- **Underweight**: Do not enter; set rejection_category to the primary concern
- **Hold**: Evidence genuinely balanced; do not enter now
- **Sell**: Avoid this name; set rejection_category

Both entry tiers open a real position, so size them for what you actually believe."""


def rating_scale_for(evaluation_mode: str | None) -> str:
    """The rating-scale block matching what the caller is deciding."""
    return NEW_ENTRY_SCALE if evaluation_mode == "new_entry" else HELD_REVIEW_SCALE


def create_portfolio_manager(llm, memory_log=None, similar_lessons: int = 2):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        # TODOS #168/M2: lessons whose recorded analyst reports resemble THIS
        # run's. Computed here, not with past_context (which is built before
        # the analysts have run): the four reports are the query. Decoration
        # only -- any failure drops the section and the PM proceeds.
        if memory_log is not None and similar_lessons > 0:
            try:
                query = "\n\n".join(
                    state.get(k) or ""
                    for k in ("market_report", "sentiment_report", "news_report", "fundamentals_report")
                )
                similar = memory_log.get_similar_lessons(
                    query, state.get("company_of_interest", ""), n=similar_lessons
                )
            except Exception:
                similar = ""
            if similar:
                past_context = f"{past_context}\n\n{similar}" if past_context else similar
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        total_capital_usd = state.get("total_capital_usd")
        max_positions = state.get("max_positions")
        # Defaults reproduce the exact range this prompt hardcoded before
        # 2026-09-04 -- see this module's docstring.
        pm_size_range_low = state.get("pm_size_range_low") or 0.5
        pm_size_range_high = state.get("pm_size_range_high") or 2.0
        rating_scale = rating_scale_for(state.get("evaluation_mode"))
        # 2026-09-22 (TODOS #166): the forced-exit horizon, which the PM was
        # never told. Measured before adding it: of 16 debates where the PM
        # stated a holding period, 15 were longer than the cap in force
        # ("3-6 months" five times, "12-18 months", "13-14 months") and one
        # fitted -- so stop/take levels and sizing were being set against a
        # thesis the system would close at a fraction of its horizon.
        #
        # Deliberately states only the horizon. The candidate's volatility is
        # already in front of the PM -- the market analyst supplies ATR in 60
        # of 61 debates and the PM mentions volatility in 3 -- so repeating it
        # in a second place was judged unlikely to change anything by itself.
        # That half waits for this one to be measured first
        # (DATA_VALIDATION_QUEUE.md), so the two changes do not land together
        # and become impossible to tell apart.
        # 2026-09-22 (TODOS #166 item 2): hold_days lets the PM shorten this
        # per-ticker -- a value longer than max_hold_days is silently
        # clamped back down downstream (position_sizer.clamp_hold_days()),
        # so this is a ceiling, not a target the PM can extend.
        max_hold_days = state.get("max_hold_days")
        horizon_line = (
            f"- Holding horizon: this position is force-closed {max_hold_days} calendar "
            f"days after entry regardless of how the thesis is developing. Set "
            f"stop_loss/take_profit and size for that window. If your conviction "
            f"calls for closing it out sooner than that, say so with the hold_days "
            f"field -- a longer number there cannot extend the {max_hold_days}-day cap.\n"
            if max_hold_days
            else ""
        )
        held_horizon = state.get("held_horizon")
        if state.get("evaluation_mode") == "held_review" and held_horizon:
            # 2026-10-23 (TODOS #166-3/A9): a review of a position already held.
            # hold_days here counts from TODAY and may lengthen the position up
            # to the cap -- the entry wording above ("shorter only, else leave
            # blank") made a PM that followed it return nothing, so an
            # extension could never happen, and a PM that read it as "from
            # entry" could name a date already in the past.
            horizon_line = (
                f"- Holding horizon: this position is currently scheduled to be "
                f"force-closed in {held_horizon['current_remaining_days']} calendar days. "
                f"It may be kept for at most {held_horizon['max_remaining_days']} more "
                f"calendar days from today. Set hold_days to the number of calendar days "
                f"FROM TODAY you want to keep holding (1 to "
                f"{held_horizon['max_remaining_days']}); a larger number is clamped down. "
                f"Leave it unset to keep the current schedule. To exit now, rate it "
                f"Sell instead -- hold_days is not an exit signal.\n"
            )
        sizing_context_line = ""
        if total_capital_usd and max_positions:
            baseline_usd = total_capital_usd / max_positions
            sizing_context_line = (
                f"- Portfolio scale: total capital ${total_capital_usd:,.2f} across up to "
                f"{max_positions} positions. Equal-weight baseline for an average-conviction "
                f"Buy: ${baseline_usd:,.2f} — this is a reference point, not a fixed target. "
                f"Size larger than this for stronger conviction, smaller for weaker conviction, "
                f"at your discretion — typically within roughly "
                f"{pm_size_range_low:g}x-{pm_size_range_high:g}x the baseline, and both "
                f"directions should see real use, not just the downward one. (This baseline does "
                f"not account for today's actual remaining cash — a downstream system enforces "
                f"that limit separately, so size based on conviction, not on guessing what's "
                f"left to spend.)\n"
            )

        if state.get("evaluation_mode") == "held_review" and total_capital_usd and max_positions:
            # 2026-10-23 (TODOS #164/A10a): the entry sizing wording above
            # ("Equal-weight baseline for an average-conviction Buy ... size
            # larger for stronger conviction") reads as an opening size; on a
            # held-position review position_size_usd is only the amount ADDED
            # now, and nothing is added unless the rating is Buy/Overweight.
            sizing_context_line = (
                f"- Add-on sizing: this position is already held. position_size_usd here "
                f"means the dollar amount to ADD to it now, not its total size. Rate it "
                f"Buy or Overweight and set a positive position_size_usd only if you want "
                f"to add; otherwise leave it 0 or unset -- Hold, Underweight and Sell "
                f"never add. An add-on happens at most once per position and is skipped "
                f"when the position context says it is unavailable. Equal-weight baseline "
                f"per position: ${total_capital_usd / max_positions:,.2f} (a reference, "
                f"not a target; a downstream system separately caps the position's total "
                f"and the cash actually left).\n"
            )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

{rating_scale}

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{horizon_line}{sizing_context_line}{lessons_line}
**Risk Analysts Debate History:**
{history}

---

The Research Manager's rating above is a recommendation, not a constraint: it
was formed before the risk analysts spoke. Where their debate supports a
different conclusion, say so and rate accordingly.

Be decisive and ground every conclusion in specific evidence from the analysts.
Set concrete stop_loss/take_profit/position_size_usd values — consider the
Trader's proposal above but decide the final numbers yourself; these are the
numbers that will actually be used to size and manage the position.

If your rating is Underweight or Sell, set rejection_category to the single
primary reason (see field description for the fixed category list) — this
is used to explain the rejection to you again if this ticker comes back for
re-review later.

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
