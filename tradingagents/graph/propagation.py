# TradingAgents/graph/propagation.py

from typing import Any

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        total_capital_usd: float | None = None,
        max_positions: int | None = None,
        max_hold_days: int | None = None,
        pm_size_range_low: float | None = None,
        pm_size_range_high: float | None = None,
        evaluation_mode: str | None = None,
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.

        ``total_capital_usd``/``max_positions``: portfolio-scale context for
        the Portfolio Manager's equal-weight sizing baseline (office-hours
        design session, 2026-08-08) -- see portfolio_manager.py's docstring
        for why ``available_budget_usd`` is deliberately excluded here.

        ``max_hold_days`` (2026-09-22): the calendar-day horizon after which
        the position is force-closed regardless of thesis -- the PM sets
        stop_loss/take_profit and sizes against it. Previously not passed at
        all, so the PM routinely reasoned over 6-18 month theses the system
        would close at 90 days.

        ``pm_size_range_low``/``pm_size_range_high`` (2026-09-04): the
        multiple-of-baseline range the PM prompt tells the PM to typically
        size within -- see portfolio_manager.py's docstring for why this is
        threaded through rather than hardcoded in the prompt text.

        ``evaluation_mode`` (2026-09-16): ``"new_entry"`` when nothing is held
        yet, ``"held_review"`` when re-underwriting an open position. Selects
        the rating-scale wording in the Research Manager and Portfolio Manager
        prompts; ``None`` keeps the original, position-management wording.
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "total_capital_usd": total_capital_usd,
            "max_positions": max_positions,
            "max_hold_days": max_hold_days,
            "pm_size_range_low": pm_size_range_low,
            "pm_size_range_high": pm_size_range_high,
            "evaluation_mode": evaluation_mode,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
