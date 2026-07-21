"""Behavioral coverage for the analyst/debate nodes rewritten to use
cached_blocks() (prompt_caching.py) instead of a plain interpolated string.

These nodes previously had no dedicated behavioral tests at all — only
source-grep checks (test_i18n_coverage.py, test_news_analyst_prompt.py).
Since the rewrite changed how each node builds and sends its prompt
(ChatPromptTemplate -> manual SystemMessage/HumanMessage, string invoke ->
message-list invoke), this file locks in: (1) each node still returns the
right state on both an Anthropic and a non-Anthropic LLM, and (2) the cache
boundary lands where it's supposed to — stable content cached, per-call
content (ticker/instrument_context, growing debate history) never cached.
"""

from unittest.mock import MagicMock

import pytest
from langchain_anthropic import ChatAnthropic

from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator


def _anthropic_llm():
    return ChatAnthropic(model="claude-haiku-4-5", api_key="test-key-not-real")


def _mock_method(llm, name: str, mock: MagicMock) -> None:
    """Attach a MagicMock as a method on a real ChatAnthropic instance.

    Pydantic v2's __setattr__ rejects plain ``llm.invoke = mock`` (`invoke`
    isn't a declared field), so this goes through ``object.__setattr__`` to
    bypass that check — same trick either way, just made reusable.
    """
    object.__setattr__(llm, name, mock)


def _mock_tool_calling_llm(content: str):
    """A MagicMock standing in for a non-Anthropic provider through the
    llm.bind_tools(tools).invoke(messages) path."""
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.return_value = MagicMock(tool_calls=[], content=content)
    llm.bind_tools.return_value = bound
    return llm, bound


def _market_state():
    return {
        "trade_date": "2026-07-21",
        "company_of_interest": "NVDA",
        "asset_type": "stock",
        "messages": [],
    }


@pytest.mark.unit
class TestMarketAnalystCaching:
    def test_non_anthropic_llm_still_produces_report(self):
        llm, bound = _mock_tool_calling_llm("Momentum is strong.")
        result = create_market_analyst(llm)(_market_state())
        assert result["market_report"] == "Momentum is strong."
        # Non-Anthropic: content must stay a plain string, never a block list
        # with a cache_control key (unrecognized-field risk on that provider).
        sent_messages = bound.invoke.call_args[0][0]
        system_content = sent_messages[0].content
        assert isinstance(system_content, str)
        assert "NVDA" in system_content

    def test_anthropic_llm_caches_stable_prefix_not_instrument_context(self):
        llm = _anthropic_llm()
        bound = MagicMock()
        bound.invoke.return_value = MagicMock(tool_calls=[], content="report")
        _mock_method(llm, "bind_tools", MagicMock(return_value=bound))

        create_market_analyst(llm)(_market_state())

        sent_messages = bound.invoke.call_args[0][0]
        blocks = sent_messages[0].content
        assert isinstance(blocks, list)
        # First block: role + indicator instructions - stable across every
        # ticker in a day's batch - must carry a cache breakpoint.
        assert "cache_control" in blocks[0]
        assert "trading assistant" in blocks[0]["text"]
        # Last block: instrument_context (ticker-specific) must never be
        # cached, or every ticker's request would look like a cache write
        # instead of a hit and none of them would ever actually hit.
        assert "cache_control" not in blocks[-1]
        assert "NVDA" in blocks[-1]["text"]

    def test_tool_call_in_progress_returns_empty_report(self):
        """When the model is still mid tool-call loop (tool_calls non-empty),
        market_report must stay empty rather than leaking a partial string."""
        llm = MagicMock()
        bound = MagicMock()
        bound.invoke.return_value = MagicMock(tool_calls=[{"name": "get_stock_data"}], content="")
        llm.bind_tools.return_value = bound
        result = create_market_analyst(llm)(_market_state())
        assert result["market_report"] == ""


def _bull_state():
    return {
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "market_report": "Market strong.",
        "sentiment_report": "Bullish.",
        "news_report": "Positive news.",
        "fundamentals_report": "Solid fundamentals.",
        "company_of_interest": "NVDA",
        "asset_type": "stock",
    }


@pytest.mark.unit
class TestBullResearcherCaching:
    def test_non_anthropic_llm_returns_argument(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Strong growth ahead.")
        result = create_bull_researcher(llm)(_bull_state())
        assert "Bull Analyst: Strong growth ahead." in result["investment_debate_state"]["history"]
        sent = llm.invoke.call_args[0][0][0].content
        assert isinstance(sent, str)

    def test_anthropic_caches_resources_not_growing_history(self):
        llm = _anthropic_llm()
        _mock_method(llm, "invoke", MagicMock(return_value=MagicMock(content="argument")))

        state = _bull_state()
        state["investment_debate_state"]["history"] = "Bear Analyst: prior round.\n"
        create_bull_researcher(llm)(state)

        blocks = llm.invoke.call_args[0][0][0].content
        assert isinstance(blocks, list)
        # Role instructions + this run's reports repeat every debate turn -> cached.
        assert "cache_control" in blocks[0]
        assert "Market research report: Market strong." in blocks[0]["text"]
        # Debate history/current_response grow every turn -> never cached.
        assert "cache_control" not in blocks[-1]
        assert "prior round" in blocks[-1]["text"]


def _aggressive_state():
    return {
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
        "fundamentals_report": "f",
        "trader_investment_plan": "Buy 100 shares.",
        "company_of_interest": "NVDA",
        "asset_type": "stock",
    }


@pytest.mark.unit
class TestAggressiveDebatorCaching:
    def test_non_anthropic_llm_returns_argument(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="High reward ahead.")
        result = create_aggressive_debator(llm)(_aggressive_state())
        assert "Aggressive Analyst: High reward ahead." in result["risk_debate_state"]["history"]

    def test_anthropic_caches_trader_decision_and_reports(self):
        llm = _anthropic_llm()
        _mock_method(llm, "invoke", MagicMock(return_value=MagicMock(content="argument")))
        create_aggressive_debator(llm)(_aggressive_state())

        blocks = llm.invoke.call_args[0][0][0].content
        assert "cache_control" in blocks[0]
        assert "Buy 100 shares." in blocks[0]["text"]
        assert "cache_control" not in blocks[-1]
