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

import tradingagents.graph.setup as graph_setup_module
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.graph.conditional_logic import ConditionalLogic


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

    def test_cache_role_and_resources_false_skips_the_breakpoint(self):
        """max_debate_rounds==1 (graph/setup.py passes cache_role_and_resources
        =False in that case): this node fires exactly once, so a cache
        breakpoint here would only pay Anthropic's write premium with no
        later turn to ever read it back (TODOS #85 A/B test, 2026-09-02,
        found exactly this waste live: 3/3 tickers wrote ~18-19K cache
        tokens, 0 ever read)."""
        llm = _anthropic_llm()
        _mock_method(llm, "invoke", MagicMock(return_value=MagicMock(content="argument")))

        create_bull_researcher(llm, cache_role_and_resources=False)(_bull_state())

        blocks = llm.invoke.call_args[0][0][0].content
        assert isinstance(blocks, list)
        assert "cache_control" not in blocks[0]
        assert "Market research report: Market strong." in blocks[0]["text"]
        assert "cache_control" not in blocks[-1]


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

    def test_cache_role_and_resources_false_skips_the_breakpoint(self):
        """max_risk_discuss_rounds==1 (graph/setup.py passes
        cache_role_and_resources=False in that case): same reasoning as
        TestBullResearcherCaching's equivalent test."""
        llm = _anthropic_llm()
        _mock_method(llm, "invoke", MagicMock(return_value=MagicMock(content="argument")))
        create_aggressive_debator(llm, cache_role_and_resources=False)(_aggressive_state())

        blocks = llm.invoke.call_args[0][0][0].content
        assert "cache_control" not in blocks[0]
        assert "Buy 100 shares." in blocks[0]["text"]
        assert "cache_control" not in blocks[-1]


@pytest.mark.unit
class TestSetupGraphCacheWiring:
    """graph/setup.py must derive cache_role_and_resources from the actual
    configured round counts, not hardcode it -- this is what makes the
    node-level behavior above (should_cache follows the flag) actually reach
    a real graph. Builds a real GraphSetup/ConditionalLogic (pure local
    graph construction, no LLM/network calls) with the 5 caching node
    factories monkeypatched to record what they were called with."""

    @staticmethod
    def _tool_nodes():
        return {key: (lambda state: state) for key in ("market", "social", "news", "fundamentals")}

    def _build(self, monkeypatch, max_debate_rounds, max_risk_discuss_rounds):
        captured = {}

        def _capturing_factory(name):
            def factory(llm, cache_role_and_resources=True):
                captured[name] = cache_role_and_resources
                return lambda state: state
            return factory

        for target, name in (
            ("create_bull_researcher", "bull_researcher"),
            ("create_bear_researcher", "bear_researcher"),
            ("create_aggressive_debator", "aggressive_analyst"),
            ("create_conservative_debator", "conservative_analyst"),
            ("create_neutral_debator", "neutral_analyst"),
        ):
            monkeypatch.setattr(graph_setup_module, target, _capturing_factory(name))

        graph_setup = graph_setup_module.GraphSetup(
            quick_thinking_llm="QUICK",
            deep_thinking_llm="DEEP",
            tool_nodes=self._tool_nodes(),
            conditional_logic=ConditionalLogic(
                max_debate_rounds=max_debate_rounds,
                max_risk_discuss_rounds=max_risk_discuss_rounds,
            ),
        )
        graph_setup.setup_graph()
        return captured

    def test_single_round_disables_caching_on_all_five_nodes(self, monkeypatch):
        """This project's production default (max_debate_rounds=
        max_risk_discuss_rounds=1) -- every debate node fires exactly once,
        so none of them should mark anything cacheable."""
        captured = self._build(monkeypatch, max_debate_rounds=1, max_risk_discuss_rounds=1)
        assert captured == {
            "bull_researcher": False,
            "bear_researcher": False,
            "aggressive_analyst": False,
            "conservative_analyst": False,
            "neutral_analyst": False,
        }

    def test_multi_round_enables_caching_on_all_five_nodes(self, monkeypatch):
        captured = self._build(monkeypatch, max_debate_rounds=2, max_risk_discuss_rounds=2)
        assert captured == {
            "bull_researcher": True,
            "bear_researcher": True,
            "aggressive_analyst": True,
            "conservative_analyst": True,
            "neutral_analyst": True,
        }

    def test_debate_and_risk_rounds_are_independent(self, monkeypatch):
        """A config with only one side multi-round (unusual, but not
        excluded by default_config.py) must not couple the two flags."""
        captured = self._build(monkeypatch, max_debate_rounds=1, max_risk_discuss_rounds=2)
        assert captured["bull_researcher"] is False
        assert captured["bear_researcher"] is False
        assert captured["aggressive_analyst"] is True
        assert captured["conservative_analyst"] is True
        assert captured["neutral_analyst"] is True
