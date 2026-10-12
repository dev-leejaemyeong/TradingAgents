"""Tests for parallel_analysts.py (TODOS #21): the four analysts run inside
one "Analyst Team" node, each in its own isolated subgraph, instead of
chaining them one after another.
"""

import time

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage

from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.parallel_analysts import build_analyst_team_node


def _agent_no_tools(report_key, content):
    """An analyst that answers immediately, no tool calls."""
    def node(state):
        return {"messages": [AIMessage(content=content)], report_key: content}
    return node


def _agent_one_tool_round(report_key, content, calls):
    """An analyst that calls a tool once, then answers on its next turn."""
    def node(state):
        calls.append(report_key)
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, ToolMessage):
            return {"messages": [AIMessage(content=content)], report_key: content}
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "lookup", "args": {}, "id": f"{report_key}-1"},
        ])]}
    return node


def _tool_node_stub(state):
    last = state["messages"][-1]
    return {"messages": [
        ToolMessage(content="tool result", tool_call_id=tc["id"]) for tc in last.tool_calls
    ]}


def _slow_agent_no_tools(report_key, content, delay):
    def node(state):
        time.sleep(delay)
        return {"messages": [AIMessage(content=content)], report_key: content}
    return node


def _initial_state(extra=None):
    state = {"messages": [], "company_of_interest": "AAPL", "trade_date": "2026-01-01"}
    state.update(extra or {})
    return state


@pytest.mark.unit
class TestMergesEveryAnalystsReport:
    def test_two_analysts_no_tool_calls(self):
        plan = build_analyst_execution_plan(["market", "news"])
        factories = {
            "market": lambda: _agent_no_tools("market_report", "market says buy"),
            "news": lambda: _agent_no_tools("news_report", "news says hold"),
        }
        tool_nodes = {"market": _tool_node_stub, "news": _tool_node_stub}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        result = node(_initial_state())

        assert result["market_report"] == "market says buy"
        assert result["news_report"] == "news says hold"

    def test_analyst_that_calls_a_tool_once_still_reports(self):
        plan = build_analyst_execution_plan(["market"])
        calls = []
        factories = {"market": lambda: _agent_one_tool_round("market_report", "final answer", calls)}
        tool_nodes = {"market": _tool_node_stub}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        result = node(_initial_state())

        assert result["market_report"] == "final answer"
        assert calls == ["market_report", "market_report"]  # agent invoked twice: tool round + final

    def test_report_key_missing_from_subgraph_output_defaults_to_empty_string(self):
        # An analyst node that never writes its report key (same tolerance
        # the old chained design had) must not raise KeyError in the merge.
        plan = build_analyst_execution_plan(["market"])
        factories = {"market": lambda: (lambda state: {"messages": [AIMessage(content="x")]})}
        tool_nodes = {"market": _tool_node_stub}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        result = node(_initial_state())

        assert result["market_report"] == ""

    def test_messages_left_in_a_clean_anchored_state_after_merge(self):
        # Downstream (Bull/Bear) reads only the report fields, but the shared
        # messages channel must still end up clean, not holding any
        # analyst's internal tool-calling scratch.
        plan = build_analyst_execution_plan(["market"])
        factories = {"market": lambda: _agent_no_tools("market_report", "report")}
        tool_nodes = {"market": _tool_node_stub}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        result = node(_initial_state())

        assert "messages" in result
        # create_msg_delete()'s anchor placeholder names the ticker/date.
        assert "AAPL" in result["messages"][-1].content


@pytest.mark.unit
class TestAnalystsAreIsolatedFromEachOther:
    def test_one_analysts_tool_calling_scratch_never_reaches_another(self):
        # If the two analysts' message histories were shared, "news"'s report
        # would see "market"'s tool call/result in state["messages"].
        plan = build_analyst_execution_plan(["market", "news"])
        seen_by_news = []

        def news_node(state):
            seen_by_news.extend(state["messages"])
            return {"messages": [AIMessage(content="news report")], "news_report": "news report"}

        calls = []
        factories = {
            "market": lambda: _agent_one_tool_round("market_report", "market report", calls),
            "news": lambda: news_node,
        }
        tool_nodes = {"market": _tool_node_stub, "news": _tool_node_stub}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        node(_initial_state())

        # news_node only ever saw the pristine initial state's messages (none
        # here), never market's tool call or tool result.
        assert seen_by_news == []


@pytest.mark.unit
class TestRunConcurrently:
    def test_four_slow_analysts_finish_in_about_one_delay_not_four(self):
        plan = build_analyst_execution_plan(["market", "social", "news", "fundamentals"])
        delay = 0.3
        factories = {
            spec.key: (lambda rk=spec.report_key: _slow_agent_no_tools(rk, rk, delay))
            for spec in plan.specs
        }
        tool_nodes = {spec.key: _tool_node_stub for spec in plan.specs}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        started = time.monotonic()
        result = node(_initial_state())
        elapsed = time.monotonic() - started

        assert elapsed < delay * 2  # concurrent: ~1x delay, not 4x
        for spec in plan.specs:
            assert result[spec.report_key] == spec.report_key


@pytest.mark.unit
class TestErrorPropagation:
    def test_one_analyst_raising_fails_the_whole_node_not_silently(self):
        plan = build_analyst_execution_plan(["market", "news"])

        def broken(state):
            raise RuntimeError("boom")

        factories = {
            "market": lambda: broken,
            "news": lambda: _agent_no_tools("news_report", "ok"),
        }
        tool_nodes = {"market": _tool_node_stub, "news": _tool_node_stub}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        with pytest.raises(RuntimeError, match="boom"):
            node(_initial_state())


class _RecordingHandler(BaseCallbackHandler):
    def __init__(self):
        self.started_nodes = []

    def on_chain_start(self, serialized, inputs, *, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node is not None:
            self.started_nodes.append(node)


@pytest.mark.unit
class TestConfigPropagation:
    def test_callbacks_reach_every_analysts_own_subgraph_nodes(self):
        # Callbacks (cost_logger, pixel_office) are threaded through each
        # subgraph's own RunnableConfig, not re-constructed -- otherwise
        # every analyst's node/LLM calls would go unobserved under parallel
        # execution. Real LangGraph callback wiring (not a stand-in), since
        # that's exactly the plumbing this proves.
        plan = build_analyst_execution_plan(["market", "news"])
        factories = {
            "market": lambda: _agent_no_tools("market_report", "m"),
            "news": lambda: _agent_no_tools("news_report", "n"),
        }
        tool_nodes = {"market": _tool_node_stub, "news": _tool_node_stub}
        node = build_analyst_team_node(plan, factories, tool_nodes, ConditionalLogic())

        handler = _RecordingHandler()
        node(_initial_state(), config={"callbacks": [handler]})

        # Each analyst's own node name (set by its own subgraph, via
        # spec.agent_node) must have been observed -- proving the config
        # this node received was actually forwarded into both subgraphs.
        assert "Market Analyst" in handler.started_nodes
        assert "News Analyst" in handler.started_nodes
