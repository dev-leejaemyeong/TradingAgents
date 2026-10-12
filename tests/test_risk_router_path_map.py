"""Shared-router / path_map completeness (#1088).

Both `should_continue_risk_analysis` (three risk edges) and
`should_continue_debate` (two research-debate edges) are single routers whose
return set is larger than any one edge previously mapped. Each edge now shares a
complete path map (`RISK_ANALYSIS_PATH_MAP` / `DEBATE_PATH_MAP`), so a
fall-through return can never hit a missing entry -- which would crash LangGraph
mid-run on prompt/i18n/refactor drift in the speaker labels.
"""
import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import DEBATE_PATH_MAP, RISK_ANALYSIS_PATH_MAP


def _state(latest_speaker, count=0):
    return {"risk_debate_state": {"latest_speaker": latest_speaker, "count": count}}


def _debate_state(current_response, count=0):
    return {"investment_debate_state": {"current_response": current_response, "count": count}}


@pytest.mark.unit
@pytest.mark.parametrize("latest_speaker", [
    "Aggressive", "Aggressive Analyst",
    "Conservative", "Conservative Analyst",
    "Neutral", "Neutral Analyst",
    "",                          # drift: empty label
    "Aggressive Risk Analyst",   # drift: node renamed
    "Agresivo",                  # drift: i18n / translated label
])
def test_router_return_always_routable(latest_speaker):
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    target = logic.should_continue_risk_analysis(_state(latest_speaker))
    assert target in RISK_ANALYSIS_PATH_MAP


@pytest.mark.unit
def test_router_terminates_at_round_limit():
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    # count >= 3 * rounds routes to the Portfolio Manager (debate ends)
    assert logic.should_continue_risk_analysis(_state("Neutral", count=3)) == "Portfolio Manager"


@pytest.mark.unit
def test_path_map_covers_full_router_range():
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    returns = {
        logic.should_continue_risk_analysis(_state(s, c))
        for s in ("Aggressive", "Conservative", "Neutral", "drift")
        for c in (0, 99)
    }
    # Every value the router can emit is a key in the shared map...
    assert returns <= set(RISK_ANALYSIS_PATH_MAP)
    # ...and the terminal target is reachable.
    assert "Portfolio Manager" in returns


@pytest.mark.unit
@pytest.mark.parametrize("current_response", [
    "Bull", "Bull Researcher", "Bear", "Bear Researcher",
    "",                       # drift: empty label
    "Optimista",              # drift: i18n / translated label
])
def test_debate_router_return_always_routable(current_response):
    logic = ConditionalLogic(max_debate_rounds=1)
    target = logic.should_continue_debate(_debate_state(current_response))
    assert target in DEBATE_PATH_MAP


@pytest.mark.unit
def test_debate_path_map_covers_full_router_range():
    logic = ConditionalLogic(max_debate_rounds=1)
    returns = {
        logic.should_continue_debate(_debate_state(s, c))
        for s in ("Bull", "Bear", "drift")
        for c in (0, 99)
    }
    assert returns <= set(DEBATE_PATH_MAP)
    assert "Research Manager" in returns  # terminal reachable


# --- who opens the bull/bear debate (2026-09-22) ----------------------------

from tradingagents.graph.conditional_logic import debate_first_speaker


@pytest.mark.unit
def test_first_speaker_is_deterministic_and_routable():
    a = debate_first_speaker("VST", "2026-09-21")
    assert a == debate_first_speaker("VST", "2026-09-21")
    assert a in DEBATE_PATH_MAP and a != "Research Manager"


@pytest.mark.unit
def test_first_speaker_is_roughly_balanced():
    """Alternation must actually alternate -- a hash that always says Bull
    would silently keep the old bias while looking like it was fixed."""
    picks = [debate_first_speaker(f"T{i}", "2026-09-21") for i in range(400)]
    bull = picks.count("Bull Researcher")
    assert 150 < bull < 250, bull


@pytest.mark.unit
def test_first_speaker_varies_with_date_for_same_ticker():
    picks = {debate_first_speaker("VST", f"2026-09-{d:02d}") for d in range(1, 29)}
    assert picks == {"Bull Researcher", "Bear Researcher"}


@pytest.mark.unit
def test_first_debate_speaker_reads_ticker_and_date_from_state():
    logic = ConditionalLogic()
    state = {"company_of_interest": "VST", "trade_date": "2026-09-21"}
    assert logic.first_debate_speaker(state) == debate_first_speaker("VST", "2026-09-21")


def _run_stubbed_graph(monkeypatch, ticker, trade_date):
    """Compile the real GraphSetup wiring with stub nodes and run it, returning
    the order the two researchers actually spoke in."""
    from langchain_core.messages import AIMessage
    from tradingagents.graph import parallel_analysts as parallel_analysts_module
    from tradingagents.graph import setup as graph_setup_module

    spoken = []

    def researcher(label):
        def factory(llm, cache_role_and_resources=True):
            def node(state):
                d = state["investment_debate_state"]
                spoken.append(label)
                return {"investment_debate_state": {
                    **d, "current_response": f"{label} Analyst: x", "count": d["count"] + 1,
                }}
            return node
        return factory

    def risk(name):
        def factory(llm, cache_role_and_resources=True):
            def node(state):
                r = state["risk_debate_state"]
                return {"risk_debate_state": {**r, "latest_speaker": name, "count": r["count"] + 1}}
            return node
        return factory

    noop = lambda *a, **k: (lambda state: {})  # noqa: E731
    stubs = {
        "create_bull_researcher": researcher("Bull"),
        "create_bear_researcher": researcher("Bear"),
        "create_aggressive_debator": risk("Aggressive"),
        "create_conservative_debator": risk("Conservative"),
        "create_neutral_debator": risk("Neutral"),
        "create_market_analyst": lambda llm: (lambda state: {"messages": [AIMessage(content="m")]}),
        "create_research_manager": noop,
        "create_trader": noop,
        "create_portfolio_manager": noop,
    }
    for target, stub in stubs.items():
        monkeypatch.setattr(graph_setup_module, target, stub)
    # The analyst phase's clear/reset step now lives in parallel_analysts.py,
    # not setup.py, since each analyst runs in its own subgraph (TODOS #21).
    monkeypatch.setattr(parallel_analysts_module, "create_msg_delete", lambda: (lambda state: {}))

    workflow = graph_setup_module.GraphSetup(
        quick_thinking_llm="Q", deep_thinking_llm="D",
        tool_nodes={"market": lambda state: state},
        conditional_logic=ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1),
    ).setup_graph(selected_analysts=("market",))
    workflow.compile().invoke({
        "messages": [], "company_of_interest": ticker, "trade_date": trade_date,
        "investment_debate_state": {"current_response": "", "count": 0},
        "risk_debate_state": {"latest_speaker": "", "count": 0},
    })
    return spoken


def _ticker_opening_with(side):
    return next(
        f"T{i}" for i in range(100)
        if debate_first_speaker(f"T{i}", "2026-09-21") == f"{side} Researcher"
    )


@pytest.mark.unit
@pytest.mark.parametrize("side,expected", [("Bull", ["Bull", "Bear"]), ("Bear", ["Bear", "Bull"])])
def test_real_graph_wiring_runs_opener_then_the_other_exactly_once_each(monkeypatch, side, expected):
    assert _run_stubbed_graph(monkeypatch, _ticker_opening_with(side), "2026-09-21") == expected
