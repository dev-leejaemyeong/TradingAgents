"""Runs the analyst phase's four independent analysts concurrently instead of
chaining them one after another (TODOS #21).

Each analyst (market/social/news/fundamentals) reads only the run's shared
task context (ticker, date, instrument identity) and writes its own report
key -- it never reads another analyst's report (confirmed by reading every
analyst node function, 2026-10-12). The one thing they used to share, the
graph's ``messages`` scratch-pad, existed only so each analyst's own
tool-calling loop had a private turn before ``create_msg_delete()`` wiped it
for the next analyst -- under concurrent execution that wipe would race with
another analyst still mid-loop (``delete_messages()`` in agent_utils.py
removes *every* message by id, including ones a concurrent analyst is still
accumulating). So each analyst instead gets its own tiny, separately
compiled subgraph with its own private ``messages``, run in its own thread;
only the resulting report string is merged back into the shared graph state.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import create_msg_delete

from .analyst_execution import AnalystExecutionPlan, AnalystNodeSpec


def _build_analyst_subgraph(spec: AnalystNodeSpec, agent_node: Callable, tool_node: ToolNode, should_continue: Callable):
    """Compile one analyst's own agent/tool-call loop as an isolated subgraph.

    Identical in shape to that analyst's slice of the old chained graph
    (agent_node <-> tool_node until it stops calling tools, then clear_node)
    -- only the surrounding wiring changes, never the analyst's own behavior.
    """
    builder = StateGraph(AgentState)
    builder.add_node(spec.agent_node, agent_node)
    builder.add_node(spec.tool_node, tool_node)
    builder.add_node(spec.clear_node, create_msg_delete())
    builder.add_edge(START, spec.agent_node)
    builder.add_conditional_edges(
        spec.agent_node, should_continue, [spec.tool_node, spec.clear_node],
    )
    builder.add_edge(spec.tool_node, spec.agent_node)
    builder.add_edge(spec.clear_node, END)
    return builder.compile()


def build_analyst_team_node(
    plan: AnalystExecutionPlan,
    analyst_factories: dict[str, Callable[[], Callable]],
    tool_nodes: dict[str, ToolNode],
    conditional_logic: Any,
) -> Callable[..., dict]:
    """Return a single node function that runs every selected analyst's
    subgraph concurrently and merges their reports into one state update.

    This replaces ``plan.specs``' individually chained nodes with one
    "Analyst Team" node in the outer graph. The analysts' subgraphs never
    read each other's state (see module docstring), so this node is the only
    synchronization point: it blocks until every one has produced its report.
    """
    subgraphs = {
        spec.key: _build_analyst_subgraph(
            spec,
            analyst_factories[spec.key](),
            tool_nodes[spec.key],
            getattr(conditional_logic, f"should_continue_{spec.key}"),
        )
        for spec in plan.specs
    }
    clear_node = create_msg_delete()

    def analyst_team_node(state: dict, config: RunnableConfig | None = None) -> dict:
        def run_one(spec: AnalystNodeSpec) -> dict:
            # A fresh dict per thread: each subgraph gets its own private
            # messages/report keys to write into, never the caller's object.
            return subgraphs[spec.key].invoke(dict(state), config=config)

        with ThreadPoolExecutor(max_workers=len(plan.specs)) as pool:
            results = dict(zip(
                (spec.key for spec in plan.specs),
                pool.map(run_one, plan.specs),
            ))

        # .get(..., "") rather than a direct subscript: an analyst node that
        # never wrote its report key (same tolerance the old chained design
        # had -- a TypedDict key no node ever set simply stays absent from
        # state) must not turn into a KeyError here.
        update = {
            spec.report_key: results[spec.key].get(spec.report_key, "")
            for spec in plan.specs
        }
        # Leave the shared messages scratch-pad anchored the same way
        # create_msg_delete() left it after the last analyst of the old
        # chained design -- nothing downstream reads an analyst's own
        # tool-calling messages, only its report (see module docstring).
        update.update(clear_node(state))
        return update

    return analyst_team_node
