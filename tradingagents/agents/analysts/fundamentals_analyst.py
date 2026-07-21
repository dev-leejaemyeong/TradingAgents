from langchain_core.messages import SystemMessage

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.prompt_caching import cached_blocks


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + get_language_instruction()
        )

        tool_names = ", ".join(tool.name for tool in tools)
        # See prompt_caching.py: stable content first so a cache hit is possible
        # at all. role/tool boilerplate + this node's instructions are cached
        # across every ticker in today's batch; the date is cached across that
        # same batch; instrument_context is unique per ticker and stays last/uncached.
        role_and_instructions = (
            "You are a helpful AI assistant, collaborating with other assistants."
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK; another assistant with different tools"
            " will help where you left off. Execute what you can to make progress."
            " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
            " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
            f" You have access to the following tools: {tool_names}.\n"
            + system_message
        )
        run_date = (
            f" Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges.\n"
        )
        per_ticker = f" {instrument_context}\n"

        content = cached_blocks(
            llm,
            (role_and_instructions, True),
            (run_date, True),
            (per_ticker, False),
        )
        messages = [SystemMessage(content=content), *state["messages"]]

        result = llm.bind_tools(tools).invoke(messages)

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
