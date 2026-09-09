import logging
import re
from datetime import datetime, timezone
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.fred import get_macro_data
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.kalshi_fed_rate import get_fed_rate_probability

logger = logging.getLogger(__name__)

# 2026-09-09, TODOS #114: the configured prediction_markets vendor (Manifold,
# a play-money crowd market) was found to diverge sharply from a real
# published model for recession specifically (8% vs. FRED's NY Fed model's
# 76% for the same period) -- a genuinely misleading number if the analyst
# happened to phrase its query as "recession" without separately thinking to
# call get_macro_indicators instead. Rather than rely on the analyst
# remembering to prefer the better tool (a prompt hint alone, easy to miss),
# these two topics are detected here and the verified source is *always*
# prepended with an explicit "trust this over the numbers below" note --
# a structural guarantee, not just a hint, regardless of which tool the LLM
# reached for or how it phrased the topic.
_RECESSION_RE = re.compile(r"\brecession\b", re.IGNORECASE)
_FED_RATE_RE = re.compile(
    r"\b(fed|fomc|federal reserve)\b.*\b(rate|hike|cut)\b|\brate (cut|hike)\b",
    re.IGNORECASE,
)


def _recession_supplement() -> str | None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        fred_report = get_macro_data("recession_probability", today, look_back_days=180)
    except Exception as e:  # noqa: BLE001 -- this is a best-effort supplement, never let it break the main tool call
        logger.warning("Recession-probability FRED supplement failed: %s", e)
        return None
    return (
        "**Note: a real published recession-probability model exists for this "
        "question and is more reliable than the crowd-market numbers below "
        "(2026-09-09 finding: an 8%-vs-76% divergence was observed between "
        "them for the same period) -- prefer this figure:**\n\n"
        f"{fred_report}\n---\n\n"
    )


def _fed_rate_supplement() -> str | None:
    try:
        kalshi_report = get_fed_rate_probability()
    except Exception as e:  # noqa: BLE001 -- best-effort supplement, never breaks the main call
        logger.warning("Fed-rate Kalshi supplement failed: %s", e)
        return None
    return (
        "**Note: a real regulated-exchange market exists for this exact "
        "question and is more reliable than crowd-market numbers below -- "
        "prefer this figure:**\n\n"
        f"{kalshi_report}\n---\n\n"
    )


@tool
def get_prediction_markets(
    topic: Annotated[
        str,
        "Event topic/keyword, e.g. 'Fed rate cut', 'recession 2026', "
        "'US election', or a sector/company event.",
    ],
    limit: Annotated[int | None, "Max markets to return; omit for a default of 6"] = None,
) -> str:
    """
    Retrieve live, market-implied probabilities for forward-looking events from
    prediction markets: Fed decisions, recession, elections, geopolitics,
    crypto. Returns the most-traded open markets matching the topic, each
    with its implied probability, traded volume, and resolution date. Uses
    the configured prediction_markets vendor -- except recession and Fed
    rate cut/hike questions, which are always supplemented with a real
    verified source (a published recession-probability model; a real
    regulated exchange's rate-decision market) ahead of the vendor's own
    crowd-market numbers.

    Args:
        topic (str): Event keyword(s) to search
        limit (int): Max markets to return; omit for a default of 6

    Returns:
        str: A formatted markdown report of matching prediction markets
    """
    vendor_result = route_to_vendor("get_prediction_markets", topic, limit)

    supplement = None
    if _RECESSION_RE.search(topic):
        supplement = _recession_supplement()
    elif _FED_RATE_RE.search(topic):
        supplement = _fed_rate_supplement()

    if supplement is None:
        return vendor_result
    return supplement + vendor_result
