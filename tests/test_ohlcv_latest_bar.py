"""The latest trading day's bar must not silently vanish (#1201).

yfinance can return the newest in-range bar with a NaN close (an unsettled or
glitched session). The old path parsed dates without normalizing timezone and
dropped every NaN-close row before applying the curr_date cutoff, so the latest
bar disappeared and the previous trading day looked like the latest. Now dates
are normalized, and a latest in-range bar with no close raises rather than
silently falling back.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows import stockstats_utils as su
from tradingagents.dataflows.symbol_utils import NoMarketDataError

# --- date normalization -----------------------------------------------------


@pytest.mark.unit
def test_normalize_dates_strips_tz_and_normalizes_to_midnight():
    aware = pd.Series(pd.to_datetime(
        ["2026-05-08 09:30:00-04:00", "2026-05-09 16:00:00-04:00"]
    ))
    out = su._normalize_dates(aware)
    assert out.dt.tz is None
    assert list(out) == [pd.Timestamp("2026-05-08"), pd.Timestamp("2026-05-09")]


@pytest.mark.unit
def test_normalize_dates_leaves_naive_dates_at_midnight():
    naive = pd.Series(pd.to_datetime(["2026-05-08 14:30:00", "2026-05-09 00:00:00"]))
    out = su._normalize_dates(naive)
    assert out.dt.tz is None
    assert list(out) == [pd.Timestamp("2026-05-08"), pd.Timestamp("2026-05-09")]


@pytest.mark.unit
def test_normalize_dates_handles_mixed_dst_offsets():
    # 5y of US bars span DST; via a cache CSV they arrive as mixed-offset
    # strings, which pd.to_datetime can't unify. Each keeps its own local date.
    mixed = pd.Series([
        "2026-01-08 00:00:00-05:00",  # EST
        "2026-06-08 00:00:00-04:00",  # EDT
        "not-a-date",                 # -> NaT
    ])
    out = su._normalize_dates(mixed)
    assert out.iloc[0] == pd.Timestamp("2026-01-08")
    assert out.iloc[1] == pd.Timestamp("2026-06-08")
    assert pd.isna(out.iloc[2])


@pytest.mark.unit
def test_normalize_dates_keeps_positive_offset_local_date():
    # A bar at local midnight (+09:00) must stay on its own calendar day, not
    # shift to the previous UTC day (which utc=True parsing would cause).
    jst = pd.Series(["2026-05-08 00:00:00+09:00"])
    assert su._normalize_dates(jst).iloc[0] == pd.Timestamp("2026-05-08")


# --- fill vs guard responsibilities -----------------------------------------


@pytest.mark.unit
def test_clean_dataframe_keeps_nan_close_for_the_caller_to_inspect():
    # _clean_dataframe normalizes but no longer drops the NaN close itself.
    df = pd.DataFrame({"Date": ["2026-05-08", "2026-05-09"], "Close": [100.0, float("nan")]})
    cleaned = su._clean_dataframe(df)
    assert len(cleaned) == 2
    assert pd.isna(cleaned["Close"].iloc[-1])


@pytest.mark.unit
def test_fill_price_gaps_drops_nan_close_rows():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-05-07", "2026-05-08"]),
                       "Close": [float("nan"), 100.0]})
    filled = su._fill_price_gaps(df)
    assert len(filled) == 1
    assert filled["Close"].iloc[0] == 100.0


# --- end to end via load_ohlcv ----------------------------------------------


@pytest.mark.unit
def test_load_ohlcv_raises_when_latest_in_range_bar_has_no_close(tmp_path, monkeypatch):
    """A historical request (cache served directly, no refetch) whose latest
    in-range row has a NaN close must raise, not silently serve the prior day."""
    monkeypatch.setattr(su, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    today = pd.Timestamp("2026-07-20")
    monkeypatch.setattr(su.pd.Timestamp, "today", staticmethod(lambda: today))

    start = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_file = tmp_path / f"AAPL-YFin-data-{start}-{end}.csv"
    pd.DataFrame({
        "Date": ["2026-07-17", "2026-07-18"],
        "Close": [100.0, float("nan")],
    }).to_csv(cache_file, index=False)

    with pytest.raises(NoMarketDataError, match="no closing price"):
        su.load_ohlcv("AAPL", "2026-07-18")


@pytest.mark.unit
def test_load_ohlcv_drops_a_nan_close_before_the_latest_row_not_raises(tmp_path, monkeypatch):
    # A NaN close on an OLDER in-range row (not the latest) is dropped, same as
    # before #1201 -- only the latest row's NaN close gets the raise treatment.
    monkeypatch.setattr(su, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    today = pd.Timestamp("2026-07-20")
    monkeypatch.setattr(su.pd.Timestamp, "today", staticmethod(lambda: today))

    start = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_file = tmp_path / f"AAPL-YFin-data-{start}-{end}.csv"
    pd.DataFrame({
        "Date": ["2026-07-16", "2026-07-17", "2026-07-18"],
        "Close": [100.0, float("nan"), 102.0],
    }).to_csv(cache_file, index=False)

    out = su.load_ohlcv("AAPL", "2026-07-18")
    assert len(out) == 2
    assert list(out["Close"]) == [100.0, 102.0]
