"""
tests/test_costs.py
===================

Unit tests for the execution-cost model and a regression test proving that
turning costs off reproduces the original cost-free engine exactly.

Run with:
    python -m pytest tests/ -v
"""

import math
import os
import sys

import pandas as pd
from pandas.testing import assert_series_equal

# Allow importing the project modules from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import _simulate_strategy
from costs import commission_cost, slippage_fill_price


def test_commission_deducted_on_known_trade():
    """Commission is 0.05% of trade value: trading $10,000 costs $5.00.

        commission = trade_value * rate = 10_000 * 0.0005 = 5.0
    """
    assert commission_cost(10_000.0, 0.0005) == 5.0

    # And it scales linearly / is side-agnostic (a $50k trade at 10 bps).
    assert math.isclose(commission_cost(50_000.0, 0.0010), 50.0)


def test_slippage_raises_buy_and_lowers_sell():
    """Buys fill above the close, sells fill below, each by slippage_bps.

    At 5 bps on a $100 stock:
        buy  -> 100 * (1 + 0.0005) = 100.05
        sell -> 100 * (1 - 0.0005) =  99.95
    """
    close = 100.0

    buy_fill = slippage_fill_price(close, "buy", 5)
    sell_fill = slippage_fill_price(close, "sell", 5)

    assert math.isclose(buy_fill, 100.05)
    assert math.isclose(sell_fill, 99.95)
    # The defining property: a buy costs more than the close, a sell earns less.
    assert buy_fill > close > sell_fill


def _reference_gross_returns(prices, top_n, lookback):
    """A frozen copy of the ORIGINAL cost-free engine loop.

    This is the behavior the cost-aware engine must still reproduce when costs
    are switched off. Kept here verbatim so the regression test has a fixed
    reference that doesn't move when backtest.py is refactored.
    """
    monthly_returns = prices.pct_change()
    momentum = prices / prices.shift(lookback) - 1.0
    dates = prices.index

    out = {}
    for i in range(lookback, len(dates) - 1):
        scores = momentum.loc[dates[i]].dropna()
        if scores.empty:
            continue
        winners = scores.sort_values(ascending=False).head(top_n).index.tolist()
        next_month = monthly_returns.loc[dates[i + 1], winners].dropna()
        out[dates[i + 1]] = 0.0 if next_month.empty else float(next_month.mean())

    return pd.Series(out).sort_index()


def _synthetic_prices():
    """Deterministic monthly price panel with shifting momentum leaders.

    Different growth paths make the top-N winners change over time, so the
    strategy actually trades — a meaningful case to regression-test.
    """
    index = pd.date_range("2020-01-31", periods=8, freq="ME")
    return pd.DataFrame(
        {
            "AAA": [100, 105, 111, 120, 132, 140, 150, 165],
            "BBB": [100, 98, 101, 99, 104, 110, 118, 121],
            "CCC": [100, 110, 108, 115, 113, 112, 120, 119],
            "DDD": [100, 101, 103, 102, 108, 116, 121, 130],
        },
        index=index,
        dtype=float,
    )


def test_costs_off_matches_original_engine_exactly():
    """Regression test: costs OFF reproduces the original engine bit-for-bit.

    With commission == 0 and slippage == 0, the cost drag each month is exactly
    0.0, so the net returns must equal the reference gross returns exactly.
    """
    prices = _synthetic_prices()
    top_n, lookback = 2, 2

    sim = _simulate_strategy(
        prices, top_n=top_n, lookback=lookback, commission=0.0, slippage_rate=0.0
    )
    reference = _reference_gross_returns(prices, top_n, lookback)

    assert_series_equal(sim["returns"], reference)

    # Sanity: with no costs, nothing is tallied as commission or slippage.
    assert sim["total_commission"] == 0.0
    assert sim["total_slippage"] == 0.0
