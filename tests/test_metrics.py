"""
tests/test_metrics.py
=====================

Unit tests for the three metric functions, each using hand-picked inputs whose
correct output can be worked out on paper. Run with:

    python -m pytest tests/ -v
"""

import math
import os
import sys

import pandas as pd

# Allow importing metrics.py from the project root when tests are run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import annualized_return, max_drawdown, sharpe_ratio


def test_annualized_return_known_doubling():
    """$1 -> $1.21 over exactly 24 monthly steps (2 years) is 10%/yr.

    (1.21 / 1.00) ** (1 / 2) - 1 = 0.10
    We build a 25-point curve (24 steps => 2 years at 12 periods/yr) that
    begins at 1.00 and ends at 1.21.
    """
    n_points = 25  # 24 steps == 2 years
    equity = pd.Series([1.0 + (1.21 - 1.0) * i / (n_points - 1) for i in range(n_points)])

    result = annualized_return(equity, periods_per_year=12)

    assert math.isclose(result, 0.10, abs_tol=1e-9)


def test_sharpe_ratio_constant_returns():
    """A constant monthly return of 1% has zero volatility.

    With std == 0, the Sharpe ratio is undefined/infinite, and our function is
    defined to return 0.0 in that degenerate case.
    """
    returns = pd.Series([0.01] * 12)

    assert sharpe_ratio(returns, periods_per_year=12) == 0.0


def test_sharpe_ratio_known_value():
    """Alternating returns give a Sharpe we can compute by hand.

    returns = [+2%, -1%, +2%, -1%, ...] (12 values, six of each).
      mean = (0.02 + -0.01) / 2 = 0.005
      sample std (ddof=1): each point deviates +/-0.015 from the mean, and
        std = sqrt( sum((x-mean)^2) / (n-1) )
            = sqrt( 12 * 0.015^2 / 11 )
      per-period sharpe = 0.005 / std
      annualized       = per-period sharpe * sqrt(12)
    """
    returns = pd.Series([0.02, -0.01] * 6)

    mean = 0.005
    std = math.sqrt(12 * (0.015 ** 2) / 11)
    expected = (mean / std) * math.sqrt(12)

    result = sharpe_ratio(returns, periods_per_year=12, risk_free_rate=0.0)

    assert math.isclose(result, expected, rel_tol=1e-9)


def test_max_drawdown_known_dip():
    """Peak at 120, trough at 90 gives a 25% drawdown.

    Curve: 100 -> 120 -> 90 -> 110
      running peak at the trough is 120
      drawdown = 90 / 120 - 1 = -0.25
    The later recovery to 110 does not undo the worst historical dip.
    """
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])

    result = max_drawdown(equity)

    assert math.isclose(result, -0.25, abs_tol=1e-12)
