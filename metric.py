"""
metrics.py
==========

Performance metrics for a backtested strategy.

Each function takes plain pandas/numpy inputs and returns a single number so
that the math is easy to read, test, and explain out loud in an interview.

Two shapes of input show up here:

* an **equity curve**: the value of $1 invested in the strategy over time, e.g.
  ``[1.00, 1.02, 1.05, 1.03, ...]`` (one point per period).
* a series of **periodic returns**: the fractional change each period, e.g.
  ``[0.02, 0.03, -0.019, ...]``. If equity[t] = equity[t-1] * (1 + r[t]),
  then r is the return series of that equity curve.

``periods_per_year`` describes how many observations make up a year. For the
monthly momentum strategy in this project that is 12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 12) -> float:
    """Compound Annual Growth Rate (CAGR) implied by an equity curve.

    Formula:
        CAGR = (ending_value / starting_value) ** (1 / years) - 1
    where
        years = number_of_periods / periods_per_year

    Intuition: what constant yearly growth rate, compounded, would take you
    from the first value to the last value over the same amount of time?

    Example: growing 1.00 -> 1.21 over 24 monthly periods (2 years) gives
        (1.21 / 1.00) ** (1 / 2) - 1 = 0.10  (10% per year)
    """
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) < 2:
        return 0.0

    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]

    # Number of *periods* elapsed is one less than the number of data points
    # (n points => n-1 steps between them), converted into a count of years.
    n_periods = len(equity_curve) - 1
    years = n_periods / periods_per_year
    if years <= 0 or start <= 0:
        return 0.0

    return (end / start) ** (1.0 / years) - 1.0


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio of a series of *periodic* returns.

    Formula:
        excess      = returns - (risk_free_rate / periods_per_year)
        sharpe      = mean(excess) / std(excess)          # per-period Sharpe
        annualized  = sharpe * sqrt(periods_per_year)     # scale up to a year

    Intuition: reward-to-risk. How much return did you earn per unit of
    volatility, above the risk-free rate? Higher is better; a value around 1
    is decent, 2+ is strong.

    Notes:
    * ``risk_free_rate`` is an *annual* rate, so we divide it by
      ``periods_per_year`` to get the per-period hurdle.
    * We annualize by sqrt(periods_per_year) because returns compound
      (mean) linearly with time but volatility (std) grows with the square
      root of time.
    * We use the sample standard deviation (ddof=1), the usual convention.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0

    per_period_rf = risk_free_rate / periods_per_year
    excess = returns - per_period_rf

    std = excess.std(ddof=1)
    # Treat an effectively-flat return series as zero volatility. Floating
    # point leaves a tiny non-zero std for constant inputs, so compare the
    # std to the scale of the returns rather than to exactly 0.
    scale = max(abs(excess.mean()), 1.0)
    if std <= 1e-12 * scale:
        return 0.0

    per_period_sharpe = excess.mean() / std
    return per_period_sharpe * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown of an equity curve (a negative number, or 0.0).

    Formula:
        running_peak = cumulative maximum of the equity curve
        drawdown[t]  = equity[t] / running_peak[t] - 1     # <= 0 at each point
        max_drawdown = min(drawdown)                       # the worst dip

    Intuition: the largest peak-to-trough percentage loss you would have
    lived through if you had held the strategy the whole time. A max drawdown
    of -0.30 means the portfolio fell 30% from a previous high at its worst.

    Returned as a negative fraction (e.g. -0.30). Returns 0.0 for a curve that
    only ever goes up.
    """
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) < 2:
        return 0.0

    # Highest value seen up to and including each point in time.
    running_peak = equity_curve.cummax()

    # How far below the running peak we are at each point (0 at new highs,
    # negative when underwater).
    drawdown = equity_curve / running_peak - 1.0

    return float(drawdown.min())
