"""
backtest.py
===========

A from-scratch backtesting engine for a monthly cross-sectional momentum
strategy on S&P 500 stocks. No Backtrader / Zipline / etc. — the portfolio
construction and return accounting are written out explicitly so every step
can be explained in an interview.

The strategy
------------
1. Work on a monthly calendar (month-end prices).
2. On the last trading day of each month (a "rebalance date"):
     * For every stock, compute its trailing 12-month total return
       (price_now / price_12_months_ago - 1). This is the momentum signal.
     * Rank all stocks by that signal and select the top N (default 10).
3. Hold that equal-weighted basket for the next month.
4. Repeat: at the next month-end, recompute momentum and rebalance.

The return we earn in a month comes from the basket we *chose at the start of
that month* — so there is no look-ahead bias: the signal at the end of month M
determines the holdings that earn month M+1's return.

We then chain those monthly portfolio returns into an equity curve and compare
it against SPY (buy-and-hold the S&P 500 ETF) over the same window.

Run it
------
    python backtest.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from metrics import annualized_return, max_drawdown, sharpe_ratio

# How many months of history the momentum signal looks back over.
MOMENTUM_LOOKBACK_MONTHS = 12
# How many top-ranked stocks we hold each month.
TOP_N = 10
# Monthly strategy => 12 periods per year for annualizing metrics.
PERIODS_PER_YEAR = 12


@dataclass
class BacktestResult:
    """Everything a caller needs to report on or plot a run."""

    equity_curve: pd.Series          # strategy value of $1 over time (month-end)
    benchmark_curve: pd.Series       # SPY value of $1 over the same dates
    monthly_returns: pd.Series       # strategy per-month returns
    benchmark_returns: pd.Series     # SPY per-month returns
    holdings: dict = field(default_factory=dict)  # rebalance date -> [tickers]

    @property
    def stats(self) -> dict:
        """Headline metrics for the strategy and the benchmark."""
        return {
            "strategy": _summary(self.equity_curve, self.monthly_returns),
            "benchmark": _summary(self.benchmark_curve, self.benchmark_returns),
        }


def _summary(equity_curve: pd.Series, returns: pd.Series) -> dict:
    return {
        "annualized_return": annualized_return(equity_curve, PERIODS_PER_YEAR),
        "sharpe_ratio": sharpe_ratio(returns, PERIODS_PER_YEAR),
        "max_drawdown": max_drawdown(equity_curve),
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        if len(equity_curve) > 1
        else 0.0,
    }


def download_monthly_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data via yfinance and reduce it to month-end close prices.

    Returns a DataFrame indexed by month-end date, one column per ticker,
    containing adjusted close prices. Columns whose data is entirely missing
    (e.g. a bad ticker) are dropped.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjust for splits/dividends so returns are total returns
        progress=False,
    )

    # With multiple tickers yfinance returns a column MultiIndex like
    # ("Close", "AAPL"); with a single ticker it returns flat columns.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    # Resample to month-end and take the last available close in each month.
    monthly = close.resample("ME").last()

    # Drop tickers with no usable data at all.
    monthly = monthly.dropna(axis=1, how="all")
    return monthly


def compute_momentum(prices: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK_MONTHS) -> pd.DataFrame:
    """Trailing total return over ``lookback`` months for every stock.

    momentum[t] = price[t] / price[t - lookback] - 1

    The first ``lookback`` rows are NaN because there isn't enough history yet.
    """
    return prices / prices.shift(lookback) - 1.0


def run_backtest(
    tickers: list[str],
    start: str,
    end: str,
    top_n: int = TOP_N,
    lookback: int = MOMENTUM_LOOKBACK_MONTHS,
    benchmark: str = "SPY",
) -> BacktestResult:
    """Run the momentum backtest and return a :class:`BacktestResult`.

    Parameters
    ----------
    tickers : list of S&P 500 tickers to choose from.
    start, end : ISO date strings ("YYYY-MM-DD") bounding the backtest.
    top_n : how many top-momentum stocks to hold each month.
    lookback : momentum lookback window in months.
    benchmark : ticker to compare against (default SPY).
    """
    prices = download_monthly_prices(tickers, start, end)
    if prices.shape[1] == 0:
        raise ValueError("No price data downloaded — check tickers/date range.")

    # Monthly simple returns for every stock: what you earn holding it for the
    # month ending at that row's date.
    monthly_returns = prices.pct_change()

    # Momentum signal known *as of* each month-end.
    momentum = compute_momentum(prices, lookback)

    # --- The core loop -----------------------------------------------------
    # For each rebalance date t (where a full lookback exists), rank stocks by
    # momentum and hold the top N. That basket earns the return realized over
    # the *next* month (t+1), so we look the return up one row forward.
    dates = prices.index
    strat_returns = {}
    holdings: dict = {}

    for i in range(lookback, len(dates) - 1):
        rebalance_date = dates[i]
        next_date = dates[i + 1]

        # Momentum scores available at the rebalance date; drop stocks we
        # can't score yet.
        scores = momentum.loc[rebalance_date].dropna()
        if scores.empty:
            continue

        # Pick the top N by momentum (highest trailing return).
        winners = scores.sort_values(ascending=False).head(top_n).index.tolist()
        holdings[rebalance_date] = winners

        # Next month's realized return for each held name; equal-weighted.
        next_month_returns = monthly_returns.loc[next_date, winners].dropna()
        if next_month_returns.empty:
            strat_returns[next_date] = 0.0
        else:
            strat_returns[next_date] = float(next_month_returns.mean())

    strat_returns = pd.Series(strat_returns).sort_index()
    if strat_returns.empty:
        raise ValueError(
            "No strategy returns produced — the date range is likely shorter "
            "than the momentum lookback."
        )

    # Chain monthly returns into an equity curve for $1 of starting capital.
    equity_curve = (1.0 + strat_returns).cumprod()

    # --- Benchmark ---------------------------------------------------------
    bench_prices = download_monthly_prices([benchmark], start, end)
    bench_returns = bench_prices[benchmark].pct_change()
    # Align the benchmark to the same months the strategy was invested in.
    bench_returns = bench_returns.reindex(strat_returns.index).fillna(0.0)
    bench_curve = (1.0 + bench_returns).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=bench_curve,
        monthly_returns=strat_returns,
        benchmark_returns=bench_returns,
        holdings=holdings,
    )


def print_report(result: BacktestResult) -> None:
    """Pretty-print the headline stats and the month-by-month equity curve."""
    stats = result.stats

    def fmt(section: dict) -> str:
        return (
            f"  Annualized return : {section['annualized_return']:>8.2%}\n"
            f"  Total return      : {section['total_return']:>8.2%}\n"
            f"  Sharpe ratio      : {section['sharpe_ratio']:>8.2f}\n"
            f"  Max drawdown      : {section['max_drawdown']:>8.2%}"
        )

    print("\n=== Momentum strategy ===")
    print(fmt(stats["strategy"]))
    print("\n=== SPY benchmark ===")
    print(fmt(stats["benchmark"]))

    print("\n=== Month-by-month equity curve (strategy vs SPY) ===")
    table = pd.DataFrame(
        {
            "strategy": result.equity_curve,
            "SPY": result.benchmark_curve,
        }
    )
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        print(table.to_string())


# A small, liquid slice of the S&P 500 used as the default universe when the
# file is run directly. Pass your own list to run_backtest() for a full run.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "DIS",
    "CSCO", "ADBE", "CRM", "NFLX", "INTC", "AMD", "QCOM", "TXN", "ORCL", "IBM",
]


def main() -> None:
    start = "2015-01-01"
    end = "2024-12-31"

    print(f"Running momentum backtest on {len(DEFAULT_UNIVERSE)} tickers "
          f"from {start} to {end} ...")
    result = run_backtest(DEFAULT_UNIVERSE, start, end)
    print_report(result)

    # Draw the equity-curve comparison and save it to results.html.
    try:
        from plot import plot_equity_curve

        out = plot_equity_curve(result, filename="results.html")
        print(f"\nSaved interactive chart to {out}")
    except Exception as exc:  # plotting is optional; never fail the backtest for it
        print(f"\n(Could not render chart: {exc})")


if __name__ == "__main__":
    main()
