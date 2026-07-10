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

Execution costs
---------------
Every rebalance trades stock, and trading isn't free. Two frictions are
modeled (see costs.py):

* **Commission** — a fraction of the value traded, charged on buys and sells.
* **Slippage**   — filling a few basis points worse than the close.

Both scale with **turnover**: how much of the portfolio changes each month.
The costs are charged as a drag on that month's return and also tallied in
dollar terms so we can see exactly how much friction ate.

We chain the monthly net returns into an equity curve and compare against SPY.

Run it
------
    python backtest.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from costs import BPS, drift_weights, rebalance_turnover
from metrics import annualized_return, max_drawdown, sharpe_ratio

# How many months of history the momentum signal looks back over.
MOMENTUM_LOOKBACK_MONTHS = 12
# How many top-ranked stocks we hold each month.
TOP_N = 10
# Monthly strategy => 12 periods per year for annualizing metrics.
PERIODS_PER_YEAR = 12

# Default execution-cost assumptions.
DEFAULT_COMMISSION = 0.0005   # 0.05% of trade value, per buy and per sell
DEFAULT_SLIPPAGE_BPS = 5.0    # fill 5 bps worse than the close on each trade


@dataclass
class BacktestResult:
    """Everything a caller needs to report on or plot a run."""

    equity_curve: pd.Series          # strategy value of $1 over time (month-end)
    benchmark_curve: pd.Series       # SPY value of $1 over the same dates
    monthly_returns: pd.Series       # strategy per-month (net) returns
    benchmark_returns: pd.Series     # SPY per-month returns
    holdings: dict = field(default_factory=dict)  # rebalance date -> [tickers]

    # --- execution-cost accounting (all dollar figures are per $1 invested) ---
    num_trades: int = 0
    total_commission: float = 0.0    # summed commission paid over the run
    total_slippage: float = 0.0      # summed slippage cost paid over the run
    commission_rate: float = 0.0     # the assumption used, for reference
    slippage_bps: float = 0.0        # the assumption used, for reference

    @property
    def stats(self) -> dict:
        """Headline metrics for the strategy and the benchmark."""
        return {
            "strategy": _summary(self.equity_curve, self.monthly_returns),
            "benchmark": _summary(self.benchmark_curve, self.benchmark_returns),
        }

    @property
    def costs(self) -> dict:
        """Cost totals and cost as a fraction of the final portfolio value."""
        final_value = (
            float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else 1.0
        )
        total = self.total_commission + self.total_slippage
        return {
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_cost": total,
            "cost_pct_of_final": total / final_value if final_value else 0.0,
            "num_trades": self.num_trades,
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


def _simulate_strategy(
    prices: pd.DataFrame,
    top_n: int = TOP_N,
    lookback: int = MOMENTUM_LOOKBACK_MONTHS,
    commission: float = 0.0,
    slippage_rate: float = 0.0,
) -> dict:
    """Run the momentum strategy on a price panel and return net returns + costs.

    This is the pure core of the engine: it takes a prices DataFrame (no network
    access) so it can be unit-tested directly. The gross return computation is
    intentionally identical to the original cost-free engine — with
    ``commission == 0`` and ``slippage_rate == 0`` the ``cost_frac`` below is
    exactly 0.0, so the net returns match the original bit-for-bit (this is what
    the regression test pins down).

    ``commission`` and ``slippage_rate`` are both plain fractions of traded
    value (e.g. 0.0005 == 5 bps). They are added together because they apply to
    the same traded value each rebalance.
    """
    monthly_returns = prices.pct_change()
    momentum = compute_momentum(prices, lookback)
    dates = prices.index

    strat_returns: dict = {}
    holdings: dict = {}

    prev_weights: dict[str, float] = {}   # drifted weights we currently hold
    equity = 1.0                          # running portfolio value (starts at $1)
    total_commission = 0.0
    total_slippage = 0.0
    num_trades = 0

    for i in range(lookback, len(dates) - 1):
        rebalance_date = dates[i]
        next_date = dates[i + 1]

        # Momentum scores available at the rebalance date; drop unscored stocks.
        scores = momentum.loc[rebalance_date].dropna()
        if scores.empty:
            continue

        # Pick the top N by momentum and target an equal weight in each.
        winners = scores.sort_values(ascending=False).head(top_n).index.tolist()
        holdings[rebalance_date] = winners
        target_weights = {t: 1.0 / len(winners) for t in winners}

        # How much we must trade to move from what we hold to the new target,
        # and the cost of doing so (a fraction of current portfolio value).
        turnover, trades = rebalance_turnover(prev_weights, target_weights)
        num_trades += trades
        commission_frac = commission * turnover
        slippage_frac = slippage_rate * turnover
        cost_frac = commission_frac + slippage_frac
        total_commission += commission_frac * equity
        total_slippage += slippage_frac * equity

        # Gross next-month return — equal-weighted mean of the held names'
        # realized returns. (Identical to the original engine.)
        next_month_returns = monthly_returns.loc[next_date, winners].dropna()
        if next_month_returns.empty:
            gross = 0.0
        else:
            gross = float(next_month_returns.mean())

        # Net return after paying this month's trading costs up front.
        net = gross - cost_frac
        strat_returns[next_date] = net
        equity *= 1.0 + net

        # Drift the equal-weighted target by this month's returns to get the
        # weights we'll actually hold (and compare against) next rebalance.
        realized = monthly_returns.loc[next_date, winners].fillna(0.0).to_dict()
        prev_weights = drift_weights(target_weights, realized)

    strat_returns = pd.Series(strat_returns).sort_index()
    if strat_returns.empty:
        raise ValueError(
            "No strategy returns produced — the date range is likely shorter "
            "than the momentum lookback."
        )

    return {
        "returns": strat_returns,
        "equity_curve": (1.0 + strat_returns).cumprod(),
        "holdings": holdings,
        "num_trades": num_trades,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
    }


def _benchmark_series(
    benchmark: str, start: str, end: str, index: pd.Index
) -> tuple[pd.Series, pd.Series]:
    """Download the benchmark and align it to the strategy's month index."""
    bench_prices = download_monthly_prices([benchmark], start, end)
    bench_returns = bench_prices[benchmark].pct_change()
    bench_returns = bench_returns.reindex(index).fillna(0.0)
    bench_curve = (1.0 + bench_returns).cumprod()
    return bench_curve, bench_returns


def _build_result(
    sim: dict,
    bench_curve: pd.Series,
    bench_returns: pd.Series,
    commission: float,
    slippage_bps: float,
) -> BacktestResult:
    return BacktestResult(
        equity_curve=sim["equity_curve"],
        benchmark_curve=bench_curve,
        monthly_returns=sim["returns"],
        benchmark_returns=bench_returns,
        holdings=sim["holdings"],
        num_trades=sim["num_trades"],
        total_commission=sim["total_commission"],
        total_slippage=sim["total_slippage"],
        commission_rate=commission,
        slippage_bps=slippage_bps,
    )


def run_backtest(
    tickers: list[str],
    start: str,
    end: str,
    top_n: int = TOP_N,
    lookback: int = MOMENTUM_LOOKBACK_MONTHS,
    benchmark: str = "SPY",
    commission: float = DEFAULT_COMMISSION,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BacktestResult:
    """Run the momentum backtest and return a :class:`BacktestResult`.

    Parameters
    ----------
    tickers : list of S&P 500 tickers to choose from.
    start, end : ISO date strings ("YYYY-MM-DD") bounding the backtest.
    top_n : how many top-momentum stocks to hold each month.
    lookback : momentum lookback window in months.
    benchmark : ticker to compare against (default SPY).
    commission : per-trade commission as a fraction of trade value
        (default 0.0005 == 0.05%). Set to 0 to turn commissions off.
    slippage_bps : slippage per trade in basis points (default 5). Set to 0
        to turn slippage off.
    """
    prices = download_monthly_prices(tickers, start, end)
    if prices.shape[1] == 0:
        raise ValueError("No price data downloaded — check tickers/date range.")

    sim = _simulate_strategy(
        prices,
        top_n=top_n,
        lookback=lookback,
        commission=commission,
        slippage_rate=slippage_bps * BPS,
    )
    bench_curve, bench_returns = _benchmark_series(
        benchmark, start, end, sim["returns"].index
    )
    return _build_result(sim, bench_curve, bench_returns, commission, slippage_bps)


def compare_costs(
    tickers: list[str],
    start: str,
    end: str,
    top_n: int = TOP_N,
    lookback: int = MOMENTUM_LOOKBACK_MONTHS,
    benchmark: str = "SPY",
    commission: float = DEFAULT_COMMISSION,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> tuple[BacktestResult, BacktestResult]:
    """Run the strategy twice — costs off then costs on — on the same data.

    Downloads the price data once and simulates both scenarios so the only
    difference between the two results is the execution-cost assumption.

    Returns ``(result_costs_off, result_costs_on)``.
    """
    prices = download_monthly_prices(tickers, start, end)
    if prices.shape[1] == 0:
        raise ValueError("No price data downloaded — check tickers/date range.")

    sim_off = _simulate_strategy(
        prices, top_n=top_n, lookback=lookback, commission=0.0, slippage_rate=0.0
    )
    sim_on = _simulate_strategy(
        prices,
        top_n=top_n,
        lookback=lookback,
        commission=commission,
        slippage_rate=slippage_bps * BPS,
    )

    bench_curve, bench_returns = _benchmark_series(
        benchmark, start, end, sim_off["returns"].index
    )

    result_off = _build_result(sim_off, bench_curve, bench_returns, 0.0, 0.0)
    result_on = _build_result(
        sim_on, bench_curve, bench_returns, commission, slippage_bps
    )
    return result_off, result_on


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


def comparison_table(
    result_off: BacktestResult, result_on: BacktestResult
) -> pd.DataFrame:
    """Build a side-by-side costs-off vs costs-on metrics table."""
    off = result_off.stats["strategy"]
    on = result_on.stats["strategy"]

    return pd.DataFrame(
        {
            "Costs OFF": {
                "Total return": f"{off['total_return']:.2%}",
                "CAGR": f"{off['annualized_return']:.2%}",
                "Sharpe ratio": f"{off['sharpe_ratio']:.2f}",
                "Max drawdown": f"{off['max_drawdown']:.2%}",
                "Number of trades": f"{result_off.num_trades:d}",
            },
            "Costs ON": {
                "Total return": f"{on['total_return']:.2%}",
                "CAGR": f"{on['annualized_return']:.2%}",
                "Sharpe ratio": f"{on['sharpe_ratio']:.2f}",
                "Max drawdown": f"{on['max_drawdown']:.2%}",
                "Number of trades": f"{result_on.num_trades:d}",
            },
        }
    )


def print_comparison(result_off: BacktestResult, result_on: BacktestResult) -> None:
    """Print the costs-off vs costs-on comparison and the cost breakdown."""
    print("\n=== Costs OFF vs Costs ON ===")
    print(comparison_table(result_off, result_on).to_string())

    c = result_on.costs
    print(
        f"\nExecution costs (per $1 invested, "
        f"{result_on.commission_rate:.4%} commission + "
        f"{result_on.slippage_bps:.0f} bps slippage):"
    )
    print(f"  Total commissions : {c['total_commission']:.4f}")
    print(f"  Total slippage    : {c['total_slippage']:.4f}")
    print(
        f"  Combined cost     : {c['total_cost']:.4f} "
        f"({c['cost_pct_of_final']:.2%} of final portfolio value)"
    )


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
    print("Simulating twice: once with costs off, once with costs on "
          f"({DEFAULT_COMMISSION:.4%} commission + {DEFAULT_SLIPPAGE_BPS:.0f} bps slippage).")

    result_off, result_on = compare_costs(DEFAULT_UNIVERSE, start, end)
    print_comparison(result_off, result_on)

    # Draw both equity curves against SPY and save to results.html.
    try:
        from plot import plot_cost_comparison

        out = plot_cost_comparison(result_off, result_on, filename="results.html")
        print(f"\nSaved interactive chart to {out}")
    except Exception as exc:  # plotting is optional; never fail the backtest for it
        print(f"\n(Could not render chart: {exc})")


if __name__ == "__main__":
    main()
