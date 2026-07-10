# Momentum Backtesting Engine

A from-scratch backtesting engine for a monthly **cross-sectional momentum**
strategy on S&P 500 stocks, written in plain Python + pandas. No Backtrader,
Zipline, or other backtesting library — the portfolio construction and return
accounting are all hand-written so every step can be explained in an interview.

## What it does

1. Downloads historical daily OHLCV data for a list of tickers with `yfinance`.
2. Reduces it to month-end prices.
3. Each month, ranks stocks by trailing 12-month return and buys the top 10.
4. Holds that equal-weighted basket for one month, then rebalances.
5. Chains the monthly returns into an equity curve and reports the headline
   metrics.
6. Compares the strategy against **SPY** (buy-and-hold the S&P 500) and renders
   an interactive Plotly chart to `results.html`.

## How the momentum strategy works

Momentum is the empirical tendency of recent winners to keep outperforming
recent losers over the following month. The signal here is the simplest version
of it:

```
momentum(stock, t) = price(t) / price(t - 12 months) - 1
```

The loop, run once per month-end `t`:

1. Compute each stock's trailing 12-month return (the momentum signal).
2. Rank all stocks and select the **top 10**.
3. Hold them **equal-weighted** for the next month.
4. At `t+1`, throw the basket away and repeat from step 1.

**No look-ahead bias:** the signal measured at the *end* of month `M` chooses
the basket that earns month `M+1`'s return. We never use information from a
month to trade within that same month.

The first 12 months of any run produce no trades — there isn't enough history
to compute the lookback yet.

## What each metric means

All metrics live in `metrics.py`, one clearly-commented function each.

- **Annualized return (CAGR)** — the constant yearly growth rate that, compounded,
  turns the starting equity into the ending equity over the same span.
  `(end / start) ** (1 / years) - 1`. Answers "how fast did money grow per year?"

- **Sharpe ratio** — reward per unit of risk: the average excess return divided
  by its volatility, scaled to a year by `√12`. Higher is better; ~1 is decent,
  2+ is strong. Answers "was the return worth the bumpiness?"

- **Max drawdown** — the largest peak-to-trough drop the strategy ever suffered,
  as a negative percentage. `min(equity / running_peak - 1)`. Answers "what's the
  worst loss I'd have had to sit through?"

## Modeling execution costs

A backtest that assumes you trade for free will always look better than reality.
This engine models two real frictions (`costs.py`), both configurable arguments
to `run_backtest()`:

- **Commission** — a fraction of the value traded, charged on every buy and sell
  (default `0.05%` of trade value).
- **Slippage** — you don't fill at the close you see; buys fill a few basis
  points higher and sells a few lower (default `5 bps`), because your order has
  to cross the spread.

Both scale with **turnover** — how much of the portfolio changes each month.
Each rebalance, the cost is charged as a drag on that month's return and also
tallied in dollars so you can see the total friction paid.

## Why backtests lie

The same momentum strategy, run with costs off vs. costs on (0.05% commission +
5 bps slippage), 30-stock universe, 2015–2024:

| Metric | Costs OFF | Costs ON |
|--------|-----------|----------|
| Total return | 769.64% | 737.57% |
| CAGR | 27.74% | 27.20% |
| Sharpe ratio | 1.27 | 1.24 |
| Max drawdown | -28.41% | -28.72% |
| Number of trades | 1,244 | 1,244 |

Combined execution cost over the run: **1.73% of final portfolio value**, split
evenly between commissions and slippage.

**What it means:** the frictionless backtest overstates the strategy — ~32
percentage points of total return and a Sharpe of 1.27 that's really 1.24
evaporate once you pay to trade. The effect is modest *here* because this
universe is large, liquid, and cheap to trade; on a higher-turnover strategy,
smaller/less-liquid stocks, or a real taxable account, the same accounting would
bite far harder. The lesson isn't "costs are small" — it's that **any headline
backtest number should be quoted after costs, or it isn't real.**

## Files

| File | Purpose |
|------|---------|

| File | Purpose |
|------|---------|
| `backtest.py` | Core engine: data download, the momentum loop, cost accounting, benchmark, reporting. |
| `metrics.py`  | Standalone metric functions (Sharpe, max drawdown, annualized return). |
| `costs.py`    | Execution-cost primitives: commission, slippage, turnover, weight drift. |
| `plot.py`     | Renders the costs-off vs costs-on vs SPY equity curves to `results.html`. |
| `tests/test_metrics.py` | Unit tests for the metric functions using hand-checked inputs. |
| `tests/test_costs.py`   | Unit tests for the cost model + a regression test pinning costs-off to the original engine. |

## How to run it

Install dependencies:

```bash
pip install yfinance pandas numpy plotly pytest
```

Run the default backtest (a 30-stock universe, 2015–2024) and generate the chart:

```bash
python backtest.py
```

This prints the strategy vs SPY stats and a month-by-month equity curve, and
writes `results.html` — open it in any browser for the interactive chart.

Run the tests:

```bash
python -m pytest tests/ -v
```

### Using your own universe and dates

```python
from backtest import run_backtest, print_report
from plot import plot_equity_curve

result = run_backtest(
    tickers=["AAPL", "MSFT", "NVDA", "JPM", "XOM", ...],  # S&P 500 tickers
    start="2015-01-01",
    end="2024-12-31",
    top_n=10,        # how many stocks to hold each month
    lookback=12,     # momentum lookback in months
)

print_report(result)
plot_equity_curve(result, "results.html")
print(result.stats)  # dict of strategy + benchmark metrics
```

## Caveats (things an interviewer might poke at)

- **Survivorship bias:** the universe is today's tickers, so companies that were
  delisted are absent. This flatters historical returns.
- **No transaction costs / slippage:** rebalancing every month isn't free in
  reality.
- **Equal weighting** and a fixed top-10 are deliberately simple choices, not
  optimized.
- Returns use adjusted close (`auto_adjust=True`), so dividends and splits are
  already baked in as total returns.
