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

## Files

| File | Purpose |
|------|---------|
| `backtest.py` | Core engine: data download, the momentum loop, benchmark, reporting. |
| `metrics.py`  | Standalone metric functions (Sharpe, max drawdown, annualized return). |
| `plot.py`     | Renders the strategy-vs-SPY equity curve to `results.html`. |
| `tests/test_metrics.py` | Unit tests for the metric functions using hand-checked inputs. |

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
