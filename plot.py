"""
plot.py
=======

Render the strategy equity curve against the SPY benchmark as an interactive
Plotly chart and save it to an HTML file you can open in any browser.
"""

from __future__ import annotations

import plotly.graph_objects as go

from backtest import BacktestResult


def plot_equity_curve(result: BacktestResult, filename: str = "results.html") -> str:
    """Build and save the equity-curve comparison chart.

    Both curves start at $1 so their shapes are directly comparable: the line
    that ends higher earned more per dollar invested over the window.

    Returns the path the HTML file was written to.
    """
    strat = result.equity_curve
    bench = result.benchmark_curve

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=strat.index,
            y=strat.values,
            mode="lines",
            name="Momentum strategy",
            line=dict(color="#2563eb", width=2),
            hovertemplate="%{x|%b %Y}<br>$%{y:.2f}<extra>Momentum</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bench.index,
            y=bench.values,
            mode="lines",
            name="SPY (S&P 500)",
            line=dict(color="#6b7280", width=2, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>$%{y:.2f}<extra>SPY</extra>",
        )
    )

    stats = result.stats["strategy"]
    subtitle = (
        f"Annualized {stats['annualized_return']:.1%} | "
        f"Sharpe {stats['sharpe_ratio']:.2f} | "
        f"Max DD {stats['max_drawdown']:.1%}"
    )

    fig.update_layout(
        title=dict(
            text=f"Momentum Strategy vs SPY — Growth of $1<br>"
            f"<sup>{subtitle}</sup>",
            x=0.5,
        ),
        xaxis_title="Date",
        yaxis_title="Value of $1 invested",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    # Embed the full plotly.js library in the HTML (rather than linking a CDN)
    # so the chart renders even with no internet / a blocked CDN. Makes the
    # file larger (~3-4 MB) but fully self-contained and portable.
    fig.write_html(filename, include_plotlyjs=True)
    return filename


def plot_cost_comparison(
    result_off,
    result_on,
    filename: str = "results.html",
) -> str:
    """Plot costs-off vs costs-on strategy curves against SPY on one chart.

    Three lines, all starting at $1 so they're directly comparable:
    * the strategy with no execution costs (the flattering backtest),
    * the same strategy after commissions + slippage (what you'd really get),
    * SPY as the market benchmark.

    The gap between the two strategy lines *is* the cost of trading. Returns the
    path the HTML file was written to.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result_off.equity_curve.index,
            y=result_off.equity_curve.values,
            mode="lines",
            name="Momentum (no costs)",
            line=dict(color="#93c5fd", width=2, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>$%{y:.2f}<extra>No costs</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result_on.equity_curve.index,
            y=result_on.equity_curve.values,
            mode="lines",
            name="Momentum (with costs)",
            line=dict(color="#2563eb", width=2),
            hovertemplate="%{x|%b %Y}<br>$%{y:.2f}<extra>With costs</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result_on.benchmark_curve.index,
            y=result_on.benchmark_curve.values,
            mode="lines",
            name="SPY (S&P 500)",
            line=dict(color="#6b7280", width=2, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>$%{y:.2f}<extra>SPY</extra>",
        )
    )

    off = result_off.stats["strategy"]
    on = result_on.stats["strategy"]
    subtitle = (
        f"No costs: {off['annualized_return']:.1%} CAGR, Sharpe {off['sharpe_ratio']:.2f}"
        f" | With costs: {on['annualized_return']:.1%} CAGR, Sharpe {on['sharpe_ratio']:.2f}"
    )

    fig.update_layout(
        title=dict(
            text="Momentum Strategy vs SPY — Growth of $1<br>"
            f"<sup>{subtitle}</sup>",
            x=0.5,
        ),
        xaxis_title="Date",
        yaxis_title="Value of $1 invested",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    fig.write_html(filename, include_plotlyjs=True)
    return filename
