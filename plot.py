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

    fig.write_html(filename, include_plotlyjs="cdn")
    return filename
