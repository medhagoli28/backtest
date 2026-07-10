"""
costs.py
========

Execution-cost primitives for the backtester, kept separate from the strategy
logic so each piece can be unit-tested and explained on its own.

Two real-world frictions are modeled:

* **Commission** — a broker fee charged as a fraction of the value traded, on
  both buys and sells (e.g. 0.05% of trade value).
* **Slippage** — the gap between the price you *see* (the close) and the price
  you actually *fill* at. Buys fill a little higher than the close and sells a
  little lower, because your order has to cross the bid/ask spread and can push
  the price against you. Measured in basis points (1 bp = 0.01% = 0.0001).

The strategy rebalances a portfolio of weights each month, so costs are driven
by **turnover**: how much of the portfolio you have to trade to move from the
weights you currently hold to the new target weights.
"""

from __future__ import annotations

# 1 basis point = 0.01% = 0.0001 in fractional terms.
BPS = 1e-4


def commission_cost(trade_value: float, commission_rate: float) -> float:
    """Commission charged on a single trade.

    commission = trade_value * commission_rate

    ``trade_value`` is the dollar (or weight) value changing hands, and
    ``commission_rate`` is a fraction (0.0005 == 0.05%). Charged identically on
    buys and sells.

    Example: trading $10,000 at 0.05% costs $10,000 * 0.0005 = $5.00.
    """
    return abs(trade_value) * commission_rate


def slippage_fill_price(close_price: float, side: str, slippage_bps: float) -> float:
    """The price you actually fill at, given the quoted close and a side.

    Buys fill *higher* than the close and sells fill *lower*, each by
    ``slippage_bps`` basis points:

        buy  fill = close * (1 + slippage_bps / 10_000)
        sell fill = close * (1 - slippage_bps / 10_000)

    ``side`` is "buy" or "sell". This is the per-share view of slippage; the
    portfolio-level cost is the same fraction applied to the value traded.

    Example: a $100 stock with 5 bps of slippage fills at $100.05 on a buy and
    $99.95 on a sell.
    """
    side = side.lower()
    move = slippage_bps * BPS
    if side == "buy":
        return close_price * (1.0 + move)
    if side == "sell":
        return close_price * (1.0 - move)
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def rebalance_turnover(
    prev_weights: dict[str, float],
    target_weights: dict[str, float],
) -> tuple[float, int]:
    """How much trading it takes to go from one portfolio to another.

    For every name that appears in either portfolio, the amount traded is the
    absolute change in its weight, ``|target - prev|``. Summing those gives the
    total fraction of the portfolio that changes hands (buys *and* sells):

        turnover = sum_over_names( |target_weight - prev_weight| )

    A brand-new position (prev = 0) contributes a full buy; a dropped position
    (target = 0) contributes a full sell. Buying an entirely fresh 10-name
    portfolio from cash has turnover 1.0; fully replacing one portfolio with a
    disjoint one has turnover 2.0 (sell 1.0 + buy 1.0).

    Returns ``(turnover, num_trades)`` where ``num_trades`` counts the names
    whose weight actually changed (each is one buy or sell execution).
    """
    names = set(prev_weights) | set(target_weights)

    turnover = 0.0
    num_trades = 0
    for name in names:
        change = target_weights.get(name, 0.0) - prev_weights.get(name, 0.0)
        if abs(change) > 1e-12:
            turnover += abs(change)
            num_trades += 1

    return turnover, num_trades


def drift_weights(
    target_weights: dict[str, float],
    period_returns: dict[str, float],
) -> dict[str, float]:
    """Weights after holding ``target_weights`` through one period of returns.

    We rebalance to equal weight, but over the following month winners grow and
    losers shrink, so by the *next* rebalance the weights have drifted. A name
    that started at weight w and returned r becomes worth w * (1 + r); we then
    renormalize so the weights sum to 1 again:

        drifted_i = w_i * (1 + r_i) / sum_j( w_j * (1 + r_j) )

    Those drifted weights are what we compare the next target against to get an
    honest turnover figure (you don't pay to trade the drift you didn't make).
    """
    grown = {
        name: weight * (1.0 + period_returns.get(name, 0.0))
        for name, weight in target_weights.items()
    }
    total = sum(grown.values())
    if total <= 0:
        return dict(target_weights)
    return {name: value / total for name, value in grown.items()}
