import logging

import numpy as np
import pandas as pd

from models.allocator import PORTFOLIO_ASSETS, NEUTRAL_ALLOCATION, build_weight_series
from models.quadrant_classifier import classify_quadrant

logger = logging.getLogger(__name__)

TRANSACTION_COST_BPS = 10
RISK_FREE_ANNUAL = 0.04
BACKTEST_START = "2007-04-11"  # HYG inception date


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str = BACKTEST_START,
    end_date: str | None = None,
    tc_bps: float = TRANSACTION_COST_BPS,
    rebalance_freq: str = "ME",
) -> dict:
    """
    Vectorized backtest. Returns dict with equity curve, returns, metrics, etc.
    weights must already be lagged (signal from T, applied at T+1).
    """
    # Align prices and weights
    asset_prices = prices[PORTFOLIO_ASSETS].copy()
    daily_returns = asset_prices.pct_change()

    start = pd.Timestamp(start_date)
    if end_date:
        end = pd.Timestamp(end_date)
        daily_returns = daily_returns.loc[start:end]
        weights = weights.loc[start:end]
    else:
        daily_returns = daily_returns.loc[start:]
        weights = weights.loc[start:]

    # Align indices
    common = daily_returns.index.intersection(weights.index)
    daily_returns = daily_returns.loc[common].fillna(0.0)
    target_weights = weights.loc[common].fillna(0.0)

    # Ensure weights sum to 1 (normalize)
    row_sums = target_weights.sum(axis=1).replace(0, 1)
    target_weights = target_weights.div(row_sums, axis=0)

    # Identify monthly rebalance dates
    rebalance_dates = set(
        daily_returns.resample(rebalance_freq).last().index.intersection(daily_returns.index)
    )

    # Vectorized simulation with weight drift
    n_days = len(daily_returns)
    n_assets = len(PORTFOLIO_ASSETS)
    portfolio_returns = np.zeros(n_days)
    actual_weights = np.zeros((n_days, n_assets))
    turnover = np.zeros(n_days)

    current_weights = target_weights.iloc[0].values.copy()
    actual_weights[0] = current_weights

    for i, (date, ret_row) in enumerate(daily_returns.iterrows()):
        if i == 0:
            portfolio_returns[i] = np.dot(current_weights, ret_row.values)
            continue

        # Drift weights with previous day's returns
        prev_w = actual_weights[i - 1]
        prev_ret = daily_returns.iloc[i - 1].values
        drifted = prev_w * (1 + prev_ret)
        drifted_sum = drifted.sum()
        if drifted_sum > 0:
            drifted /= drifted_sum
        else:
            drifted = np.ones(n_assets) / n_assets

        if date in rebalance_dates:
            new_target = target_weights.loc[date].values
            to = np.abs(new_target - drifted).sum()
            turnover[i] = to
            tc_drag = to * tc_bps / 10_000
            current_weights = new_target
        else:
            tc_drag = 0.0
            current_weights = drifted

        gross_return = np.dot(current_weights, ret_row.values)
        portfolio_returns[i] = gross_return - tc_drag
        actual_weights[i] = current_weights

    returns_series = pd.Series(portfolio_returns, index=daily_returns.index)
    cumulative = (1 + returns_series).cumprod()
    weights_df = pd.DataFrame(actual_weights, index=daily_returns.index, columns=PORTFOLIO_ASSETS)
    turnover_series = pd.Series(turnover, index=daily_returns.index)

    metrics = compute_metrics(returns_series)

    return {
        "returns": returns_series,
        "cumulative": cumulative,
        "weights_history": weights_df,
        "turnover": turnover_series,
        "metrics": metrics,
    }


def compute_metrics(
    returns: pd.Series,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> dict[str, float]:
    """Compute standard portfolio performance metrics."""
    n_days = len(returns)
    if n_days == 0:
        return {}

    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1

    # CAGR
    total_return = (1 + returns).prod()
    years = n_days / 252
    cagr = total_return ** (1 / years) - 1 if years > 0 else 0.0

    # Volatility (annualized)
    vol = returns.std() * np.sqrt(252)

    # Sharpe
    excess = returns - rf_daily
    sharpe = (excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    # Sortino
    downside = returns[returns < rf_daily]
    sortino_denom = downside.std() * np.sqrt(252) if len(downside) > 0 else float("nan")
    sortino = ((returns.mean() - rf_daily) * 252 / sortino_denom) if sortino_denom and not np.isnan(sortino_denom) else 0.0

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate monthly
    monthly_ret = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly_ret > 0).mean()

    # Avg turnover (annualized) — expects turnover from run_backtest
    # Here we just report vol and other metrics

    # VaR / CVaR (95%)
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

    best_year = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1).max()
    worst_year = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1).min()

    return {
        "CAGR": round(cagr, 4),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Max Drawdown": round(max_dd, 4),
        "Volatility": round(vol, 4),
        "Win Rate (Monthly)": round(win_rate, 3),
        "VaR 95%": round(var_95, 4),
        "CVaR 95%": round(cvar_95, 4),
        "Best Year": round(best_year, 4),
        "Worst Year": round(worst_year, 4),
    }


def build_benchmarks(
    prices: pd.DataFrame,
    start_date: str = BACKTEST_START,
    end_date: str | None = None,
) -> dict[str, pd.Series]:
    """Build 4 benchmark return series."""
    asset_prices = prices[PORTFOLIO_ASSETS].copy()
    daily_returns = asset_prices.pct_change()

    start = pd.Timestamp(start_date)
    if end_date:
        daily_returns = daily_returns.loc[start:pd.Timestamp(end_date)]
    else:
        daily_returns = daily_returns.loc[start:]

    daily_returns = daily_returns.fillna(0.0)

    # 1. SPY buy-and-hold
    spy_bh = daily_returns["SPY"]

    # 2. 60/40 (SPY/TLT) monthly rebalanced
    w_6040 = pd.DataFrame(0.0, index=daily_returns.index, columns=PORTFOLIO_ASSETS)
    w_6040["SPY"] = 0.60
    w_6040["TLT"] = 0.40
    result_6040 = run_backtest(w_6040, prices, start_date=start_date, end_date=end_date, tc_bps=5)
    ret_6040 = result_6040["returns"]

    # 3. Equal weight (1/6 each)
    w_ew = pd.DataFrame(1 / len(PORTFOLIO_ASSETS), index=daily_returns.index, columns=PORTFOLIO_ASSETS)
    result_ew = run_backtest(w_ew, prices, start_date=start_date, end_date=end_date, tc_bps=5)
    ret_ew = result_ew["returns"]

    return {
        "SPY B&H": spy_bh,
        "60/40": ret_6040,
        "Equal Weight": ret_ew,
    }


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    random_state: int = 42,
) -> tuple[float, float]:
    """Vectorized bootstrap confidence interval for Sharpe ratio."""
    rng = np.random.default_rng(random_state)
    n = len(returns)
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = (returns - rf_daily).values

    # Sample n_resamples x n matrix of indices
    idx = rng.integers(0, n, (n_resamples, n))
    samples = excess[idx]  # shape (n_resamples, n)

    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1)
    # Avoid division by zero
    valid = sample_stds > 0
    sharpe_dist = np.zeros(n_resamples)
    sharpe_dist[valid] = sample_means[valid] / sample_stds[valid] * np.sqrt(252)

    alpha = 1 - confidence
    lo = float(np.percentile(sharpe_dist, alpha / 2 * 100))
    hi = float(np.percentile(sharpe_dist, (1 - alpha / 2) * 100))
    return lo, hi


def run_sensitivity_analysis(
    prices: pd.DataFrame,
    zscores: pd.DataFrame,
    directions: pd.DataFrame,
    hmm_probabilities: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Re-runs quadrant classification + backtest for different threshold values.
    Returns DataFrame: rows = thresholds, columns = metrics.
    """
    from models.ratios import compute_composite_signals

    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 1.0, 1.5]

    composite = compute_composite_signals(zscores, directions)
    growth = composite["growth_score"]
    inflation = composite["inflation_score"]

    results = []
    for t in thresholds:
        quad_labels = classify_quadrant(growth, inflation, threshold=t)
        weights = build_weight_series(quad_labels, hmm_probabilities)
        bt = run_backtest(weights, prices)
        row = {"threshold": t}
        row.update(bt["metrics"])
        results.append(row)

    return pd.DataFrame(results).set_index("threshold")


def run_subperiod_analysis(
    returns: pd.Series,
    sub_periods: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Compute metrics for each sub-period."""
    if sub_periods is None:
        sub_periods = {
            "GFC (2007-2009)":    ("2007-04-11", "2009-12-31"),
            "Recovery (2009-2015)": ("2010-01-01", "2015-12-31"),
            "Late Cycle (2015-2020)": ("2016-01-01", "2020-03-31"),
            "Post-COVID (2020-2026)": ("2020-04-01", "2026-06-16"),
        }

    rows = []
    for label, (start, end) in sub_periods.items():
        slice_ = returns.loc[start:end]
        if len(slice_) < 20:
            continue
        m = compute_metrics(slice_)
        rows.append({"Period": label, **m})

    df = pd.DataFrame(rows).set_index("Period") if rows else pd.DataFrame()
    return df
