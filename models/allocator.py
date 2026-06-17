import pandas as pd
import numpy as np

PORTFOLIO_ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XLE", "GSG"]

ALLOCATIONS: dict[int, dict[str, float]] = {
    1: {"SPY": 0.30, "QQQ": 0.10, "GLD": 0.20, "TLT": 0.00, "XLE": 0.25, "GSG": 0.15},  # Growth + Inflation
    2: {"SPY": 0.35, "QQQ": 0.50, "GLD": 0.00, "TLT": 0.10, "XLE": 0.03, "GSG": 0.02},  # Growth + Deflation (V3b — 85% equity)
    3: {"SPY": 0.10, "QQQ": 0.00, "GLD": 0.40, "TLT": 0.00, "XLE": 0.30, "GSG": 0.20},  # Stagflation
    4: {"SPY": 0.05, "QQQ": 0.05, "GLD": 0.35, "TLT": 0.50, "XLE": 0.00, "GSG": 0.05},  # Recession
}

NEUTRAL_ALLOCATION: dict[str, float] = {
    "SPY": 0.33, "QQQ": 0.00, "GLD": 0.33, "TLT": 0.34, "XLE": 0.00, "GSG": 0.00,
}

CONFIDENCE_THRESHOLD = 0.60


def get_allocation(quadrant: int, hmm_max_proba: float) -> dict[str, float]:
    """Returns allocation dict. Falls back to neutral if HMM confidence is low."""
    if hmm_max_proba < CONFIDENCE_THRESHOLD:
        return NEUTRAL_ALLOCATION.copy()
    return ALLOCATIONS.get(quadrant, NEUTRAL_ALLOCATION).copy()


def build_weight_series(
    quadrant_labels: pd.Series,
    hmm_probabilities: pd.DataFrame,
    rebalance_freq: str = "ME",
    apply_lag: bool = True,
) -> pd.DataFrame:
    """
    Builds daily weight DataFrame for backtesting.
    - Weights only change on month-end rebalance dates
    - apply_lag=True shifts signals by 1 day (avoid lookahead bias)
    - NaN weights (from lag or pre-signal) filled with NEUTRAL_ALLOCATION
    """
    # Determine target allocation for each day based on quadrant + HMM confidence
    target_weights = pd.DataFrame(index=quadrant_labels.index, columns=PORTFOLIO_ASSETS, dtype=float)

    for date in quadrant_labels.index:
        q = quadrant_labels.loc[date]
        if pd.isna(q):
            alloc = NEUTRAL_ALLOCATION
        else:
            if date in hmm_probabilities.index and not hmm_probabilities.loc[date].isna().all():
                max_proba = float(hmm_probabilities.loc[date].max())
            else:
                max_proba = 0.0
            alloc = get_allocation(int(q), max_proba)
        for asset in PORTFOLIO_ASSETS:
            target_weights.loc[date, asset] = alloc.get(asset, 0.0)

    # Resample to rebalance frequency: only update weights on rebalance dates
    # Between rebalance dates, use last known weight (will drift in backtest engine)
    rebalance_dates = target_weights.resample(rebalance_freq).last().index
    monthly_weights = target_weights.reindex(rebalance_dates)

    # Forward-fill to all business days
    daily_weights = monthly_weights.reindex(quadrant_labels.index).ffill()

    # Fill any remaining NaN at the start with neutral
    neutral_row = pd.Series(NEUTRAL_ALLOCATION, dtype=float)
    daily_weights = daily_weights.fillna(neutral_row)

    if apply_lag:
        # Shift by 1 day: signal from day T applied starting day T+1
        daily_weights = daily_weights.shift(1)
        daily_weights.iloc[0] = neutral_row

    return daily_weights.astype(float)
