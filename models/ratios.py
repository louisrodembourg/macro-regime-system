import numpy as np
import pandas as pd

RATIO_DEFINITIONS: dict[str, tuple[str, str]] = {
    "SP500_Oil":   ("^GSPC", "CL=F"),
    "Nasdaq_Gold": ("^IXIC", "GC=F"),
    "Copper_Gold": ("HG=F",  "GC=F"),
    "HYG_TLT":    ("HYG",   "TLT"),
    "SP500_Gold":  ("^GSPC", "GC=F"),
    "Oil_Gold":    ("CL=F",  "GC=F"),
    "Gold_TLT":    ("GC=F",  "TLT"),
}

ZSCORE_WINDOW = 252
DIRECTION_WINDOW = 20
MA_7Y_WINDOW = 252 * 7  # ~1764 trading days


def compute_ratios(prices: pd.DataFrame) -> pd.DataFrame:
    """Raw ratio levels for each defined ratio pair."""
    out = {}
    for name, (num, den) in RATIO_DEFINITIONS.items():
        if num in prices.columns and den in prices.columns:
            # Avoid division by zero
            denom = prices[den].replace(0, float("nan"))
            out[name] = prices[num] / denom
        else:
            out[name] = pd.Series(float("nan"), index=prices.index)
    return pd.DataFrame(out, index=prices.index)


def compute_zscores(
    ratios: pd.DataFrame,
    window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Rolling z-score: (x - rolling_mean) / rolling_std. First (window-1) rows are NaN."""
    roll = ratios.rolling(window, min_periods=window)
    return (ratios - roll.mean()) / roll.std()


def compute_long_term_ma(
    ratios: pd.DataFrame,
    window: int = MA_7Y_WINDOW,
) -> pd.DataFrame:
    """Rolling 7-year mean of ratio levels. min_periods=window//2 to show from ~3.5y of data."""
    return ratios.rolling(window, min_periods=window // 2).mean()


def compute_directions(
    ratios: pd.DataFrame,
    window: int = DIRECTION_WINDOW,
) -> pd.DataFrame:
    """Sign of (ratio / ratio.shift(window) - 1): +1 trending up, -1 trending down."""
    pct_chg = ratios / ratios.shift(window) - 1
    return np.sign(pct_chg)


def compute_composite_signals(
    zscores: pd.DataFrame,
    directions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Composite signals used by the quadrant classifier.
    - risk_on_score: mean of 4 risk-on proxies
    - growth_score: weighted Copper/Gold + HYG/TLT + SP500/Oil direction
    - inflation_score: Oil_Gold direction + Gold_TLT z-score
    """
    risk_on = pd.DataFrame({
        "SP500_Oil_z":   zscores["SP500_Oil"],
        "Nasdaq_Gold_z": zscores["Nasdaq_Gold"],
        "Copper_Gold_z": zscores["Copper_Gold"],
        "HYG_TLT_z":    zscores["HYG_TLT"],
    })
    risk_on_score = risk_on.mean(axis=1)

    growth_score = (
        0.4 * zscores["Copper_Gold"]
        + 0.4 * zscores["HYG_TLT"]
        + 0.2 * directions["SP500_Oil"]
    )

    inflation_score = (
        0.5 * directions["Oil_Gold"]
        + 0.5 * zscores["Gold_TLT"]
    )

    return pd.DataFrame({
        "risk_on_score":   risk_on_score,
        "growth_score":    growth_score,
        "inflation_score": inflation_score,
        # Individual z-scores for dashboard KPIs
        "copper_gold_z":  zscores["Copper_Gold"],
        "sp500_oil_z":    zscores["SP500_Oil"],
        "hyg_tlt_z":      zscores["HYG_TLT"],
        "gold_tlt_z":     zscores["Gold_TLT"],
    })
