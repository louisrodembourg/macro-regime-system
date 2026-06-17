"""
Backtest V1 vs V2 — 3 améliorations appliquées:

  #1 — Smooth confidence blend
       Remplace le hard switch à 60% par une interpolation linéaire
       alloc = confidence * regime_alloc + (1-confidence) * neutral_alloc

  #2 — Momentum filter
       Plafonne SPY+QQQ à 40% quand SPY < SMA200j
       Signal anti-overfitting: 50+ ans d'evidence out-of-sample (Faber 2007)

  #3 — Yield curve (TLT/SHY) comme 5e feature HMM
       Courbe des taux inversée = meilleur leading indicator de récession
       Information orthogonale à VIX/vol/credit spread
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import fetch_prices
from models.ratios import (
    compute_ratios, compute_zscores,
    compute_directions, compute_composite_signals,
)
from models.hmm_detector import (
    build_hmm_features, run_walk_forward_hmm,
    HMM_N_STATES, _prep_X, _fit_hmm, _get_state_order, _relabel,
)
from models.quadrant_classifier import classify_quadrant
from models.allocator import (
    ALLOCATIONS, NEUTRAL_ALLOCATION, PORTFOLIO_ASSETS,
)
from backtest.engine import (
    run_backtest, compute_metrics, build_benchmarks, bootstrap_sharpe_ci,
)

# ─────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────

def annual_cagr(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    return float((1 + returns).prod() ** (252 / n) - 1)


def format_metric(v: float, key: str) -> str:
    pct_keys = {"CAGR", "Max Drawdown", "Volatility", "Best Year", "Worst Year", "Win Rate (Monthly)"}
    if key in pct_keys:
        return f"{v * 100:.1f}%"
    return f"{v:.2f}"


# ─────────────────────────────────────────────
# V2 — Amélioration #1 : Smooth confidence blend
# ─────────────────────────────────────────────

def smooth_blend_allocation(
    quadrant: int,
    hmm_confidence: float,
) -> dict[str, float]:
    """Interpolation linéaire entre regime et neutral selon la confiance HMM."""
    regime_alloc = ALLOCATIONS.get(quadrant, NEUTRAL_ALLOCATION)
    blended = {}
    for asset in PORTFOLIO_ASSETS:
        r = regime_alloc.get(asset, 0.0)
        n = NEUTRAL_ALLOCATION.get(asset, 0.0)
        blended[asset] = hmm_confidence * r + (1.0 - hmm_confidence) * n
    return blended


# ─────────────────────────────────────────────
# V2 — Amélioration #2 : Momentum filter (SPY vs SMA200)
# ─────────────────────────────────────────────

def apply_momentum_filter(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    equity_cap_bearish: float = 0.40,
) -> pd.DataFrame:
    """
    Quand SPY < SMA200 : plafonne SPY+QQQ à equity_cap_bearish.
    L'excédent est redistribué 50/50 sur GLD et TLT.
    """
    sma200 = prices["SPY"].rolling(200, min_periods=100).mean()
    bearish = (prices["SPY"] < sma200).reindex(weights_df.index).fillna(False)

    result = weights_df.copy()
    bear_dates = result.index[bearish]

    for date in bear_dates:
        equity = result.loc[date, "SPY"] + result.loc[date, "QQQ"]
        if equity > equity_cap_bearish:
            excess = equity - equity_cap_bearish
            scale = equity_cap_bearish / equity
            result.loc[date, "SPY"] *= scale
            result.loc[date, "QQQ"] *= scale
            result.loc[date, "GLD"] = min(1.0, result.loc[date, "GLD"] + excess * 0.5)
            result.loc[date, "TLT"] = min(1.0, result.loc[date, "TLT"] + excess * 0.5)

    # Renormaliser pour sommer à 1
    row_sums = result.sum(axis=1).replace(0, 1.0)
    return result.div(row_sums, axis=0)


# ─────────────────────────────────────────────
# V2 — Amélioration #3 : 5e feature HMM (yield curve TLT/SHY)
# ─────────────────────────────────────────────

def fetch_shy(start: str = "2004-01-01") -> pd.Series:
    """Télécharge SHY (1-3Y Treasuries) comme proxy short-end yield."""
    try:
        raw = yf.download("SHY", start=start, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"]["SHY"]
        return raw["Close"]
    except Exception:
        return pd.Series(dtype=float)


def build_hmm_features_v2(
    prices: pd.DataFrame,
    zscores: pd.DataFrame,
    shy: pd.Series,
) -> pd.DataFrame:
    """
    5 features (v2) : ajoute le z-score TLT/SHY (courbe des taux).
    Courbe inversée (z-score bas) → signal récession.
    """
    base = build_hmm_features(prices, zscores)

    if not shy.empty:
        tlt = prices["TLT"]
        shy_aligned = shy.reindex(tlt.index).ffill().bfill()
        tlt_shy_ratio = tlt / shy_aligned.replace(0, np.nan)
        roll_mean = tlt_shy_ratio.rolling(252, min_periods=63).mean()
        roll_std = tlt_shy_ratio.rolling(252, min_periods=63).std()
        tlt_shy_z = (tlt_shy_ratio - roll_mean) / roll_std
        base["yield_curve_z"] = tlt_shy_z
    else:
        base["yield_curve_z"] = 0.0

    return base


def run_walk_forward_hmm_v2(features: pd.DataFrame) -> object:
    """Walk-forward HMM identique à v1 mais avec 5 features."""
    return run_walk_forward_hmm(features)


# ─────────────────────────────────────────────
# Builders de poids
# ─────────────────────────────────────────────

def build_v1_weights(
    quad_labels: pd.Series,
    hmm_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """V1 : hard switch à 60%, pas de momentum filter."""
    from models.allocator import build_weight_series
    return build_weight_series(quad_labels, hmm_probabilities)


def build_v2_weights(
    quad_labels: pd.Series,
    hmm_probabilities: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """V2 : smooth blend + momentum filter."""
    target = pd.DataFrame(index=quad_labels.index, columns=PORTFOLIO_ASSETS, dtype=float)

    for date in quad_labels.index:
        q = quad_labels.loc[date]
        if pd.isna(q):
            alloc = NEUTRAL_ALLOCATION
        else:
            if date in hmm_probabilities.index and not hmm_probabilities.loc[date].isna().all():
                confidence = float(hmm_probabilities.loc[date].max())
            else:
                confidence = 0.0
            alloc = smooth_blend_allocation(int(q), confidence)
        for asset in PORTFOLIO_ASSETS:
            target.loc[date, asset] = alloc.get(asset, 0.0)

    # Rebalancement mensuel
    rebalance_dates = target.resample("ME").last().index
    monthly = target.reindex(rebalance_dates)
    daily = monthly.reindex(quad_labels.index).ffill()
    neutral_row = pd.Series(NEUTRAL_ALLOCATION, dtype=float)
    daily = daily.fillna(neutral_row)

    # Lag d'1 jour
    daily = daily.shift(1)
    daily.iloc[0] = neutral_row

    # Momentum filter
    daily = apply_momentum_filter(daily.astype(float), prices)

    return daily.astype(float)


# ─────────────────────────────────────────────
# Script principal
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BACKTEST V1 vs V2 — Améliorations macro régime")
    print("=" * 60)

    # 1. Données
    print("\n[1/6] Chargement des données (2004–2026)...")
    prices = fetch_prices(start="2004-01-01")

    print("[2/6] Téléchargement SHY (yield curve)...")
    shy = fetch_shy(start="2004-01-01")
    if shy.empty:
        print("      ⚠ SHY indisponible — feature yield curve désactivée")

    # 2. Signaux communs
    print("[3/6] Calcul des ratios & signaux...")
    ratios = compute_ratios(prices)
    zscores = compute_zscores(ratios)
    directions = compute_directions(ratios)
    composite = compute_composite_signals(zscores, directions)
    growth = composite["growth_score"]
    inflation = composite["inflation_score"]
    quad_labels = classify_quadrant(growth, inflation)

    # 3. HMM V1 (4 features)
    print("[4/6] HMM V1 — walk-forward (4 features)...")
    features_v1 = build_hmm_features(prices, zscores)
    result_v1 = run_walk_forward_hmm(features_v1)

    # 4. HMM V2 (5 features avec yield curve)
    print("[5/6] HMM V2 — walk-forward (5 features avec yield curve)...")
    features_v2 = build_hmm_features_v2(prices, zscores, shy)
    result_v2 = run_walk_forward_hmm_v2(features_v2)

    # 5. Construction des poids
    weights_v1 = build_v1_weights(quad_labels, result_v1.probabilities)
    weights_v2 = build_v2_weights(quad_labels, result_v2.probabilities, prices)

    # 6. Backtests
    print("[6/6] Backtests...")
    bt_v1 = run_backtest(weights_v1, prices)
    bt_v2 = run_backtest(weights_v2, prices)
    benchmarks = build_benchmarks(prices)
    bench_metrics = {k: compute_metrics(v) for k, v in benchmarks.items()}

    m_v1 = bt_v1["metrics"]
    m_v2 = bt_v2["metrics"]

    # ── Tableau comparatif ────────────────────────────────────
    print("\n")
    print("=" * 72)
    print("  RÉSULTATS — 2007 à 2026")
    print("=" * 72)

    metrics_order = [
        "CAGR", "Sharpe", "Sortino", "Calmar",
        "Max Drawdown", "Volatility", "Win Rate (Monthly)",
        "Best Year", "Worst Year",
    ]
    positive = {"CAGR", "Sharpe", "Sortino", "Calmar", "Win Rate (Monthly)", "Best Year"}

    header = f"{'Métrique':<26}{'V1 (actuel)':>13}{'V2 (amélioré)':>14}{'SPY B&H':>10}{'60/40':>10}"
    print(header)
    print("-" * 73)

    for m in metrics_order:
        v1 = m_v1.get(m, float("nan"))
        v2 = m_v2.get(m, float("nan"))
        spy = bench_metrics.get("SPY B&H", {}).get(m, float("nan"))
        b40 = bench_metrics.get("60/40", {}).get(m, float("nan"))

        all_vals = {
            "V1": v1, "V2": v2, "SPY": spy, "60/40": b40,
        }
        valid = {k: val for k, val in all_vals.items() if not np.isnan(val)}
        if valid:
            best_key = max(valid, key=lambda k: valid[k] if m in positive else -valid[k])
        else:
            best_key = None

        def tag(k, val):
            if np.isnan(val):
                return "  —"
            s = format_metric(val, m)
            return f"{s} {'🏆' if k == best_key else '  '}"

        row = (
            f"{m:<26}"
            f"{tag('V1', v1):>15}"
            f"{tag('V2', v2):>15}"
            f"{tag('SPY', spy):>12}"
            f"{tag('60/40', b40):>12}"
        )
        print(row)

    # ── Delta V2 vs V1 ──────────────────────────────────────
    print("\n")
    print("=" * 50)
    print("  DELTA V2 vs V1")
    print("=" * 50)
    deltas = {
        "CAGR":              (m_v2["CAGR"] - m_v1["CAGR"]) * 100,
        "Sharpe":            m_v2["Sharpe"] - m_v1["Sharpe"],
        "Max Drawdown":      (m_v2["Max Drawdown"] - m_v1["Max Drawdown"]) * 100,
        "Volatility":        (m_v2["Volatility"] - m_v1["Volatility"]) * 100,
        "Calmar":            m_v2["Calmar"] - m_v1["Calmar"],
    }
    for k, d in deltas.items():
        arrow = "↑" if (d > 0 and k != "Volatility" and k != "Max Drawdown") or \
                       (d < 0 and k in ("Volatility", "Max Drawdown")) else "↓"
        sign = "+" if d > 0 else ""
        unit = "pp" if k in ("CAGR", "Max Drawdown", "Volatility") else ""
        print(f"  {k:<22} {sign}{d:.2f}{unit}  {arrow}")

    # ── Rendements annuels ───────────────────────────────────
    print("\n")
    print("=" * 55)
    print("  RENDEMENTS ANNUELS")
    print("=" * 55)
    ret_v1 = bt_v1["returns"]
    ret_v2 = bt_v2["returns"]
    ret_spy = benchmarks["SPY B&H"]

    ann_v1 = (ret_v1 + 1).resample("YE").prod() - 1
    ann_v2 = (ret_v2 + 1).resample("YE").prod() - 1
    ann_spy = (ret_spy + 1).resample("YE").prod() - 1

    print(f"{'Année':<8}{'V1':>9}{'V2':>9}{'SPY':>9}  {'Meilleur':>9}")
    print("-" * 44)
    for year in ann_v1.index:
        y = year.year
        v1_r = float(ann_v1.get(year, float("nan")))
        v2_r = float(ann_v2.get(year, float("nan"))) if year in ann_v2.index else float("nan")
        spy_r = float(ann_spy.get(year, float("nan"))) if year in ann_spy.index else float("nan")

        valid_yr = [(v1_r, "V1"), (v2_r, "V2"), (spy_r, "SPY")]
        valid_yr = [(r, n) for r, n in valid_yr if not np.isnan(r)]
        best_yr = max(valid_yr, key=lambda x: x[0])[1] if valid_yr else "?"

        v1_s = f"{v1_r*100:+.1f}%" if not np.isnan(v1_r) else "—"
        v2_s = f"{v2_r*100:+.1f}%" if not np.isnan(v2_r) else "—"
        spy_s = f"{spy_r*100:+.1f}%" if not np.isnan(spy_r) else "—"

        # Highlight des années où V2 bat V1
        flag = " ← V2>V1" if (not np.isnan(v2_r) and not np.isnan(v1_r) and v2_r > v1_r) else ""
        print(f"{y:<8}{v1_s:>9}{v2_s:>9}{spy_s:>9}  {best_yr:>6}{flag}")

    # ── Simulation 10 000€ ─────────────────────────────────
    print("\n")
    print("=" * 50)
    print("  SIMULATION : 10 000€ investis en avril 2007")
    print("=" * 50)
    final_vals = {
        "V2 (amélioré)":  10000 * float((1 + ret_v2).prod()),
        "V1 (actuel)":    10000 * float((1 + ret_v1).prod()),
        "SPY B&H":        10000 * float((1 + ret_spy).prod()),
        "60/40":          10000 * float((1 + benchmarks["60/40"]).prod()),
    }
    for name, val in sorted(final_vals.items(), key=lambda x: -x[1]):
        bar = "█" * int(val / 5000)
        print(f"  {name:<22} → {val:>10,.0f} €  {bar}")

    # ── Bootstrap Sharpe CI ───────────────────────────────
    print("\n")
    print("=" * 50)
    print("  INTERVALLE DE CONFIANCE SHARPE (Bootstrap 95%)")
    print("=" * 50)
    lo_v1, hi_v1 = bootstrap_sharpe_ci(ret_v1)
    lo_v2, hi_v2 = bootstrap_sharpe_ci(ret_v2)
    print(f"  V1 : Sharpe = {m_v1['Sharpe']:.2f}  IC 95% [{lo_v1:.2f}, {hi_v1:.2f}]")
    print(f"  V2 : Sharpe = {m_v2['Sharpe']:.2f}  IC 95% [{lo_v2:.2f}, {hi_v2:.2f}]")

    # ── Verdict ──────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("  VERDICT")
    print("=" * 60)
    sharpe_better = m_v2["Sharpe"] > m_v1["Sharpe"]
    cagr_better = m_v2["CAGR"] > m_v1["CAGR"]
    dd_better = m_v2["Max Drawdown"] > m_v1["Max Drawdown"]

    wins = sum([sharpe_better, cagr_better, dd_better])
    if wins >= 2:
        print("  ✅ V2 est une AMÉLIORATION nette sur la majorité des métriques")
    else:
        print("  ⚠  V2 est MIXTE — amélioration sur certains axes, recul sur d'autres")

    print(f"  CAGR   : V2 {'>' if cagr_better else '<'} V1  ({m_v2['CAGR']*100:.1f}% vs {m_v1['CAGR']*100:.1f}%)")
    print(f"  Sharpe : V2 {'>' if sharpe_better else '<'} V1  ({m_v2['Sharpe']:.2f} vs {m_v1['Sharpe']:.2f})")
    print(f"  Max DD : V2 {'>' if dd_better else '<'} V1  ({m_v2['Max Drawdown']*100:.1f}% vs {m_v1['Max Drawdown']*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
