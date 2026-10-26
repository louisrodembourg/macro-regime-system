import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.ratios import RATIO_DEFINITIONS
from models.quadrant_classifier import QUADRANT_COLORS, QUADRANT_NAMES

PLOTLY_TEMPLATE = "plotly_dark"

RATIO_DESCRIPTIONS = {
    "SP500_Oil":   ("S&P 500 / Oil (WTI)", "Risk-on vs. inflation pétrolière"),
    "Nasdaq_Gold": ("Nasdaq / Gold", "Growth tech vs. safe-haven monétaire"),
    "Copper_Gold": ("Copper / Gold (Dr. Copper)", "Expansion industrielle vs. contraction"),
    "HYG_TLT":    ("HYG / TLT (High Yield / Treasuries 20Y)", "Appétit au risque obligataire"),
    "SP500_Gold":  ("S&P 500 / Gold", "Risk-on équités vs. store of value"),
    "Oil_Gold":    ("Oil / Gold (WTI)", "Inflation réelle vs. refuge monétaire"),
    "Gold_TLT":   ("Gold / TLT", "Or vs. obligations longues — proxy inflation"),
}


def _add_regime_background(
    fig: go.Figure,
    regime_labels: pd.Series,
    row: int,
    col: int = 1,
) -> go.Figure:
    """Add colored vertical bands for each regime period."""
    if regime_labels is None or regime_labels.dropna().empty:
        return fig

    valid = regime_labels.dropna()
    prev_date = None
    prev_quad = None

    for date, quad in valid.items():
        if prev_quad is None:
            prev_date = date
            prev_quad = quad
            continue
        if quad != prev_quad:
            fig.add_vrect(
                x0=prev_date,
                x1=date,
                fillcolor=QUADRANT_COLORS.get(int(prev_quad), "#888888"),
                opacity=0.08,
                line_width=0,
                row=row,
                col=col,
            )
            prev_date = date
            prev_quad = quad

    if prev_date is not None and prev_quad is not None:
        fig.add_vrect(
            x0=prev_date,
            x1=valid.index[-1],
            fillcolor=QUADRANT_COLORS.get(int(prev_quad), "#888888"),
            opacity=0.08,
            line_width=0,
            row=row,
            col=col,
        )
    return fig


def plot_composite_signal(
    composite: pd.DataFrame,
    regime_labels: pd.Series | None = None,
) -> go.Figure:
    """Multi-line chart of normalized ratio z-scores with regime background shading."""
    fig = go.Figure()

    signal_cols = {
        "risk_on_score":  ("Risk-On Score", "#10B981", 2),
        "copper_gold_z":  ("Copper/Gold Z", "#F59E0B", 1),
        "sp500_oil_z":    ("SP500/Oil Z", "#60A5FA", 1),
        "hyg_tlt_z":      ("HYG/TLT Z", "#A78BFA", 1),
    }

    for col, (name, color, width) in signal_cols.items():
        if col in composite.columns:
            fig.add_trace(go.Scatter(
                x=composite.index,
                y=composite[col],
                name=name,
                line=dict(color=color, width=width),
                opacity=0.9 if col == "risk_on_score" else 0.6,
            ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)

    # Regime background
    if regime_labels is not None:
        valid = regime_labels.dropna()
        prev_date = None
        prev_quad = None
        for date, quad in valid.items():
            if prev_quad is None:
                prev_date = date
                prev_quad = quad
                continue
            if quad != prev_quad:
                fig.add_vrect(
                    x0=prev_date, x1=date,
                    fillcolor=QUADRANT_COLORS.get(int(prev_quad), "#888"),
                    opacity=0.07, line_width=0,
                )
                prev_date = date
                prev_quad = quad
        if prev_date and prev_quad is not None:
            fig.add_vrect(
                x0=prev_date, x1=valid.index[-1],
                fillcolor=QUADRANT_COLORS.get(int(prev_quad), "#888"),
                opacity=0.07, line_width=0,
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Signaux macro normalisés (z-scores rolling 252j)",
        height=420,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Z-score")
    return fig


def plot_ratio_with_zscore(
    ratios: pd.DataFrame,
    zscores: pd.DataFrame,
    ratio_name: str,
    regime_labels: pd.Series | None = None,
    long_term_ma: pd.DataFrame | None = None,
) -> go.Figure:
    """3-panel chart: raw ratio level / z-score with Bollinger bands / regime distribution."""
    label, description = RATIO_DESCRIPTIONS.get(ratio_name, (ratio_name, ""))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{label} — Niveau", f"{label} — Z-Score (252j)"],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
    )

    ratio_series = ratios[ratio_name].dropna()
    zscore_series = zscores[ratio_name].dropna()

    # Panel 1: raw ratio
    fig.add_trace(go.Scatter(
        x=ratio_series.index,
        y=ratio_series.values,
        name=label,
        line=dict(color="#60A5FA", width=1.5),
    ), row=1, col=1)

    # Panel 1: 7-year moving average overlay
    if long_term_ma is not None and ratio_name in long_term_ma.columns:
        ma_series = long_term_ma[ratio_name].dropna()
        if not ma_series.empty:
            fig.add_trace(go.Scatter(
                x=ma_series.index,
                y=ma_series.values,
                name="MM 7 ans",
                line=dict(color="#F59E0B", width=1.8, dash="dot"),
                opacity=0.85,
            ), row=1, col=1)

    # Regime background on raw ratio
    if regime_labels is not None:
        fig = _add_regime_background(fig, regime_labels, row=1)

    # Panel 2: z-score + Bollinger bands (±1σ, ±2σ)
    fig.add_trace(go.Scatter(
        x=zscore_series.index,
        y=zscore_series.values,
        name="Z-Score",
        line=dict(color="#10B981", width=1.5),
    ), row=2, col=1)

    # add_hline requires row="all" or row/col as keyword; use row=2, col=1
    for sigma, color, dash in [(1, "#F59E0B", "dash"), (2, "#EF4444", "dot"), (-1, "#F59E0B", "dash"), (-2, "#EF4444", "dot")]:
        fig.add_hline(y=sigma, line_dash=dash, line_color=color, opacity=0.4, row=2, col=1)

    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.2, row=2, col=1)

    if regime_labels is not None:
        fig = _add_regime_background(fig, regime_labels, row=2)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=550,
        showlegend=True,
        legend=dict(orientation="h", y=-0.12),
        hovermode="x unified",
        margin=dict(t=60, b=10),
    )
    return fig, description


def auto_interpret_ratio(ratio_name: str, current_zscore: float) -> str:
    """Generate automatic text interpretation for a ratio's current z-score."""
    _, desc = RATIO_DESCRIPTIONS.get(ratio_name, (ratio_name, ""))

    if abs(current_zscore) < 0.5:
        level = "dans sa moyenne historique (zone neutre)"
        signal = "Pas de signal directionnel fort."
    elif current_zscore >= 2:
        level = "**extrêmement élevé** (+2σ) — zone de surachat"
        signal = "Signal risk-on très fort. Potentiel retour à la moyenne."
    elif current_zscore >= 1:
        level = "**élevé** (+1σ) — tendance risk-on"
        signal = "Conditions favorables aux actifs risqués."
    elif current_zscore <= -2:
        level = "**extrêmement bas** (-2σ) — zone de survente"
        signal = "Signal risk-off très fort. Or, TLT et cash favorisés."
    elif current_zscore <= -1:
        level = "**bas** (-1σ) — tendance risk-off"
        signal = "Rotation vers les actifs refuges en cours."
    else:
        level = "légèrement hors de sa moyenne"
        signal = "Signal modéré, à surveiller."

    return (
        f"**{ratio_name.replace('_', '/')}** — {desc}\n\n"
        f"Z-score actuel : **{current_zscore:.2f}** — Le ratio est {level}.\n\n"
        f"{signal}"
    )
