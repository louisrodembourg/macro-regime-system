import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.hmm_detector import REGIME_COLORS, REGIME_LABELS, HMM_N_STATES
from models.quadrant_classifier import QUADRANT_COLORS, QUADRANT_NAMES

PLOTLY_TEMPLATE = "plotly_dark"


def plot_hmm_timeline(
    regime_series: pd.Series,
    prices: pd.DataFrame,
) -> go.Figure:
    """
    Dual-panel: top = ^GSPC price line with regime shading, bottom = regime bars.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["S&P 500 avec régimes HMM", "Régime HMM détecté"],
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Panel 1: SP500
    sp500 = prices["^GSPC"].dropna()
    fig.add_trace(go.Scatter(
        x=sp500.index,
        y=sp500.values,
        name="S&P 500",
        line=dict(color="#60A5FA", width=1.5),
        showlegend=True,
    ), row=1, col=1)

    # Regime background shading on SP500
    _add_hmm_vrects(fig, regime_series, row=1)

    # Panel 2: regime coloring bars
    valid = regime_series.dropna()
    for state, label in REGIME_LABELS.items():
        mask = valid == state
        state_dates = valid[mask].index
        if len(state_dates) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=state_dates,
            y=np.ones(len(state_dates)) * state,
            mode="markers",
            marker=dict(
                symbol="square",
                size=4,
                color=REGIME_COLORS.get(state, "#888"),
            ),
            name=f"{label} (HMM)",
            showlegend=True,
        ), row=2, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=60, b=10),
    )
    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(
        title_text="Régime",
        tickvals=[0, 1, 2],
        ticktext=["Bull", "Transition", "Crisis"],
        row=2, col=1,
    )
    return fig


def _add_hmm_vrects(
    fig: go.Figure,
    regime_series: pd.Series,
    row: int,
) -> None:
    """Add vertical regime bands to a figure panel."""
    valid = regime_series.dropna()
    if valid.empty:
        return

    prev_date = None
    prev_state = None

    for date, state in valid.items():
        if prev_state is None:
            prev_date = date
            prev_state = state
            continue
        if state != prev_state:
            fig.add_vrect(
                x0=prev_date, x1=date,
                fillcolor=REGIME_COLORS.get(int(prev_state), "#888"),
                opacity=0.12, line_width=0,
                row=row, col=1,
            )
            prev_date = date
            prev_state = state

    if prev_date is not None:
        fig.add_vrect(
            x0=prev_date, x1=valid.index[-1],
            fillcolor=REGIME_COLORS.get(int(prev_state), "#888"),
            opacity=0.12, line_width=0,
            row=row, col=1,
        )


def plot_regime_returns_heatmap(
    prices: pd.DataFrame,
    quadrant_labels: pd.Series,
) -> go.Figure:
    """Heatmap of average monthly returns per asset grouped by quadrant."""
    asset_returns = prices[["SPY", "QQQ", "GLD", "TLT", "XLE", "GSG"]].pct_change()
    monthly_returns = (asset_returns + 1).resample("ME").prod() - 1
    monthly_quad = quadrant_labels.resample("ME").last()

    combined = monthly_quad.rename("quadrant").to_frame().join(monthly_returns, how="inner")
    combined = combined.dropna(subset=["quadrant"])

    rows_data = []
    for q in sorted(QUADRANT_NAMES.keys()):
        sub = combined[combined["quadrant"] == q]
        if len(sub) == 0:
            continue
        row = {"Régime": f"Q{q}: {QUADRANT_NAMES[q]}", "N mois": len(sub)}
        for col in monthly_returns.columns:
            row[col] = round(sub[col].mean() * 100, 2)
        rows_data.append(row)

    if not rows_data:
        return go.Figure()

    df_heatmap = pd.DataFrame(rows_data).set_index("Régime")
    assets = [c for c in df_heatmap.columns if c != "N mois"]
    z_values = df_heatmap[assets].values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z_values,
        x=assets,
        y=df_heatmap.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in z_values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Rend. moy. (%)", tickformat=".1f"),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Rendements mensuels moyens par régime macro",
        height=280,
        margin=dict(t=50, b=10),
        xaxis_title="Actif",
    )
    return fig


def plot_transition_matrix(model) -> go.Figure:
    """Heatmap of HMM transition probability matrix."""
    labels = [REGIME_LABELS[i] for i in range(HMM_N_STATES)]
    transmat = model.transmat_

    fig = go.Figure(go.Heatmap(
        z=transmat,
        x=labels,
        y=labels,
        colorscale="Blues",
        zmin=0, zmax=1,
        text=[[f"{v:.1%}" for v in row] for row in transmat],
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="Probabilité", tickformat=".0%"),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Matrice de transition HMM",
        height=300,
        margin=dict(t=50, b=10),
        xaxis_title="État suivant",
        yaxis_title="État actuel",
    )
    return fig


def plot_gmm_confusion(confusion_df: pd.DataFrame) -> go.Figure:
    """Heatmap comparing GMM clusters vs. quadrant labels."""
    if confusion_df.empty:
        return go.Figure()

    z = confusion_df.values
    fig = go.Figure(go.Heatmap(
        z=z,
        x=confusion_df.columns.tolist(),
        y=confusion_df.index.tolist(),
        colorscale="Purples",
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Count"),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Accord HMM/Quadrant vs. GMM Clustering",
        height=300,
        margin=dict(t=50, b=10),
        xaxis_title="Quadrant macro",
        yaxis_title="Cluster GMM",
    )
    return fig


def compute_regime_stats(
    hmm_states: pd.Series,
    quadrant_labels: pd.Series,
) -> pd.DataFrame:
    """Compute average duration and frequency per regime."""
    rows = []

    for s, lbl in REGIME_LABELS.items():
        mask = hmm_states == s
        # Count consecutive runs
        runs = []
        in_run = False
        count = 0
        for val in mask:
            if val:
                in_run = True
                count += 1
            elif in_run:
                runs.append(count)
                count = 0
                in_run = False
        if in_run and count > 0:
            runs.append(count)

        n_days = mask.sum()
        total = hmm_states.notna().sum()
        rows.append({
            "Régime": lbl,
            "Couleur": REGIME_COLORS.get(s, "#888"),
            "% du temps": round(n_days / total * 100, 1) if total > 0 else 0,
            "Durée moy. (jours)": round(np.mean(runs), 1) if runs else 0,
            "Nb épisodes": len(runs),
        })

    return pd.DataFrame(rows).set_index("Régime")
