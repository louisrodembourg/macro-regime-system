import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_TEMPLATE = "plotly_dark"

BENCHMARK_COLORS = {
    "Stratégie Régime": "#10B981",
    "SPY B&H":          "#60A5FA",
    "60/40":            "#F59E0B",
    "Equal Weight":     "#A78BFA",
}


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    cum = _equity_curve(returns)
    return cum / cum.cummax() - 1


def plot_equity_curves(
    strategy_returns: pd.Series,
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    """Equity curves normalized to 1.0 at start."""
    fig = go.Figure()

    all_curves = {"Stratégie Régime": strategy_returns, **benchmarks}
    for name, ret in all_curves.items():
        curve = _equity_curve(ret.dropna())
        color = BENCHMARK_COLORS.get(name, "#888888")
        width = 2.5 if name == "Stratégie Régime" else 1.2
        fig.add_trace(go.Scatter(
            x=curve.index,
            y=curve.values,
            name=name,
            line=dict(color=color, width=width),
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Performance comparative — Courbes de capital (base 1.0)",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=10),
        yaxis_title="Capital (base 1.0)",
    )
    return fig


def plot_drawdown(
    strategy_returns: pd.Series,
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    """Underwater (drawdown) chart for strategy + benchmarks."""
    fig = go.Figure()

    all_rets = {"Stratégie Régime": strategy_returns, **benchmarks}
    for name, ret in all_rets.items():
        dd = _drawdown(ret.dropna())
        color = BENCHMARK_COLORS.get(name, "#888888")
        width = 2.0 if name == "Stratégie Régime" else 1.0
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            name=name,
            line=dict(color=color, width=width),
            fill="tozeroy" if name == "Stratégie Régime" else None,
            fillcolor="rgba(16,185,129,0.08)" if name == "Stratégie Régime" else None,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Drawdowns (%)",
        height=320,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(t=50, b=10),
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1f",
    )
    return fig


def plot_metrics_table(
    strategy_metrics: dict[str, float],
    benchmark_metrics: dict[str, dict[str, float]],
) -> go.Figure:
    """Comparison table with color-coded cells (green if strategy beats benchmark)."""
    metric_order = [
        "CAGR", "Sharpe", "Sortino", "Calmar",
        "Max Drawdown", "Volatility", "Win Rate (Monthly)",
        "VaR 95%", "CVaR 95%", "Best Year", "Worst Year",
    ]
    metric_order = [m for m in metric_order if m in strategy_metrics]

    headers = ["Métrique", "Stratégie Régime"] + list(benchmark_metrics.keys())
    rows: list[list] = []

    # Format helpers
    def fmt(val: float, metric: str) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if metric in ("CAGR", "Max Drawdown", "Volatility", "VaR 95%", "CVaR 95%",
                       "Win Rate (Monthly)", "Best Year", "Worst Year"):
            return f"{val * 100:.1f}%"
        return f"{val:.2f}"

    col_cells: list[list] = [metric_order]
    strat_vals = [fmt(strategy_metrics.get(m, float("nan")), m) for m in metric_order]
    col_cells.append(strat_vals)

    # For color coding: compare strategy vs each benchmark
    fill_colors: list[list[str]] = [
        ["#1a1d2e"] * len(metric_order),  # metric name column — neutral
        [],  # strategy column — will be filled below
    ]
    strat_color_col = []

    for m in metric_order:
        strat_v = strategy_metrics.get(m, float("nan"))
        bench_vals = [bm.get(m, float("nan")) for bm in benchmark_metrics.values()]
        # Strategy is green if better than ALL benchmarks for positive metrics
        positive_metrics = {"CAGR", "Sharpe", "Sortino", "Calmar", "Win Rate (Monthly)", "Best Year"}
        negative_metrics = {"Max Drawdown", "Volatility", "VaR 95%", "CVaR 95%", "Worst Year"}

        if all(not np.isnan(v) for v in [strat_v] + bench_vals):
            if m in positive_metrics:
                better = all(strat_v >= v for v in bench_vals)
            elif m in negative_metrics:
                better = all(strat_v >= v for v in bench_vals)  # drawdown is negative, higher = less bad
            else:
                better = False
            strat_color_col.append("rgba(16,185,129,0.25)" if better else "#1a1d2e")
        else:
            strat_color_col.append("#1a1d2e")

    fill_colors[1] = strat_color_col

    for bench_name, bm in benchmark_metrics.items():
        bench_fmts = [fmt(bm.get(m, float("nan")), m) for m in metric_order]
        col_cells.append(bench_fmts)
        fill_colors.append(["#1a1d2e"] * len(metric_order))

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#0f1117",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=col_cells,
            fill_color=fill_colors,
            font=dict(color="white", size=11),
            align=["left"] + ["center"] * (len(headers) - 1),
            height=28,
        ),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(t=10, b=10),
    )
    return fig


def plot_sensitivity_heatmap(sensitivity_df: pd.DataFrame) -> go.Figure:
    """Heatmap: rows=thresholds, columns=metrics, colored by value."""
    if sensitivity_df.empty:
        return go.Figure()

    display_cols = ["CAGR", "Sharpe", "Sortino", "Max Drawdown", "Win Rate (Monthly)"]
    display_cols = [c for c in display_cols if c in sensitivity_df.columns]

    z = sensitivity_df[display_cols].values
    text = [[f"{v:.2f}" for v in row] for row in z]
    y_labels = [f"Threshold={t}" for t in sensitivity_df.index]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=display_cols,
        y=y_labels,
        colorscale="RdYlGn",
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Valeur"),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Analyse de sensibilité — Métriques selon le seuil de classification",
        height=300,
        margin=dict(t=50, b=10),
    )
    return fig


def plot_annual_returns_bar(
    strategy_returns: pd.Series,
    benchmarks: dict[str, pd.Series],
) -> go.Figure:
    """Grouped bar chart of annual returns per year."""
    fig = go.Figure()

    all_rets = {"Stratégie Régime": strategy_returns, **benchmarks}
    for name, ret in all_rets.items():
        annual = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        fig.add_trace(go.Bar(
            x=annual.index.year.astype(str),
            y=annual.values * 100,
            name=name,
            marker_color=BENCHMARK_COLORS.get(name, "#888888"),
            opacity=0.85,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Rendements annuels (%)",
        height=350,
        barmode="group",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=10),
        yaxis_title="Rendement (%)",
        yaxis_tickformat=".1f",
    )
    return fig
