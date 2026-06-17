import hashlib
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # Suppress hmmlearn ConvergenceWarning in UI

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path when run from subdirectory
sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import fetch_prices
from models.allocator import (
    ALLOCATIONS,
    PORTFOLIO_ASSETS,
    build_weight_series,
    get_allocation,
)
from models.hmm_detector import (
    REGIME_COLORS,
    REGIME_LABELS,
    build_hmm_features,
    fit_gmm,
    run_walk_forward_hmm,
)
from models.quadrant_classifier import (
    QUADRANT_COLORS,
    QUADRANT_DESCRIPTIONS,
    QUADRANT_NAMES,
    align_gmm_to_quadrants,
    classify_quadrant,
)
from models.ratios import (
    RATIO_DEFINITIONS,
    compute_composite_signals,
    compute_directions,
    compute_ratios,
    compute_zscores,
)
from backtest.engine import (
    bootstrap_sharpe_ci,
    build_benchmarks,
    compute_metrics,
    run_backtest,
    run_sensitivity_analysis,
    run_subperiod_analysis,
)
from visualizations.portfolio_perf import (
    plot_annual_returns_bar,
    plot_drawdown,
    plot_equity_curves,
    plot_metrics_table,
    plot_sensitivity_heatmap,
)
from visualizations.ratio_charts import (
    auto_interpret_ratio,
    plot_composite_signal,
    plot_ratio_with_zscore,
)
from visualizations.regime_timeline import (
    compute_regime_stats,
    plot_gmm_confusion,
    plot_hmm_timeline,
    plot_regime_returns_heatmap,
    plot_transition_matrix,
)

logging.basicConfig(level=logging.INFO)

PERIOD_OPTIONS = {
    "1 an":   1,
    "3 ans":  3,
    "5 ans":  5,
    "10 ans": 10,
    "Max (2004)": 22,
}

REBALANCE_OPTIONS = {
    "Mensuel":       "ME",
    "Trimestriel":   "QE",
}


# ──────────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────────

def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]


@st.cache_data(ttl=86400, show_spinner=False)
def load_prices(start: str, end: str) -> pd.DataFrame:
    return fetch_prices(start=start, end=end)


@st.cache_data(show_spinner=False)
def load_signals(prices_hash: str, prices: pd.DataFrame) -> dict:
    ratios = compute_ratios(prices)
    zscores = compute_zscores(ratios)
    directions = compute_directions(ratios)
    composite = compute_composite_signals(zscores, directions)
    return {
        "ratios": ratios,
        "zscores": zscores,
        "directions": directions,
        "composite": composite,
    }


@st.cache_data(show_spinner=False)
def load_hmm(features_hash: str, prices: pd.DataFrame, zscores: pd.DataFrame) -> dict:
    features = build_hmm_features(prices, zscores)
    result = run_walk_forward_hmm(features)
    gmm_model, gmm_labels = fit_gmm(features)
    return {
        "states": result.states,
        "probabilities": result.probabilities,
        "model": result.model,
        "features": features,
        "gmm_model": gmm_model,
        "gmm_labels": gmm_labels,
    }


@st.cache_data(show_spinner=False)
def load_quadrants(
    signals_hash: str,
    hmm_hash: str,
    growth: pd.Series,
    inflation: pd.Series,
    hmm_probabilities: pd.DataFrame,
    gmm_labels: pd.Series,
) -> dict:
    quadrant_labels = classify_quadrant(growth, inflation)
    weights = build_weight_series(quadrant_labels, hmm_probabilities)
    _, confusion_df = align_gmm_to_quadrants(gmm_labels, quadrant_labels)
    return {
        "quadrant_labels": quadrant_labels,
        "weights": weights,
        "confusion_df": confusion_df,
    }


@st.cache_data(show_spinner=False)
def load_backtest(
    weights_hash: str,
    prices_hash: str,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_freq: str,
) -> dict:
    bt_result = run_backtest(weights, prices, rebalance_freq=rebalance_freq)
    benchmarks_rets = build_benchmarks(prices)
    benchmark_metrics = {
        name: compute_metrics(ret)
        for name, ret in benchmarks_rets.items()
    }
    return {
        "bt_result": bt_result,
        "benchmarks_rets": benchmarks_rets,
        "benchmark_metrics": benchmark_metrics,
    }


# ──────────────────────────────────────────────
# Shared sidebar
# ──────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.title("Paramètres")
        period_label = st.selectbox("Période d'analyse", list(PERIOD_OPTIONS.keys()), index=3)
        rebalance_label = st.selectbox("Rebalancement", list(REBALANCE_OPTIONS.keys()), index=0)

        if st.button("Rafraîchir les données", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(
            "⚠️ **Avertissement** : Cet outil est fourni à des fins éducatives et de recherche "
            "uniquement. Il ne constitue pas un conseil en investissement. Les performances passées "
            "ne préjugent pas des performances futures."
        )

    years = PERIOD_OPTIONS[period_label]
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    rebalance_freq = REBALANCE_OPTIONS[rebalance_label]
    return start, today, rebalance_freq


def render_disclaimer():
    st.caption(
        "⚠️ *Ceci n'est pas un conseil en investissement. Les performances passées ne préjugent "
        "pas des performances futures.*"
    )


# ──────────────────────────────────────────────
# Page 1: Macro Dashboard
# ──────────────────────────────────────────────

def page_dashboard():
    st.title("📊 Dashboard Macro")
    start, today, rebalance_freq = render_sidebar()

    with st.spinner("Chargement des données de marché..."):
        prices = load_prices(start="2004-01-01", end=today)

    with st.spinner("Calcul des signaux macro..."):
        signals = load_signals(_df_hash(prices), prices)
        zscores = signals["zscores"]
        directions = signals["directions"]
        composite = signals["composite"]

    with st.spinner("Détection des régimes (HMM walk-forward)..."):
        hmm_data = load_hmm(_df_hash(prices), prices, zscores)

    hmm_states = hmm_data["states"]
    hmm_proba = hmm_data["probabilities"]
    quadrant_data = load_quadrants(
        _df_hash(zscores),
        _df_hash(hmm_proba),
        composite["growth_score"],
        composite["inflation_score"],
        hmm_proba,
        hmm_data["gmm_labels"],
    )
    quadrant_labels = quadrant_data["quadrant_labels"]

    # Current values
    last_valid_hmm = hmm_states.dropna()
    last_valid_quad = quadrant_labels.dropna()

    current_hmm_state = int(last_valid_hmm.iloc[-1]) if not last_valid_hmm.empty else 1
    current_quad = int(last_valid_quad.iloc[-1]) if not last_valid_quad.empty else 2
    current_proba = hmm_proba.dropna().iloc[-1].values if not hmm_proba.dropna().empty else [0.33, 0.34, 0.33]
    hmm_confidence = float(np.max(current_proba)) * 100

    current_alloc = get_allocation(current_quad, hmm_confidence / 100)

    # ── Header ──────────────────────────
    q_color = QUADRANT_COLORS.get(current_quad, "#888")
    regime_color = REGIME_COLORS.get(current_hmm_state, "#888")
    st.markdown(
        f"""
        <div style="display:flex; gap:16px; margin-bottom:16px;">
            <div style="background:{q_color}22; border:1.5px solid {q_color}; border-radius:8px;
                        padding:12px 20px; flex:1; text-align:center;">
                <div style="font-size:12px; color:#aaa;">QUADRANT MACRO</div>
                <div style="font-size:18px; font-weight:700; color:{q_color};">
                    Q{current_quad} — {QUADRANT_NAMES.get(current_quad, '?')}
                </div>
                <div style="font-size:11px; color:#ccc;">{QUADRANT_DESCRIPTIONS.get(current_quad, '')}</div>
            </div>
            <div style="background:{regime_color}22; border:1.5px solid {regime_color}; border-radius:8px;
                        padding:12px 20px; flex:1; text-align:center;">
                <div style="font-size:12px; color:#aaa;">RÉGIME HMM</div>
                <div style="font-size:18px; font-weight:700; color:{regime_color};">
                    {REGIME_LABELS.get(current_hmm_state, '?')}
                </div>
                <div style="font-size:11px; color:#ccc;">Confiance : {hmm_confidence:.0f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI metrics ──────────────────────────
    last_composite = composite.dropna()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cg_z = float(last_composite["copper_gold_z"].iloc[-1]) if not last_composite.empty else 0.0
        st.metric("Copper/Gold Z-Score", f"{cg_z:.2f}", help="Dr. Copper — expansion vs. contraction. >0 = signal croissance")
    with col2:
        so_z = float(last_composite["sp500_oil_z"].iloc[-1]) if not last_composite.empty else 0.0
        st.metric("SP500/Oil Z-Score", f"{so_z:.2f}", help="Risk-on vs. inflation énergie. >0 = favoris équités")
    with col3:
        vix_level = float(prices["^VIX"].dropna().iloc[-1]) if "^VIX" in prices.columns else 0.0
        vix_prev = float(prices["^VIX"].dropna().iloc[-2]) if len(prices["^VIX"].dropna()) > 1 else vix_level
        st.metric("VIX", f"{vix_level:.1f}", delta=f"{vix_level - vix_prev:+.1f}", delta_color="inverse",
                  help="Indice de volatilité. <18 = calme, >28 = crise")
    with col4:
        st.metric("Confiance HMM", f"{hmm_confidence:.0f}%",
                  help="Probabilité max du régime HMM actuel. <60% → allocation neutre activée")

    st.divider()

    # ── Composite signal chart ──────────────────────────
    st.subheader("Signaux macro normalisés")
    # Filter composite to selected period
    composite_period = composite.loc[start:] if start in composite.index or composite.index[0] <= pd.Timestamp(start) else composite
    quad_period = quadrant_labels.loc[start:] if not quadrant_labels.empty else quadrant_labels
    st.plotly_chart(plot_composite_signal(composite_period, quad_period), use_container_width=True)

    # ── Allocation ──────────────────────────
    st.divider()
    col_alloc, col_table = st.columns([1, 1])

    with col_alloc:
        st.subheader("Allocation recommandée")
        uncertain = hmm_confidence / 100 < 0.60
        if uncertain:
            st.warning("Confiance HMM < 60% — allocation neutre activée")

        labels = [k for k, v in current_alloc.items() if v > 0]
        values = [v for v in current_alloc.values() if v > 0]
        colors = ["#10B981", "#60A5FA", "#F59E0B", "#A78BFA", "#EF4444", "#FB923C"]

        pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=colors[:len(labels)]),
            textinfo="label+percent",
            textfont_size=12,
        ))
        pie.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(pie, use_container_width=True)

    with col_table:
        st.subheader("Poids par actif")
        alloc_df = pd.DataFrame([
            {"Actif": k, "Poids": f"{v * 100:.0f}%"}
            for k, v in current_alloc.items()
        ])
        st.dataframe(alloc_df, use_container_width=True, hide_index=True, height=250)

    render_disclaimer()


# ──────────────────────────────────────────────
# Page 2: Ratio Analysis
# ──────────────────────────────────────────────

def page_ratios():
    st.title("📈 Analyse des Ratios")
    start, today, _ = render_sidebar()

    with st.spinner("Chargement..."):
        prices = load_prices(start="2004-01-01", end=today)
        signals = load_signals(_df_hash(prices), prices)
        hmm_data = load_hmm(_df_hash(prices), prices, signals["zscores"])
        quadrant_data = load_quadrants(
            _df_hash(signals["zscores"]),
            _df_hash(hmm_data["probabilities"]),
            signals["composite"]["growth_score"],
            signals["composite"]["inflation_score"],
            hmm_data["probabilities"],
            hmm_data["gmm_labels"],
        )

    ratio_name = st.selectbox(
        "Sélectionner un ratio",
        options=list(RATIO_DEFINITIONS.keys()),
        format_func=lambda r: r.replace("_", "/"),
    )

    ratios_period = signals["ratios"].loc[start:]
    zscores_period = signals["zscores"].loc[start:]
    quad_period = quadrant_data["quadrant_labels"].loc[start:]

    current_z = float(zscores_period[ratio_name].dropna().iloc[-1]) if not zscores_period[ratio_name].dropna().empty else 0.0

    chart, description = plot_ratio_with_zscore(ratios_period, zscores_period, ratio_name, quad_period)
    st.plotly_chart(chart, use_container_width=True)

    with st.expander("Interprétation du ratio", expanded=True):
        st.markdown(auto_interpret_ratio(ratio_name, current_z))

    render_disclaimer()


# ──────────────────────────────────────────────
# Page 3: Historical Regimes
# ──────────────────────────────────────────────

def page_regimes():
    st.title("🗺️ Régimes Historiques")
    start, today, _ = render_sidebar()

    with st.spinner("Calcul des régimes historiques..."):
        prices = load_prices(start="2004-01-01", end=today)
        signals = load_signals(_df_hash(prices), prices)
        hmm_data = load_hmm(_df_hash(prices), prices, signals["zscores"])
        quadrant_data = load_quadrants(
            _df_hash(signals["zscores"]),
            _df_hash(hmm_data["probabilities"]),
            signals["composite"]["growth_score"],
            signals["composite"]["inflation_score"],
            hmm_data["probabilities"],
            hmm_data["gmm_labels"],
        )

    hmm_states = hmm_data["states"]
    quad_labels = quadrant_data["quadrant_labels"]

    # ── HMM Timeline ──────────────────────────
    st.subheader("Timeline des régimes HMM (2004–2026)")
    states_period = hmm_states.loc[start:]
    prices_period = prices.loc[start:]
    st.plotly_chart(plot_hmm_timeline(states_period, prices_period), use_container_width=True)

    # ── Regime stats ──────────────────────────
    st.subheader("Statistiques des régimes")
    stats_df = compute_regime_stats(hmm_states.dropna(), quad_labels.dropna())
    col1, col2 = st.columns(2)
    with col1:
        stats_display = stats_df[["% du temps", "Durée moy. (jours)", "Nb épisodes"]].copy()
        st.dataframe(stats_display, use_container_width=True)
    with col2:
        if hmm_data["model"] is not None:
            st.plotly_chart(plot_transition_matrix(hmm_data["model"]), use_container_width=True)

    # ── Returns heatmap by regime ──────────────────────────
    st.subheader("Rendements mensuels moyens par quadrant")
    st.plotly_chart(plot_regime_returns_heatmap(prices, quad_labels), use_container_width=True)

    # ── GMM vs Quadrant confusion ──────────────────────────
    st.subheader("Accord GMM vs. Classification quadrant")
    st.markdown(
        "Matrice de confusion entre le clustering non supervisé (GMM, 4 composantes) "
        "et la classification basée sur les règles macro. Un fort accord valide la robustesse des régimes détectés."
    )
    st.plotly_chart(plot_gmm_confusion(quadrant_data["confusion_df"]), use_container_width=True)

    render_disclaimer()


# ──────────────────────────────────────────────
# Page 4: Backtest
# ──────────────────────────────────────────────

def page_backtest():
    st.title("⚡ Backtest — Stratégie Régime vs. Benchmarks")
    start, today, rebalance_freq = render_sidebar()

    with st.spinner("Calcul du backtest..."):
        prices = load_prices(start="2004-01-01", end=today)
        signals = load_signals(_df_hash(prices), prices)
        hmm_data = load_hmm(_df_hash(prices), prices, signals["zscores"])
        quadrant_data = load_quadrants(
            _df_hash(signals["zscores"]),
            _df_hash(hmm_data["probabilities"]),
            signals["composite"]["growth_score"],
            signals["composite"]["inflation_score"],
            hmm_data["probabilities"],
            hmm_data["gmm_labels"],
        )
        bt_data = load_backtest(
            _df_hash(quadrant_data["weights"]),
            _df_hash(prices),
            quadrant_data["weights"],
            prices,
            rebalance_freq,
        )

    bt_result = bt_data["bt_result"]
    benchmarks_rets = bt_data["benchmarks_rets"]
    benchmark_metrics = bt_data["benchmark_metrics"]
    strategy_returns = bt_result["returns"]
    strategy_metrics = bt_result["metrics"]

    # ── KPI row ──────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CAGR (Stratégie)", f"{strategy_metrics.get('CAGR', 0)*100:.1f}%")
    with col2:
        st.metric("Sharpe", f"{strategy_metrics.get('Sharpe', 0):.2f}")
    with col3:
        st.metric("Max Drawdown", f"{strategy_metrics.get('Max Drawdown', 0)*100:.1f}%")
    with col4:
        st.metric("Calmar", f"{strategy_metrics.get('Calmar', 0):.2f}")

    # ── Equity curves ──────────────────────────
    st.subheader("Courbes de capital")
    st.plotly_chart(plot_equity_curves(strategy_returns, benchmarks_rets), use_container_width=True)

    # ── Annual returns ──────────────────────────
    st.plotly_chart(plot_annual_returns_bar(strategy_returns, benchmarks_rets), use_container_width=True)

    # ── Drawdown ──────────────────────────
    st.subheader("Drawdowns")
    st.plotly_chart(plot_drawdown(strategy_returns, benchmarks_rets), use_container_width=True)

    # ── Metrics table ──────────────────────────
    st.subheader("Tableau de métriques comparatives")
    st.plotly_chart(
        plot_metrics_table(strategy_metrics, benchmark_metrics),
        use_container_width=True,
    )

    # ── Advanced analysis ──────────────────────────
    with st.expander("Analyse de sensibilité (seuils de classification)"):
        st.markdown("Impact du seuil de classification des quadrants sur les métriques.")
        with st.spinner("Calcul de la sensibilité..."):
            try:
                sens_df = run_sensitivity_analysis(
                    prices,
                    signals["zscores"],
                    signals["directions"],
                    hmm_data["probabilities"],
                )
                st.plotly_chart(plot_sensitivity_heatmap(sens_df), use_container_width=True)
                st.dataframe(sens_df.style.format("{:.3f}"), use_container_width=True)
            except Exception as e:
                st.warning(f"Analyse de sensibilité impossible : {e}")

    with st.expander("Analyse par sous-période"):
        subperiod_df = run_subperiod_analysis(strategy_returns)
        if not subperiod_df.empty:
            st.dataframe(subperiod_df.style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("Données insuffisantes pour l'analyse par sous-période.")

    with st.expander("Intervalle de confiance du Sharpe (Bootstrap 1000 itérations)"):
        with st.spinner("Bootstrap..."):
            lo, hi = bootstrap_sharpe_ci(strategy_returns)
        st.metric(
            "Sharpe — IC 95%",
            f"[{lo:.2f}, {hi:.2f}]",
            help="Si l'intervalle inclut 0, le Sharpe n'est pas statistiquement significatif.",
        )
        if lo > 0:
            st.success("Le Sharpe est statistiquement positif au seuil 95%.")
        else:
            st.warning("L'intervalle de confiance inclut 0 — résultat potentiellement dû à la chance.")

    render_disclaimer()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Macro Regime System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pg = st.navigation([
        st.Page(page_dashboard, title="Dashboard Macro",    icon="📊", default=True),
        st.Page(page_ratios,    title="Analyse des Ratios", icon="📈"),
        st.Page(page_regimes,   title="Régimes Historiques", icon="🗺️"),
        st.Page(page_backtest,  title="Backtest",           icon="⚡"),
    ])
    pg.run()


if __name__ == "__main__":
    main()
