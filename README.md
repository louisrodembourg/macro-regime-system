# Macro Regime Detection & Tactical Asset Allocation System

A Streamlit application that detects macroeconomic regimes from inter-asset ratios and generates tactical asset allocation recommendations, backed by a walk-forward validated backtest (2007–2026).

## Installation

```bash
cd macro_regime_system
pip install -r requirements.txt
```

> Note: `hmmlearn` requires a C compiler. On macOS: `xcode-select --install`. On Linux: `apt install build-essential`.

## Launch

```bash
streamlit run app.py
```

The app will fetch ~22 years of financial data on first launch (cached locally for 24h). The HMM walk-forward computation takes ~30–60s on first run and is then cached.

## Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard Macro | Regime badge, 4 KPI metrics, composite ratio signal chart, allocation pie chart |
| 📈 Analyse des Ratios | 3-panel ratio chart (raw / z-score / Bollinger), auto-interpretation |
| 🗺️ Régimes Historiques | HMM timeline, returns heatmap by regime, transition matrix, GMM comparison |
| ⚡ Backtest | Equity curves, drawdowns, metrics table vs 3 benchmarks, sensitivity analysis, bootstrap Sharpe CI |

## Architecture

```
data/fetcher.py              # yfinance + pickle cache
models/ratios.py             # 7 inter-asset ratios, z-scores, directions
models/hmm_detector.py       # GaussianHMM walk-forward + GMM clustering
models/quadrant_classifier.py # 4-quadrant classification (Bridgewater framework)
models/allocator.py          # Allocation tables per regime
backtest/engine.py           # Vectorized backtest, metrics, benchmarks, robustness tests
visualizations/              # Plotly chart factories (dark mode)
app.py                       # Streamlit multi-page app
```

## Theoretical Framework

### The 4 Macro Quadrants (Bridgewater/Dalio)

| Quadrant | Growth | Inflation | Favored Assets |
|----------|--------|-----------|----------------|
| Q1 — Expansion inflationniste | ↗ | ↗ | Commodities, Energy, Gold, Value stocks |
| Q2 — Expansion saine | ↗ | ↘ | Tech/Growth stocks, Bonds |
| Q3 — Stagflation | ↘ | ↗ | Gold, Energy — avoid Tech & TLT |
| Q4 — Récession/Déflation | ↘ | ↘ | TLT, Gold, Cash |

### Regime Detection

- **HMM (3 states)**: GaussianHMM fitted on VIX log-change, SP500 20d vol, Copper/Gold z-score, HYG/TLT z-score. States ordered by volatility: Bull (0) / Transition (1) / Crisis (2).
- **Walk-forward**: 5-year train window, 1-year test window, 6-month step — no lookahead bias.
- **GMM (4 clusters)**: Compared against quadrant classification using the Hungarian algorithm.
- **Uncertainty**: If HMM max probability < 60%, neutral allocation (33% SPY / 34% TLT / 33% GLD) is used.

### Backtest Methodology

- **Period**: 2007-04-11 → today (limited by HYG availability)
- **Rebalancing**: Monthly (last business day)
- **Transaction costs**: 10 bps per rebalance
- **Lookahead protection**: Signals shifted by 1 day before backtest
- **Benchmarks**: SPY B&H, 60/40 (SPY/TLT), Equal Weight

### Robustness Tests

- Sensitivity analysis: threshold variation [0.3, 0.5, 0.7, 1.0, 1.5]
- Sub-period analysis: GFC / Recovery / Late Cycle / Post-COVID
- Bootstrap Sharpe CI: 1000 resamples, 95% confidence interval

## Disclaimer

This tool is for educational and research purposes only. It does not constitute investment advice. Past performance is not indicative of future results.
