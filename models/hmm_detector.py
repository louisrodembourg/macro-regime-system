import logging
import warnings
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn import hmm as _hmmlearn_hmm
    _GaussianHMM = _hmmlearn_hmm.GaussianHMM
except Exception:
    # hmmlearn has compiled C extensions that break on some Python versions
    # (e.g. Python 3.14 on Streamlit Cloud). Fall back to pure-NumPy implementation.
    from models._hmm_numpy import GaussianHMM as _GaussianHMM
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.warning(
        "hmmlearn not available — using pure-NumPy GaussianHMM fallback"
    )

logger = logging.getLogger(__name__)

HMM_N_STATES = 3
REGIME_LABELS = {0: "Bull", 1: "Transition", 2: "Crisis"}
REGIME_COLORS = {0: "#10B981", 1: "#F59E0B", 2: "#EF4444"}


class HMMResult(NamedTuple):
    states: pd.Series           # semantic regime label (0/1/2) per date
    probabilities: pd.DataFrame # shape (n_dates, 3), columns = [0,1,2]
    model: Any                  # last fitted model (for transition matrix viz)
    state_order: np.ndarray     # relabeling mapping from last window


def build_hmm_features(
    prices: pd.DataFrame,
    zscores: pd.DataFrame,
) -> pd.DataFrame:
    """
    4-feature matrix for HMM:
      vix_log_change, sp500_20d_vol, copper_gold_zscore, hyg_tlt_zscore
    """
    vix_log_change = np.log(prices["^VIX"].replace(0, float("nan"))).diff()
    sp500_20d_vol = (
        prices["^GSPC"].pct_change()
        .rolling(20, min_periods=10)
        .std()
        * np.sqrt(252)
    )

    features = pd.DataFrame({
        "vix_log_change":     vix_log_change,
        "sp500_20d_vol":      sp500_20d_vol,
        "copper_gold_zscore": zscores["Copper_Gold"],
        "hyg_tlt_zscore":     zscores["HYG_TLT"],
    }, index=prices.index)

    return features


def _fit_hmm(X: np.ndarray, n_iter: int = 200) -> Any:
    model = _GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, lengths=[len(X)])
    if not model.monitor_.converged:
        logger.warning("HMM did not converge (n_iter=%d)", n_iter)
    return model


def _get_state_order(model: Any) -> np.ndarray:
    """Sort states by sp500_20d_vol mean (feature index 1) ascending.
    Returns state_order where state_order[semantic_label] = internal_label.
    Bull=0 (lowest vol), Transition=1, Crisis=2 (highest vol)."""
    vol_means = model.means_[:, 1]
    return np.argsort(vol_means)


def _relabel(
    raw_states: np.ndarray,
    raw_proba: np.ndarray,
    state_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map internal HMM state indices to semantic labels 0/1/2."""
    relabel_map = {old: new for new, old in enumerate(state_order)}
    labeled_states = np.array([relabel_map[s] for s in raw_states])
    # Reorder probability columns to semantic order
    labeled_proba = raw_proba[:, state_order]
    return labeled_states, labeled_proba


def _prep_X(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """Drop NaN rows and return C-contiguous float64 array + valid dates."""
    valid = df.dropna()
    X = np.ascontiguousarray(valid.values.astype(np.float64))
    return X, valid.index


def run_walk_forward_hmm(
    features: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 6,
) -> HMMResult:
    """
    Walk-forward HMM: fit on train window, predict on test window only.
    No lookahead bias — future data never used in model training.
    Returns HMMResult with states/probabilities aligned to features.index.
    """
    valid_index = features.dropna().index
    if len(valid_index) == 0:
        raise ValueError("No valid (non-NaN) rows in features DataFrame")

    data_start = valid_index[0]
    data_end = valid_index[-1]

    output_states = pd.Series(np.nan, index=features.index, dtype=float)
    output_proba = pd.DataFrame(
        np.nan,
        index=features.index,
        columns=[0, 1, 2],
    )

    last_model = None
    last_state_order = None

    # Generate walk-forward windows
    windows = []
    t = data_start
    while True:
        train_end = t + pd.DateOffset(years=train_years)
        test_end = t + pd.DateOffset(years=train_years + test_years)
        if train_end > data_end:
            break
        test_end = min(test_end, data_end)
        windows.append((t, train_end, test_end))
        t = t + pd.DateOffset(months=step_months)

    if not windows:
        raise ValueError("Not enough data for even one walk-forward window")

    logger.info("Running walk-forward HMM with %d windows", len(windows))

    for i, (train_start, train_end, test_end) in enumerate(windows):
        # Fit on train window only
        train_data = features.loc[train_start:train_end]
        X_train, train_dates = _prep_X(train_data)
        if len(X_train) < 50:
            logger.warning("Window %d: too few training samples (%d), skipping", i, len(X_train))
            continue

        model = _fit_hmm(X_train)
        state_order = _get_state_order(model)
        last_model = model
        last_state_order = state_order

        # For the first window, also backfill the training period (indicative only)
        if i == 0:
            raw_train = model.predict(X_train, lengths=[len(X_train)])
            raw_proba_train = model.predict_proba(X_train, lengths=[len(X_train)])
            labeled_train, labeled_proba_train = _relabel(raw_train, raw_proba_train, state_order)
            output_states.loc[train_dates] = labeled_train.astype(float)
            output_proba.loc[train_dates] = labeled_proba_train

        # Predict on test window only
        test_slice_start = train_end + pd.Timedelta(days=1)
        test_data = features.loc[test_slice_start:test_end]
        X_test, test_dates = _prep_X(test_data)
        if len(X_test) == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_test = model.predict(X_test, lengths=[len(X_test)])
            raw_proba_test = model.predict_proba(X_test, lengths=[len(X_test)])

        labeled_test, labeled_proba_test = _relabel(raw_test, raw_proba_test, state_order)
        output_states.loc[test_dates] = labeled_test.astype(float)
        output_proba.loc[test_dates] = labeled_proba_test

    # Convert to int where non-NaN
    valid_mask = output_states.notna()
    output_states[valid_mask] = output_states[valid_mask].astype(int)

    return HMMResult(
        states=output_states,
        probabilities=output_proba,
        model=last_model,
        state_order=last_state_order,
    )


def fit_gmm(
    features: pd.DataFrame,
    n_components: int = 4,
    random_state: int = 42,
) -> tuple[GaussianMixture, pd.Series]:
    """
    Fit GaussianMixture on feature matrix.
    Returns (fitted_gmm, cluster_labels_series) aligned to features.index.
    """
    X, valid_dates = _prep_X(features)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
    )
    gmm.fit(X)
    raw_labels = gmm.predict(X)
    label_series = pd.Series(np.nan, index=features.index, dtype=float)
    label_series.loc[valid_dates] = raw_labels.astype(float)
    return gmm, label_series
