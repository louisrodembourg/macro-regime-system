import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

QUADRANT_NAMES = {
    1: "Growth + Inflation",
    2: "Growth + Deflation",
    3: "Stagflation",
    4: "Recession",
}

QUADRANT_COLORS = {
    1: "#F59E0B",
    2: "#10B981",
    3: "#EF4444",
    4: "#6366F1",
}

QUADRANT_DESCRIPTIONS = {
    1: "Expansion inflationniste — favoriser commodités, énergie, actions value",
    2: "Expansion saine — risk-on pur, surpondérer tech/growth et actions",
    3: "Stagflation — le pire régime, or dominant, fuir tech et obligations longues",
    4: "Récession/Déflation — flight to quality, TLT et or",
}


def classify_quadrant(
    growth_score: pd.Series,
    inflation_score: pd.Series,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Assigns macro quadrant 1–4 based on growth and inflation scores.
    threshold enables sensitivity analysis.
    """
    conditions = [
        (growth_score > threshold) & (inflation_score > threshold),   # Q1
        (growth_score > threshold) & (inflation_score <= threshold),   # Q2
        (growth_score <= threshold) & (inflation_score > threshold),   # Q3
        (growth_score <= threshold) & (inflation_score <= threshold),  # Q4
    ]
    choices = [1, 2, 3, 4]

    result = pd.Series(np.nan, index=growth_score.index, dtype=float)
    for cond, val in zip(conditions, choices):
        valid = cond & growth_score.notna() & inflation_score.notna()
        result[valid] = val

    return result


def align_gmm_to_quadrants(
    gmm_labels: pd.Series,
    quadrant_labels: pd.Series,
    n_components: int = 4,
) -> tuple[dict[int, int], pd.DataFrame]:
    """
    Aligns GMM cluster labels to quadrant labels using the Hungarian algorithm.
    Returns (label_mapping, confusion_matrix_df).
    """
    # Align on common valid dates
    common = gmm_labels.dropna().index.intersection(quadrant_labels.dropna().index)
    if len(common) == 0:
        empty_cm = pd.DataFrame(
            0,
            index=[f"GMM {i}" for i in range(n_components)],
            columns=[QUADRANT_NAMES[q] for q in sorted(QUADRANT_NAMES)],
        )
        return {}, empty_cm

    gmm_arr = gmm_labels.loc[common].astype(int).values
    quad_arr = quadrant_labels.loc[common].astype(int).values

    # Build confusion matrix: rows = GMM clusters, cols = quadrants (1-4)
    quad_labels_sorted = sorted(QUADRANT_NAMES.keys())
    gmm_clusters = list(range(n_components))

    cm = np.zeros((n_components, len(quad_labels_sorted)), dtype=int)
    for gi in gmm_clusters:
        for qi, q in enumerate(quad_labels_sorted):
            cm[gi, qi] = np.sum((gmm_arr == gi) & (quad_arr == q))

    # Hungarian algorithm: maximize agreement → minimize negative cm
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {int(r): int(quad_labels_sorted[c]) for r, c in zip(row_ind, col_ind)}

    cm_df = pd.DataFrame(
        cm,
        index=[f"GMM {i}" for i in gmm_clusters],
        columns=[QUADRANT_NAMES[q] for q in quad_labels_sorted],
    )
    return label_mapping, cm_df
