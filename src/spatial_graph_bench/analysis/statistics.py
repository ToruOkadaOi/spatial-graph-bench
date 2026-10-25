"""Paired seed inference with explicit global multiple-testing correction."""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return monotone Holm adjusted p-values in the original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def paired_tests(lifts: list[float], epsilon: float) -> dict:
    """Test superiority/inferiority beyond ±epsilon and TOST inside the band."""
    a = np.asarray(lifts, dtype=float)
    if len(a) < 2 or not np.all(np.isfinite(a)) or epsilon <= 0:
        raise ValueError("At least two finite paired lifts and positive epsilon are required")
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    se = sd / np.sqrt(len(a))
    if se == 0:
        # Degenerate empirical distribution; boundary hypotheses are not rejected.
        p_sup = 0.0 if mean > epsilon else 1.0
        p_inf = 0.0 if mean < -epsilon else 1.0
        p_tost = 0.0 if -epsilon < mean < epsilon else 1.0
        low = high = mean
    else:
        lower = (mean + epsilon) / se
        upper = (mean - epsilon) / se
        p_sup = float(stats.t.sf(upper, len(a) - 1))
        p_inf = float(stats.t.cdf(lower, len(a) - 1))
        p_tost = float(max(stats.t.sf(lower, len(a) - 1), stats.t.cdf(upper, len(a) - 1)))
        radius = float(stats.t.ppf(0.95, len(a) - 1) * se)
        low, high = mean - radius, mean + radius
    return {
        "n": len(a),
        "mean_lift": mean,
        "std_lift": sd,
        "ci_90_low": low,
        "ci_90_high": high,
        "p_sup": p_sup,
        "p_inf": p_inf,
        "p_tost": p_tost,
    }


def correct_grid(rows: list[dict]) -> None:
    """One retrospective family: all three hypotheses in every dataset/configuration.

    This conservative 336-test family includes secondary topologies. It is an
    explicit analysis amendment, not a claim that this rule was pre-registered.
    """
    keys = ("p_sup", "p_inf", "p_tost")
    adjusted = holm_adjust([r[k] for r in rows for k in keys])
    for i, r in enumerate(rows):
        for j, key in enumerate(keys):
            r[key + "_holm"] = adjusted[i * 3 + j]
        r["family_size"] = len(adjusted)
        if r["p_sup_holm"] <= 0.05:
            r["decision"] = "POSITIVE LIFT"
        elif r["p_inf_holm"] <= 0.05:
            r["decision"] = "NEGATIVE LIFT"
        elif r["p_tost_holm"] <= 0.05:
            r["decision"] = "EQUIVALENT"
        else:
            r["decision"] = "INCONCLUSIVE"
