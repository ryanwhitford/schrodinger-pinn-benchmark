"""Error metrics and comparison helpers for validating and benchmarking solvers.

All wavefunction comparisons remove the global sign ambiguity before measuring error.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import align_sign, normalize_1d


def energy_error(e_numerical: float, e_exact: float) -> float:
    """Absolute energy error |E_num - E_exact|."""
    return float(abs(e_numerical - e_exact))


def wavefunction_l2_error(psi: np.ndarray, reference: np.ndarray, x: np.ndarray) -> float:
    """L2 error sqrt(int |psi - ref|^2 dx) after normalization and sign alignment."""
    p = normalize_1d(psi, x)
    r = normalize_1d(reference, x)
    p = align_sign(p, r)
    return float(np.sqrt(np.trapezoid((p - r) ** 2, x)))


def normalization_error(psi: np.ndarray, x: np.ndarray) -> float:
    """|<psi|psi> - 1| on a 1D grid."""
    return float(abs(np.trapezoid(psi**2, x) - 1.0))


def relative_potential_error(v_pred: np.ndarray, v_true: np.ndarray,
                             x: np.ndarray) -> float:
    """Relative L2 error of a reconstructed potential (used in the inverse problem).

    Potentials recovered from an eigenproblem are only defined up to an additive
    constant (V -> V + c shifts every energy by c), so we align the mean before
    comparing.
    """
    shift = np.trapezoid(v_true - v_pred, x) / (x[-1] - x[0])
    v = v_pred + shift
    num = np.sqrt(np.trapezoid((v - v_true) ** 2, x))
    den = np.sqrt(np.trapezoid(v_true**2, x))
    return float(num / (den + 1e-300))


def summarize_seeds(values: list[float]) -> dict:
    """Mean/std/min/max summary for multi-seed robustness reporting."""
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "n": len(a)}
