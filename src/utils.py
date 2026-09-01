"""Shared utilities: reproducibility, timing, memory, integration, I/O, plotting.

All physics uses dimensionless (natural) units with hbar = m = 1.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------
# Reproducibility & hardware
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device if available and requested, else CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hardware_info() -> dict:
    """Record hardware/software details for reproducibility manifests."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# --------------------------------------------------------------------------
# Timing (CUDA-aware)
# --------------------------------------------------------------------------
def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize CUDA before/after timing so elapsed time is meaningful."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Context manager returning wall-clock seconds, CUDA-synchronized."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.device)
        self.elapsed = time.perf_counter() - self._t0


def best_of(fn: Callable[[], object], repeats: int = 3) -> tuple[object, float]:
    """Run fn repeats times; return (last_output, best/min wall time in s)."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


# --------------------------------------------------------------------------
# Numerical helpers
# --------------------------------------------------------------------------
def trapezoid_weights(n: int, dx: float) -> np.ndarray:
    """1D trapezoidal quadrature weights on a uniform grid of n points."""
    w = np.full(n, dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def normalize_1d(psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalize a real 1D wavefunction so that integral |psi|^2 dx = 1."""
    norm = np.sqrt(np.trapezoid(psi**2, x))
    return psi / (norm + 1e-300)


def align_sign(psi: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove the global +/- sign ambiguity relative to a reference vector."""
    return psi * np.sign(np.sum(psi * reference) + 1e-30)


def l2_error(psi: np.ndarray, reference: np.ndarray, x: np.ndarray) -> float:
    """L2 wavefunction error after sign alignment: sqrt(int |psi-ref|^2 dx)."""
    p = align_sign(normalize_1d(psi, x), normalize_1d(reference, x))
    r = normalize_1d(reference, x)
    return float(np.sqrt(np.trapezoid((p - r) ** 2, x)))


def normalization_error_1d(psi: np.ndarray, x: np.ndarray) -> float:
    """|<psi|psi> - 1| on a 1D grid."""
    return float(abs(np.trapezoid(psi**2, x) - 1.0))


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def project_root() -> str:
    """Absolute path to the repository root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_dir(name: str) -> str:
    """Absolute path to results/<name>/, created if missing."""
    d = os.path.join(project_root(), "results", name)
    os.makedirs(d, exist_ok=True)
    return d


def save_csv(rows: list[dict], path: str) -> None:
    """Write a list of flat dicts to CSV (keys of the first row are columns)."""
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: dict, path: str) -> None:
    """Pretty-print a dict to JSON (used for hyperparameter/hardware manifests)."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# --------------------------------------------------------------------------
# Plotting style (publication-quality defaults, colorblind-friendly)
# --------------------------------------------------------------------------
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}


def set_plot_style() -> None:
    """Apply a clean, readable Matplotlib style used across all figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": "white",
    })
