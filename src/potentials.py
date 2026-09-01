"""Potentials and analytic references for the time-independent Schrodinger equation.

Dimensionless units: hbar = m = 1. The stationary equation solved throughout is

    -1/2 laplacian(psi) + V(x) psi = E psi.

Each `Potential` bundles the potential callable (acting on a torch tensor of shape
(N, dim)), the spatial domain, the dimensionality, and -- where they exist -- the
closed-form energies and wavefunctions used for validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, pi, sqrt
from typing import Callable, Optional

import numpy as np
import torch
from numpy.polynomial.hermite import hermval


@dataclass
class Potential:
    """A quantum system: potential, domain, dimensionality, optional analytics."""

    name: str
    V: Callable[[torch.Tensor], torch.Tensor]      # V(x): (N, dim) -> (N, 1)
    dim: int = 1
    domain: tuple[float, float] = (-6.0, 6.0)        # per-axis box [a, b]^dim
    analytic_energy: Optional[Callable[[int], float]] = None      # E(n)
    analytic_wavefn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1D infinite square well on [a, b], width L = b - a
#   E_n = (n+1)^2 pi^2 / (2 L^2)          (n = 0, 1, 2, ...; ground state n=0)
#   psi_n(x) = sqrt(2/L) sin((n+1) pi (x-a)/L)
# --------------------------------------------------------------------------
def infinite_square_well(a: float = 0.0, b: float = 1.0) -> Potential:
    L = b - a

    def V(x: torch.Tensor) -> torch.Tensor:
        # zero inside the well; boundary conditions imposed by the domain/BC loss
        return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)

    def energy(n: int) -> float:
        return (n + 1) ** 2 * pi**2 / (2.0 * L**2)

    def wavefn(x: np.ndarray, n: int) -> np.ndarray:
        psi = np.sqrt(2.0 / L) * np.sin((n + 1) * pi * (x - a) / L)
        psi = np.where((x > a) & (x < b), psi, 0.0)
        return psi

    return Potential("Infinite Square Well", V, dim=1, domain=(a, b),
                     analytic_energy=energy, analytic_wavefn=wavefn,
                     params={"a": a, "b": b, "L": L})


# --------------------------------------------------------------------------
# 1D harmonic oscillator  V = 1/2 omega^2 x^2
#   E_n = omega (n + 1/2)
#   psi_n = (omega/pi)^{1/4} / sqrt(2^n n!) H_n(sqrt(omega) x) exp(-omega x^2/2)
# --------------------------------------------------------------------------
def harmonic_oscillator(omega: float = 1.0, domain: tuple[float, float] = (-6.0, 6.0)) -> Potential:
    def V(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * omega**2 * x**2).sum(dim=1, keepdim=True)

    def energy(n: int) -> float:
        return omega * (n + 0.5)

    def wavefn(x: np.ndarray, n: int) -> np.ndarray:
        xi = sqrt(omega) * x
        Hn = hermval(xi, [0] * n + [1])
        norm = (omega / pi) ** 0.25 / sqrt(2.0**n * factorial(n))
        return norm * Hn * np.exp(-omega * x**2 / 2.0)

    return Potential("Harmonic Oscillator", V, dim=1, domain=domain,
                     analytic_energy=energy, analytic_wavefn=wavefn,
                     params={"omega": omega})


# --------------------------------------------------------------------------
# 1D anharmonic / quartic oscillator  V = 1/2 omega^2 x^2 + lambda x^4
# No closed form -> validated against the FDM reference solver.
# --------------------------------------------------------------------------
def anharmonic_oscillator(omega: float = 1.0, lam: float = 0.1,
                          domain: tuple[float, float] = (-6.0, 6.0)) -> Potential:
    def V(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * omega**2 * x**2 + lam * x**4).sum(dim=1, keepdim=True)

    return Potential("Anharmonic (quartic) Oscillator", V, dim=1, domain=domain,
                     analytic_energy=None, analytic_wavefn=None,
                     params={"omega": omega, "lambda": lam})


def anharmonic_potential_lambda(x: torch.Tensor, lam: float, omega: float = 1.0) -> torch.Tensor:
    """Parameterized quartic potential V(x; lambda) used by the parameterized PINN."""
    return 0.5 * omega**2 * x**2 + lam * x**4


# --------------------------------------------------------------------------
# d-dimensional isotropic harmonic oscillator  V = 1/2 omega^2 sum_i x_i^2
#   Separable: E = omega (sum_i n_i + d/2); ground-state energy = omega d/2.
# --------------------------------------------------------------------------
def nd_harmonic_oscillator(dim: int, omega: float = 1.0,
                           domain: tuple[float, float] = (-6.0, 6.0)) -> Potential:
    def V(x: torch.Tensor) -> torch.Tensor:
        return (0.5 * omega**2 * x**2).sum(dim=1, keepdim=True)

    def ground_energy(n: int = 0) -> float:
        # ground state (all n_i = 0): E0 = omega * d / 2
        return omega * dim / 2.0

    return Potential(f"{dim}D Harmonic Oscillator", V, dim=dim, domain=domain,
                     analytic_energy=ground_energy, analytic_wavefn=None,
                     params={"omega": omega, "dim": dim})
