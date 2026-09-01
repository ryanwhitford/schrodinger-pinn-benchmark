"""Physics-informed neural network for the time-independent Schrodinger equation.

The network maps coordinates to a scalar wavefunction, x -> psi_theta(x), and the
energy eigenvalue E is a *trainable* parameter (the analytic energy is never used
during training). The loss is the PDE residual of

    R(x) = -1/2 laplacian(psi) + V(x) psi - E psi,

plus normalization, boundary-condition and (for excited states) orthogonality terms.

Note on the residual formulation: for a fixed psi the residual-minimizing E equals the
Rayleigh quotient <psi|H|psi>/<psi|psi>, so E tracks the current state's energy. Which
eigenstate is found depends on initialization; a smooth small-weight initialization
biases the network toward the nodeless ground state. Excited states are targeted by
penalizing overlap with previously converged states -- an approach whose limitations
(sensitivity to weights, imperfect orthogonality) are documented in the experiments.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

_ACTIVATIONS = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU, "sin": None}


class Sine(nn.Module):
    """Sine activation (used optionally for smooth high-frequency states)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SchrodingerPINN(nn.Module):
    """MLP wavefunction psi_theta(x) with a trainable energy eigenvalue E."""

    def __init__(self, dim: int = 1, width: int = 64, depth: int = 3,
                 activation: str = "tanh", e_init: float = 1.0):
        super().__init__()
        self.dim = dim
        if activation == "sin":
            Act: Callable[[], nn.Module] = Sine
        else:
            Act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = [nn.Linear(dim, width), Act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), Act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.E = nn.Parameter(torch.tensor([float(e_init)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def laplacian(psi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sum of unmixed second derivatives  sum_i d^2 psi / d x_i^2  via autograd.

    Args:
        psi: (N, 1) network output, built from x with create_graph enabled.
        x:   (N, dim) input coordinates with requires_grad=True.
    Returns:
        (N, 1) Laplacian of psi.
    """
    grad = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    lap = torch.zeros_like(psi)
    for i in range(x.shape[1]):
        gi = grad[:, i:i + 1]
        d2 = torch.autograd.grad(gi, x, torch.ones_like(gi), create_graph=True)[0][:, i:i + 1]
        lap = lap + d2
    return lap


def pde_residual(model: SchrodingerPINN, x: torch.Tensor,
                 V: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (residual R(x), psi) for the stationary Schrodinger equation."""
    psi = model(x)
    lap = laplacian(psi, x)
    Vx = V(x)
    R = -0.5 * lap + Vx * psi - model.E * psi
    return R, psi


def domain_volume(domain: tuple[float, float], dim: int) -> float:
    """Volume of the box domain [a, b]^dim."""
    a, b = domain
    return (b - a) ** dim
