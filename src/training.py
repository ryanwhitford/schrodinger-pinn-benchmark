"""Training loop and configuration for the Schrodinger PINN.

Handles collocation sampling, the composite loss (PDE residual + normalization +
boundary condition + orthogonality), optimization, history logging and CUDA-aware
timing. Designed to be reused unchanged across the dimensional-scaling, parameterized
and inverse experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .pinn import SchrodingerPINN, pde_residual, laplacian, domain_volume
from .potentials import Potential
from .utils import Timer, set_seed, get_device


@dataclass
class PINNConfig:
    """Hyperparameters for a single PINN training run."""

    dim: int = 1
    width: int = 64
    depth: int = 3
    activation: str = "tanh"
    lr: float = 3e-3
    epochs: int = 3000
    n_collocation: int = 512
    n_boundary: int = 64
    w_pde: float = 1.0
    w_norm: float = 1.0
    w_bc: float = 1.0
    w_ortho: float = 20.0
    e_init: float = 1.0
    integration: str = "auto"      # 'grid' (deterministic, 1D), 'mc', or 'auto'
    resample: bool = False         # if MC, resample the collocation set each epoch
    seed: int = 0
    record_every: int = 50
    device: str = "cpu"


@dataclass
class PINNResult:
    model: SchrodingerPINN
    energy: float
    history: dict
    train_time: float
    n_params: int
    config: PINNConfig


def _resolve_integration(mode: str, dim: int) -> str:
    """'auto' -> deterministic grid quadrature in 1D, Monte-Carlo otherwise."""
    if mode != "auto":
        return mode
    return "grid" if dim == 1 else "mc"


def _build_collocation(domain: tuple[float, float], dim: int, n: int, mode: str,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (points, quadrature weights) for evaluating spatial integrals.

    grid: uniform 1D grid with trapezoidal weights (deterministic, low variance).
    mc:   uniform samples with weight vol/N (needed when a grid is infeasible).
    """
    a, b = float(domain[0]), float(domain[1])
    if mode == "grid":
        assert dim == 1, "grid integration is only implemented for dim=1"
        axis = torch.linspace(a, b, n, device=device).view(-1, 1)
        x = axis.clone().detach().requires_grad_(True)
        dx = (b - a) / (n - 1)
        w = torch.full((n, 1), dx, device=device)
        w[0] *= 0.5
        w[-1] *= 0.5
    elif mode == "gaussian":
        # importance sampling from an isotropic Gaussian proposal centered in the box.
        # Weights 1/(N p(x)) give an unbiased integral estimate concentrated where the
        # bound-state wavefunction has support (uniform grids waste points in high d).
        sigma = (b - a) / 6.0
        x_np = sigma * torch.randn(n, dim, device=device)
        x = x_np.clone().detach().requires_grad_(True)
        logp = (-0.5 * (x_np / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))).sum(dim=1, keepdim=True)
        w = torch.exp(-logp) / n
    else:  # uniform Monte Carlo
        x = (a + (b - a) * torch.rand(n, dim, device=device)).clone().detach().requires_grad_(True)
        vol = (b - a) ** dim
        w = torch.full((n, 1), vol / n, device=device)
    return x, w


def _sample_interior(domain: tuple[float, float], dim: int, n: int,
                     device: torch.device) -> torch.Tensor:
    a, b = float(domain[0]), float(domain[1])
    x = a + (b - a) * torch.rand(n, dim, device=device)
    return x.clone().detach().requires_grad_(True)


def _sample_boundary(domain: tuple[float, float], dim: int, n: int,
                     device: torch.device) -> torch.Tensor:
    """Sample points on the faces of the box [a, b]^dim (for the psi=0 BC)."""
    a, b = float(domain[0]), float(domain[1])
    x = a + (b - a) * torch.rand(n, dim, device=device)
    # for each point pin a random axis to a random face (a or b)
    axes = torch.randint(0, dim, (n,), device=device)
    faces = torch.where(torch.rand(n, device=device) < 0.5, a, b).to(x.dtype)
    x[torch.arange(n, device=device), axes] = faces
    return x


def train_pinn(potential: Potential, config: PINNConfig,
               prev_models: Optional[list[SchrodingerPINN]] = None,
               verbose: bool = False) -> PINNResult:
    """Train a PINN for one eigenstate of `potential`.

    Args:
        potential: system to solve (provides V, dim, domain).
        config: hyperparameters.
        prev_models: previously converged states to be orthogonal to (excited states).
        verbose: print progress.
    """
    set_seed(config.seed)
    device = get_device() if config.device == "auto" else torch.device(config.device)
    dim = potential.dim
    domain = potential.domain
    mode = _resolve_integration(config.integration, dim)
    prev_models = prev_models or []
    for m in prev_models:
        for p in m.parameters():
            p.requires_grad_(False)

    model = SchrodingerPINN(dim, config.width, config.depth,
                            config.activation, config.e_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.epochs)
    hist = {"epoch": [], "energy": [], "loss": [], "loss_pde": []}

    x, w = _build_collocation(domain, dim, config.n_collocation, mode, device)
    xb = _sample_boundary(domain, dim, config.n_boundary, device)
    # cache normalized reference wavefunctions of previously found states on x
    prev_psi = []
    for pm in prev_models:
        with torch.no_grad():
            pp = pm(x)
            pp = pp / torch.sqrt(torch.sum(w * pp**2) + 1e-12)
        prev_psi.append(pp)

    with Timer(device) as timer:
        for epoch in range(config.epochs):
            if config.resample and mode == "mc":
                x, w = _build_collocation(domain, dim, config.n_collocation, mode, device)
                xb = _sample_boundary(domain, dim, config.n_boundary, device)
                prev_psi = []
                for pm in prev_models:
                    with torch.no_grad():
                        pp = pm(x)
                        pp = pp / torch.sqrt(torch.sum(w * pp**2) + 1e-12)
                    prev_psi.append(pp)
            opt.zero_grad()
            R, psi = pde_residual(model, x, potential.V)
            loss_pde = torch.mean(R**2)
            norm = torch.sum(w * psi**2)                       # <psi|psi>
            loss_norm = (norm - 1.0) ** 2
            psi_b = model(xb)
            loss_bc = torch.mean(psi_b**2)
            loss = (config.w_pde * loss_pde + config.w_norm * loss_norm
                    + config.w_bc * loss_bc)
            for pp in prev_psi:                                 # orthogonality
                overlap = torch.sum(w * psi * pp)
                loss = loss + config.w_ortho * overlap**2 / (norm + 1e-12)
            loss.backward()
            opt.step()
            sched.step()
            if epoch % config.record_every == 0:
                hist["epoch"].append(epoch)
                hist["energy"].append(float(model.E.detach()))
                hist["loss"].append(float(loss.detach()))
                hist["loss_pde"].append(float(loss_pde.detach()))
            if verbose and epoch % max(1, config.epochs // 5) == 0:
                print(f"    epoch {epoch:5d}  E={float(model.E.detach()):+.5f}  "
                      f"loss={float(loss.detach()):.3e}")

    n_params = sum(p.numel() for p in model.parameters())
    return PINNResult(model=model, energy=float(model.E.detach()), history=hist,
                      train_time=timer.elapsed, n_params=n_params, config=config)


def solve_states_pinn(potential: Potential, config: PINNConfig, n_states: int = 1,
                      verbose: bool = False) -> list[PINNResult]:
    """Solve the lowest `n_states` eigenstates sequentially via orthogonality."""
    results: list[PINNResult] = []
    prev: list[SchrodingerPINN] = []
    for k in range(n_states):
        cfg = PINNConfig(**{**config.__dict__, "seed": config.seed + k})
        res = train_pinn(potential, cfg, prev_models=prev, verbose=verbose)
        results.append(res)
        prev = prev + [res.model]
        if verbose:
            print(f"  state {k}: E = {res.energy:+.5f}")
    return results


@torch.no_grad()
def evaluate_on_grid(model: SchrodingerPINN, axis: np.ndarray,
                     domain: tuple[float, float]) -> np.ndarray:
    """Evaluate a trained 1D PINN on a coordinate axis and normalize it."""
    device = next(model.parameters()).device
    x = torch.tensor(axis.reshape(-1, 1), dtype=torch.float32, device=device)
    psi = model(x).cpu().numpy().reshape(-1)
    norm = np.sqrt(np.trapezoid(psi**2, axis))
    return psi / (norm + 1e-300)


def inference_time(model: SchrodingerPINN, n_points: int = 10000,
                   repeats: int = 5) -> float:
    """Best-of wall time to evaluate the network at n_points (forward pass)."""
    device = next(model.parameters()).device
    x = torch.rand(n_points, model.dim, device=device)
    best = float("inf")
    for _ in range(repeats):
        with Timer(device) as t, torch.no_grad():
            model(x)
        best = min(best, t.elapsed)
    return best
