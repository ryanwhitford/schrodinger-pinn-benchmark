"""Experiment 3 -- Inverse problem: recover an unknown potential from sparse data.

Given only a handful of noisy samples of a ground-state wavefunction, reconstruct the
potential V(x) that produced it. Two networks are learned jointly: psi_theta(x) fitting
the data and V_phi(x) as the unknown potential, tied together by the Schrodinger
residual. This is a setting where the physics constraint does real work -- ordinary
interpolation of a few points cannot recover a potential, but the PDE can.

The true system is V(x) = 1/2 x^2 + lambda x^4; "observations" come from the FDM
ground state. We sweep the number of measurements N_data in {10, 25, 50, 100}.

Note: an eigen-problem fixes V only up to an additive constant (V -> V+c shifts E by c),
and V is only constrained where |psi| is appreciable -- errors are reported on the
physically supported region and the potential is mean-aligned before comparison.

Run:  python experiments/inverse_problem.py
Outputs: results/inverse/{inverse.csv, inverse.json, inverse.png}
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import potentials as P
from src.fdm import solve_fdm
from src.evaluation import relative_potential_error, wavefunction_l2_error
from src.utils import (set_seed, results_dir, save_csv, save_json, set_plot_style,
                       hardware_info, Timer, CB)


def mlp(width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


class InversePINN(nn.Module):
    """psi_theta(x) fits data; V_phi(x) is the reconstructed potential; E is scalar."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        self.psi_net = mlp(width, depth)
        self.V_net = mlp(width, depth)
        self.E = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.psi_net(x)


def train_inverse(x_data, psi_data, domain, n_coll, width, depth, epochs, lr,
                  w_data, w_pde, w_norm, w_bc, w_smooth, seed, device="cpu"):
    set_seed(seed)
    dev = torch.device(device)
    a, b = domain
    xd = torch.tensor(x_data.reshape(-1, 1), dtype=torch.float32, device=dev)
    pd = torch.tensor(psi_data.reshape(-1, 1), dtype=torch.float32, device=dev)
    axis = torch.linspace(a, b, n_coll, device=dev).view(-1, 1)
    xc = axis.clone().detach().requires_grad_(True)
    dx = (b - a) / (n_coll - 1)
    w = torch.full((n_coll, 1), dx, device=dev); w[0] *= 0.5; w[-1] *= 0.5

    model = InversePINN(width, depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    with Timer(dev) as timer:
        for epoch in range(epochs):
            opt.zero_grad()
            # data term
            loss_data = torch.mean((model.psi_net(xd) - pd) ** 2)
            # physics residual on collocation grid
            psi = model.psi_net(xc)
            psi_x = torch.autograd.grad(psi, xc, torch.ones_like(psi), create_graph=True)[0]
            psi_xx = torch.autograd.grad(psi_x, xc, torch.ones_like(psi_x), create_graph=True)[0]
            V = model.V_net(xc)
            R = -0.5 * psi_xx + V * psi - model.E * psi
            loss_pde = torch.mean(R**2)
            norm = torch.sum(w * psi**2)
            loss_norm = (norm - 1.0) ** 2
            loss_bc = psi[0, 0] ** 2 + psi[-1, 0] ** 2
            # mild smoothness prior on V (second derivative) to tame the unconstrained tails
            Vx = torch.autograd.grad(V, xc, torch.ones_like(V), create_graph=True)[0]
            Vxx = torch.autograd.grad(Vx, xc, torch.ones_like(Vx), create_graph=True)[0]
            loss_smooth = torch.mean(Vxx**2)
            loss = (w_data * loss_data + w_pde * loss_pde + w_norm * loss_norm
                    + w_bc * loss_bc + w_smooth * loss_smooth)
            loss.backward(); opt.step(); sched.step()

    return model, timer.elapsed


@torch.no_grad()
def eval_inverse(model, axis, device="cpu"):
    dev = torch.device(device)
    x = torch.tensor(axis.reshape(-1, 1), dtype=torch.float32, device=dev)
    psi = model.psi_net(x).cpu().numpy().reshape(-1)
    V = model.V_net(x).cpu().numpy().reshape(-1)
    return psi, V


def main(args):
    set_plot_style()
    outdir = results_dir("inverse")
    domain = (-6.0, 6.0)
    lam = args.lam

    # ground-truth system and its FDM ground state
    sysm = P.anharmonic_oscillator(lam=lam, domain=domain)
    fdm = solve_fdm(sysm, n_per_dim=1000, k=1)
    axis = fdm.axis
    psi_true = fdm.wavefunctions[0]
    psi_true *= np.sign(psi_true[len(psi_true) // 2] + 1e-12)   # make positive bump
    V_true = 0.5 * axis**2 + lam * axis**4

    # region where the wavefunction has appreciable support (V observable there)
    support = psi_true**2 > 1e-2 * (psi_true**2).max()
    weight = psi_true**2                    # physical weighting: know V where |psi|^2 lives

    def weighted_V_error(V_pred):
        # potentials are defined up to an additive constant -> align the weighted mean
        shift = np.trapezoid((V_true - V_pred) * weight, axis) / np.trapezoid(weight, axis)
        num = np.sqrt(np.trapezoid((V_pred + shift - V_true) ** 2 * weight, axis))
        den = np.sqrt(np.trapezoid(V_true ** 2 * weight, axis))
        return float(num / (den + 1e-300))

    seeds = list(range(args.seeds))
    rows = []
    recon = {}   # N_data -> (psi_pred, V_pred) for best seed (plotting)
    for N in args.n_data:
        errsV, errsPsi = [], []
        best = None
        for s in seeds:
            rng = np.random.default_rng(s)
            # sample measurement locations preferentially where psi is supported
            probs = psi_true**2 / psi_true.dot(psi_true)
            idx = rng.choice(len(axis), size=N, replace=False, p=probs)
            x_data = axis[idx]
            psi_data = psi_true[idx] + args.noise * rng.standard_normal(N)

            model, _ = train_inverse(
                x_data, psi_data, domain, args.n_coll, args.width, args.depth,
                args.epochs, args.lr, args.w_data, args.w_pde, args.w_norm,
                args.w_bc, args.w_smooth, seed=s)
            psi_pred, V_pred = eval_inverse(model, axis)
            eV = weighted_V_error(V_pred)
            ePsi = wavefunction_l2_error(psi_pred, psi_true, axis)
            errsV.append(eV); errsPsi.append(ePsi)
            if best is None or eV <= min(errsV):
                best = (psi_pred, V_pred, x_data, psi_data)
        recon[N] = best
        rows.append({
            "n_data": N, "V_rel_error_mean": float(np.mean(errsV)),
            "V_rel_error_std": float(np.std(errsV)),
            "V_rel_error_min": float(np.min(errsV)),
            "psi_l2_error_mean": float(np.mean(errsPsi)), "n_seeds": len(seeds),
        })
        print(f"  [inverse] N_data={N:3d}  V_rel_err={np.mean(errsV):.2e}"
              f"+/-{np.std(errsV):.1e}  psi_L2={np.mean(errsPsi):.2e}")

    save_csv(rows, os.path.join(outdir, "inverse.csv"))
    save_json({"hardware": hardware_info(), "lambda": lam, "seeds": seeds,
               "budget": vars(args)}, os.path.join(outdir, "inverse.json"))

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # (a) reconstructed potential for the largest N_data
    Nbig = args.n_data[-1]
    psi_pred, V_pred, x_data, psi_data = recon[Nbig]
    shift = np.trapezoid(V_true[support] - V_pred[support], axis[support]) / (
        axis[support][-1] - axis[support][0])
    ax[0].plot(axis, V_true, color=CB["black"], lw=2, label="true $V(x)$")
    ax[0].plot(axis, V_pred + shift, "--", color=CB["red"], lw=2, label="reconstructed")
    ax[0].set_ylim(-1, V_true[support].max() * 1.2)
    ax[0].set_xlim(-5, 5)
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$V(x)$")
    ax[0].set_title(f"Recovered potential ($N_{{data}}={Nbig}$)"); ax[0].legend()

    # (b) wavefunction + observations
    ax[1].plot(axis, psi_true, color=CB["black"], lw=2, label="true $\\psi$ (FDM)")
    ax[1].plot(axis, psi_pred * np.sign(psi_pred[len(psi_pred)//2]), "--",
               color=CB["red"], lw=2, label="PINN $\\psi$")
    ax[1].scatter(x_data, psi_data * np.sign(psi_data.mean() + 1e-9), s=25,
                  color=CB["blue"], zorder=3, label=f"observations ($N={Nbig}$)")
    ax[1].set_xlim(-5, 5); ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$\\psi(x)$")
    ax[1].set_title("Wavefunction fit"); ax[1].legend()

    # (c) error vs number of measurements
    Ns = [r["n_data"] for r in rows]
    ax[2].errorbar(Ns, [r["V_rel_error_mean"] for r in rows],
                   yerr=[r["V_rel_error_std"] for r in rows], fmt="o-",
                   color=CB["purple"], capsize=3, label="V relative error")
    ax[2].set_xlabel("number of measurements $N_{data}$")
    ax[2].set_ylabel("relative $V$ error (supported region)")
    ax[2].set_title("Reconstruction quality vs data"); ax[2].legend()

    fig.suptitle("Inverse problem: recovering an unknown potential from sparse data",
                 fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "inverse.png"))
    print(f"[inverse] wrote results to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inverse problem: potential reconstruction")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--n_data", type=int, nargs="+", default=[10, 25, 50, 100])
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--n_coll", type=int, default=500)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w_data", type=float, default=100.0)
    ap.add_argument("--w_pde", type=float, default=1.0)
    ap.add_argument("--w_norm", type=float, default=1.0)
    ap.add_argument("--w_bc", type=float, default=1.0)
    ap.add_argument("--w_smooth", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args())
