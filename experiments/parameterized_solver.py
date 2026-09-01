"""Experiment 2 -- Parameterized PINN and amortized inference.

Instead of solving a single equation, train ONE PINN to represent a whole family of
solutions psi(x, lambda) and energies E(lambda) for the quartic oscillator
V(x; lambda) = 1/2 x^2 + lambda x^4, with lambda as an extra network input. After
training we evaluate at held-out lambda values (no supervised wavefunction data is
ever used) and compare against per-instance FDM solves.

Key question (amortization): total cost is
    T_FDM(N)  = N * t_fdm
    T_PINN(N) = t_train + N * t_infer
Is there a crossover N beyond which the trained PINN is cheaper? Reported honestly,
including "no crossover in the tested range" if that is what the data show.

Run:  python experiments/parameterized_solver.py
Outputs: results/parameterized/{param.csv, amortization.csv, param.json, *.png}
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import potentials as P
from src.fdm import solve_fdm
from src.evaluation import energy_error, wavefunction_l2_error
from src.utils import (set_seed, results_dir, save_csv, save_json, set_plot_style,
                       hardware_info, Timer, CB)


class ParamPINN(nn.Module):
    """psi(x, lambda) via one MLP. The energy E(lambda) is not a separate network:
    it is the variational Rayleigh quotient of psi(., lambda), a well-defined function
    of lambda that is read off after (and during) training. Minimizing the mean energy
    over lambda drives each slice to its ground state (Rayleigh-Ritz)."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.psi_net = nn.Sequential(*layers)

    def psi(self, X: torch.Tensor) -> torch.Tensor:      # X: (M, 2) = (x, lambda)
        return self.psi_net(X)


def V_quartic(x: torch.Tensor, lam: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    return 0.5 * omega**2 * x**2 + lam * x**4


def train_parametric(domain, lam_range, nx, nlam, width, depth, epochs, lr, seed,
                     device="cpu"):
    set_seed(seed)
    dev = torch.device(device)
    a, b = domain
    x_axis = torch.linspace(a, b, nx, device=dev)
    dx = (b - a) / (nx - 1)
    w = torch.full((nx,), dx, device=dev); w[0] *= 0.5; w[-1] *= 0.5
    lam_axis = torch.linspace(lam_range[0], lam_range[1], nlam, device=dev)

    # build (nlam*nx, 2) collocation grid, x fastest within each lambda block
    XX, LL = torch.meshgrid(x_axis, lam_axis, indexing="xy")   # (nlam, nx)
    X = torch.stack([XX.reshape(-1), LL.reshape(-1)], dim=1)
    X = X.clone().detach().requires_grad_(True)

    model = ParamPINN(width, depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"epoch": [], "loss": []}

    with Timer(dev) as timer:
        for epoch in range(epochs):
            opt.zero_grad()
            psi = model.psi(X)                                  # (nlam*nx, 1)
            psi_x = torch.autograd.grad(psi, X, torch.ones_like(psi),
                                        create_graph=True)[0][:, 0:1]
            Vx = V_quartic(X[:, 0:1], X[:, 1:2])
            psi_g = psi.view(nlam, nx)
            psix_g = psi_x.view(nlam, nx)
            V_g = Vx.view(nlam, nx)
            kinetic = 0.5 * torch.sum(w * psix_g**2, dim=1)     # (nlam,)
            potential = torch.sum(w * V_g * psi_g**2, dim=1)
            norm = torch.sum(w * psi_g**2, dim=1)
            E_var = (kinetic + potential) / norm                # variational E(lambda)
            loss_energy = torch.mean(E_var)
            loss_norm = torch.mean((norm - 1.0) ** 2)
            loss_bc = torch.mean(psi_g[:, 0] ** 2 + psi_g[:, -1] ** 2)
            loss = loss_energy + 5.0 * loss_norm + 1.0 * loss_bc
            loss.backward(); opt.step(); sched.step()
            if epoch % 50 == 0:
                hist["epoch"].append(epoch); hist["loss"].append(float(loss.detach()))

    return model, timer.elapsed, hist, (x_axis.cpu().numpy(), w.cpu().numpy())


@torch.no_grad()
def eval_parametric(model, x_axis, lam, device="cpu"):
    """Return (E(lambda), normalized psi) at a given lambda via the Rayleigh quotient."""
    dev = torch.device(device)
    x = torch.tensor(x_axis, dtype=torch.float32, device=dev).view(-1, 1)
    lamc = torch.full_like(x, float(lam))
    psi = model.psi(torch.cat([x, lamc], dim=1)).cpu().numpy().reshape(-1)
    norm = np.sqrt(np.trapezoid(psi**2, x_axis))
    psi_n = psi / (norm + 1e-300)
    psi_x = np.gradient(psi_n, x_axis)
    V = 0.5 * x_axis**2 + lam * x_axis**4
    kinetic = 0.5 * np.trapezoid(psi_x**2, x_axis)
    potential = np.trapezoid(V * psi_n**2, x_axis)
    E = float(kinetic + potential)
    return E, psi_n


def main(args):
    set_plot_style()
    outdir = results_dir("parameterized")
    domain = (-6.0, 6.0)

    model, t_train, hist, (x_axis, w) = train_parametric(
        domain, (args.lam_min, args.lam_max), args.nx, args.nlam,
        args.width, args.depth, args.epochs, args.lr, args.seed)
    print(f"[parameterized] trained psi(x,lambda) in {t_train:.1f}s")

    # held-out test lambdas (midpoints between training nodes)
    train_lams = np.linspace(args.lam_min, args.lam_max, args.nlam)
    test_lams = 0.5 * (train_lams[:-1] + train_lams[1:])
    fine = np.linspace(domain[0], domain[1], 800)

    rows = []
    t_fdm_list = []
    for lam in test_lams:
        sysm = P.anharmonic_oscillator(lam=float(lam), domain=domain)
        # best-of-5 FDM solve time (one full eigen-solve per instance)
        tfd = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            fdm = solve_fdm(sysm, n_per_dim=args.fdm_n, k=1)
            tfd = min(tfd, time.perf_counter() - t0)
        t_fdm_list.append(tfd)
        E_fdm = float(fdm.energies[0])
        psi_fdm = np.interp(fine, fdm.axis, fdm.wavefunctions[0])
        E_pinn, psi_pinn = eval_parametric(model, fine, lam)
        rows.append({
            "lambda": float(lam), "E_fdm": E_fdm, "E_pinn": E_pinn,
            "energy_error": energy_error(E_pinn, E_fdm),
            "wavefn_l2_error": wavefunction_l2_error(psi_pinn, psi_fdm, fine),
        })
    save_csv(rows, os.path.join(outdir, "param.csv"))

    # clean single-instance inference time: best-of forward pass on the same grid size
    dev = torch.device("cpu")
    Xq = torch.rand(fine.size, 2, device=dev)
    t_inf = float("inf")
    for _ in range(10):
        with Timer(dev) as tq, torch.no_grad():
            model.psi(Xq)     # one forward pass = evaluate psi(.,lambda) on the grid
        t_inf = min(t_inf, tq.elapsed)
    t_fdm = float(np.mean(t_fdm_list))
    Ns = [1, 10, 100, 1000, 10000]
    amort = []
    crossover = None
    for N in Ns:
        T_fdm = N * t_fdm
        T_pinn = t_train + N * t_inf
        amort.append({"N_problems": N, "T_fdm_s": T_fdm, "T_pinn_s": T_pinn})
        if crossover is None and T_pinn < T_fdm:
            crossover = N
    # exact crossover (continuous) if PINN inference is cheaper per-solve
    cross_cont = (t_train / (t_fdm - t_inf)) if t_fdm > t_inf else float("inf")
    save_csv(amort, os.path.join(outdir, "amortization.csv"))
    save_json({"hardware": hardware_info(), "t_train_s": t_train,
               "t_fdm_single_s": t_fdm, "t_infer_single_s": t_inf,
               "crossover_N_discrete": crossover,
               "crossover_N_continuous": cross_cont,
               "budget": vars(args)}, os.path.join(outdir, "param.json"))

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    lams = [r["lambda"] for r in rows]
    ax[0].plot(lams, [r["E_fdm"] for r in rows], "o-", color=CB["blue"], label="FDM")
    ax[0].plot(lams, [r["E_pinn"] for r in rows], "s--", color=CB["red"], label="PINN")
    ax[0].set_xlabel(r"$\lambda$"); ax[0].set_ylabel("ground-state energy $E_0(\\lambda)$")
    ax[0].set_title("Energy vs quartic strength (held-out $\\lambda$)"); ax[0].legend()

    ax[1].semilogy(lams, [r["energy_error"] for r in rows], "o-", color=CB["purple"],
                   label="energy error")
    ax[1].semilogy(lams, [r["wavefn_l2_error"] for r in rows], "s-", color=CB["green"],
                   label="wavefn $L_2$ error")
    ax[1].set_xlabel(r"$\lambda$"); ax[1].set_ylabel("error vs FDM (log)")
    ax[1].set_title("PINN accuracy at held-out $\\lambda$"); ax[1].legend()

    ax[2].loglog([a["N_problems"] for a in amort], [a["T_fdm_s"] for a in amort],
                 "o-", color=CB["blue"], label="FDM: $N\\,t_{fdm}$")
    ax[2].loglog([a["N_problems"] for a in amort], [a["T_pinn_s"] for a in amort],
                 "s-", color=CB["red"], label="PINN: $t_{train}+N\\,t_{infer}$")
    if np.isfinite(cross_cont):
        ax[2].axvline(cross_cont, color="gray", ls="--")
        ax[2].text(cross_cont, min(a["T_pinn_s"] for a in amort),
                   f"  crossover\n  N$\\approx${cross_cont:.0f}", fontsize=9, va="bottom")
    ax[2].set_xlabel("number of PDE instances $N$"); ax[2].set_ylabel("total time (s, log)")
    ax[2].set_title("Amortization: total cost vs #instances"); ax[2].legend(fontsize=9)

    fig.suptitle("Parameterized PINN over the quartic-oscillator family", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "parameterized.png"))

    print(f"[parameterized] mean energy error {np.mean([r['energy_error'] for r in rows]):.2e}, "
          f"t_train={t_train:.1f}s t_fdm={t_fdm*1e3:.2f}ms t_infer={t_inf*1e3:.2f}ms")
    print(f"[parameterized] amortization crossover N ~ "
          f"{cross_cont:.0f} instances" if np.isfinite(cross_cont)
          else "[parameterized] no crossover (PINN inference not cheaper than FDM)")
    print(f"[parameterized] wrote results to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parameterized PINN + amortization")
    ap.add_argument("--lam_min", type=float, default=0.0)
    ap.add_argument("--lam_max", type=float, default=0.5)
    ap.add_argument("--nx", type=int, default=192)
    ap.add_argument("--nlam", type=int, default=21)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fdm_n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
