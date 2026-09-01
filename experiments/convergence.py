"""Experiment 4 -- Convergence studies.

FDM error vs grid spacing dx, and PINN error vs collocation count, network width,
depth and training iterations. Comparing methods at arbitrary settings is not
meaningful, so this experiment maps out how each method converges.

Run:  python experiments/convergence.py
Outputs: results/convergence/{convergence.csv, convergence.json, convergence.png}
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import potentials as P
from src.fdm import solve_fdm
from src.training import PINNConfig, train_pinn, evaluate_on_grid
from src.evaluation import energy_error
from src.utils import (set_seed, results_dir, save_csv, save_json, set_plot_style,
                       hardware_info, CB)


def fdm_convergence(system, n_list, k=1):
    exact = system.analytic_energy(0)
    rows = []
    for n in n_list:
        r = solve_fdm(system, n_per_dim=n, k=k)
        rows.append({"n": n, "dx": r.dx, "E0": float(r.energies[0]),
                     "energy_error": energy_error(r.energies[0], exact),
                     "runtime_s": r.runtime})
    return rows


def pinn_convergence(system, axis_setting, values, base, seeds):
    """Sweep one PINN hyperparameter; return mean/std energy error over seeds."""
    exact = system.analytic_energy(0)
    rows = []
    for v in values:
        errs, times = [], []
        for s in seeds:
            kw = dict(base)
            kw[axis_setting] = v
            kw["seed"] = s
            cfg = PINNConfig(dim=1, **kw)
            r = train_pinn(system, cfg)
            errs.append(energy_error(r.energy, exact))
            times.append(r.train_time)
        errs = np.array(errs)
        rows.append({"setting": axis_setting, "value": v,
                     "energy_error_mean": float(errs.mean()),
                     "energy_error_std": float(errs.std()),
                     "energy_error_min": float(errs.min()),
                     "runtime_mean_s": float(np.mean(times)), "n_seeds": len(seeds)})
    return rows


def main(args):
    set_plot_style()
    set_seed(0)
    outdir = results_dir("convergence")
    ho = P.harmonic_oscillator(domain=(-6, 6))
    seeds = list(range(args.seeds))

    # --- FDM: error vs dx ---
    fdm_rows = fdm_convergence(ho, args.fdm_n)

    # --- PINN sweeps (hold others at base, vary one) ---
    # base is a well-behaved config; the width sweep separately shows that naively
    # enlarging the network (width 64+) *hurts* at a fixed training budget.
    base = dict(width=32, depth=3, epochs=args.epochs, lr=5e-3,
                n_collocation=512, activation="tanh")
    coll_rows = pinn_convergence(ho, "n_collocation", args.collocation, base, seeds)
    width_rows = pinn_convergence(ho, "width", args.widths, base, seeds)
    depth_rows = pinn_convergence(ho, "depth", args.depths, base, seeds)
    iter_rows = pinn_convergence(ho, "epochs", args.iters, base, seeds)

    # --- save CSVs ---
    save_csv(fdm_rows, os.path.join(outdir, "fdm_convergence.csv"))
    save_csv(coll_rows + width_rows + depth_rows + iter_rows,
             os.path.join(outdir, "pinn_convergence.csv"))
    save_json({"hardware": hardware_info(), "seeds": seeds,
               "base_pinn": base}, os.path.join(outdir, "convergence.json"))

    # --- plots ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    dx = [r["dx"] for r in fdm_rows]
    fe = [r["energy_error"] for r in fdm_rows]
    ax[0, 0].loglog(dx, fe, "o-", color=CB["blue"], label="FDM")
    ref = np.array(dx, float)
    ax[0, 0].loglog(dx, fe[-1] * (ref / ref[-1]) ** 2, "k--", lw=1, label=r"$\propto \Delta x^2$")
    ax[0, 0].set_xlabel(r"grid spacing $\Delta x$"); ax[0, 0].set_ylabel("energy error")
    ax[0, 0].set_title("FDM convergence (2nd order)"); ax[0, 0].legend()

    def errbar(ax, rows, xkey, title, xlabel, logx=True):
        xs = [r["value"] for r in rows]
        m = [r["energy_error_mean"] for r in rows]
        sd = [r["energy_error_std"] for r in rows]
        ax.errorbar(xs, m, yerr=sd, fmt="o-", color=CB["red"], capsize=3)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("energy error"); ax.set_title(title)

    errbar(ax[0, 1], coll_rows, "value", "PINN vs collocation points", "collocation points")
    errbar(ax[0, 2], iter_rows, "value", "PINN vs training iterations", "epochs")
    errbar(ax[1, 0], width_rows, "value", "PINN vs network width", "width")
    errbar(ax[1, 1], depth_rows, "value", "PINN vs network depth", "hidden layers", logx=False)

    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.9, "Convergence summary", fontsize=12, weight="bold")
    txt = (f"FDM: error ~ dx^2 (2nd order), reaches\n"
           f"  {min(fe):.1e} at dx={dx[np.argmin(fe)]:.1e}\n\n"
           f"PINN (mean over {len(seeds)} seeds):\n"
           f"  best collocation error {min(r['energy_error_mean'] for r in coll_rows):.1e}\n"
           f"  best iterations error  {min(r['energy_error_mean'] for r in iter_rows):.1e}\n\n"
           "PINN error saturates (optimization/\n"
           "stochastic floor); FDM keeps improving\n"
           "with resolution until round-off.")
    ax[1, 2].text(0.0, 0.05, txt, fontsize=10, va="bottom", family="monospace")

    fig.suptitle("Convergence: FDM vs PINN (1D harmonic oscillator)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "convergence.png"))
    print(f"[convergence] wrote results to {outdir}")
    print(f"  FDM best energy error: {min(fe):.2e}")
    print(f"  PINN best energy error (collocation sweep): "
          f"{min(r['energy_error_mean'] for r in coll_rows):.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FDM/PINN convergence study")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fdm_n", type=int, nargs="+",
                    default=[50, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--collocation", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512])
    ap.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--iters", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    main(ap.parse_args())
