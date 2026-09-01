"""Experiment 1 -- Dimensional scaling (the original hypothesis test).

Compares the finite-difference method (tensor-product grid, N = n^d points) against
the PINN as the dimensionality d grows. Measures runtime, memory, degrees of freedom
and accuracy for both, at clearly reported computational budgets.

The purpose is NOT to make the PINN look good: it is to test whether avoiding an
explicit n^d grid actually yields a computational advantage, and at what cost in
accuracy. FDM is expected to win at low d and to hit an exponential wall; the PINN
avoids the grid but its optimization becomes hard and unreliable in high d.

Run:  python experiments/dimensional_scaling.py
Outputs: results/scaling/{fdm_scaling.csv, pinn_scaling.csv, scaling.json, scaling.png}
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
from src.training import PINNConfig, train_pinn, inference_time
from src.evaluation import energy_error, summarize_seeds
from src.utils import (results_dir, save_csv, save_json, set_plot_style,
                       hardware_info, CB)


def fdm_scaling(dims, n_fixed, n_feasible):
    """Fixed n points per axis so the grid N = n^d grows exactly as the curse of
    dimensionality dictates. Dimensions whose grid exceeds `n_feasible` are not run;
    their grid size and memory footprint are projected instead, marking the wall."""
    rows = []
    for d in dims:
        n_grid = n_fixed ** d
        system = P.nd_harmonic_oscillator(d, domain=(-6, 6))
        exact = system.analytic_energy(0)      # ground state = d/2
        if n_grid <= n_feasible:
            r = solve_fdm(system, n_per_dim=n_fixed, k=1)
            rows.append({
                "dim": d, "n_per_dim": n_fixed, "n_grid": r.n_grid, "feasible": True,
                "runtime_s": r.runtime, "memory_MB": r.memory_bytes / 1e6,
                "E0": float(r.energies[0]), "E0_exact": exact,
                "energy_error": energy_error(r.energies[0], exact),
            })
            print(f"  [FDM] d={d} n={n_fixed} N={r.n_grid:.2e} t={r.runtime:.2f}s "
                  f"mem={r.memory_bytes/1e6:.1f}MB Eerr={rows[-1]['energy_error']:.2e}")
        else:
            # project storage: sparse Hamiltonian ~ (2d+1) nonzeros/row, 16 bytes each
            proj_mem = n_grid * (2 * d + 1) * 16 / 1e6
            rows.append({
                "dim": d, "n_per_dim": n_fixed, "n_grid": n_grid, "feasible": False,
                "runtime_s": float("nan"), "memory_MB": proj_mem,
                "E0": float("nan"), "E0_exact": exact, "energy_error": float("nan"),
            })
            print(f"  [FDM] d={d} n={n_fixed} N={n_grid:.2e} INFEASIBLE "
                  f"(projected mem {proj_mem/1e3:.1f} GB)")
    return rows


def pinn_scaling(dims, epochs, n_coll, seeds):
    rows = []
    for d in dims:
        system = P.nd_harmonic_oscillator(d, domain=(-6, 6))
        exact = system.analytic_energy(0)
        errs, ttimes, itimes, params = [], [], [], 0
        for s in seeds:
            cfg = PINNConfig(dim=d, epochs=epochs, lr=3e-3, n_collocation=n_coll,
                             integration="mc", w_norm=1.0, seed=s)
            r = train_pinn(system, cfg)
            errs.append(energy_error(r.energy, exact))
            ttimes.append(r.train_time)
            itimes.append(inference_time(r.model, n_points=10000))
            params = r.n_params
        es = summarize_seeds(errs)
        rows.append({
            "dim": d, "n_collocation": n_coll, "n_params": params, "epochs": epochs,
            "train_time_mean_s": float(np.mean(ttimes)),
            "inference_time_s": float(np.min(itimes)),
            "energy_error_mean": es["mean"], "energy_error_std": es["std"],
            "energy_error_min": es["min"], "n_seeds": len(seeds),
        })
        print(f"  [PINN] d={d} params={params} t_train={np.mean(ttimes):.1f}s "
              f"Eerr={es['mean']:.2e}+/-{es['std']:.1e}")
    return rows


def main(args):
    set_plot_style()
    outdir = results_dir("scaling")
    seeds = list(range(args.seeds))

    print("[scaling] FDM tensor-product grid:")
    fdm_rows = fdm_scaling(args.fdm_dims, args.n_fixed, args.n_feasible)
    print("[scaling] PINN:")
    pinn_rows = pinn_scaling(args.pinn_dims, args.epochs, args.collocation, seeds)

    save_csv(fdm_rows, os.path.join(outdir, "fdm_scaling.csv"))
    save_csv(pinn_rows, os.path.join(outdir, "pinn_scaling.csv"))
    save_json({"hardware": hardware_info(), "seeds": seeds,
               "fdm_budget": {"n_fixed": args.n_fixed, "n_feasible": args.n_feasible},
               "pinn_budget": {"epochs": args.epochs, "collocation": args.collocation}},
              os.path.join(outdir, "scaling.json"))

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    feas = [r for r in fdm_rows if r["feasible"]]
    fdf = [r["dim"] for r in feas]
    pd_ = [r["dim"] for r in pinn_rows]

    # (a) cost vs dimension (only measured/feasible FDM points)
    ax[0].semilogy(fdf, [r["runtime_s"] for r in feas], "o-", color=CB["blue"],
                   label="FDM eigensolve")
    ax[0].semilogy(pd_, [r["train_time_mean_s"] for r in pinn_rows], "s-", color=CB["red"],
                   label="PINN training")
    ax[0].semilogy(pd_, [r["inference_time_s"] for r in pinn_rows], "^--", color=CB["orange"],
                   label="PINN inference")
    ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel("wall-clock time (s, log)")
    ax[0].set_title(f"Computational cost vs dimension  (FDM n={args.n_fixed}/axis)")
    ax[0].legend(fontsize=9)

    # (b) degrees of freedom / memory vs dimension (all dims: measured + projected)
    allf = [r["dim"] for r in fdm_rows]
    ax[1].semilogy(allf, [r["n_grid"] for r in fdm_rows], "o-", color=CB["blue"],
                   label=r"FDM grid points $N=n^d$")
    infeas = [r for r in fdm_rows if not r["feasible"]]
    if infeas:
        ax[1].semilogy([r["dim"] for r in infeas], [r["n_grid"] for r in infeas],
                       "x", color=CB["blue"], ms=11, mew=2, label="FDM (infeasible, projected)")
    ax[1].semilogy(pd_, [r["n_params"] for r in pinn_rows], "s-", color=CB["red"],
                   label="PINN parameters")
    ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel("degrees of freedom (log)")
    ax[1].set_title("Representation size: $n^d$ grid vs PINN parameters")
    ax[1].legend(fontsize=9)

    # (c) accuracy vs dimension
    ax[2].semilogy(fdf, [max(r["energy_error"], 1e-12) for r in feas], "o-",
                   color=CB["blue"], label="FDM")
    pe = [r["energy_error_mean"] for r in pinn_rows]
    ps = [r["energy_error_std"] for r in pinn_rows]
    ax[2].errorbar(pd_, pe, yerr=ps, fmt="s-", color=CB["red"], capsize=3,
                   label=f"PINN (mean$\\pm$std, {len(seeds)} seeds)")
    ax[2].set_xlabel("dimension $d$"); ax[2].set_ylabel("ground-state energy error (log)")
    ax[2].set_title("Accuracy vs dimension"); ax[2].legend(fontsize=9)

    fig.suptitle("Dimensional scaling: FDM tensor grid vs PINN (isotropic harmonic oscillator)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scaling.png"))
    print(f"[scaling] wrote results to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dimensional scaling: FDM vs PINN")
    ap.add_argument("--fdm_dims", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pinn_dims", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--collocation", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_fixed", type=int, default=40)
    ap.add_argument("--n_feasible", type=int, default=150000)
    main(ap.parse_args())
