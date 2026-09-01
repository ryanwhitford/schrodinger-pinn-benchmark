"""Generate the three project notebooks. Run from the repo root: python notebooks/build_notebooks.py"""
import nbformat as nbf
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def nb_from(cells):
    nb = nbf.v4.new_notebook()
    out = []
    for kind, src in cells:
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                   else nbf.v4.new_code_cell(src))
    nb["cells"] = out
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}}
    return nb


BOOT = (
    "import sys, os\n"
    "sys.path.insert(0, os.path.abspath('..'))  # repo root, so `import src` works\n"
    "import numpy as np, matplotlib.pyplot as plt\n"
    "from src.utils import set_plot_style, CB\n"
    "set_plot_style()"
)

# ==========================================================================
# 01 -- FDM vs PINN validation
# ==========================================================================
nb1 = nb_from([
    ("md", "# 01 · FDM vs PINN on the 1D harmonic oscillator\n\n"
           "A side-by-side validation of the two solvers against the analytic spectrum "
           "$E_n=n+\\tfrac12$. This notebook runs live (a few seconds for FDM, ~15 s for a "
           "short PINN). It establishes that both solvers are correct before any benchmarking."),
    ("code", BOOT),
    ("code",
     "from src import potentials as P\n"
     "from src.fdm import solve_fdm\n"
     "from src.training import PINNConfig, solve_states_pinn, evaluate_on_grid\n"
     "from src.evaluation import energy_error, wavefunction_l2_error, normalization_error\n\n"
     "ho = P.harmonic_oscillator(domain=(-6, 6))\n"
     "axis = np.linspace(-6, 6, 1000)\n"
     "exact = np.array([ho.analytic_energy(n) for n in range(4)])\n"
     "exact"),
    ("md", "## Finite differences\nBuild the tridiagonal Hamiltonian and diagonalize."),
    ("code",
     "fdm = solve_fdm(ho, n_per_dim=800, k=4)\n"
     "print('FDM energies:', np.round(fdm.energies, 5))\n"
     "print('exact       :', exact)\n"
     "print('max |error| :', np.abs(fdm.energies - exact).max())\n"
     "print('runtime     : %.2f ms' % (fdm.runtime*1e3))"),
    ("md", "## PINN (variational energy + orthogonality)\n"
           "We solve the ground state and first two excited states. The third excited state "
           "is where the vanilla orthogonality ladder becomes unreliable (see the scaling / "
           "robustness discussion) so we validate n = 0, 1, 2 here."),
    ("code",
     "cfg = PINNConfig(dim=1, epochs=2000, lr=5e-3, n_collocation=512, w_ortho=20.0, seed=0)\n"
     "states = solve_states_pinn(ho, cfg, n_states=3, verbose=False)\n"
     "E_pinn = np.array([s.energy for s in states])\n"
     "for n, s in enumerate(states):\n"
     "    psi = evaluate_on_grid(s.model, axis, (-6, 6))\n"
     "    ref = ho.analytic_wavefn(axis, n)\n"
     "    print(f'n={n}  E={s.energy:+.5f} (exact {exact[n]:.1f})  '\n"
     "          f'Eerr={energy_error(s.energy, exact[n]):.2e}  '\n"
     "          f'L2={wavefunction_l2_error(psi, ref, axis):.2e}')"),
    ("md", "## Wavefunctions: analytic vs FDM vs PINN"),
    ("code",
     "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
     "for n in range(3):\n"
     "    ref = ho.analytic_wavefn(axis, n)\n"
     "    fref = ho.analytic_wavefn(fdm.axis, n)\n"
     "    fpsi = fdm.wavefunctions[n] * np.sign((fdm.wavefunctions[n]*fref).sum())\n"
     "    ppsi = evaluate_on_grid(states[n].model, axis, (-6, 6))\n"
     "    ppsi *= np.sign((ppsi*ref).sum())\n"
     "    ax = axes[n]\n"
     "    ax.plot(axis, ref, color=CB['black'], lw=2.2, label='analytic')\n"
     "    ax.plot(fdm.axis, fpsi, color=CB['blue'], lw=1.3, ls='--', label='FDM')\n"
     "    ax.plot(axis, ppsi, color=CB['red'], lw=1.3, ls=':', label='PINN')\n"
     "    ax.set_title(f'n={n}'); ax.set_xlabel('x'); ax.set_xlim(-6, 6)\n"
     "    if n == 0: ax.set_ylabel(r'$\\psi_n(x)$'); ax.legend()\n"
     "fig.suptitle('Harmonic-oscillator eigenfunctions: analytic vs FDM vs PINN')\n"
     "fig.tight_layout(); plt.show()"),
    ("md", "**Takeaway.** Both solvers reproduce the analytic states. FDM reaches ~$10^{-4}$ "
           "energy error in milliseconds; the PINN reaches ~$10^{-3}$ in seconds. In 1D the "
           "finite-difference method is decisively better -- see notebook 02 for how this "
           "changes (or does not) with dimension."),
])

# ==========================================================================
# 02 -- Dimensional scaling (loads experiment results)
# ==========================================================================
nb2 = nb_from([
    ("md", "# 02 · Dimensional scaling: does the PINN escape the curse of dimensionality?\n\n"
           "This notebook loads the results produced by\n"
           "`python experiments/dimensional_scaling.py` and interprets them. Re-run that "
           "script to regenerate the CSVs."),
    ("code", BOOT + "\nimport pandas as pd"),
    ("code",
     "fdm = pd.read_csv('../results/scaling/fdm_scaling.csv')\n"
     "pinn = pd.read_csv('../results/scaling/pinn_scaling.csv')\n"
     "display(fdm); display(pinn)"),
    ("md", "## The finite-difference grid explodes as $N=n^d$\n"
           "With $n$ points per axis fixed, the grid grows exponentially in $d$ and quickly "
           "becomes infeasible in memory -- the curse of dimensionality made concrete."),
    ("code",
     "fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))\n"
     "ax[0].semilogy(fdm['dim'], fdm['n_grid'], 'o-', color=CB['blue'], label='grid points $n^d$')\n"
     "ax[0].semilogy(pinn['dim'], pinn['n_params'], 's-', color=CB['red'], label='PINN parameters')\n"
     "ax[0].set_xlabel('dimension d'); ax[0].set_ylabel('degrees of freedom'); ax[0].legend()\n"
     "ax[0].set_title('Representation size')\n"
     "feas = fdm[fdm['feasible']]\n"
     "ax[1].semilogy(feas['dim'], feas['runtime_s'], 'o-', color=CB['blue'], label='FDM solve')\n"
     "ax[1].semilogy(pinn['dim'], pinn['train_time_mean_s'], 's-', color=CB['red'], label='PINN train')\n"
     "ax[1].set_xlabel('dimension d'); ax[1].set_ylabel('time (s)'); ax[1].legend()\n"
     "ax[1].set_title('Wall-clock cost')\n"
     "fig.tight_layout(); plt.show()"),
    ("md", "## But avoiding the grid is not the same as solving the problem\n"
           "The PINN runs at every dimension, but its **accuracy degrades and becomes erratic** "
           "as $d$ grows -- optimization and collocation coverage get harder. Avoiding an explicit "
           "$n^d$ grid does **not** eliminate the curse of dimensionality; it relocates it into "
           "the optimization."),
    ("code",
     "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
     "feas = fdm[fdm['feasible']]\n"
     "ax.semilogy(feas['dim'], feas['energy_error'], 'o-', color=CB['blue'], label='FDM')\n"
     "ax.errorbar(pinn['dim'], pinn['energy_error_mean'], yerr=pinn['energy_error_std'],\n"
     "            fmt='s-', color=CB['red'], capsize=3, label='PINN (mean±std)')\n"
     "ax.set_xlabel('dimension d'); ax.set_ylabel('ground-state energy error'); ax.legend()\n"
     "ax.set_title('Accuracy vs dimension'); plt.show()"),
    ("md", "See `results/scaling/scaling.png` for the full three-panel figure and "
           "`results/scaling/scaling.json` for the exact budgets and hardware."),
])

# ==========================================================================
# 03 -- Inverse problem (loads experiment results + a small live demo)
# ==========================================================================
nb3 = nb_from([
    ("md", "# 03 · Inverse problem: recovering a potential from sparse data\n\n"
           "Loads the sweep produced by `python experiments/inverse_problem.py` and also runs "
           "one small live reconstruction so the mechanism is visible."),
    ("code", BOOT + "\nimport pandas as pd"),
    ("code",
     "inv = pd.read_csv('../results/inverse/inverse.csv'); display(inv)\n"
     "fig, ax = plt.subplots(figsize=(7, 4.3))\n"
     "ax.errorbar(inv['n_data'], inv['V_rel_error_mean'], yerr=inv['V_rel_error_std'],\n"
     "            fmt='o-', color=CB['purple'], capsize=3)\n"
     "ax.set_xlabel('number of measurements $N_{data}$')\n"
     "ax.set_ylabel('relative V error (|psi|^2-weighted)')\n"
     "ax.set_title('Reconstruction quality vs data'); plt.show()"),
    ("md", "## Live demo: reconstruct V from a handful of points"),
    ("code",
     "from src import potentials as P\n"
     "from src.fdm import solve_fdm\n"
     "from experiments.inverse_problem import train_inverse, eval_inverse\n"
     "lam = 0.3; domain = (-6, 6)\n"
     "fdm = solve_fdm(P.anharmonic_oscillator(lam=lam, domain=domain), n_per_dim=1000, k=1)\n"
     "axis = fdm.axis; psi_true = fdm.wavefunctions[0]; psi_true *= np.sign(psi_true[500]+1e-9)\n"
     "V_true = 0.5*axis**2 + lam*axis**4\n"
     "rng = np.random.default_rng(0); probs = psi_true**2/psi_true.dot(psi_true)\n"
     "idx = rng.choice(len(axis), size=15, replace=False, p=probs)\n"
     "model, _ = train_inverse(axis[idx], psi_true[idx], domain, 500, 64, 4, 2500,\n"
     "                         3e-3, 100., 1., 1., 1., 1e-3, seed=0)\n"
     "psi_p, V_p = eval_inverse(model, axis)"),
    ("code",
     "supp = psi_true**2 > 1e-2*(psi_true**2).max()\n"
     "shift = np.trapezoid((V_true[supp]-V_p[supp]), axis[supp])/(axis[supp][-1]-axis[supp][0])\n"
     "fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))\n"
     "ax[0].plot(axis, V_true, color=CB['black'], lw=2, label='true V')\n"
     "ax[0].plot(axis, V_p+shift, '--', color=CB['red'], lw=2, label='reconstructed')\n"
     "ax[0].set_xlim(-5, 5); ax[0].set_ylim(-1, V_true[supp].max()*1.2)\n"
     "ax[0].set_xlabel('x'); ax[0].set_ylabel('V(x)'); ax[0].legend(); ax[0].set_title('Potential (15 points)')\n"
     "ax[1].plot(axis, psi_true, color=CB['black'], lw=2, label='true psi')\n"
     "ax[1].plot(axis, psi_p*np.sign(psi_p[500]), '--', color=CB['red'], lw=2, label='PINN psi')\n"
     "ax[1].scatter(axis[idx], psi_true[idx], color=CB['blue'], zorder=3, label='observations')\n"
     "ax[1].set_xlim(-5, 5); ax[1].set_xlabel('x'); ax[1].set_ylabel('psi'); ax[1].legend()\n"
     "ax[1].set_title('Wavefunction fit'); fig.tight_layout(); plt.show()"),
    ("md", "**Takeaway.** With the Schrodinger equation as a constraint, even ~10-25 scattered "
           "measurements recover the potential's shape -- something plain interpolation of so few "
           "points cannot do. Quality saturates with more data because the bottleneck is the joint "
           "psi/V optimization, not the number of measurements."),
])

for name, nb in [("01_fdm_vs_pinn.ipynb", nb1), ("02_scaling.ipynb", nb2),
                 ("03_inverse_problem.ipynb", nb3)]:
    path = os.path.join(HERE, name)
    nbf.write(nb, path)
    print("wrote", path, "-", len(nb["cells"]), "cells")
