# PINNs vs. Finite Differences for the Schrödinger Equation

A scientific computational study to discover when physics-informed neural
networks (PINNs) are actually useful for solving the time-independent Schrödinger
equation versus when conventional finite-difference methods (FDM) remain the better
tool. All conclusions below are drawn from measurements produced by the scripts in
this repository.

> **Key Findings.** For ordinary forward solves the finite-difference method wins
> without question. It is orders of magnitude faster and more accurate in 1–3 dimensions.
> The PINN does **not** get the curse of dimensionality as clearly: avoiding an explicit
> `n^d` grid simply relocates the difficulty into optimization, where accuracy becomes
> erratic as `d` grows. PINNs pay off in *different* regimes: **Amortized inference**
> over a family of problems (a measured cost crossover near **~1600 instances**) and
> **physics-constrained inverse problems**, where a handful of measurements suffice to
> reconstruct an unknown potential.

---

## 1 · Motivation

PINNs solve differential equations by training a neural network whose loss is the PDE
residual itself, using automatic differentiation for the derivatives. They are
mesh-free, produce a continuous differentiable solution, and can absorb data and
unknown parameters as easily as boundary conditions. That flexibility has made them
popular. However, flexibility does not equal efficiency. For the Schrödinger equation,
where fast, accurate classical eigensolvers already exist, the question is not
"can a PINN do it?" but "**when is it worth it?**"

## 2 · Research question

> **When are PINNs actually useful for solving quantum-mechanical PDEs, and how do their
> computational trade-offs compare with conventional finite-difference methods?**

## 3 · Physics

Dimensionless units throughout ($\hbar = m = 1$). We solve the stationary equation

$$-\tfrac12 \nabla^2 \psi(\mathbf{x}) + V(\mathbf{x})\,\psi(\mathbf{x}) = E\,\psi(\mathbf{x}).$$

Systems implemented (`src/potentials.py`): 1D infinite square well, 1D harmonic
oscillator $V=\tfrac12\omega^2x^2$ (exact $E_n=n+\tfrac12$), 1D anharmonic/quartic
oscillator $V=\tfrac12\omega^2x^2+\lambda x^4$, and the $d$-dimensional isotropic
harmonic oscillator (exact ground energy $E_0=d/2$).

## 4 · Methods

**Finite differences (`src/fdm.py`).** Second-order central differences turn $\hat H$
into a (tensor-product) sparse matrix; SciPy's `eigsh` returns the lowest eigenpairs.
In $d$ dimensions the kinetic term is a Kronecker sum, so the grid has $N=n^d$ points. 
Dense, tridiagonal and sparse variants are provided.

**PINN (`src/pinn.py`, `src/training.py`).** A tanh MLP maps $\mathbf{x}\mapsto\psi_\theta(\mathbf{x})$;
the Laplacian is taken by automatic differentiation. The energy $E$ is a **trainable
parameter** (the analytic value is never used). The loss combines the PDE residual, a
normalization term $\big(\int|\psi|^2-1\big)^2$, a boundary term, and, for excited
states, an orthogonality penalty against previously found states:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{PDE}} + \lambda_N \mathcal{L}_{\mathrm{norm}} + \lambda_{\mathrm{BC}} \mathcal{L}_{\mathrm{BC}} + \lambda_O \sum_{m<n} \langle \psi_n, \psi_m \rangle^2 .
$$

Spatial integrals use deterministic trapezoidal quadrature in 1D and Monte-Carlo
sampling in higher dimensions. *Documented limitation:* the residual/orthogonality
ladder is reliable for the ground and first couple of excited states, then degrades —
the third excited state frequently collapses onto a lower level or overshoots. This is
an inherent difficulty of the formulation.

**Parameterized PINN (`experiments/parameterized_solver.py`).** One network
$\psi_\theta(x,\lambda)$ represents the whole quartic-oscillator family; the energy
$E(\lambda)$ is the variational Rayleigh quotient of each slice (a well-defined function
of $\lambda$), and minimizing the mean energy drives every slice to its ground state.

**Inverse PINN (`experiments/inverse_problem.py`).** Two networks $\psi_\theta(x)$ and
$V_\phi(x)$ are trained jointly to fit sparse wavefunction samples *and* satisfy the
Schrödinger residual, recovering an unknown potential.

## 5 · Experimental results

All numbers below were measured on the hardware recorded in each experiment's
`*.json` manifest (CPU sandbox; see manifests for details). Budgets are intentionally
modest ("lightweight but honest") and are recorded alongside the results.

### 5.1 Forward solve — FDM wins in low dimensions

On the 1D harmonic oscillator, FDM reaches $\sim\!10^{-4}$ energy error in
milliseconds; the PINN reaches $\sim\!10^{-3}$ in seconds (notebook `01`). For a single
low-dimensional forward solve, **FDM is the right tool by a wide margin.**

### 5.2 Dimensional scaling

Fixed $n=40$ points per axis, so the FDM grid is exactly $N=n^d$
(`results/scaling/`):

| $d$ | FDM grid $N=n^d$ | FDM time | FDM energy err | PINN time (3 seeds) | PINN energy err |
|----:|-----------------:|---------:|---------------:|--------------------:|----------------:|
| 1 | 40 | 0.005 s | 3.0×10⁻³ | 9.3 s | 0.040 ± 0.009 |
| 2 | 1 600 | 0.009 s | 6.0×10⁻³ | 14.3 s | 0.303 ± 0.028 |
| 3 | 64 000 | 0.36 s | 8.9×10⁻³ | 20.9 s | 0.143 ± 0.069 |
| 4 | 2 560 000 | *not run* (grid > budget; ≈0.37 GB sparse) | — | 45.2 s | 0.858 ± 0.001 |
| 5 | 102 400 000 | *not run* (≈18 GB sparse) | — | — | — |

The FDM grid grows exactly as $n^d$ and blows past the study's feasibility budget by
$d=4$; storing even the *sparse* Hamiltonian needs ~18 GB at $d=5$ (a dense one would
need $(n^d)^2$ — astronomically more). The PINN, by contrast, keeps a fixed ~8.5k
parameters and runs at every dimension. **But its energy error is large and
non-monotonic** (0.04 → 0.30 → 0.14 → 0.86): avoiding the grid does not confer accuracy.
The curse of dimensionality reappears as an optimization/coverage problem.
See `results/scaling/scaling.png`.

### 5.3 Parameterized PINN

One network trained over $\lambda\in[0,0.5]$ (17 nodes, 23.5 s) predicts the ground-state
energy at 16 **held-out** $\lambda$ values with mean error $5\times10^{-3}$ and
wavefunction $L_2$ error $\sim\!10^{-2}$ (`results/parameterized/`). Timing:

- FDM single solve: **14.9 ms**  ·  PINN inference: **0.23 ms**  ·  PINN training: **23.5 s**

Total-cost model $T_\text{FDM}=N\,t_\text{fdm}$ vs. $T_\text{PINN}=t_\text{train}+N\,t_\text{infer}$
gives a **measured crossover at $N\approx1600$ instances**: below it, just run FDM;
above it, the amortized network is cheaper — and it additionally yields a continuous,
differentiable $\psi(x,\lambda)$. This is the clearest regime in which the PINN pays off,
and the crossover was measured, not assumed.

### 5.4 Inverse problem — physics constraint enables sparse reconstruction

From as few as **10** noisy wavefunction samples, the joint $\psi_\theta/V_\phi$ network
reconstructs the unknown quartic potential's shape (relative, $|\psi|^2$-weighted error
on the supported region):

| $N_\text{data}$ | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| $V$ rel. error | 0.34 ± 0.01 | 0.31 ± 0.04 | 0.31 ± 0.03 | 0.29 ± 0.01 |

Quality is roughly flat in $N$: the Schrödinger constraint does the heavy lifting, so
even ~10 points suffice (plain interpolation of 10 points could not recover a
potential), and adding data saturates because the bottleneck is the joint optimization.
See `results/inverse/inverse.png`.

### 5.5 Convergence — methods must be compared at matched, converged settings

FDM converges at the expected second-order rate $\propto\Delta x^2$, down to
$4.4\times10^{-7}$ energy error. PINN accuracy improves cleanly with training
(0.064 → 1.5×10⁻³ → 3.5×10⁻⁴ → 6.2×10⁻⁵ at 500 → 1000 → 2000 → 4000 epochs) but then
hits an optimization floor, and it is **strikingly sensitive to architecture**: at a
fixed 2000-epoch budget, width 32 gives $3.5\times10^{-4}$ while width 128 gives
**0.41** — a wider network is dramatically *worse* because it trains more slowly.
Collocation count matters little once integration is fine enough (the optimization
floor dominates). This is exactly why comparing the two methods at arbitrary settings is
meaningless — each must be run at converged, matched settings. See
`results/convergence/convergence.png`.

## 6 · Conclusions

- **Single forward solve (low $d$):** **FDM wins**, decisively, in both speed and
  accuracy. Use finite differences.
- **High-dimensional scaling:** FDM hits the $n^d$ wall in time and memory, but the PINN
  does **not** automatically defeat the curse of dimensionality. Avoiding the grid
  moves the difficulty into optimization, where accuracy degrades and becomes erratic.
  Avoiding an explicit tensor-product grid is not the same as eliminating this problem.
- **Parameterized families:** the PINN's amortized inference **does** pay off past a
  measured crossover (~1600 solves here), and uniquely provides a continuous solution
  over the parameter.
- **Inverse problems:** embedding the Schrödinger equation as a constraint lets a PINN
  reconstruct an unknown potential from very sparse data — a setting where classical
  forward solvers do not directly apply.

## 7 · A conceptual distinction

**PDE solve vs. PINN inference.** Running FDM computes a *new* solution each time. Once a
PINN is trained, evaluating $\psi_\theta(\mathbf{x})$ is cheap **inference**. It is only
fair to compare against the *full* PINN cost,
$$T_\text{PINN} = t_\text{train} + N_\text{queries}\,t_\text{inference},$$
which is exactly the amortization analysis in §5.3. Comparing FDM solve time against PINN
inference time alone (ignoring training) would be misleading.

## 8 · Reproducibility

Every experiment fixes random seeds, records hyperparameters and hardware to a JSON
manifest, saves numerical results to CSV, and regenerates its figures. Run from the repo
root:

```bash
pip install -r requirements.txt
python experiments/dimensional_scaling.py     # -> results/scaling/
python experiments/parameterized_solver.py    # -> results/parameterized/
python experiments/inverse_problem.py          # -> results/inverse/
python experiments/convergence.py              # -> results/convergence/
python notebooks/build_notebooks.py            # (re)generate the notebooks
```

Each script takes `--help` for its budget knobs (epochs, collocation, seeds, dimensions).
Defaults are sized for a CPU laptop.

## 9 · Repository layout

```
src/           potentials, FDM solver, PINN model, training loop, evaluation, utils
experiments/   dimensional_scaling, parameterized_solver, inverse_problem, convergence
results/       CSVs, JSON manifests and figures written by the experiments
notebooks/     01_fdm_vs_pinn, 02_scaling, 03_inverse_problem
```

## 10 · Limitations & caveats

- Keep in mind that budgets here are modest (CPU sandbox); absolute timings are hardware-specific, but the
  *scaling trends and crossovers* are the findings.
- The residual/orthogonality excited-state ladder is reliable only for the lowest few
  states (§4).
- The dimensional-scaling FDM crossover to "infeasible" reflects this study's runtime
  budget ($N\le1.5\times10^5$); larger machines push the wall out by a dimension or two.
