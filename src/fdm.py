"""Finite-difference Schrodinger solver (the classical baseline).

The Hamiltonian H = -1/2 D^(2) + V is built with second-order central differences
on a uniform grid and diagonalized with SciPy. In d dimensions the kinetic operator
is assembled as a Kronecker (tensor-product) sum, so the grid has n^d points --
this is the exact mechanism behind the curse of dimensionality that the scaling
experiment probes.

Returns eigenvalues, normalized eigenvectors, the grid, runtime and a memory estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .potentials import Potential
from .utils import Timer


@dataclass
class FDMResult:
    energies: np.ndarray                 # (k,) lowest eigenvalues
    wavefunctions: np.ndarray            # (k, N) normalized eigenvectors on the grid
    grid: np.ndarray                     # (N, dim) grid coordinates (1D: (N,))
    axis: np.ndarray                     # (n,) 1D coordinate axis
    n_per_dim: int
    n_grid: int                          # N = n^dim
    dim: int
    runtime: float
    memory_bytes: int                    # sparse Hamiltonian storage estimate
    dx: float


def _second_derivative_1d(n: int, dx: float) -> sp.csr_matrix:
    """Second-order central-difference second-derivative operator, Dirichlet BC."""
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    D2 = sp.diags([off, main, off], [-1, 0, 1], format="csr") / dx**2
    return D2


def _potential_on_grid(potential: Potential, coords: np.ndarray) -> np.ndarray:
    """Evaluate V on an (N, dim) array of grid coordinates -> (N,)."""
    with torch.no_grad():
        v = potential.V(torch.tensor(coords, dtype=torch.float64)).numpy().reshape(-1)
    return v


def solve_fdm(potential: Potential, n_per_dim: int = 400, k: int = 4,
              domain: Optional[tuple[float, float]] = None) -> FDMResult:
    """Solve for the k lowest eigenstates by finite differences.

    Args:
        potential: the quantum system (carries V, dim, domain).
        n_per_dim: grid points along each axis (total grid = n_per_dim**dim).
        k: number of lowest eigenstates to return.
        domain: optional (a, b) override for the per-axis box.
    """
    dim = potential.dim
    a, b = domain if domain is not None else potential.domain
    axis = np.linspace(a, b, n_per_dim)
    dx = axis[1] - axis[0]

    with Timer() as timer:
        D2 = _second_derivative_1d(n_per_dim, dx)
        T1 = -0.5 * D2                                   # 1D kinetic operator

        if dim == 1:
            coords = axis.reshape(-1, 1)
            T = T1
        else:
            # Kinetic energy as a Kronecker sum: T = sum_i I x .. x T1 x .. x I
            eye = sp.identity(n_per_dim, format="csr")
            T = sp.csr_matrix((n_per_dim**dim, n_per_dim**dim))
            for axis_i in range(dim):
                factors = [T1 if j == axis_i else eye for j in range(dim)]
                kron = factors[0]
                for f in factors[1:]:
                    kron = sp.kron(kron, f, format="csr")
                T = T + kron
            # coordinates in C-order matching the Kronecker layout
            mesh = np.meshgrid(*([axis] * dim), indexing="ij")
            coords = np.stack([m.reshape(-1) for m in mesh], axis=1)

        Vvec = _potential_on_grid(potential, coords)
        H = T + sp.diags(Vvec, 0, format="csr")
        H = H.tocsr()

        # k lowest eigenpairs; 'SA' = smallest algebraic eigenvalues
        kk = min(k, H.shape[0] - 2)
        energies, vecs = spla.eigsh(H, k=kk, which="SA")
        order = np.argsort(energies)
        energies, vecs = energies[order], vecs[:, order]

    # normalize eigenvectors so that sum |psi|^2 dx^dim = 1
    cell = dx**dim
    norms = np.sqrt((vecs**2).sum(axis=0) * cell)
    wavefunctions = (vecs / norms).T                      # (k, N)

    mem = int(H.data.nbytes + H.indptr.nbytes + H.indices.nbytes)
    return FDMResult(
        energies=energies, wavefunctions=wavefunctions,
        grid=coords, axis=axis, n_per_dim=n_per_dim, n_grid=n_per_dim**dim,
        dim=dim, runtime=timer.elapsed, memory_bytes=mem, dx=float(dx),
    )
