#!/usr/bin/env python3
"""Convex SOCP code listing (runnable transcription).

This script is a lightly cleaned transcription of the manuscript code listing
(phase‑window constrained program). Changes from the literal listing:

- Uses ASCII quotes and PEP8-ish formatting.
- Adds a `main()` entry point.
- Adds solver fallback (Clarabel → ECOS → SCS) for portability.

The optimization model and update logic match the listing.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional

import numpy as np
import numpy.linalg as nl
import numpy.fft as nf
import numpy.random as nr
import scipy.linalg as sl
import cvxpy as cp


def getH(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build the block matrix [[X, Y], [-Y*, X*]] where X,Y are circulant from x,y."""
    X = sl.circulant(x)
    Y = sl.circulant(y)
    YT = np.conj(Y.T)
    XT = np.conj(X.T)
    return np.block([[X, Y], [-YT, XT]])


def get_lp_prog(n: int, N: int, q: int) -> Tuple[cp.Problem, List[cp.Variable], cp.Parameter, cp.Parameter, cp.Parameter]:
    """Construct the CVXPY problem used in the paper's code listing."""

    # Regular 2q-gon vertex directions (as in the listing)
    qq = np.exp(1j * np.pi * np.array([(0.5 + k) / q for k in range(q)]))

    # DFT matrix (O(n^2) memory, consistent with the listing)
    F = nf.fft(np.eye(n))

    # Objective direction and continuation parameters
    r = cp.Parameter(N * n, complex=True)
    t = cp.Parameter(nonneg=True)  # τ in the paper (named t in the listing)
    s = cp.Parameter(nonneg=True)  # cos(pi/(2q)) + σ in the paper

    # Decision variables: N complex vectors length n
    x = [cp.Variable(n, complex=True) for _ in range(N)]
    X = cp.hstack(x)  # shape (n, N)

    # Joint spectral energy: sum_j |F x_j|^2 (elementwise in frequency index)
    FXYZW = cp.sum([cp.square(cp.abs(F @ xj)) for xj in x])

    constr = [
        cp.abs(X) <= t,
        FXYZW <= N * n,
    ] + [
        cp.abs(cp.real(X) * _q.real + cp.imag(X) * _q.imag) <= s for _q in qq
    ]

    objective = cp.Maximize(cp.real(X) @ cp.real(r) + cp.imag(X) @ cp.imag(r))
    prob = cp.Problem(objective, constr)
    return prob, x, r, t, s


def solve_with_fallback(prob: cp.Problem, preferred: str = "CLARABEL") -> float:
    """Solve with a preferred solver, then fall back to common SOC solvers."""
    solvers = [preferred, "ECOS", "SCS"]
    last_err: Optional[Exception] = None
    for s in solvers:
        try:
            return float(prob.solve(solver=s))
        except Exception as e:  # pragma: no cover (depends on local solver installs)
            last_err = e
    raise RuntimeError(f"All solvers failed: {last_err}")


def main() -> None:
    # Parameters from the listing
    n = 53
    N = 2
    L0 = 10
    L = 100
    q = 2
    eps = 1e-5

    # Phase-window slack (σ); listing uses cos(pi/(2q)) + 0.2
    sigma = 0.2

    prob, x, r, t, s = get_lp_prog(n, N, q)

    nr.seed(1)
    r.value = nr.randn(N * n) + 1j * nr.randn(N * n)
    s.value = float(np.cos(np.pi / (2 * q)) + sigma)

    t0 = time.time()

    # Presolve loop: tighten τ (named t here) and update r to the solution
    for it in range(L0):
        t.value = 1 / (it + 1)
        solve_with_fallback(prob, preferred="CLARABEL")
        r.value = np.hstack([xj.value for xj in x])

    # Main loop at τ = 1
    t.value = 1.0
    rcur = 1.0  # matches listing initialization
    for _ in range(L):
        solve_with_fallback(prob, preferred="CLARABEL")
        r.value = np.hstack([xj.value for xj in x])
        if nl.norm(rcur - r.value) < eps:
            break
        rcur = r.value

    t1 = time.time()

    # Build the block-circulant matrix (N=2)
    H = getH(*[xj.value for xj in x])

    print(f"Time (s):    {t1 - t0:5.1f}")
    print(f"Max |H(j,k)| {max(abs(H.flatten())):5.5e}")
    print(f"Min |H(j,k)| {min(abs(H.flatten())):5.5e}")
    print(f"Size H       {H.shape}")
    print(f"Condition(H) {nl.cond(H):5.5e}")


if __name__ == "__main__":
    main()
