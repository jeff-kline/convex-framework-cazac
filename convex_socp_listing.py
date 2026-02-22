#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Convex SOCP code listing (paper companion)

This script presents the reference implementation of the paper, showing how
CAZAC-style spectral flatness constraints can be enforced via a small second-order
cone program (SOCP) in CVXPY, and how the resulting sequences can be assembled into
a structured block-circulant matrix.

Model (high level)
------------------
Given N length-n complex sequences {x_j}, the program enforces:
  • Time-domain amplitude bound:            |x_j(k)| ≤ τ   (parameter τ tightened in a presolve loop)
  • Joint spectral flatness (energy):       Σ_j |FFT(x_j)(m)|^2 ≤ N n   for all frequency bins m
  • Optional quantized-phase side-constraint:
        entries lie within a neighborhood of vertices of a regular 2q-gon on the unit circle
        (implemented as linear/second-order cone inequalities using directions exp(iπ(0.5+k)/q)
        with slack s = cos(π/(2q)) + σ).

Sequential refinement
---------------------
The objective maximizes alignment with a running reference direction r. After each solve,
r is updated to the current solution. A short continuation phase tightens τ (the listing’s
“presolve”), followed by a main loop at τ = 1 until convergence.

N=2 block construction
----------------------
For N=2, the output sequences (x,y) define circulant matrices X=circulant(x), Y=circulant(y),
and the script forms the 2n×2n block matrix
    H = [[X,  Y],
         [-Y*, X*]],
which is unitary/complex-Hadamard when the constraints are met at equality (numerically).
The script prints basic diagnostics (entry magnitudes, size, condition number).

Requirements
------------
    numpy
    scipy
    cvxpy
    clarabel   (preferred SOC solver; ECOS/SCS are used as fallbacks)


Sample output
----------------------------
Time (s):      5.7
Max |H(j,k)| 1.00000e+00
Min |H(j,k)| 1.00000e+00
Size H       (106, 106)
Condition(H) 1.00000e+00
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
