# Reference scripts

Two small reference implementations of the algorithm analyzed in
`../theorems/convergence.md`.

- `cazac-socp-solver.py` — CVXPY-based SOCP refinement that enforces an
  amplitude bound and a coupled spectral-flatness (Fourier energy) bound
  across `N` sequences. Optional phase/quantized-phase side constraints are
  supported. For `N=2`, the script also assembles the associated
  `2n × 2n` block-circulant matrix and prints basic diagnostics.

- `cazac-projected-solver.py` — NumPy-only iterative-projection (IPUC-style)
  baseline that alternates time-domain unit-modulus normalization with
  per-frequency coupled power normalization. Included as a comparison
  point: it is a different (nonconvex, projection-based) heuristic for the
  same feasibility problem, not the algorithm the theorems are about.
  Includes a small test harness and supports `N>1` and an optional
  `real=True` mode.

See the docstrings and in-file comments for parameter meanings.

## Quick start

Edit parameters at the top of the file you want to run, then:

```
python cazac-socp-solver.py
python cazac-projected-solver.py
```

## Requirements

- SOCP script: `numpy`, `scipy`, `cvxpy`, plus a cone solver (e.g.,
  `clarabel`; other solvers may work depending on your setup).
- Projection baseline: `numpy`.

```
pip install -r requirements.txt
```
