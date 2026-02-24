# Convex refinement for CAZAC feasibility

This repository contains small reference scripts accompanying the manuscript
“A Convex Programming Framework for Constructing CAZAC Sequences”.

## Files

- `cazac-socp-solver.py`  
  CVXPY-based SOCP refinement that enforces an amplitude bound and a coupled
  spectral-flatness (Fourier energy) bound across `N` sequences. Optional
  phase/quantized-phase side constraints are supported. For `N=2`, the script
  also assembles the associated `2n × 2n` block-circulant matrix and prints
  basic diagnostics.

- `cazac-projected-solver.py`  
  NumPy-only iterative projection (IPUC-style) baseline that alternates
  time-domain unit-modulus normalization with per-frequency coupled power
  normalization. Includes a small test harness and supports `N>1` and an
  optional `real=True` mode.

See the docstrings and in-file comments for details and parameter meanings.

## Quick start

Edit parameters at the top of the file you want to run, then:

- `python cazac-socp-solver.py`
- `python cazac-projected-solver.py`

## Requirements

- SOCP script: `numpy`, `scipy`, `cvxpy`, plus a cone solver (e.g., `clarabel`;
  other solvers may work depending on your setup).
- Projection baseline: `numpy`.

## License

GPL-3.0 (add a `LICENSE` file with the GPL v3 text if you make the repo public).

## Citation

If you use this code, please cite the associated manuscript (and the final
venue version once available).
