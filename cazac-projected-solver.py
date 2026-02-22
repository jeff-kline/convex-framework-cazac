"""
IPUC-style iterative projection for CAZAC feasibility + small test harness.

This module implements an alternating normalization scheme to construct
length-n constant-amplitude sequences with (coupled) spectral flatness.
It is intended as a minimal, NumPy-only reference script for GitHub.

Constraints
-----------
For sequences x_j (j=1..N):
  (i)  |x_j[t]| = 1  for all samples t
  (ii) sum_{j=1}^N |FFT(x_j)[k]|^2 = N*n  for all frequency bins k

Algorithm
---------
`generate_cazac_family` alternates between:
  1) time-domain unit-modulus normalization (elementwise), and
  2) frequency-domain per-bin rescaling so the total power equals N*n,
     then returns to time domain via IFFT.

Extensions beyond the original single-sequence formulation
----------------------------------------------------------
This implementation explicitly supports:
  - N>1 coupled families via the per-bin total-power constraint, and
  - an optional `real=True` mode that forces real-valued iterates (yielding
    approximately {±1} sequences after unit-modulus normalization).

Testing regime
--------------
`run_regime` repeats independent trials (distinct seeds) and reports the
success fraction under the criterion `discrepancy_N(seqs) <= eps` within
`max_iter` iterations.

Reference: Amis et al., “CAZAC sequence generation of any length with iterative
projection onto unit circle: principle and first results,” arXiv:2509.05097.
"""
