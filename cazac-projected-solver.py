# Copyright (C) 2026 Jeff Kline
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
IPUC-style iterative projection for CAZAC feasibility + small test harness.

This module implements an alternating normalization scheme to construct
length-n constant-amplitude sequences with (coupled) spectral flatness.
It is a minimal, NumPy-only reference script for GitHub.

Constraints
-----------
For sequences x_j (j=1..N):
  (i)  |x_j[t]| = 1  for all samples t
  (ii) sum_{j=1}^N |FFT(x_j)[k]|^2 = N*n  for all frequency bins k

Algorithm
---------
`generate_cazac_family` alternates between:
  1) time-domain unit-modulus normalization (elementwise), and
  2) frequency-domain per-bin rescaling so total power equals N*n,
then returns to time domain via IFFT.

Extensions beyond the original single-sequence formulation
----------------------------------------------------------
This implementation supports:
  - N>1 coupled families via the per-bin total-power constraint, and
  - an optional `real=True` mode that forces real-valued iterates (yielding
    approximately {±1} sequences after unit-modulus normalization).

Testing regime
--------------
`run_regime` repeats independent trials (distinct seeds) and reports the
success fraction under the criterion `discrepancy_N(seqs) <= eps` within
`max_iter` iterations.

Sample output (verbatim)
------------------------
=== Testing Regime Summary ===
attempts: 50
successes: 15
success_fraction: 0.3
n: 167
N: 1
max_iter: 100000
eps: 1e-05
real: False
discrepancy_min: 9.996231881359563e-06
discrepancy_median: 0.014590377160075718
discrepancy_max: 1.620940523052525
success_discrepancy_max: 9.999859940990063e-06

Failed trials: 35
Failed seeds: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 33, 35, 37, 38, 40, 41, 42, 44, 45, 46, 47, 49]

Reference: Amis et al., “CAZAC sequence generation of any length with iterative
projection onto unit circle: principle and first results,” arXiv:2509.05097.
"""
import numpy as np

def discrepancy_N(seqs):
    """
    Max deviation of sum_j |F(seqs[j])|^2 from N*n.
    """
    n = len(seqs[0])
    spectra = [np.fft.fft(s) for s in seqs]
    S = sum(np.abs(Sj) ** 2 for Sj in spectra)
    return np.max(np.abs(S - len(seqs) * n))


def generate_cazac_family(
    n,
    N=4,                 # <-- number of sequences
    eps=1e-8,
    max_iter=10000,
    seed=None,
    verbose=True,
    real=False,
):
    """
    Iterative projection to generate N length-n sequences such that:
      (i)  |x_j(k)| = 1  for all j,k
      (ii) sum_j |F(x_j)|^2 = N*n  (per frequency bin)

    Returns
    -------
    seqs : list of length-N ndarrays of shape (n,)
    """

    rng = np.random.default_rng(seed)
    tiny = 1e-15

    def rand_unit_phasor(size):
        return np.exp(1j * 2 * np.pi * rng.random(size))

    def maybe_real(v):
        return np.real(v) if real else v

    # --- Initialization ---
    spectra = [rand_unit_phasor(n) for _ in range(N)]
    seqs = [maybe_real(np.fft.ifft(S)) for S in spectra]

    seqs = [s / (np.abs(s) + tiny) for s in seqs]

    converged = False
    iters = 0

    for k in range(1, max_iter + 1):

        if discrepancy_N(seqs) <= eps:
            converged = True
            iters = k - 1
            break

        # --- Projection 1: unit modulus ---
        seqs = [s / (np.abs(s) + tiny) for s in seqs]

        # --- Frequency domain ---
        spectra = [np.fft.fft(s) for s in seqs]

        # --- Projection 2: enforce per-bin total power = N*n ---
        P = sum(np.abs(Sj) ** 2 for Sj in spectra) + tiny
        s = np.sqrt((N * n) / P)

        spectra = [Sj * s for Sj in spectra]

        # --- Back to time domain ---
        seqs = [maybe_real(np.fft.ifft(Sj)) for Sj in spectra]
        seqs = [s / (np.abs(s) + tiny) for s in seqs]

        iters = k

    if real:
        seqs = [np.real(s) for s in seqs]
    # Final normalization
    seqs = [s / (np.abs(s) + tiny) for s in seqs]

    
    if verbose:
        print("Iterations performed:", iters)
        print("Converged:", converged)
        print("Final discrepancy:", discrepancy_N(seqs))
        mags = [np.abs(s) for s in seqs]
        print("Max |.|:", max(np.max(m) for m in mags))
        print("Min |.|:", min(np.min(m) for m in mags))
        print("N =", N, " Real mode:", real)

    return seqs
    
def run_regime(
    attempts=5,
    n=167,
    N=1,
    max_iter=100000,
    eps=1e-5,
    real=False,
    base_seed=12345,
):
    """
    Runs 'attempts' independent trials and reports success fraction.

    Success criterion:
      discrepancy_N(seqs) <= eps  (achieved within max_iter iterations)
    """
    results = []
    successes = 0

    for i in range(attempts):
        seed = base_seed + i

        # Silence per-run prints to keep output readable.
        seqs = generate_cazac_family(
            n=n,
            N=N,
            eps=eps,
            max_iter=max_iter,
            seed=seed,
            verbose=False,
            real=real,
        )

        disc = float(discrepancy_N(seqs))
        ok = disc <= eps
        successes += int(ok)

        results.append(
            {
                "trial": i,
                "seed": seed,
                "success": ok,
                "final_discrepancy": disc,
            }
        )

    frac = successes / attempts

    # Summaries
    discs = np.array([r["final_discrepancy"] for r in results], dtype=float)
    success_discs = np.array([r["final_discrepancy"] for r in results if r["success"]], dtype=float)

    summary = {
        "attempts": attempts,
        "successes": successes,
        "success_fraction": frac,
        "n": n,
        "N": N,
        "max_iter": max_iter,
        "eps": eps,
        "real": real,
        "discrepancy_min": float(discs.min()),
        "discrepancy_median": float(np.median(discs)),
        "discrepancy_max": float(discs.max()),
        "success_discrepancy_max": (float(success_discs.max()) if len(success_discs) else None),
    }

    return summary, results



summary, results = run_regime(
    attempts=50,
    n=167,
    N=1,            # change if you want N=1 or N=2, etc.
    max_iter=100000,
    eps=1e-5,
    real=False,
    base_seed=1
)

print("=== Testing Regime Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# Optional: quick list of failed seeds (useful for debugging)
failed = [r for r in results if not r["success"]]
print("\nFailed trials:", len(failed))
if failed:
    print("Failed seeds:", [r["seed"] for r in failed])
    
