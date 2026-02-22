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