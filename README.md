# Convergence of iterated SOCP refinement for CAZAC feasibility

**Release status: admitted.** Version `0.1.0` is preserved at tag
[`v0.1.0`](https://github.com/jeff-kline/convex-framework-cazac/releases/tag/v0.1.0)
and archived at
[`10.5281/zenodo.21766714`](https://doi.org/10.5281/zenodo.21766714).
Admission under the project's
[public research standard](https://jeff-kline.github.io/posts/research-program/index.html)
is a process verdict, not peer review or a correctness certificate.

This repository studies an iterative method for constructing
constant-amplitude zero-autocorrelation (CAZAC) sequences. A CAZAC sequence
has constant magnitude in time and a flat Fourier magnitude. The method
replaces those nonconvex equalities by convex inequalities and repeatedly
solves a second-order cone program (SOCP), using the previous optimizer as the
next objective direction.

The iteration always converges to one fixed point and has finite total path
length. It does **not**, however, always converge to a CAZAC point: the paper
constructs exposed non-CAZAC fixed points that no deterministic selection rule
can escape once reached.

> **Main result.** For each fixed relaxation parameter `τ ≥ 1`, every
> trajectory from a nonzero input converges to a single fixed point. Near a
> regular CAZAC limit, the trajectory reaches that point exactly after finitely
> many steps. At a Zadoff--Chu tuple, the required regularity holds exactly when
> the sequence length is squarefree.

The proof is in [paper/main.pdf](paper/main.pdf); its authoritative source is
[paper/main.tex](paper/main.tex).

## What is proved

The paper separates four principal results.

1. **Global convergence (Theorem A).** For every relaxation parameter
   `τ ≥ 1`, every nonzero initialization, and every measurable choice of
   SOCP optimizer, the iterates have finite length and converge to a single
   fixed point. The Kurdyka--Łojasiewicz descent mechanism is a special case of
   the general framework of Aktaş and Kroer (2025). The CAZAC-specific step is
   the reduction of feasibility to maximizing the squared norm over the convex
   relaxation.
2. **An obstruction (Theorem B).** Some fixed points are not CAZAC points, and
   some are exposed: they uniquely maximize their own linear objective. Thus
   global convergence does not imply convergence to the CAZAC set, and no
   deterministic optimizer-selection rule can provide that guarantee.
3. **Finite local arrival (Theorem C).** If the limiting CAZAC point satisfies
   the stated regularity hypothesis `(R)`, then the local constraint slack has
   sharp linear growth, the relevant KL exponent is zero, and the iteration
   reaches the limit exactly after finitely many steps. This theorem is proved
   conditional on `(R)`.
4. **Regularity at Zadoff--Chu points (Theorem D).** If
   `n = ∏ p^(a_p)` and `d_max = ∏ p^floor(a_p/2)`, then the
   active-gradient dependency space at every Zadoff--Chu tuple has dimension
   `d_max`. Consequently `(R)` holds there exactly when `n` is squarefree,
   for odd and even lengths alike.

The general convergence argument also applies to the elliptope of correlation
matrices. It proves convergence to a single fixed point even when the
fixed-point set is a continuum. Which fixed point is reached is only partly
understood: a rank-one attraction dichotomy follows from Felzenszwalb,
Klivans, and Paul (2021), while the claim that the exceptional basin has
measure zero remains open.

## What is new—and what is not

The descent template behind Theorem A is prior work, not a new optimization
principle. The contribution here is its CAZAC formulation, the obstruction to
deterministic convergence to the target set, the conditional finite-arrival
theorem, and the exact dependency count at Zadoff--Chu tuples.

Earlier constructions of Backelin (1989), Popović (1992), and Björck--Saffari
(1995) already produce continuous CAZAC families at non-squarefree lengths.
They anticipate the lower-bound, or existence, side of the degeneracy counted
in Theorem D. The paper's narrower contribution is the exact count at every
Zadoff--Chu tuple, its connection to the convergence hypothesis, and the
chirp-Fourier mechanism used to obtain it. The precise identification between
those older families and the dependency space computed here has not been
proved.

For odd prime lengths, Benoist (2024, Proposition A.2) proves projective
transversality at Gaussian chirps, the prime-length counterpart of the
regularity condition used here. The paper's Zadoff--Chu calculation gives an
exact dependency count for every length, both parities, every coprime root,
and coupled systems with `N ≥ 1`. The paper also proves a partial statement for
Björck's construction on a symmetry-invariant subspace; full regularity for
Björck sequences is available from Benoist (2024, Proposition A.3) and is not
claimed here.

Novelty claims are limited to the sources cited in the paper and the
repository's existing audit record; they do not claim global priority.

## Evidence and limitations

The repository distinguishes the support for its claims:

- **Proof.** The paper proves Theorems A, B, and D unconditionally. Theorem C is
  proved conditional on `(R)`, which Theorem D verifies for squarefree
  Zadoff--Chu tuples.
- **Exact or numerical checks.** The paper names scripts used to test the
  Zadoff--Chu rank formula, Björck calculations, and elliptope trajectories.
  Those checks support the proofs or motivate open questions; they do not
  replace the proofs.
- **Reference implementation.** The tracked `code/` directory contains the
  SOCP iteration and a projection-based comparison method. The SOCP script
  encodes optional phase-window constraints, but its release default makes
  them redundant, so the default feasible sets are exactly the `C_τ` sets
  analyzed in the paper.
- **Audit history.** [AUDIT-LEDGER.md](AUDIT-LEDGER.md) records earlier
  AI-assisted mathematical, citation, numerical, prose, and privacy checks.
  It is a historical record, not peer review.
- **Archived release.** The Zenodo `v0.1.0` file is byte-identical to the
  canonical GitHub tag zipball and has SHA-256
  `4d5861d335d2752251be03f5ecb4b831eed0e9600c6bfc5aab392237da4e8905`.

Important release limitations remain:

- Several numerical claims in the paper refer to scripts in the companion
  `cazac-algorithm` repository, which the paper states is not publicly
  fetchable at the time of writing. Those claims are not independently
  reproducible from this checkout alone.
- Theorem C does not give a global basin-of-attraction theorem.
- Theorem B rules out a deterministic guarantee of reaching a CAZAC point; it
  does not rule out probabilistic guarantees over initializations.
- The cubic-or-higher-phase discussion obstructs only the specific
  completing-the-square mechanism used for quadratic chirps, under the stated
  coprimality condition. It is not a general impossibility theorem.
- Process-separated AI audits can expose errors, but they are not independent
  expert review or a correctness certificate.

## Reproduce the tracked checks

Build the paper from the repository root:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

Run the reference SOCP implementation in a local virtual environment:

```bash
cd code
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python cazac-socp-solver.py
```

The current default was reproduced on 2026-08-02. Runtime is machine-specific;
the structural diagnostics were:

```text
Time (s):     62.1
Max |H(j,k)| 1.00000e+00
Min |H(j,k)| 1.00000e+00
Size H       (106, 106)
Condition(H) 1.00000e+00
```

The projection baseline is a different algorithm and is not covered by the
convergence theorems:

```bash
cd code
.venv/bin/python cazac-projected-solver.py
```

See [code/README.md](code/README.md) for parameters, dependencies, and scope.

## Repository map

- [paper/main.tex](paper/main.tex) and [paper/main.pdf](paper/main.pdf) -- the
  paper, proofs, prior-work discussion, open problems, and claim-tier ledger.
- [code/cazac-socp-solver.py](code/cazac-socp-solver.py) -- reference
  implementation of the SOCP iteration analyzed in the paper.
- [code/cazac-projected-solver.py](code/cazac-projected-solver.py) --
  projection-based comparison baseline, not the analyzed iteration.
- [code/README.md](code/README.md) -- code setup and scope.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) -- verified environment, commands,
  expected outputs, and the boundary between tracked and unavailable evidence.
- [CITATION.cff](CITATION.cff) -- machine-readable citation metadata for
  version `0.1.0` and its DOI.
- [CORRECTIONS.md](CORRECTIONS.md) -- version history and the correction,
  withdrawal, and supersession policy.
- [AUDIT-LEDGER.md](AUDIT-LEDGER.md) -- historical audit findings and
  dispositions.
- [audit/reports/opus-release-audit-draft.md](audit/reports/opus-release-audit-draft.md)
  -- preserved read-only release audit; its central disposition is recorded
  separately in
  [audit/reports/opus-release-audit-disposition.md](audit/reports/opus-release-audit-disposition.md).
- [LICENSE](LICENSE) -- GNU General Public License v3.0.

## AI assistance and responsibility

Large language models substantially assisted with the mathematics, code,
literature search, exposition, and adversarial checks. Model agreement is
evidence about the checking process, not independent validation. Jeffery Kline
directs the work and is responsible for claims released under his name.

## Citation

For reproducible citation, use version `0.1.0` and its version DOI:

```bibtex
@misc{kline2026cazacconvergence,
  author       = {Kline, Jeffery},
  title        = {Convergence of iterated SOCP refinement for CAZAC feasibility},
  year         = {2026},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.21766714},
  url          = {https://github.com/jeff-kline/convex-framework-cazac},
  note         = {Archived research software and paper}
}
```

This project is licensed under the GNU General Public License v3.0; see
[LICENSE](LICENSE).
