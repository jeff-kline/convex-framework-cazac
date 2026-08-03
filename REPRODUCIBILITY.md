# Reproducibility record

This record separates what was reproduced from this checkout from evidence
that remains outside it. Version `0.1.0` is frozen at commit
`8e2441dd670c6262ed685b1d0fbeaa7484ce04cd`, tagged as `v0.1.0`, and archived
under version DOI `10.5281/zenodo.21766714`.

## Paper

From the repository root:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Verified result: `paper/main.pdf`, 21 pages. The build has no overfull boxes,
unresolved references, or unresolved citations. Three underfull bibliography
line-wrap warnings remain; they do not alter content or visibility.

## Reference SOCP solver

Create a repository-local environment and run the default case:

```bash
cd code
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python cazac-socp-solver.py
```

Verified environment:

| Component | Version |
|---|---:|
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| CVXPY | 1.7.5 |
| Clarabel | 0.11.1 |

Verified output on 2026-08-02:

```text
Time (s):     62.1
Max |H(j,k)| 1.00000e+00
Min |H(j,k)| 1.00000e+00
Size H       (106, 106)
Condition(H) 1.00000e+00
```

Runtime is machine-specific. The release checks are the matrix size, unit
entry magnitudes to the printed tolerance, and condition number 1 to the
printed tolerance.

The script encodes optional phase-window inequalities. At the release default
`sigma = 1.0`, they are redundant: the schedule has `|X| <= 1`, while their
right-hand side is greater than 1. The default run therefore uses exactly the
convex sets analyzed in the paper. Lowering `sigma` activates a stricter
experimental variant that the theorems do not cover.

The first ten solves use `τ = 1, 1/2, ..., 1/10`; the main phase then resets
to fixed `τ = 1`. The theorems apply to the fixed-`τ` iteration. The
preliminary solves only select the main phase's input direction, and no
theoretical benefit is claimed for them.

## Projection baseline

`code/cazac-projected-solver.py` implements a different IPUC-style alternating
projection method. Its embedded `15/50` success count at `n = 167` belongs to
that baseline. It is not evidence for the SOCP basin-of-attraction question in
the paper and is not used to support any theorem.

## Evidence boundary

| Item | Status in this checkout |
|---|---|
| Theorems A--D | Written proofs in `paper/main.tex`; adversarially rederived in the preserved Opus audit |
| Paper PDF | Rebuilt locally and visually checked |
| Default SOCP example | Reproduced locally as recorded above |
| Projection baseline sample | Tracked in code; not evidence for the SOCP theorems |
| Named experiments in companion `cazac-algorithm` repository | Not publicly fetchable during this release preparation; not independently reproduced here |
| Permanent archive and DOI | Verified: Zenodo record `21766714`; DOI `10.5281/zenodo.21766714` |

Unavailable companion artifacts are a disclosed reproducibility limitation.
They support secondary numerical checks and open questions, not the principal
proofs.

## Archive verification

The canonical GitHub API zipball and the downloaded Zenodo file are
byte-identical:

| Field | Verified value |
|---|---|
| Tag | `v0.1.0` |
| Commit | `8e2441dd670c6262ed685b1d0fbeaa7484ce04cd` |
| Version DOI | `10.5281/zenodo.21766714` |
| Concept DOI | `10.5281/zenodo.21766713` |
| Archived file | `jeff-kline/convex-framework-cazac-v0.1.0.zip` |
| File size | 493,141 bytes |
| Provider checksum | `md5:b391a134c6c8e29a7b939837025b1c2c` |
| SHA-256 | `4d5861d335d2752251be03f5ecb4b831eed0e9600c6bfc5aab392237da4e8905` |

Two separately downloaded GitHub tag zipballs matched each other, and the
Zenodo file matched them byte-for-byte. Two deterministic `git archive` ZIP
files also matched each other, with SHA-256
`af1559b3e92f98db4faaca4ba15f1dec6164164c96456b1d00ced232f72fd24f`;
they are a secondary check and are not expected to be byte-identical to the
GitHub/Zenodo zipball.
