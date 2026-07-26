# Convergence of iterated SOCP refinement for CAZAC feasibility

CAZAC sequences satisfy a constant-modulus condition in time and a flat
spectral-magnitude condition in frequency — a nonconvex intersection of two
tori. This repository's primary artifact is `main.tex`, a self-contained
paper proving a convergence theory for a convex relaxation of that
feasibility problem: replace the equality constraints with convex
inequalities (bounded amplitude, bounded spectral energy) and iterate a
second-order cone program (SOCP) against a running reference direction
that resets to each new optimizer. Four named results. **Theorem A**
(unconditional): the iterate sequence always converges to a single fixed
point, not merely to a set of limit points — and, because the proof never
uses CAZAC-specific structure, it transfers verbatim to a second example,
the elliptope of correlation matrices, closing a gap Felzenszwalb–Klivans–
Paul's (2021) own machinery leaves open on their own worked example.
**Theorem B** (unconditional, obstruction): the limit need not be a CAZAC
point — there exist non-CAZAC fixed points that no measurable selection
rule can escape, so no *deterministic* convergence-to-CAZAC theorem is
possible. **Theorem C** (proved conditional on a checkable regularity
hypothesis; unconditional at squarefree lengths of either parity): near a
regular CAZAC point the iteration lands on it **exactly** after finitely
many steps, including a complete contradiction-based proof of the finite-
arrival step. **Theorem D**: the regularity hypothesis is proved at
Zadoff–Chu points for *all* lengths $n$; a related partial result at
Björck's quadratic-phase construction reduces the same question to one
unproved, numerically-confirmed number-theoretic lemma.

## Read this first

`main.tex` / `main.pdf` — the paper (build with `pdflatex main.tex`).
`theorems/convergence.md` — the same theorems and proofs, mirrored in
markdown. `STATUS.md` — every principal claim with its proof tier (proved /
proved-conditional / numerically verified / conjectural). `problems/` —
the open questions this leaves (P1–P8; `main.tex`'s own condensed
Open Problems section labels the same list Q1–Q7 for its shorter,
paper-facing form), stated precisely enough to attack.

## How to use this repository

This repository was written with substantial assistance from large
language models and is designed, in part, for other language models to
ingest. The intended workflow:

1. Give an AI agent access to the repository.
2. Ask it to trace the proof structure — which propositions each theorem
   depends on, where a hypothesis is used, and what exactly remains open.
3. Interrogate its answers as a human reader: request derivations, exact
   statements, and the specific line in `theorems/convergence.md` a claim
   comes from.

The proofs are written to be precise enough for an agent to navigate while
remaining readable by a human. They should not be treated as an automatic
guarantee of correctness — read them.

### Ask your agent

- Which theorem, if any, guarantees the algorithm reaches a CAZAC point,
  and under what hypothesis?
- What exactly does hypothesis (R) require, and where is it proved to
  hold?
- Why does Theorem A's proof not already rule out Theorem B's obstruction?
- What would it take to upgrade Theorem C from "near a regular point" to
  a global statement?
- Which claims in `STATUS.md` are proved vs. merely numerically observed?

## Verify something in a minute

Build the paper:
```
pdflatex main.tex && pdflatex main.tex
```
Run the reference SOCP solver on its built-in default parameters (needs
`cvxpy` and a cone solver, e.g. `clarabel`; `pip install -r code/requirements.txt`):
```
cd code && python cazac-socp-solver.py
```
Recorded output (ran 2026-07-26, ~56s at the script's default size):
```
Time (s):     56.1
Max |H(j,k)| 1.00000e+00
Min |H(j,k)| 1.00000e+00
Size H       (106, 106)
Condition(H) 1.00000e+00
```
(Harmless `cvxpy`/`ortools` GLOP-PDLP import warnings on newer `ortools`
versions may print first; they do not affect the CLARABEL-based result
above. See `code/README.md` for both scripts' parameters.)

## Layout

- `main.tex` / `main.pdf` — the paper: setup, Theorems A–D, the elliptope
  corollary, the Björck partial result, open problems, and a tiered
  verification/provenance section, with an inline bibliography.
- `theorems/convergence.md` — the same theorems (statements and proofs),
  mirrored in markdown.
- `problems/open-questions.md` — open problems (P1–P8), stated for others
  to attack.
- `STATUS.md` — proof tier of every principal claim.
- `code/` — reference implementations of the algorithm the theorems
  analyze, plus a projection-based comparison baseline (see
  `code/README.md`).

## How to cite

If you reference this work, please cite the repository:

```bibtex
@misc{kline2026cazacconvergence,
  author       = {Kline, Jeffery},
  title        = {{Convergence of iterated SOCP refinement for CAZAC feasibility}},
  year         = {2026},
  howpublished = {\url{https://github.com/jeff-kline/convex-framework-cazac}},
  note         = {GitHub repository; written by large language models under human direction}
}
```

Plain text: Jeffery Kline, *Convergence of iterated SOCP refinement for
CAZAC feasibility*, 2026.
https://github.com/jeff-kline/convex-framework-cazac

For a reproducible reference, pin a specific commit hash or a tagged
release in the `note` field.

## License

This project is licensed under the GNU General Public License v3.0 — see
[`LICENSE`](LICENSE) for the full text.
