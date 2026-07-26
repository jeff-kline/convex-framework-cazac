# Convergence of iterated SOCP refinement for CAZAC feasibility

CAZAC sequences satisfy a constant-modulus condition in time and a flat
spectral-magnitude condition in frequency — a nonconvex intersection of two
tori. This repository's primary artifact is `main.tex`, a self-contained
paper proving a convergence theory for a convex relaxation of that
feasibility problem: replace the equality constraints with convex
inequalities (bounded amplitude, bounded spectral energy) and iterate a
second-order cone program (SOCP) against a running reference direction
that resets to each new optimizer. Four named results. The central
technical content is **Theorem D**: a regularity hypothesis needed for a
sharp local convergence rate is proved unconditionally at Zadoff–Chu
points for *all* lengths $n$ (both parities — even $n$ was not previously
closed), via a new mechanism for this problem: the DFT of a Zadoff–Chu
chirp is again a chirp, collapsing a dependency-rank computation to
counting solutions of a congruence. The same mechanism extends (via a
different closure fact) to a partial result at Björck's quadratic-phase
construction, reducing that case to one unproved, numerically-confirmed
number-theoretic lemma, and is proved *not* to extend to cubic or higher
phase sequences. This regularity result underwrites **Theorem C** (proved
conditional on the hypothesis; unconditional at squarefree lengths of
either parity): near a regular CAZAC point the iteration lands on it
**exactly** after finitely many steps, including a complete
contradiction-based proof of the finite-arrival step. The convergence
theory this rests on is **Theorem A** (unconditional): the iterate
sequence always converges to a single fixed point, not merely to a set
of limit points — via a KL/Frank-Wolfe-type descent mechanism that is
itself a known template (also independently instantiated by Aktaş
and Kroer, 2025); the paper's own contribution is the reduction of CAZAC
feasibility to a norm-maximization problem that makes the template apply.
Since neither that reduction nor the descent argument uses CAZAC-specific
structure, the argument transfers verbatim to a second example, the
elliptope of correlation matrices, closing a gap Felzenszwalb–Klivans–
Paul's (2021, "FKP") own machinery leaves open on their own worked
example. FKP's own Theorem 20 in turn settles, by direct citation, the
*pointwise* half of "does every trajectory converge to a rank-1 point?":
vertices are exactly the fixed points whose full neighborhood converges
to them, and every non-vertex point provably cannot attract one either.
A 140-trial numerical sweep finds zero exceptions to full rank-1
convergence; the residual *measure-zero* form of the claim is open
(Q7/P8), structurally the same conjecture as the paper's own central open
question (Q6) in a second domain. **Theorem B** (unconditional,
obstruction): the limit need not be a CAZAC point — there exist
non-CAZAC fixed points that no measurable selection rule can escape, so
no *deterministic* convergence-to-CAZAC theorem is possible.

## Read this first

`main.tex` / `main.pdf` — the paper (build with `pdflatex main.tex`).
`theorems/convergence.md` — the same theorems and proofs, mirrored in
markdown. `STATUS.md` — every principal claim with its proof tier (proved /
proved-conditional / numerically verified / conjectural). `problems/` —
the open questions this leaves (P1–P8; `main.tex`'s own condensed
Open Problems section labels the same list Q1–Q7 for its shorter,
paper-facing form), stated precisely enough to attack. `AUDIT-LEDGER.md` —
the record of an adversarial audit run against this repo (per-claim
findings across mathematics, citations, numerics, prose, privacy, and
this file), with fixes applied and cross-referenced.

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
- `AUDIT-LEDGER.md` — the record of adversarial audits run against this
  repo (per-claim findings across mathematics, citations, numerics,
  prose, privacy, and this file), with fixes applied and cross-referenced.

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
