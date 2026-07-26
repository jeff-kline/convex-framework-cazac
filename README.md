# Convergence of iterated SOCP refinement for CAZAC feasibility

**Scorecard** (blind, cold, per-axis grading; see `AUDIT-LEDGER.md` for
methodology, per-axis justification, and what would move each score) ---
graded at commit `b9634b2`, 2026-07-26:

| Axis | Score | One-line note |
| --- | --- | --- |
| Novelty | 7/10 | Graded against CAZAC construction (the target field), not just the optimization literature Theorem A borrows from: no prior iterative CAZAC method (e.g. IPUC, 2025) has a convergence guarantee; Theorem D's exact dimension count is unclaimed elsewhere. |
| Depth | 8/10 | Central claims fully proved and independently re-derivable; secondary claims (Theorem C, Björck) conditional on a stated hypothesis proved in an important special case. |
| Reach | 7/10 | Method genuinely reused within the paper (elliptope, Björck, cubic-phase) with real payoffs each time; Q1--Q7 are concrete, some already partially attacked. |
| Evidence | 7/10 | Flagship numerical claims reproducible with stated tolerances and disclosed negative results (the $p=71$ near-degeneracy); a couple of secondary claims were found under-cited during scoring and fixed. |

CAZAC sequences satisfy a constant-modulus condition in time and a flat
spectral-magnitude condition in frequency — a nonconvex intersection of two
tori. This repository's primary artifact is `paper/main.tex`, a
self-contained paper proving a convergence theory for a convex relaxation of
that feasibility problem: replace the equality constraints with convex
inequalities and iterate a second-order cone program (SOCP) against a
running reference direction that resets to each new optimizer. Four named
results. The central technical content is **Theorem D**: a regularity
hypothesis needed for a sharp local convergence rate is proved
unconditionally at Zadoff–Chu points for *all* lengths $n$ (both parities —
even $n$ was not previously closed), via a new mechanism: the DFT of a
Zadoff–Chu chirp is again a chirp, collapsing a dependency-rank computation
to counting solutions of a congruence. The same mechanism extends to a
partial result at Björck's Legendre-symbol-phase construction, reducing that case
to one unproved, numerically-confirmed lemma; that same specific mechanism
(not every conceivable route) provably cannot reach cubic or higher phase.
This underwrites **Theorem C** (proved conditional on the hypothesis;
unconditional when Theorem A's limit lies on a Zadoff–Chu tuple's symmetry
orbit at squarefree $n$): near a regular CAZAC point the iteration lands on
it **exactly** after finitely many steps. The theory rests on **Theorem A**
(unconditional): the iterate sequence always converges to a single fixed
point, not merely a set of limit points — via a KL/Frank-Wolfe descent
mechanism that is itself a known template, of which Aktaş and Kroer (2025,
prior art) give a general instance; the paper's own
contribution is the reduction of CAZAC feasibility to a norm-maximization
problem that makes the template apply. This reduction transfers verbatim to
a second example, the elliptope of correlation matrices, closing a gap
Felzenszwalb–Klivans–Paul's (2021, "FKP") own machinery leaves open on their
own worked example. FKP's own Theorem 20 in turn settles, by direct
citation, the *pointwise* half of "does every trajectory converge to a
rank-1 point?"; a 140-trial numerical sweep finds zero exceptions, and the
residual *measure-zero* form of the claim is open (Q7). **Theorem B**
(unconditional, obstruction): the limit need not be a CAZAC point, so no
*deterministic* convergence-to-CAZAC theorem is possible.

## Read this first

`paper/main.tex` / `paper/main.pdf` is the only content artifact — the
paper (build with `pdflatex main.tex` from inside `paper/`), including its
own proof-tier ledger and open-problems list. Earlier drafts kept the same
claims independently restated across several markdown files
(`theorems/convergence.md`, `STATUS.md`, `problems/open-questions.md`);
every one of several audit rounds found at least one place where those
copies had drifted out of sync, so they have been removed and `main.tex`
is the single source of truth. `AUDIT-LEDGER.md` is the record of the
adversarial audits run against this repo (per-claim findings across
mathematics, citations, numerics, prose, privacy, and the README), with
fixes applied and cross-referenced.

## How to use this repository

This repository was written with substantial assistance from large
language models and is designed, in part, for other language models to
ingest. The intended workflow:

1. Give an AI agent access to the repository.
2. Ask it to trace the proof structure — which propositions each theorem
   depends on, where a hypothesis is used, and what exactly remains open.
3. Interrogate its answers as a human reader: request derivations, exact
   statements, and the specific line in `main.tex` a claim comes from.

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
- Which claims in `main.tex`'s closing "Verification status" section are
  proved vs. merely numerically observed?

## Verify something in a minute

Build the paper:
```
cd paper && pdflatex main.tex && pdflatex main.tex
```
Run the reference SOCP solver on its built-in default parameters, in a
local venv (needs `cvxpy` and a cone solver, e.g. `clarabel`):
```
cd code && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python cazac-socp-solver.py
```
Recorded output (ran 2026-07-26, ~57s at the script's default size):
```
Time (s):     57.0
Max |H(j,k)| 1.00000e+00
Min |H(j,k)| 1.00000e+00
Size H       (106, 106)
Condition(H) 1.00000e+00
```
(Harmless `cvxpy`/`ortools` GLOP-PDLP import warnings on newer `ortools`
versions may print first; they do not affect the CLARABEL-based result
above. See `code/README.md` for both scripts' parameters.)

## Layout

- `paper/` — the sole content artifact: `main.tex` (setup, Theorems A–D,
  the elliptope corollary, the Björck partial result, open problems
  (Q1–Q7), and a tiered verification/provenance section, with an inline
  bibliography) and the built `main.pdf`.
- `code/` — reference implementations of the algorithm the theorems
  analyze, plus a projection-based comparison baseline (see
  `code/README.md`).
- `AUDIT-LEDGER.md` — the record of adversarial audits run against this
  repo (per-claim findings across mathematics, citations, numerics,
  prose, privacy, and this file), with fixes applied and cross-referenced.
- `LICENSE` — GNU GPL v3.0.

## How to cite

If you reference this work, please cite the repository. **Note:** the
remote below is this repository's configured `origin`, but at the time of
writing that remote is a private, out-of-date snapshot that predates this
document (an earlier version of the repository, from before its current
`paper/` layout and audit history) and local work here is currently ahead
of it — the URL will not resolve to what is described in this document
until the local state is pushed. Check before citing, or cite a local
commit hash in the interim.

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
