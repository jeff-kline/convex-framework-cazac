# Status

This file has been retired as an independent document. The proof-tier
table it used to duplicate now lives only in `main.tex`'s closing
**"Verification status and provenance"** section (build with
`pdflatex main.tex`, or read `main.pdf`), which distinguishes:

- **Unconditional** — complete proof, no open hypothesis.
- **Proved conditional on (R)** — complete proof, resting on regularity
  hypothesis (R).
- **Proved conditional on (R) and an open sub-case** — complete proof
  modulo one explicitly identified, unresolved lemma.
- **Verified numerically** — no proof; a named script in the companion
  repository (`cazac-algorithm`'s `experiments/` directory) reproduces
  the claim to a stated tolerance.
- **Conjecture** — stated as a target, not established either way.

Keeping this tier table in one place (rather than mirrored here and in
`main.tex`) is a deliberate simplification: this repo previously kept the
same claims independently restated across five files, and every one of
several audit rounds found at least one place where the copies had
drifted out of sync. `main.tex` is now the single source of truth.
