# Admission record

**Current verdict: ADMITTED**

**Release state:** ADMITTED

**Version:** 0.1.0

**Immutable release:** tag `v0.1.0`, commit
`8e2441dd670c6262ed685b1d0fbeaa7484ce04cd`

**Version DOI:** [10.5281/zenodo.21766714](https://doi.org/10.5281/zenodo.21766714)

**Admission date:** August 2, 2026

This is the living gate record for the first public research release.
Admission means that the project passed its defined prior-work, consistency,
reproducibility, archive, and stewardship checks against one frozen commit. It
is not peer review, a correctness certificate, or proof of global novelty.

## Standard applied

Jeff Kline, [“A Public Standard for This
Work”](https://jeff-kline.github.io/posts/research-program/index.html), version
0.4 draft, published August 1, 2026.

## Gate status

| Gate | Status | Evidence and disposition |
|---|---|---|
| P1 — prior work and claim boundary | PASS with named access limits | The public claims credit the optimization template, earlier continuous CAZAC families, Benoist's prime Gaussian-chirp transversality and full Björck regularity, and other cited antecedents. Novelty is bounded to the cited corpus. Some primary sources and the companion repository were unavailable; the limits remain visible. |
| A1 — adversarial consistency | PASS | The Opus audit rederived Theorems A--D and found no mathematical defect. Its prose, citation, and parameter findings were accepted and repaired. The raw report and root disposition are preserved under `audit/reports/`. |
| R1 — reproducibility and release integrity | PASS | The paper rebuilt and was visually inspected; the default solver reproduced the recorded structural output; the immutable tag resolves to the audited commit; and the public Zenodo file is byte-identical to the pinned GitHub tag zipball with a valid 16-file internal manifest. |

## Claim boundary

- **Proved unconditionally:** global convergence to one fixed point (Theorem
  A), exposed non-CAZAC fixed points (Theorem B), and the exact dependency
  count at every Zadoff--Chu tuple (Theorem D).
- **Proved conditionally:** finite local arrival (Theorem C), conditional on
  regularity hypothesis (R). Theorem D verifies (R) at squarefree
  Zadoff--Chu tuples.
- **Prior work:** the descent template, prime Gaussian-chirp transversality,
  full Björck regularity, and continuous non-squarefree CAZAC families are
  credited in the paper and README.
- **Numerical support:** named computations support selected calculations and
  motivate open questions. They do not replace proofs.
- **Not claimed:** deterministic global convergence to a CAZAC point, a global
  basin-of-attraction theorem, or global priority beyond the bounded cited
  corpus.

## Archive record

| Field | Verified value |
|---|---|
| Record status | Published; open access |
| Version DOI | `10.5281/zenodo.21766714` |
| Concept DOI | `10.5281/zenodo.21766713` |
| Record URL | `https://zenodo.org/records/21766714` |
| Title | *Convergence of iterated SOCP refinement for CAZAC feasibility* |
| Creator | Jeffery Kline |
| Publication date | 2026-08-03 |
| Version | `v0.1.0` |
| License | GNU General Public License v3.0 |
| Resource type | Software |
| Repository | `https://github.com/jeff-kline/convex-framework-cazac` |
| Archived file | `jeff-kline/convex-framework-cazac-v0.1.0.zip` |
| File size | 493,141 bytes |
| Provider checksum | `md5:b391a134c6c8e29a7b939837025b1c2c` |
| SHA-256 | `4d5861d335d2752251be03f5ecb4b831eed0e9600c6bfc5aab392237da4e8905` |

The downloaded Zenodo file is byte-identical to the GitHub `v0.1.0` tag
zipball pinned before publication. Its internal `MANIFEST.sha256` verifies all
16 release files. A deterministic `git archive` was also generated twice and
matched byte-for-byte, with SHA-256
`af1559b3e92f98db4faaca4ba15f1dec6164164c96456b1d00ced232f72fd24f`.

## State transitions

| State | Condition | Status |
|---|---|---|
| DRAFT → CANDIDATE | P1, A1, and pre-freeze R1 pass; prose and artifacts agree. | Passed August 2, 2026 |
| CANDIDATE → TAGGED | Freeze one clean commit and create one immutable semantic tag. | Passed: `v0.1.0` → `8e2441dd670c6262ed685b1d0fbeaa7484ce04cd` |
| TAGGED → ARCHIVED | Archive the exact tagged tree and verify the downloaded provider file byte-for-byte. | Passed: Zenodo record `21766714` |
| ARCHIVED → ADMITTED | Activate and verify the DOI, reconcile living surfaces, and issue the final verdict. | Passed August 2, 2026 |

## Named residual risks

- The novelty statement is bounded by the cited literature corpus and access
  limits; it is not a global priority claim.
- Named numerical checks in the unavailable companion repository were not
  independently reproduced from this checkout. They support secondary claims
  and open questions, not Theorems A--D.
- The projection baseline's embedded `15/50` result is not evidence for the
  SOCP basin question.
- Process-separated AI audits may share training data and blind spots; they
  are not independent expert review.
