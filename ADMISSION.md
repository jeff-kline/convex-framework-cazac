# Admission record

**Current verdict: NOT YET ADMITTED**

**Release state:** CANDIDATE

**Planned version:** 0.1.0

**Immutable release:** Pending

**Version DOI:** Pending

This is the living gate record for the first public research release.
Admission will mean that the project passed its defined prior-work,
consistency, reproducibility, archive, and stewardship checks against one
frozen commit. It will not mean peer review, a correctness certificate, or
proof of global novelty.

## Standard applied

Jeff Kline, [“A Public Standard for This
Work”](https://jeff-kline.github.io/posts/research-program/index.html), version
0.4 draft, published August 1, 2026.

## Gate status

| Gate | Status | Evidence and disposition |
|---|---|---|
| P1 — prior work and claim boundary | PASS with named access limits | The public claims credit the optimization template, earlier continuous CAZAC families, Benoist's prime Gaussian-chirp transversality and full Björck regularity, and other cited antecedents. Novelty is bounded to the cited corpus. Some primary sources and the companion repository were unavailable; the limits remain visible. |
| A1 — adversarial consistency | PASS | The Opus audit rederived Theorems A--D and found no mathematical defect. Its prose, citation, and parameter findings were accepted and repaired. The raw report and root disposition are preserved under `audit/reports/`. |
| R1 — reproducibility and release integrity | IN PROGRESS | The paper rebuild, PDF inspection, solver reproduction, citation metadata, correction policy, and evidence boundary are complete. A clean commit, immutable tag, permanent archive, DOI verification, and provider byte-identity check remain pending. |

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

## Planned archive route and action ownership

The proposed route is the **Zenodo GitHub integration**, consistent with the
repository owner's recent releases. Before any immutable action, the user must
approve the exact candidate commit, tag `v0.1.0`, and GitHub Release. The user
must enable the repository in the authenticated Zenodo portal; agents will not
open or operate that portal. A later, separate approval will govern any living
metadata or public-site update.

## State transitions

| State | Condition | Status |
|---|---|---|
| DRAFT → CANDIDATE | P1, A1, and pre-freeze R1 pass; prose and artifacts agree. | Passed August 2, 2026 |
| CANDIDATE → TAGGED | Freeze one clean commit and create one immutable semantic tag. | Pending explicit approval |
| TAGGED → ARCHIVED | Archive the exact tagged tree and verify the downloaded provider file byte-for-byte. | Pending |
| ARCHIVED → ADMITTED | Activate and verify the DOI, reconcile living surfaces, and issue the final verdict. | Pending separate approval |

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
