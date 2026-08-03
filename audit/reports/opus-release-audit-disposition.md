# Disposition of the Opus release audit

This is the root maintainer's disposition of
[`opus-release-audit-draft.md`](opus-release-audit-draft.md). The raw audit is
preserved unchanged. This file records which findings were accepted and how
the release candidate addressed them.

## Lane dispositions

| Lane | Audit result | Disposition |
|---|---|---|
| Claims and public prose | Partial | Accepted. Removed release-process phrasing, restored Theorem C's conditional status, and made the nonzero and fixed-`τ` scope explicit. |
| Proof and parameter audit | Pass | Accepted. No mathematical defect was found in Theorems A--D. |
| Citations and prior art | Partial | Accepted. Added Benoist (2024), Proposition A.2, as the odd-prime projective-transversality predecessor and retained Proposition A.3 credit for full Björck regularity. |
| Reproducibility and release readiness | Partial | Accepted. Corrected the paper's preliminary schedule, made the optional phase constraints redundant by default, reran the solver, removed the misattributed Q6 statistic, and added stewardship files. |

## Must-fix findings

1. Process language in the date, theorem headings, remarks, and proof prose was
   removed or replaced with mathematical status language.
2. Theorem C is labeled **proved conditional on (R)** wherever its tier is
   summarized.
3. Benoist's prime-case transversality result is credited and the broader
   scope of Theorem D is stated narrowly.
4. The paper now gives the implementation's actual preliminary schedule,
   `τ = 1, 1/2, ..., 1/10`, and says explicitly that it is outside the
   fixed-`τ` theorem.
5. The release default `sigma = 1.0` makes the encoded phase-window constraints
   redundant. A fresh default run retained the reported structural result.
6. The Q6 `15/50` value is identified as output from the projection baseline,
   not the analyzed SOCP iteration; it is no longer offered as evidence for
   Q6.
7. `CITATION.cff`, `CORRECTIONS.md`, and `REPRODUCIBILITY.md` now provide the
   initial citation, stewardship, and evidence-boundary artifacts.

## Remaining publication gates

The candidate is still a draft. It has not been committed, tagged, released,
archived, assigned a DOI, or admitted. Those externally visible and immutable
steps require final candidate verification and explicit approval.
