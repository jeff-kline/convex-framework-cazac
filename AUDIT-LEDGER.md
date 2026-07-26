# Audit ledger — convex-framework-cazac

Full adversarial audit of this repo's primary artifact (`main.tex`/`main.pdf`)
and supporting files, run 2026-07-26 against `main.tex` at the state
described below. Six axes, each run cold (no shared context) and in
parallel, each with its own per-claim findings table; this file is the
merged, durable record — see `adversarial-audit` skill conventions
(splitting by axis, forced verdict blocks, per-claim IDs).

Graded object: working tree of `/Users/klinellc/Documents/convex-framework-cazac`,
git HEAD `4a356c2` + uncommitted `main.tex`/`main.pdf` and supporting files
(same commit as all scoring passes this session). All fixes below were
applied by the coordinator after independently validating each auditor's
report against the artifact (re-reading source material, in one case
re-deriving proofs from the sibling repo's `results/` directory) — no fix
was applied on the strength of a summary alone.

## Axis verdicts (initial pass, before fixes)

| Axis | Verdict | Must-fix | Cosmetic |
|---|---|---|---|
| Mathematics/logic | MATERIAL_ISSUES | 6 | 1 |
| Citations/attribution | CLEAN | 0 | 2 |
| Numerics/reproduction | MINOR_ISSUES | 3 | 0 |
| Abstract/prose | MINOR_ISSUES | 0 | 4 |
| Privacy | MINOR_ISSUES | 1 | 2 |
| README/front door | MATERIAL_ISSUES | 2 | 1 |

## Per-claim findings — mathematics/logic (27 claims enumerated, 27 checked)

| ID | claim | outcome (initial) | fix applied | outcome (now) |
|---|---|---|---|---|
| Props 1-4 | generalized power method, max-norm=CAZAC, monotone ascent, limit points | CONFIRMED | — | CONFIRMED |
| Thm A | global convergence via ABS13 KL descent | CONFIRMED (cite-trust caveat: ABS13 itself independently fetched by citations axis, resolved) | — | CONFIRMED |
| Prop ell1-4, Thm ell | elliptope semialgebraicity/transfer/continuum/3-tier convergence | CONFIRMED | — | CONFIRMED |
| Thm B(a) | flat-spectrum fixed-point characterization | **GAP** — "direct computation" asserted, zero computation shown | Ported full KKT/support-function proof from `cazac-algorithm/results/T2/findings.md` Theorem T3 into `main.tex` | CONFIRMED |
| Thm B(b) | exposed point $x^*=(1,\sqrt2-1)$, $n=2$ | CONFIRMED, cosmetic defect (self-contradictory $\hat y$ normalization) | Dropped the spurious $/\sqrt2$ normalization; proof now uses $F(y)(0)$ directly, matching source `results/T2/findings.md` Theorem T7 | CONFIRMED, defect fixed |
| Thm B(c) | no deterministic theorem possible | CONFIRMED | — | CONFIRMED |
| Thm C(a) | sharp linear growth via Newton-type error bound | **GAP** — load-bearing Newton bound asserted, never derived or cited | Ported full Lemma (quantitative Newton-type error bound for quadratic maps) + balanced-constraint-map machinery ($\Phi_s$, zero-set/Lipschitz/(R)-equivalence lemmas) + Case-1/Case-2 proof, from `cazac-algorithm/results/T5/report.md` Lemma 2 + Theorem 1 | CONFIRMED |
| Thm C(b) | KL exponent 0 via normal-cone inequality | **GAP** — inherited from (a); constant $c_0$ not derivable as written | Ported full derivation (normal-cone inequality + Cauchy-Schwarz + explicit constant chain) from `results/T5/report.md` Theorem 2 | CONFIRMED |
| Thm C(c) | finite exact arrival, contradiction argument | CONFIRMED airtight *given* (a)/(b) | (a)/(b) now proved, so (c) no longer rests on an unproved premise | CONFIRMED, unconditionally within the theorem's own hypothesis (R) |
| Thm C remark | "genuine proof by contradiction, not a heuristic squeeze" | **OVERCLAIM** at the time (since (a)/(b) weren't shown) | No text change needed — claim is accurate now that (a)/(b) are proved | CONFIRMED |
| Thm D | $\dim K=d_{\max}$, all $n$ | CONFIRMED (two load-bearing identities independently re-derived at $n=4,9$) | — | CONFIRMED |
| Cor D1, D2 | (R) iff squarefree; tangent dimension | CONFIRMED | — | CONFIRMED |
| Legendre-closure / cubic-negative props | — | CONFIRMED | — | CONFIRMED |
| Lemma bjaffine, bjequiv | Björck affine closure, $G$-module structure | CONFIRMED | — | CONFIRMED |
| Thm E0 | $K\cap T=\mathrm{span}\{\mathbf1\}$ | **STATUS-TAG MISMATCH** — theorem header said "Proved," proof headed "Proof sketch," remark called it "complete proof" | Ported the full displayed 3-equation system (both $p\bmod4$ classes) from `cazac-algorithm/theory/06-zc-regularity.md`; relabeled proof environment from "Proof sketch" to "Proof" | CONFIRMED, all three status statements now agree |
| Björck remark, ledger | proved/unproved boundary; ledger's "full"/"complete" language | **must-fix**, tied to Thm C(a)/(b) and Thm E0 gaps above | Same fixes as above resolve this — boundary was always drawn in the right place, the "complete" claims are now actually true | CONFIRMED |

**Coverage: 27/27 claims checked** (2 residual non-error scope notes: ABS13's exact theorem statement was verified by the *citations* axis, not re-derived here; Thm D's Steps 2-3 Gauss-sum algebra was checked by numerical consistency at $n=4,9$, not full symbolic rebuild — both explicitly non-error scope limits, not findings).

## Per-claim findings — citations/attribution (10/10 checked)

| ID | outcome | must-fix |
|---|---|---|
| ABS13 | CONFIRMED (fetched PDF, Thm 2.9 verified verbatim) | no |
| BDLS07 | CONFIRMED | no |
| FKP21 | CONFIRMED — the single most load-bearing citation in the paper, independently re-verified verbatim against the primary source a third time this session | no |
| JNRS10 | CONFIRMED | no |
| LLM09 | CONFIRMED (cosmetic: preprint vs. published title wording differs, not an error) | no |
| DL12 | CONFIRMED (cosmetic: bib entry omits later journal publication) | no |
| Chu72 | CONFIRMED | no |
| Bjorck90 | CONFIRMED (page range not independently re-verified against a primary source) | no |
| CazacAlgorithmRepo bibitem | CONFIRMED no URL asserted; repo confirmed to have no git remote | no |
| README.md's own bibtex URL | CONFIRMED matches actual `origin` remote exactly | no |

**Verdict: CLEAN. No fabrication found anywhere.**

## Per-claim findings — numerics/reproduction (6/6 checked, all scripts run via required venv)

| ID | claim | outcome | fix applied |
|---|---|---|---|
| 1 | Elliptope $n=5$, 4 trajectories, escape-to-rank-1 | CONFIRMED exactly | — |
| 2 | Björck affine-closure check attribution | **must-fix** — "44 primes" wrongly attributed jointly to two scripts; only one produces that count | Reworded to attribute the 44-prime figure solely to `verify_bjorck_isotypic.py`, and separately name `verify_bjorck_regularity.py`'s disjoint 14-prime check | 
| 3 | 44-prime Björck determinant check, incl. $p=71$ near-degeneracy | **must-fix** — "60-digit precision" claimed for all 44 primes; script only reruns it at $p=71$ | Reworded to scope the 60-digit recomputation to the $p=71$ spot-check specifically |
| 4 | ZC odd-$n$, 21 configurations | CONFIRMED exactly | — |
| 5 | Even-$n$, 43 configurations | **must-fix** — "$n\in\{2,4,\dots,50\}$" notation implies the full 25-term sequence; only 13 sparse values tested | Replaced with the literal 13-element set |
| 6 | All 5 cited scripts exist at stated paths | CONFIRMED | — |

**All three must-fix items resolved in `main.tex`.**

## Per-claim findings — abstract/prose (14/14 checked)

CLEAN on all factual/tiering claims (Theorem C's conditional status, Björck's
partial/open status, the elliptope's three-way tier split all verified
three-way consistent: abstract vs. body vs. closing ledger). Full-document
grep for revision-history language found zero problematic hits. Four
cosmetic findings:
- Title page said "consolidated working draft" — **fixed** (removed, now just "July 2026").
- Provenance paragraph used internal multi-agent-process jargon — **fixed** (reworded, independently flagged by the privacy axis too).
- Two abstract sentences (Theorem C, elliptope) are dense enough that careless skimming risks misattributing which clause is conditional/numerical — **not fixed** (style-only, content independently verified accurate; left as a known readability note, not a correctness issue).

## Per-claim findings — privacy (15/15 files inventoried and checked)

No secrets, no email/phone leakage beyond the intended `\author{Jeffery Kline}`
(matches the author's own house style), no stray off-topic files, clean PDF
metadata. Two findings:
- **Must-fix:** no `.gitignore` existed; `main.log` embeds the local username `klinellc` via TeX font-cache paths, one `git add -A` away from being committed. **Fixed** — added `.gitignore` covering `*.aux .log .out .toc .fls .fdb_latexmk .synctex.gz`.
- Cosmetic: provenance paragraph's internal jargon (same finding as the abstract axis, independently) — **fixed**, same edit.

## Per-claim findings — README/front door (15/15 checked)

- **Must-fix S8:** README claimed `problems/open-questions.md` uses "Q1–Q7" headings; the file actually used `P1`–`P7` plus one orphaned `Q7` section that itself cited a nonexistent "Q6". **Fixed** — renamed the orphaned section to `P8`, fixed its internal cross-reference to `P1`, and corrected README's description to `P1–P8` (noting `main.tex`'s own condensed section separately uses `Q1–Q7` for its shorter form).
- **Must-fix S9:** `theorems/convergence.md`'s opening summary still said "Three results are established" with no distinct Theorem D section, stale relative to `main.tex`'s four-theorem structure. **Fixed** — updated the summary to "Four results," added Theorem D to the abstract paragraph, and gave it its own `## 6. Theorem D` heading (renumbering the following section to `## 7`).
- S10-S15: all CONFIRMED (compile instructions actually run and verified clean; bibtex/license/layout/status pointers all accurate).
- S12 (recommended, not a hard must-fix): no runnable quickstart with recorded output for `code/`, unlike the sibling `turyn-converse` repo's house style. **Fixed** — added a "Verify something in a minute" section to README.md with an actually-executed, verified `code/cazac-socp-solver.py` run (recorded output, timestamped).

## What was not checked (explicit)

- Björck-1990's exact normalization convention ($b(0)=1$) was accepted as standard, not independently sourced against the original 1990 paper.
- The $G$-module isotypic-dimension count in Lemma bjequiv was accepted as standard representation theory, not rederived from first principles.
- Thm D's Steps 2-3 (Gauss-sum correlation-equation algebra) were checked by numerical consistency at two values of $n$, not a full symbolic rebuild.
- `code/cazac-projected-solver.py` (the projection baseline) was not executed during this audit; only `cazac-socp-solver.py` was run for the README quickstart.

## Verdict after fixes

All must-fix findings across all six axes have been resolved and the paper
recompiled clean (`pdflatex main.tex`, twice, no errors, no undefined
references — 13 pages, up from 11 before the math-axis fixes added real
proof content). No finding was papered over with a downgrade where a real
proof was available in the sibling repo's working notes; three genuine
proof gaps (Thm B(a), Thm C(a)/(b), Thm E0) were closed by porting in
complete, previously-unported derivations rather than by softening the
paper's claims.
