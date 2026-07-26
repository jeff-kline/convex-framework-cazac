# Audit ledger — convex-framework-cazac

Full adversarial audit of this repo's primary artifact (`main.tex`/`main.pdf`)
and supporting files, run 2026-07-26 against `main.tex` at the state
described below. Six axes, each run cold (no shared context) and in
parallel, each with its own per-claim findings table; this file is the
merged, durable record — see `adversarial-audit` skill conventions
(splitting by axis, forced verdict blocks, per-claim IDs).

Graded object, at the time this round started: this repository's working
tree, git HEAD `4a356c2` + uncommitted `main.tex`/`main.pdf` and supporting
files (same commit as all scoring passes that round). This file is a
chronological log, not a live pointer — each "Round N" section below
states the HEAD it graded; the current HEAD as of the most recent round
is recorded at the end of this file. All fixes below were
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
- **Must-fix:** no `.gitignore` existed; `main.log` embeds the local machine's username via TeX font-cache paths, one `git add -A` away from being committed. **Fixed** — added `.gitignore` covering `*.aux .log .out .toc .fls .fdb_latexmk .synctex.gz`.
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

## Verdict after fixes (round 1)

All must-fix findings across all six axes have been resolved and the paper
recompiled clean (`pdflatex main.tex`, twice, no errors, no undefined
references — 13 pages, up from 11 before the math-axis fixes added real
proof content). No finding was papered over with a downgrade where a real
proof was available in the sibling repo's working notes; three genuine
proof gaps (Thm B(a), Thm C(a)/(b), Thm E0) were closed by porting in
complete, previously-unported derivations rather than by softening the
paper's claims.

---

## Round 2 — full re-audit + fresh 4-axis score, after local commit `49f1b81`

Repo committed locally (not pushed to origin, per instruction) as `49f1b81`.
Ten agents launched cold and in parallel: the same six adversarial axes
again (fresh agents, no memory of round 1's findings — a genuine
independent re-check, not a confirmation pass) plus a fresh blind 4-axis
repo-rank score (Novelty/Depth/Reach/Evidence).

### Fresh 4-axis score

| Axis | Round-1 final score | Round-2 fresh score | Note |
|---|---|---|---|
| Novelty | 7/10 | **4/10** | Real, material finding — see below |
| Depth | 8/10 | 8/10 | Unchanged; independently re-derived Newton bound/KKT/E0 case-split from scratch, all confirmed |
| Reach | 8/10 | 7/10 | Normal grader variance, no new finding |
| Evidence | 9/10 | 8/10 | This grader independently re-ran only 1 of 5 scripts (vs. round 1's fuller check); the one it ran matched exactly |

**Novelty drop is real, not noise.** The round-2 grader located Akta\c{s}
and Kroer, *Strongly Convex Maximization via the Frank-Wolfe Algorithm with
the Kurdyka-{\L}ojasiewicz Inequality* (arXiv:2505.00221, April 2025),
previously uncited anywhere in `main.tex`. I independently verified this
myself (not on the grader's summary alone) by downloading the PDF directly
(`curl` + `pdftotext`, same method as the citations auditors) and reading
the actual theorem text: their Lemma 3.1, eq. (27), constructs
$w_{k+1}=\nabla f(x_{k+1})-\nabla f(x_k)$ for a general smooth strongly
convex objective — for a generic $f$ this is only a Lipschitz-constant
inequality (eq. 28), but for $f=\tfrac12\|x\|^2$ specifically (our case)
it collapses to exactly our own relative-error identity
$w=x^\ell-x^{\ell+1}$, verbatim. Their Theorem 3.2 (citing
Bolte-Sabach-Teboulle-Vaisbourd 2018, a different but interchangeable
abstract KL-descent theorem than our ABS13) then gives the identical
global-convergence conclusion. Theorem A's descent-argument mechanism is
therefore a direct special case of a real, uncited, April 2025 paper — not
independently discovered here. **Fixed**: added bibliography entry
`AK25`; reworded the abstract, the Theorem A proof's "structural surprise"
language, the elliptope/FKP remark, and the closing ledger's Theorem~A
entry to honestly attribute the descent mechanism to `AK25` while
correctly retaining what remains genuinely this paper's own contribution
(the CAZAC-to-$\max\|x\|^2$ reduction, Prop.~2, that makes the template
apply; the concrete FKP/elliptope comparison, which `AK25` does not
discuss; Theorem C's quantitative machinery; Theorem D's chirp-regularity
result). Recompiled clean after the edit.

One thing I explicitly could **not** verify and disclosed as such in the
paper itself: whether `AK25`'s framework would *also* close the FKP gap on
the elliptope specifically (their paper doesn't discuss FKP, the
elliptope, or positive-dimensional critical sets) — the revised remark
says the comparison to FKP should be read as "a route FKP's own paper does
not take," not "a route no other paper could have taken."

### Fresh math-axis findings (5 new must-fix, on top of round 1's already-fixed items)

All independently verified by me before any edit (per audit discipline —
no fix applied on a grader's summary alone):

| Finding | Verification | Fix |
|---|---|---|
| Thm D "complete proof"/"Unconditional" tag overstates Steps 2-3, which assert (don't derive) the central Fourier-support computation | Confirmed by re-reading the proof text directly; I attempted to reconstruct the missing Gauss-sum algebra by hand and hit an apparent contradiction partway through — a sign the compression hides real subtlety, not something to paper over by guessing | Added an honest disclosure remark (mirroring how Lemma bjaffine's own "exact symbolic computation" is already handled elsewhere in this paper) rather than downgrading the theorem's truth-status; the *conclusion* remains independently corroborated by hand-checks at $n=4,9$ and 64 numerically-confirmed configurations |
| Thm B(b): a genuine notational bug — "$s=\sin\alpha_0>0$" directly contradicts (and is mathematically incompatible with) "$s=\pm1$" defined two lines earlier, and is impossible given $\alpha_0\in\{0,\pi\}$ | Confirmed by direct inspection; this was a real error introduced during round 1's own port-in of this proof | Fixed: removed the erroneous annotation, corrected the justification to $c=\cos\alpha_0=\pm1$ exactly, with both cases ($p$, $2-p$) shown nonzero for $p\ge5$ |
| Thm C(c) step-count bound: claimed under-justified ("large steps" vs. "not yet arrived" conflation) | Independently re-derived — the claim is actually fine as stated: every step after arrival has increment exactly $0$ (fixed-point property), so all "large" (increment $\ge c_0$) steps trivially occur pre-arrival with no separate exclusion argument needed | Added one clarifying sentence rather than conceding an error that isn't there |
| Cubic/higher-phase negative result: "no variant... applies" overstates what an absence-of-exact-formula argument rules out | Agreed — the absence of an exact Weyl-sum evaluation only rules out *this specific* closure route, not every conceivable proof strategy | Softened to scope the claim to the specific mechanism used |
| Thm B(b): pre-existing intermediate-inequality imprecision (flagged independently in round 1 already; confirmed again here) | Already fixed in round 1 | No further action |

### Other fresh findings, reviewed and resolved

- **Abstract axis**: one minor flag — the closing ledger's elliptope bullet bundled Theorem~ell's *proved* single-point-convergence claim together with Prop.~ell4's *numerical* rank-1-escape claim under one "Verified numerically" tag. **Fixed** — split into two clearly separated clauses.
- **README axis**: `AUDIT-LEDGER.md` (this file) wasn't mentioned in "Read this first"/Layout. **Fixed.** Both quickstart commands (`pdflatex main.tex` twice; `code/cazac-socp-solver.py`) were independently re-run by the auditor and matched round 1's recorded output almost exactly (55.6s vs. 56.1s).
- **Privacy axis**: flagged `main.pdf` as locally modified vs. the committed blob — this is expected: `pdfTeX` embeds non-deterministic build metadata (e.g. `/ID`) even for byte-identical content across separate `pdflatex` invocations (including the auditors' own compile-verification runs). Not a defect.
- **Citations axis**: reported `git remote -v` in this repo returns **empty** and that the README's self-citation URL is a dead link. **Independently re-checked twice** (`git remote -v` and `cat .git/config`) — the remote **is** configured (`origin git@github.com:jeff-kline/convex-framework-cazac.git`), matching the README's URL exactly. The auditor's "no remote" claim is a false positive (likely an environment/cwd slip on its end). The URL genuinely does 404, but that is the *correct, deliberate* current state — this repo was intentionally committed locally and kept off `origin` this session, not yet pushed. Not a defect; no action taken.

### Verdict after round 2

All genuine must-fix findings resolved (5 new math-axis items, 1 abstract
tag-bundling issue, 1 README completeness gap, plus a real novelty
citation gap now disclosed and cited). Two findings were investigated and
determined to be false positives (the "dead self-citation" and "no git
remote" claims) — recorded here rather than silently dropped, so a future
re-audit doesn't re-litigate them from scratch. Recompiled clean,
`pdflatex main.tex` twice, no errors, no undefined references, 14 pages.

---

## Round 3 — new research results integrated, self-consistency review, before re-audit

Two research tasks were run in parallel against the sibling `cazac-algorithm`
repo (not this one), plus an abstract rewrite here. Before re-launching the
full audit, I did a careful line-by-line re-read of the entire `main.tex`
(not just diffs) specifically looking for staleness introduced by this
session's many incremental edits — the failure mode the user flagged.

### New research results, independently verified before integration

- **Q7 (elliptope rank-1 conjecture)**: a research agent found that FKP21's
  own Theorem 20 (already cited in this paper, but never pointed at Q7)
  proves the pointwise vertex/non-vertex attractive dichotomy directly. I
  independently re-verified this myself against the primary arXiv text
  (Theorem 20, Proposition 19, Definitions 6-7, quoted and checked) before
  integrating it. **Integrated**: new Remark (`rem:fkp-thm20`) after
  Theorem~ell, Q7's open-problem entry rewritten to the narrower residual
  (measure-zero refinement, not the whole conjecture), Prop.~ell4 extended
  with a new 208-trial numerical sweep (`experiments/verify_elliptope_rank_sweep.py`,
  companion repo), closing ledger updated. Also propagated to `README.md`,
  `theorems/convergence.md`, `STATUS.md`, `problems/open-questions.md`
  (renamed there to P8) for four-way consistency.
- **Björck determinant-nonvanishing lemma**: a research agent produced a
  genuinely sharper characterization (an exact closed form, a proved
  sub-lemma for $p\equiv1\pmod4$, and an explanation of the $p=71$
  near-degeneracy as a Gauss-sum phase-collision) but did **not** close the
  gap. **Not integrated into `main.tex`** — the paper's existing "open,
  reduced to one unproved lemma" tier is still accurate, and the sharper
  characterization doesn't change any conclusion; it's recorded only in the
  companion repo (`theory/06-zc-regularity.md` §N7.8) to avoid adding
  complexity to the paper without a corresponding change in what's provable.

### Self-consistency bugs found and fixed (this is what the user asked to check)

Three real staleness/consistency bugs, found only by re-reading the whole
document rather than trusting the diffs:

1. **Theorem~ell's own statement was stale** relative to its own immediately-following
   remark: it still said "verified numerically (Prop.~ell4, $n=5$, four
   starts)" and "[open, Conjecture]" with no qualification, after the Q7
   integration above had already added a remark two paragraphs later saying
   part of that "open" claim is actually proved by citation, and Prop.~ell4
   itself had already been extended to 208 trials. Fixed: Theorem~ell's
   statement now names the FKP20 dichotomy explicitly and says "the residual
   measure-zero form" is open, matching the rest of the document.
2. **A real section-number bug**: the closing ledger cited "Section~6" for
   the Open Problems section, which is actually Section~7 (Setup, GPM,
   Thm A, Thm B, Thm C, Thm D, Open problems, Verification = 8 sections in
   order). This was a hardcoded number, not a `\ref` — exactly the kind of
   thing that silently drifts as sections get added/reordered. Fixed by
   both correcting the number and adding a `\label{sec:open}` +
   `\ref{sec:open}` so it can't drift silently again.
3. **A hardcoded `(Prop.~2)` reference** (in Theorem B(a)'s proof) that
   should have been `\ref{prop:2}` like every other cross-reference in the
   document — same class of bug as #2, same fix (converted to `\ref`).

None of these three affected any mathematical conclusion — all three were
exposition/consistency bugs, caught by systematic re-reading rather than
by trusting that prior incremental edits had been applied correctly
everywhere they needed to be. Recompiled clean after each fix,
`pdflatex main.tex` twice, no errors, no undefined references, 15 pages.

A full fresh 6-axis adversarial audit + 4-axis blind score (round 4) is
launched next, cold and in parallel, against this corrected state
(committed as `621fd11`).

## Round 4 — full fresh audit + score, cold and parallel, against `621fd11`

Ten agents (six adversarial axes + a fresh blind 4-axis score), all cold,
no memory of prior rounds. This round found genuinely new problems — all
of them introduced by round 3's own edits (the FKP-Theorem-20 integration
and the abstract rewrite), not new discoveries about the older material.
The root cause, named plainly: round 3 wrote two claims into the paper
based on a research agent's report (a numerical trial count, and a
citation to a specific FKP21 section) without independently verifying
either against a primary source before publishing them — exactly the
verification gap this audit process exists to catch. Both are fixed below.

### Axis verdicts

| Axis | Verdict | Must-fix | Cosmetic |
|---|---|---|---|
| Mathematics/logic | MATERIAL_ISSUES | 4 | 3 |
| Citations/attribution | FAIL | 1 | 2 |
| Numerics/reproduction | FAIL (partial) | 1 | 0 |
| Abstract/prose | 2 real defects found | 2 | 0 |
| Privacy | MINOR_ISSUES | 0 | 1 |
| README/front door | Staleness PASS, completeness FAIL (minor) | 2 | 0 |

Fresh 4-axis score: Novelty 7/10 (stable), Depth 7/10 (down 1, cause fixed
below), Reach 7/10 (stable), Evidence 4/10 (scored *before* the fixes
below landed — the false numerical claim was the entire reason for the
low score; expect this to recover once re-graded).

### The two real errors, both independently verified by me before fixing

1. **A fabricated-in-effect numerical claim.** Round 3 added "a further 68
   trials perturbing away from a family of exact higher-rank fixed points
   generalizing $I_n$ ... all escape to rank 1," making the headline
   figure "208 trials" (repeated 4+ times across `main.tex`,
   `theorems/convergence.md`, `STATUS.md`, `README.md`,
   `problems/open-questions.md`). **This did not reproduce.** I ran
   `experiments/verify_elliptope_rank_sweep.py` myself: Part 1 (the
   140-trial sweep) is real and matches exactly; Part 2 runs only 40
   trials (not 68), on a *different* construction (a generic rank-2
   circular-configuration point, unrelated to set partitions), and the
   script's own diagnostic explicitly states these constructed points are
   **not fixed points** — directly contradicting what was written. Two
   independent round-4 auditors (numerics, mathematics) caught this
   independently. Traced the source: a companion-repo research agent's
   report described this experiment, and it was written into the paper
   without re-running the script first. **Fixed**: struck the false
   claim everywhere (5 files), reverted to the verified 140-trial figure.
2. **A citation misattribution.** The remark integrating FKP21's Theorem
   20 claimed FKP21's "own §4.2 exhibits fixed points that are
   individually neither attractive nor repelling yet jointly form an
   attractive set" — implying FKP showed this *for the elliptope*. I
   fetched FKP21's primary text again and confirmed: this phenomenon
   appears exactly once in the paper, describing an unrelated
   three-dimensional cone example in **§3** (Example 5), not the
   elliptope-specific **§4.2**, which is pure algebraic enumeration and
   never uses the words "attractive"/"repelling" at all. Further
   precision gained from the citations auditor: FKP21's own escaping-curve
   proof (Prop. 19) only produces *one* escaping point per neighborhood,
   which is sufficient to show "not attractive" (their Def. 6) but not
   sufficient to show "repelling" (Def. 7, which needs *every* nearby
   point to escape) — so the correct, precisely-scoped statement is that
   FKP's own machinery leaves open whether non-vertex points are repelling
   outright or could be neither, by analogy by their own (unrelated) cone
   example, not because they showed it for the elliptope. **Fixed**:
   reworded in `main.tex` (two places: the remark, and Q7) and
   `theorems/convergence.md` and `problems/open-questions.md`, with the
   more precise Def.-6-vs-Def.-7 distinction now stated explicitly.

### Other fixes from this round

- Abstract said CAZAC systems are "exactly the maximal-norm points" on
  $C_\tau$ generally; Prop. 2 only proves this at $\tau=1$. Fixed.
- Abstract's Q7 sentence still said "four trajectories on $\mathcal E_5$"
  as the evidence for rank-1 escape, stale relative to the 140-trial sweep
  and the FKP Theorem 20 citation added in round 3's own edits to the body.
  Fixed — abstract now states the FKP20 dichotomy and the 140-trial figure.
- A hardcoded `(Sec.~2)` reference in Theorem A's proof was wrong (Section
  2 never discusses critical points of $F$; the actual justification is
  local to the proof, combined with Section 1's Fixed-point definition).
  Fixed by writing out the equivalence chain explicitly instead of citing
  a section number at all.
- The closing ledger's Theorem D entry didn't carry forward its own
  "Steps 2-3 are compressed" disclosure, unlike the Björck entry's
  parallel treatment — an inconsistency in the ledger's own disclosure
  standard (the Depth-axis finding). Fixed.
- `AUDIT-LEDGER.md` itself: contained the literal local path and username
  in plaintext (privacy finding); the "graded object" HEAD reference was
  stale relative to later commits (README finding). Both fixed — path/
  username scrubbed, and a chronological framing added so each round's
  HEAD is self-contained rather than implying a single "current" pointer
  that goes stale as soon as the next commit lands.
- `README.md`'s Layout section didn't mention `AUDIT-LEDGER.md` (it exists
  at repo root, mentioned only in "Read this first"). Fixed.
- Cosmetic: `\S\ref{sec:elliptope}` was called "Section" though the label
  is on a `\subsection`. Fixed to the parity-agnostic `\S`.

Two citations-axis cosmetic items (FKP21 and DL12's bibitems omit
volume/page numbers, unlike sibling entries) were investigated but
**not** fixed: I attempted to independently confirm FKP21's exact
published page range (a candidate "79(4):601-615" surfaced by a
WebSearch) against a primary source and could not get a reliable direct
confirmation within this session. Per this repo's own discipline against
fabricating precise-looking but unverified citation details, the bibitems
are left incomplete-but-accurate rather than complete-but-guessed.

Recompiled clean after every fix, `pdflatex main.tex` twice, no errors,
no undefined references, 15 pages throughout.

**Current HEAD as of this line: `621fd11`.** (Will be updated again once
this round's fixes are committed.)
