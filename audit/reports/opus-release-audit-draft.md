# Process-separated AI release audit — convex-framework-cazac (draft working tree)

**Auditor:** Claude Opus 5, process-separated AI audit.
**Mode:** read-only. No file was edited, staged, committed, tagged, pushed, or
published; all builds and runs were done on copies in a session scratchpad.
This report file is the single permitted write.
**Standard applied:** <https://jeff-kline.github.io/posts/research-program/index.html>
(Draft 0.4, August 2026), read in full.
**Scope:** the uncommitted working-tree edits to `README.md`, `paper/main.tex`,
and `paper/main.pdf`, plus the tracked repository state supporting them.
**Email:** no email client, connector, API, or account was opened, used, or
requested.

This is **not** peer review, **not** independent expert validation, and **not**
a correctness certificate.

---

## Premise corrections (before the lanes)

- **File ownership.** The working-tree edits are **not root-owned**:
  `README.md`, `paper/main.tex`, `paper/main.pdf` are all `klinellc:staff`,
  mode `644`. The task's "root-owned" framing does not hold.
- **`git status --short --branch`** → `## main...origin/main` with
  `M README.md`, `M paper/main.pdf`, `M paper/main.tex`. Local `main` equals
  `origin/main`; the three edits are uncommitted, so **the currently published
  README on GitHub is still the pre-edit scorecard version** (verified against
  `raw.githubusercontent.com`). No tags, no releases.
- **The remote now resolves publicly** (anonymous `GET` → `200`; API reports
  `"private": false`, `pushed_at: 2026-07-26`). The old README's caveat that
  origin was "a private, out-of-date snapshot" is now stale, so the edit
  correctly deletes it. The companion `cazac-algorithm` repo → **404**, so the
  paper's "not publicly hosted" disclosure is accurate.

---

## Lane 1 — Claim and public prose: **PARTIAL**

The new README and the new abstract/introduction are a real improvement: title,
author, principal claims, evidence tiers, prior-work credit, and novelty
boundary now agree across README ↔ abstract ↔ intro ↔ verification ledger. I
checked each of the four principal results in all four places and found no
drift. Backelin/Popović/Björck–Saffari credit now appears in the abstract itself
(`paper/main.tex:62-64`), and README `61-82` states the contribution boundary
and disclaims global priority.

**Human-comprehensibility test: passes.** `main.tex:71-143` (new Introduction)
plus `145-192` (Setup) give a reader the definition of CAZAC, the convex body
`C_τ`, the iteration `(⋆)`, and the four theorem statements without any AI
summary. Hypothesis (R) is stated informally at `661-666` and exactly at
`733-741`; `K` is defined at `865`. A mathematically informed reader can get
from cold start to Theorem D's proof unaided.

### Must-fix

**1.1 — Process-history language on the title page.** `paper/main.tex:33`

```
\date{Draft release preparation\\August 2026}
```

This renders on page 1 of the shipped PDF (confirmed via `pdftotext`: "Draft
release preparation / August 2026"). It describes the internal workflow state,
not the document. Smallest disposition — replace with:

> `\date{Draft, August 2026}`

**1.2 — Editorial instruction addressed to future editors.**
`paper/main.tex:1156`

```
\begin{remark}[What is proved and what is not --- do not upgrade]
```

"do not upgrade" is an instruction to whoever edits next, not content for a
reader. Replace with:

> `\begin{remark}[What is proved and what is not]`

**1.3 — Audit-process aside inside a proof.** `paper/main.tex:901-902`

> "…the vanishing of a convolution against the *reality-defect* kernel $e$, not
> (a subtlety a compressed first pass can miss) mere reality of
> $\sum_sc'_s\phi(s+m)$ itself…"

The parenthetical describes how a reviewer might err, not the mathematics.
Replace the clause with:

> `--- the vanishing of a convolution against the \emph{reality-defect} kernel $e$, not mere reality of $\sum_sc'_s\phi(s+m)$ itself, which would be a strictly stronger, false requirement:`

**1.4 — Tier label broken by the edit.** `paper/main.tex:746` was changed from
"Proved conditional on hypothesis (R)" to:

```
\begin{theorem}[Theorem C --- Conditional on (R)]\label{thm:C}
```

But the tier vocabulary at `1326-1329` defines the tier as ***Proved*
conditional on (R)**, and the ledger entry at `1367-1369` still reads "*Proved
conditional on hypothesis (R)*". As now written, the theorem header is the only
one in the paper that omits "Proved", which reads as a *weaker* status than the
ledger assigns. Restore:

> `\begin{theorem}[Theorem C --- Proved conditional on hypothesis (R)]\label{thm:C}`

### Optional

- `paper/main.tex:51-52` — the abstract asserts "Every nonzero trajectory
  converges to a single fixed point" without the `τ ≥ 1` scope that Theorem A
  (`294`) carries. README `32-33` states it correctly. Same for README's
  blockquote `20-21` ("every trajectory", no nonzero-initialization condition).
- `paper/main.tex:1420-1421` — "none has been upgraded in the writing of this
  paper" is an unverifiable process assertion.
- `paper/main.tex:416` — "(This restriction is load-bearing, not cosmetic: …)"
  is editorial voice; the parenthetical's `n=2` computation is worth keeping,
  the framing is not.
- `paper/main.tex:353-356` — the IPUC novelty claim is bounded only by "as far
  as we have found"; README `81-82` carries the sharper boundary sentence.
  Consider putting that sentence in the paper too, since the paper travels
  separately.
- `paper/main.tex:556-566` — Theorem B(a)'s "exactly" characterization folds its
  own hypothesis ("time-domain modulus strictly less than 1 everywhere") into
  the conclusion clause. Exposition defect only; the proof is correct.

---

## Lane 2 — Proof and parameters: **PASS**

I re-derived Theorems A–D and the supporting lemmas from the definitions.
**I found no mathematical defect.** Detail of what I checked:

- **Prop. 2 (`211-223`)** — both chains verified:
  `‖x‖² = (1/n)Σ_k S_x(k) ≤ Nn` with equality iff `S_x ≡ Nn`; separately
  `‖x‖² ≤ Nn` at `τ=1` with equality iff all `|x_j(k)|=1`. Equality in the
  single quantity forces both conditions independently, so "max-norm ⟺ CAZAC"
  holds. Correctly scoped to `τ=1` (for `τ>1` only the spectral half is tight).
- **Prop. 3 / Theorem A (`229-331`)** — ABS Thm 2.9's three hypotheses check
  out: H1 with `a=½` is exactly Prop. 3(3); H2 is the *exact* identity
  `w = x^ℓ - x^{ℓ+1} ∈ ∂F(x^{ℓ+1})`, `‖w‖ = ‖x^{ℓ+1}-x^ℓ‖`, which is genuinely
  special to `f = ½‖x‖²`; H3 from compactness. The `x^0 ∈ C_τ` gap between
  Prop. 3 and Theorem A is handled honestly by the one-step re-indexing
  paragraph (`257-265`), and `x^0 ≠ 0 ⟹ x^1 ≠ 0` holds since the support
  function is positive.
- **Theorem B (`555-649`)** — (b) verified explicitly at `n=2`: `x*=(1,t)`,
  `t=√2-1`. Feasibility holds (`S_{x*}(0)=2`, `S_{x*}(1)=6-4√2≈0.343`), the
  decomposition `⟨x*,y⟩ = (1-t)Re y(0) + t Re F(y)(0)` is an identity, the bound
  `(1-t)+t√2 = 1+t² = ‖x*‖²` is tight, and equality forces `y=x*` uniquely.
  `‖x*‖² = 4-2√2 < 2`, so it is genuinely non-CAZAC and genuinely exposed. The
  `N=2` doubling argument is not a coordinatewise repeat and is correctly argued
  through the *coupled* constraint `S_y(0) ≤ 4`. Part (a)'s converse via
  per-frequency Cauchy–Schwarz is correct. **Coupled vs. single-sequence
  handling is sound throughout**: the shared multiplier `μ_k` across blocks is
  what makes `N≥2` pin only `Σ_j|x̂_j(k)|² = Nn`, and the paper's `N=2,n=2`
  witness (`x̂_0(0)=√3, x̂_1(0)=1`) is correct.
- **Boundary/parity cases** — `n=2` handled (`cor:D1` "odd or `n=2`"); the
  `n≤3` elliptope case is correctly excluded from `prop:ell3` and I verified the
  `n=2` claim `F = {r ∈ {-1,0,1}}` directly.
- **Lemma `lem:newton` (`668-696`)** — every constant checked. Under
  `‖z_i-x*‖ ≤ σ/(4M)` and `b_i ≤ σ²/(16M)`, recursion (∗) gives
  `b_{i+1} ≤ ¼b_i + (1/32)b_i < ½b_i`; `Σ‖d_i‖ ≤ 2b_0/σ ≤ σ/(8M)`, closing the
  induction at `σ/(8M)+σ/(8M)=σ/(4M)`. Sound.
- **Theorem C (`746-824`)** — Case 1/Case 2 constants reconcile exactly: Case 2's
  threshold `D(x) > σ_s²/64` matches the ceiling `(σ_s/(2√2))·ρ = σ_s²/64` on
  the other side. Part (b)'s chain
  `‖w‖ ≥ σ_s/(4√2) - σ_s/(32√2) = (7/32)σ_s/√2 ≥ σ_s/(8√2)` is arithmetically
  correct, and the identity `‖x‖²-⟨x,z⟩ = -½(Nn-‖x‖²)+½d²` verifies. **Part (c)
  is a genuine contradiction argument, not a squeeze**: an exact equality (from
  A) against a hard floor (from (b)) is jointly satisfiable only on `CAZ`. The
  post-arrival singleton-argmax step and the step-count bound
  `(Nn-‖x^0‖²)/c_0²` are both correct.
- **Regularity-to-finite-arrival implication (`733-741`)** — the surjectivity ⟺
  `dim K = 1` restatement is right: `K` always contains the constant vector,
  which is not in `H`, so `dim(K∩H) = dim K - 1`, and the annihilator inside
  `ℝ^{Nn}×H` is trivial exactly when `dim K = 1`.
- **Theorem D (`870-931`) — fully re-derived, both parities.** Step 0's
  dependency ⟺ `K` isomorphism verified from the real gradients. Step 1
  verified: odd `n` via `T_{m+s}=T_m+T_s+ms` and `T_n ≡ 0 (mod n)`; even `n` via
  `ω_{2n}^{-u(j+n)²}=ω_{2n}^{-uj²}`, which is an exact integer identity **only
  because `n` is even** — correctly stated. Step 2's correlation reduction and
  the reality-defect kernel `e` are right, and the paper is correct that
  dropping the conjugate term would force `ĉ' ≡ 0`. Step 3: I independently
  computed `ê(t) = n(ω^{u(σ-T_σ)} - ω^{uT_σ})` (odd, using `T_{-σ}=T_σ-σ`) and
  `ê(t)=n(ω_{2n}^{-uσ²}-ω_{2n}^{uσ²})` (even) — both match the paper exactly,
  and both vanish iff `σ² ≡ 0 (mod n)`. Step 4: `ĉ'(t)=ĉ(u^{-1}t)` and
  `u·Z_0 = Z_0` give root- and block-independence.
- **`d_max = ∏_p p^⌊a_p/2⌋` verified.** `t² ≡ 0 (mod n)` ⟺
  `v_p(t) ≥ ⌈a_p/2⌉` ∀p ⟺ `t ∈ m_0ℤ_n` with `m_0 = ∏_p p^⌈a_p/2⌉`;
  `|Z_0| = n/m_0 = ∏_p p^⌊a_p/2⌋ = d_max`. And `supp ĉ ⊆ m_0ℤ_n` ⟺ `c` is
  `d_max`-periodic, so `dim_ℝ K = d_max` — the two forms in the theorem
  statement are genuinely equivalent. `dim K = 1` ⟺ all `a_p ≤ 1` ⟺ `n`
  squarefree, giving Corollary D1. Corollary D2's `(N-1)n + 1` follows from
  `2Nn - (Nn+n) + dim K`.
- **Björck (`1081-1154`)** — Lemma `lem:bjaffine`'s affine decomposition verified
  for both residue classes from Björck's definition, and the DFT closure
  `b̂ = C·1 + Bg·χ + Ap·δ_0` follows from `F(1)=pδ_0, F(χ)=gχ, F(δ_0)=1`.
  Theorem `thm:E0`'s two systems force `β=γ=0` for the stated reasons
  (`sin α_0 > 0` since `1/(1+√p) ∈ (0,1)`; `sin β_0 > 0` since
  `(1-p)/(1+p) ∈ (-1,1)`). The scope disclaimer at `1231-1243` is accurate and
  appropriately refuses to upgrade.
- **The exposed non-CAZAC fixed point and the deterministic-obstruction
  conclusion are correct as stated**, and the paper is careful (`651-657`,
  README `109-110`) that this does not rule out probabilistic guarantees.

**Exposition defects only** (not math): items 1.3, and Theorem B(a)'s
tautological clause noted above.

---

## Lane 3 — Citations and prior art: **PARTIAL**

I fetched and read the actual sources rather than matching keywords.

**Verified correct, mechanism-level:**

| Cited claim | Verification |
|---|---|
| **Aktaş–Kroer** `AK25` — title, authors, 2025, arXiv:2505.00221 | ✔ exact |
| `AK25` Lemma 3.1 eq. (27) is `w_{k+1}=∇f(x_{k+1})-∇f(x_k)` | ✔ verified in source; for `f=½‖x‖²` the Lipschitz constant is exactly 1, so the paper's "inequality becomes equality" reading is right |
| `AK25` Thm 3.2 hypotheses = "strongly convex smooth objective, nonempty compact set, KL" | ✔ Assumption 1.1 + KL, verbatim |
| `AK25` Max-Cut app. factorizes via Burer–Monteiro, **§4.3** | ✔ §4.3 is "Simple Parallel Algorithm for the Max-Cut SDP Formulation"; Burer–Monteiro used there |
| **FKP21** Thm 1 = limit points exist, connected, all fixed | ✔ verbatim; single-point convergence appears only as an **unnumbered "Note"** requiring a *finite* fixed-point set — the paper's characterization is exactly right |
| FKP Thm 20 = "The vertices of `L_n` are the attractive fixed points of `T`" | ✔ verbatim |
| FKP Thm 20's proof: `‖M-X‖<1` ⟹ arrival in one step | ✔ verbatim |
| FKP Prop. 19 gives *one* escaping point; Def. 6 (attractive) vs Def. 7 (repelling) gap | ✔ Def. 7 requires *every* nearby point — the paper's distinction is correct |
| FKP Example 5 is an unrelated **three-dimensional cone** | ✔ verbatim |
| FKP: infinitely many fixed points for `n>3`, finite for `n=3` | ✔ ("when `n > 3` there are already an infinite number", Example 17 in `L_4`) |
| **Benoist (2024)**, arXiv:2406.11529, **Prop. A.3** = transversality at Björck–Saffari functions, both residue classes | ✔ exact — and Benoist's `θ = arccos(1/(√p+1))` for `p≡1 (4)` matches `lem:bjaffine`'s `α` exactly; his `arctan(√p)` for `p≡3 (4)` equals half the paper's `β` (`cos 2θ = (1-p)/(1+p)`), i.e. the same constant |
| **Amis et al.** arXiv:2509.05097 | ✔ real authors are Karine Amis, **Eloi** Boutillon, **Emmanuel** Boutillon — the repeated "E. Boutillon" in `main.tex:1458` is correct initials, not a duplication error |
| Attouch–Bolte–Svaiter, Bolte–Daniilidis–Lewis–Shiota, Journée et al., Lewis–Luke–Malick, Drusvyatskiy–Lewis, Chu, Popović, Björck, Björck–Saffari, Haagerup | bibliographic data consistent with the published records |

### Must-fix

**3.1 — Undisclosed prior art for the prime case of Corollary D1.** Benoist
(2024) — *already cited by this paper, for the Björck case* — contains
**Proposition A.2**:

> "Let `p ≥ 3` be prime and `g₀` be the gaussian function on `F_p`,
> `x ↦ g₀(x) := e^{2iπx²/p}`. Then the intersection `T ∩ F⁻¹T` is transverse at
> `[g₀]`."

Benoist's transversality (`T_p T ∩ T_p(F⁻¹T) = {0}` in `CP^{n-1}`, defined at
his §2.1) is — for `N=1`, modulo the global-phase quotient — the same condition
as hypothesis (R). `g₀` is a quadratic chirp, i.e. a Zadoff–Chu sequence for
prime `p`. So the statement in `paper/main.tex:950-955`:

> "Hypothesis (R) holds at every Zadoff--Chu tuple iff `n` is squarefree ---
> **in particular at every prime `n`**, odd or `n=2`."

has a published predecessor for the "prime `n`, odd" half, in a source the paper
reads and cites elsewhere. Theorem D remains substantially more general (all `n`
including even and composite-squarefree, the *exact* count `d_max` rather than
transversality alone, all `N`, all coprime roots, and the connection to
Theorem C's rate), so **the novelty of Theorem D survives** — but the credit is
not currently visible. Smallest disposition: append one sentence to the remark
at `1017-1023` or to Corollary D1:

> `For $N=1$ and odd prime $n$, transversality at the Zadoff--Chu point is also proved, by an independent Floer-homological method, in Benoist \cite{Benoist24}, Prop.~A.2; Theorem~D's contribution at those lengths is the exact count $\dim K=d_{\max}$ and its extension to composite and even $n$, to $N\ge1$, and to every coprime root.`

*Hedge:* I verified Benoist's statement and his definition of transversality,
but I did not construct a formal equivalence proof between his `T ∩ F⁻¹T`
transversality at `[g₀]` and this paper's `dim K = 1`. I judge them equivalent
for `N=1`; that judgment is the one link in this finding I did not fully prove.

### Optional

- `paper/main.tex:489-503` argues FKP's Theorem 1 does not deliver single-point
  convergence — **correct**, and I confirmed it against the source. But FKP's
  *abstract* says "We prove the process always converges to a fixed point." A
  reader comparing abstracts will see a flat contradiction. One clause would
  defuse it: note that FKP's abstract states set-level convergence loosely and
  that their Theorem 1 as proved gives only a connected limit set.
- `paper/main.tex:983-999` — Backelin's `(m-1)`-parameter family and
  Björck–Saffari's "parity condition removed by a Kronecker-product trick" I
  could **not** verify: the Backelin 1989 Stockholm report and the 1995
  *C. R. Acad. Sci.* note are not openly fetchable. Popović's GCL description
  (`991-999`) matches the standard account of that paper. Recorded as residual
  risk, not a finding.

---

## Lane 4 — Reproducibility and release mechanics: **PARTIAL**

**Checks run:**

| Check | Result |
|---|---|
| `pdflatex main.tex` ×2 (scratchpad copy) | ✔ exit 0 both passes, **21 pages**, no errors, no unresolved `??`, only 3 underfull-hbox warnings in the bibliography (`main.tex:1486-1500`) |
| Tracked `paper/main.pdf` vs. fresh build | **byte-length identical** (439 736 B); SHA differs only via embedded timestamp. The tracked PDF **is** current with the edited `main.tex` |
| PDF page-1 rendering | ✔ title, author, date, abstract render; `\path{}` macros render (hyperref supplies `url`) |
| Private-path / credential scan across all tracked `.md`/`.tex`/`.py`/`.txt` | ✔ **clean** — zero hits for home paths, usernames, hostnames |
| README local links (`paper/main.tex`, `paper/main.pdf`, `code/*`, `AUDIT-LEDGER.md`, `LICENSE`) | ✔ all resolve |
| Documented SOCP run, fresh venv in scratchpad (`numpy 2.0.2`, `scipy 1.13.1`, `cvxpy 1.7.5`, `clarabel 0.11.1`) | ✔ **4 of 5 output lines reproduce exactly**: `Max \|H\| 1.00000e+00`, `Min \|H\| 1.00000e+00`, `Size H (106, 106)`, `Condition(H) 1.00000e+00`. Wall time **79.5 s** vs. recorded 57.0 s (hardware-dependent) |
| `github.com/jeff-kline/convex-framework-cazac` | 200, public, **0 tags, 0 releases** |
| `github.com/jeff-kline/cazac-algorithm` | **404** — paper's disclosure accurate |

### Must-fix

**4.1 — The paper's stated presolve schedule contradicts the shipped code.**
`paper/main.tex:177-178`:

> "The paper's practical schedule presolves at `τ = 2 + 1/ℓ` for `ℓ = 1,…,10`,
> then fixes `τ = 1` for the main phase"

`code/cazac-socp-solver.py:172-175`:

```python
for it in range(L0):
    t.value = 1 / (it + 1)
```

i.e. `τ = 1, ½, ⅓, …, 1/10` — **decreasing to 0.1, not descending from 3 to
2.1**. This also falsifies the parenthetical at `paper/main.tex:258-259`
("hands the main phase a point in `C_{2.1}\setminus C_1`, not in `C_1`"): with
the shipped code the presolve output lies *inside* `C_1`, which makes the
`x^0 ∈ C_τ` hypothesis automatic. Harmless to every theorem (all are stated for
fixed `τ ≥ 1` and any measurable selection), but the paper misdescribes its own
reference implementation. Smallest disposition — replace `main.tex:177-181`
with:

> `The reference implementation's presolve schedule tightens $\tau=1/\ell$ for $\ell=1,\dots,10$ before fixing $\tau=1$ for the main phase; all theorems below concern the main phase at fixed $\tau\ge1$ (in particular $\tau=1$), and hold for any measurable selection rule, so the finitely many presolve steps do not affect any conclusion.`

and delete the now-false clause "hands the main phase a point in
`$C_{2.1}\setminus C_1$`, not in `$C_1$`" from `258-259`.

**4.2 — The recorded run exercises constraints that are not in `C_τ`.**
`code/cazac-socp-solver.py:111-116` adds quantized-phase side constraints that
are **active by default** (`q=2`, `sigma=0.2` at `:152-161`), giving
`s = cos(π/4)+0.2 ≈ 0.9071 < 1`. For a unimodular entry `e^{iθ}` this excludes
four arcs of ±24.8° around `π/4, 3π/4, 5π/4, 7π/4` — roughly **55% of the unit
circle**. The default program therefore optimizes over
`C_τ ∩ (phase window) ⊊ C_τ`, which is not the set the theorems are about.
README `127-144` presents the resulting output under the heading "Run the
reference SOCP implementation" with no such qualification. Smallest disposition
— insert after README `134`:

> `The script's default parameters (`q=2`, `sigma=0.2`) also impose a quantized-phase side constraint, which restricts the feasible set below the convex body `C_τ` that the theorems analyze. Set `sigma` large enough to deactivate it to run the unmodified iteration.`

**4.3 — Recorded output is no longer tied to a version or date.** README `136`
now reads "The current default run was previously recorded as:". The pre-edit
README carried "(ran 2026-07-26, ~57s at the script's default size)". The
standard requires recorded outputs be tied to the code and data version that
produced them; the edit removes the only tie. Replace README `136` with:

> `Recorded output from commit `b5a4577`, run 2026-07-26 at the script's default parameters (the reported wall time is machine-dependent; a 2026 laptop reproduced the four numeric lines exactly at ~80 s):`

(substituting the actual release commit).

**4.4 — Q6's headline number matches a *different* algorithm's tracked output.**
`paper/main.tex:1290-1293` attributes to *the iteration* `(⋆)`:

> "For `N=1`, empirical success probability decays from ≈1 at small `n` to
> **≈0.3 at `n=167` (50 trials** per configuration; `docs/main.tex`, Table 1, in
> the companion repository)"

`code/cazac-projected-solver.py:51-58` documents, verbatim, for the **IPUC-style
projection baseline** — which README `146-147` and `code/README.md:12-16` both
state is *not* the analyzed iteration:

```
attempts: 50 / successes: 15 / success_fraction: 0.3 / n: 167 / N: 1
```

Same `n`, same trial count, same fraction. Either Table 1 was produced by the
projection baseline and is mis-attributed to `(⋆)`, or it is a coincidence. The
cited source is unfetchable (404), so **I cannot resolve this**. Smallest
disposition: state in Q6 which algorithm produced Table 1, or drop the figure
until it can be reproduced publicly.

**4.5 — Missing stewardship artifacts (P1/R1).** No `CITATION.cff`; no version
identifier; no correction/withdrawal/supersession policy anywhere in the repo;
the BibTeX at README `181-190` has no version, date, or correction history.
These are required by the standard's *Release and stewardship* block. README
`3-6` honestly declares the repo **not yet admitted**, so these are gating items
rather than misrepresentations — but they are the gate.

### Optional

- `code/cazac-socp-solver.py:139` — comment says "the output matrix will be
  2nN x 2nN"; `getH` returns `2n × 2n` (106×106 at `n=53, N=2`), so the comment
  is wrong by a factor of `N`.
- `paper/main.tex:1486-1500` — three underfull hboxes in the
  `CazacAlgorithmRepo` bibliography entry (long `\tt` filenames). Cosmetic.
- `paper/main.tex:1412-1421` (Provenance) asserts the companion repo's scripts
  "together these reproduce every numerical claim in this paper" while the
  bibliography entry two lines later says a reader "cannot yet independently
  fetch this repository." Both are true of the author; only the second is true
  of a reader. Consider rewording to "reproduce, for the author, every numerical
  claim."

---

## Residual risks and inaccessible sources

1. **The `cazac-algorithm` companion repository is 404.** Every numerical claim
   in the paper — 64 Zadoff–Chu configurations, the 140-trial elliptope sweep,
   the 44-prime Björck determinant check, the `p=71` mpmath recomputation, the
   Popović `n=4` family, Q6's Table 1 — rests on it. The paper discloses this
   (`1496-1499`), and README `104-107` repeats it. Honest, but it means **no
   numerical claim in this paper is currently reproducible by a reader**, and
   finding 4.4 cannot be settled.
2. **Backelin (1989)** (Stockholm University report) and **Björck–Saffari
   (1995)** (*C. R. Acad. Sci.*) were not obtainable. Their described mechanisms
   and dimension counts are unverified by me.
3. **The Benoist Prop. A.2 ⟺ hypothesis (R) equivalence** underlying finding 3.1
   is my reading, not a proof (see hedge above).
4. **`AUDIT-LEDGER.md`** (64 KB, 873 lines) is dense internal process history —
   audit rounds, commit hashes, model tiers, "coordinator"/"fork" vocabulary. I
   did not audit its claims against the paper claim-by-claim. README `97-99`
   frames it correctly as "a historical record, not peer review," which is the
   disposition the standard asks for.
5. **The paper's Bolte–Sabach–Teboulle–Vaisbourd attribution**
   (`main.tex:509-511`) for AK25's underlying abstract descent theorem was not
   checked against AK25's Theorem 2.7.
6. I did not authenticate to GitHub or any portal; all remote reads were
   anonymous HTTP GETs. No external state was changed.

---

## Verdict summary

| Lane | Verdict | Must-fix count |
|---|---|---|
| 1 — Claim and public prose | **PARTIAL** | 4 |
| 2 — Proof and parameters | **PASS** | 0 (exposition-only findings) |
| 3 — Citations and prior art | **PARTIAL** | 1 |
| 4 — Reproducibility and release mechanics | **PARTIAL** | 5 |

The mathematics is the strongest part of this artifact — I re-derived all four
theorems and every constant and found nothing wrong. The release-preparation
edits improved claim consistency and prior-work credit substantially. What
remains is concentrated in three places: process-history language that survived
into the published PDF, a reference implementation whose presolve and constraint
set do not match the paper's description of it, and one undisclosed prior-art
overlap in a source the paper already cites.

This was a process-separated AI audit. It is not peer review, not independent
expert validation, and not a correctness certificate.
