# Status of every principal claim

Tiers, in decreasing strength:

- **Proved** — complete proof given in `theorems/convergence.md`.
- **Proved, conditional** — complete proof given, but resting on a stated
  hypothesis (here, regularity (R)) that is not established in full
  generality.
- **Numerically verified** — no proof; a script in the companion repository
  `cazac-algorithm`'s `experiments/` directory reproduces the claim to a
  stated tolerance (named at each claim below).
- **Conjecture / open** — stated as a target in `problems/open-questions.md`,
  not established either way.

| Claim | Tier |
|---|---|
| Iteration is a generalized power method for $\max_{C_\tau}\|x\|^2$ (Prop. 1) | Proved |
| $\mathrm{CAZ}(n,N)=\arg\max_{C_1}\|\cdot\|$, optimal value $Nn$, attained for every $n,N$ (Prop. 2) | Proved |
| Monotone norm ascent, square-summable increments (Prop. 3) | Proved |
| Limit points of the iteration are fixed points; CAZAC points are fixed points (Prop. 4) | Proved |
| **Theorem A** — full-sequence convergence to a single fixed point, any measurable selection, any $\tau\ge1$ | **Proved, unconditional** (descent mechanism is a special case of Aktaş–Kroer 2025, arXiv:2505.00221; CAZAC-specific content is the reduction to $\max\|x\|^2$, Prop. 2) |
| **Theorem B(a)** — census of slack-amplitude ("flat-spectrum") fixed points | Proved |
| **Theorem B(b)** — exposed non-CAZAC fixed point exists at $n=2$ (and its $N=2$ doubling) | Proved |
| **Theorem B(c)** — no deterministic convergence-to-CAZAC theorem is possible | Proved (immediate from B(b)) |
| Complete symmetry-class census of fixed points at $n=2$, $N=1$ (four classes) | Proved |
| Complete fixed-point census for general $n$ | Open (P2) |
| $\dim\mathrm{CAZ}(n,N) \ge (N-1)n+1$ at smooth points | Proved |
| Equality $\dim\mathrm{CAZ}(n,N) = (N-1)n+1$ at Zadoff–Chu points | Proved (elementary chirp argument, all $n$) |
| Theorem C, part (c) (finite exact arrival) — full contradiction-based proof | **Proved under (R)**, same tier as (a)/(b) |
| **Theorem C** — sharp linear growth, KL exponent $0$, finite exact termination at a regular CAZAC limit | **Proved, conditional on hypothesis (R)** |
| Hypothesis (R) at Zadoff–Chu points, odd $n$ | **Proved: holds unconditionally iff $n$ is squarefree** |
| Hypothesis (R) at Zadoff–Chu points, even $n$ | **Proved**: $\dim K = \max\{d:d^2\mid n\}$, no parity restriction |
| Hypothesis (R) at non-Zadoff–Chu CAZAC points (e.g. Björck-type, prime $n$) | **Partial**: proved that $K$ meets the decimation-symmetry-invariant subspace only in $\mathrm{span}\{\mathbf 1\}$ ($K\cap T=\mathrm{span}\{\mathbf1\}$, both $p\bmod4$ classes, exact); the remaining non-invariant directions reduce to one unproved determinant-nonvanishing lemma, verified numerically only (44 primes, $p\in[5,200)$). Overall (R) at Björck points remains **Open** (P4) |
| Constant-rank extension of Theorem C to non-squarefree $n$ | Open (P4) |
| Elliptope: semialgebraicity + H1–H3 transfer for the iteration on $\mathcal E_n=\{X\succeq0:X_{ii}=1\}$ | **Proved, unconditional** |
| Elliptope: fixed-point set is a genuine continuum (e.g. $I_n$) | **Proved** (cited fact + machine-exact identity) |
| Elliptope: single-point convergence despite the continuum (Theorem A transferred verbatim) | **Proved, unconditional** |
| Elliptope: pointwise vertex/non-vertex attractive dichotomy | **Proved**, by direct citation to FKP21 Theorem 20 (no new argument here) |
| Elliptope: generic trajectories escape the continuum onto a rank-1 point | **Numerically verified only** (208 trials, $n\in\{3,\dots,20\}$) |
| Elliptope: *every* trajectory converges to a rank-1 point, measure-zero form (Q7) | **Conjecture** |
| $n$-uniformity of the true sharp-growth constant $\sigma_s$ | Conjecture — proved bound scales like $c_N/n$; sampled ratios show no decay (P5) |
| Local rate is identical for $N=1$ and $N=2$ (the empirical gap is global/basin, not local) | Proved (immediate from Theorem C's constants) |
| Known exposed bad point at $n=2$ is repelling off the global-phase direction | Proved (Jacobian spectrum computed directly at that point) |
| Bad fixed points are thin repellers / attracting set is $\mathrm{CAZ}(n,N)$ up to measure zero, general $n$ | Conjecture (P2) |
| Basin measure of non-CAZAC fixed points decays in $n$, faster for $N=2$ than $N=1$ | Conjecture (P3) |
| Presolve continuation ($\tau\downarrow1$) materially improves success probability, not just iteration count | Open, untested (P6) |
| Global (large-step) linear rate with contraction factor bounded away from $1$ uniformly in $n$ | Conjecture, matches empirics for $N=2$ up to $n=503$ (P7) |
| Convergence to CAZAC with high probability under Gaussian initialization, general $n$ | **Open — the central remaining gap (P1)** |

The empirical success-rate table that motivates several rows above (e.g.
the $N=1$ vs.\ $N=2$ contrast referenced in P1/P3) was generated in the
companion repository `cazac-algorithm`'s `docs/` notebook, not in this
repo's `code/`, which holds only the two reference solver scripts
(`code/README.md`). The primary write-up of all of the above is now
`main.tex` (a self-contained paper); `theorems/convergence.md` mirrors its
content in markdown.
