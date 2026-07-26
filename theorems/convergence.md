# Convergence of iterated SOCP refinement for CAZAC feasibility

This file mirrors `../main.tex` (the primary, self-contained paper); see
that file for the same content with a full bibliography and an
introductory abstract in dense results-forward register.

## Abstract

Constant-amplitude zero-autocorrelation (CAZAC) sequences satisfy a
constant-modulus condition in time and a flat-spectrum condition in
frequency simultaneously — a nonconvex intersection of two tori. A natural
heuristic for finding such sequences is to replace the equality constraints
by convex inequality relaxations (amplitude $\le \tau$, spectral energy
$\le$ its maximum) and iterate: solve a second-order cone program (SOCP)
against a running reference direction, then reset the reference direction
to the optimizer just found. This document gives a rigorous convergence
theory for that iteration. Four results are established. **Theorem A**
(unconditional): the iterate sequence always has finite length and
converges to a single fixed point — not merely to a set of limit points —
including on a second, structurally unrelated example (the elliptope of
correlation matrices). The descent mechanism behind Theorem A is itself a
known KL/Frank-Wolfe-type template (also independently instantiated by
Aktaş and Kroer, 2025, arXiv:2505.00221, for a general strongly convex
smooth objective, of which ours is a special case); this document's own
contribution is the reduction of CAZAC feasibility to a norm-maximization
problem that makes the template applicable, not the descent argument
itself. **Theorem B** (unconditional, obstruction): the
limit need not be a CAZAC point; there exist non-CAZAC fixed points that
no measurable selection rule can escape, so no *deterministic*
convergence-to-CAZAC theorem is possible. **Theorem C** (conditional on a
checkable regularity hypothesis, proved unconditionally in an important
case): near a *regular* CAZAC point, the iteration does not merely
converge at some rate — it lands on the point **exactly** after finitely
many steps. **Theorem D**: the regularity hypothesis is proved
unconditionally at Zadoff–Chu points for every $n$, with a related partial
result at Björck's quadratic-phase construction.

Everything below is elementary once the right identity (the reduction of
the whole problem to a generalized power method, Section 2) is in hand;
the interesting content is that a single quantity — the $\ell^1$ constraint
slack — simultaneously drives the global convergence proof and the local
sharp-growth estimate.

## 1. Setup

Fix $n \ge 1$ and $N \ge 1$. Let $F$ denote the (unnormalized) discrete
Fourier transform on $\mathbb{C}^n$, so Parseval's identity reads
$\|Fu\|^2 = n\|u\|^2$. The state space is
$x = (x_0,\dots,x_{N-1}) \in (\mathbb{C}^n)^N \cong \mathbb{R}^{2Nn}$, with
real inner product $\langle u,v\rangle_{\mathbb R} = \operatorname{Re}\langle u,v\rangle$
and norm $\|x\|^2 = \sum_{j,k}|x_j(k)|^2$. Write the coupled spectral energy
as $S_x(k) = \sum_{j=0}^{N-1} |F(x_j)(k)|^2$.

**Definition (coupled CAZAC system).** $x$ is CAZAC if $|x_j(k)| = 1$ for
all $j,k$ and $S_x(k) = Nn$ for all $k$. Write $\mathrm{CAZ}(n,N)$ for this
set. For $N=1$ this is the classical CAZAC condition; Zadoff–Chu sequences
show $\mathrm{CAZ}(n,N) \ne \varnothing$ for every $n,N$.

For $\tau \ge 1$, the convex relaxation is

$$C_\tau = \bigl\{\, x \in (\mathbb{C}^n)^N \;:\; |x_j(k)| \le \tau \ \forall j,k,\quad S_x(k) \le Nn \ \forall k \,\bigr\}.$$

$C_\tau$ is compact, convex, semialgebraic, and second-order-cone
representable (each defining inequality is a Euclidean-norm bound on a
linear image of $x$).

**Iteration.** Given $x^0 \ne 0$, define

$$x^{\ell+1} \in \operatorname*{arg\,max}_{x \in C_\tau} \langle x^\ell, x\rangle_{\mathbb R}. \tag{$\star$}$$

The argmax can be a face of positive dimension; $(\star)$ permits *any*
measurable selection from it — the theorems below hold regardless of which
point a particular solver returns.

**Fixed points.** $x^*$ is a fixed point of $(\star)$ if
$x^* \in \arg\max_{x\in C_\tau}\langle x^*,x\rangle_{\mathbb R}$, equivalently
$\sigma_{C_\tau}(x^*) = \|x^*\|^2$ where $\sigma_C(r)=\max_{x\in C}\langle r,x\rangle_{\mathbb R}$
is the support function of $C$.

## 2. The iteration is a generalized power method

**Proposition 1.** With $f(x) = \tfrac12\|x\|^2$ (so $\nabla f(x)=x$),
$(\star)$ is exactly
$x^{\ell+1} \in \arg\max_{x\in C_\tau}\langle \nabla f(x^\ell), x\rangle_{\mathbb R}$:
the conditional-gradient / generalized-power-method update for maximizing
the convex function $f$ over the compact convex set $C_\tau$.

*Proof.* Immediate from the definitions. $\blacksquare$

**Proposition 2 (max-norm points are exactly CAZAC points).** For every
$x \in C_\tau$, $\|x\|^2 \le Nn$. At $\tau = 1$, equality holds iff
$x \in \mathrm{CAZ}(n,N)$; the maximum $Nn$ is attained for every $n,N$.

*Proof.* By Parseval, $\|x\|^2 = \sum_j\|x_j\|^2 = \tfrac1n\sum_k S_x(k) \le \tfrac1n\cdot n\cdot Nn = Nn$,
with equality iff $S_x\equiv Nn$. Separately, at $\tau=1$,
$\|x\|^2=\sum_{j,k}|x_j(k)|^2\le Nn$ with equality iff every $|x_j(k)|=1$.
Both chains tight simultaneously forces both CAZAC conditions; conversely a
CAZAC point attains $Nn$. Attainment: $N$ copies of a Zadoff–Chu sequence.
$\blacksquare$

This reframes "find a CAZAC system" — a nonconvex feasibility problem — as
"maximize a convex function over a convex set with a known optimal value,"
namely $\max_{x\in C_1}\|x\|^2 = Nn$. It also gives a free optimality
certificate: an iterate is $\delta$-suboptimal iff $Nn-\|x^\ell\|^2=\delta$.

**Proposition 3 (monotone ascent, square-summable increments).** Along the
iteration at fixed $\tau$, with $x^\ell \ne 0$:

1. $\langle x^\ell,x^{\ell+1}\rangle_{\mathbb R} \ge \|x^\ell\|^2$;
2. $\|x^{\ell+1}\| \ge \|x^\ell\|$, and $\|x^\ell\| \uparrow \rho \le \sqrt{Nn}$;
3. $\|x^{\ell+1}-x^\ell\|^2 \le \|x^{\ell+1}\|^2 - \|x^\ell\|^2$, hence
   $\sum_\ell \|x^{\ell+1}-x^\ell\|^2 \le Nn - \|x^0\|^2$ and
   $\|x^{\ell+1}-x^\ell\| \to 0$.

*Proof.* (1) $x^\ell \in C_\tau$ is feasible for the program whose optimizer
is $x^{\ell+1}$, so the optimal value is at least
$\langle x^\ell,x^\ell\rangle_{\mathbb R} = \|x^\ell\|^2$.
(2) Cauchy–Schwarz: $\|x^\ell\|\|x^{\ell+1}\| \ge \langle x^\ell,x^{\ell+1}\rangle_{\mathbb R} \ge \|x^\ell\|^2$;
divide by $\|x^\ell\|$; monotone and bounded above by Prop. 2.
(3) Expand $\|x^{\ell+1}-x^\ell\|^2 = \|x^{\ell+1}\|^2+\|x^\ell\|^2-2\langle x^\ell,x^{\ell+1}\rangle_{\mathbb R} \le \|x^{\ell+1}\|^2-\|x^\ell\|^2$
by (1); telescope. $\blacksquare$

Square-summable increments alone do *not* imply the sequence converges
(harmonic-type sums can wander) — this is exactly the gap Theorem A closes.

**Proposition 4 (limit points are fixed points; CAZAC $\subseteq$ fixed
points).** Every limit point of $(x^\ell)$ is a fixed point of $(\star)$,
and every $x^*\in\mathrm{CAZ}(n,N)$ is a fixed point at $\tau=1$ — in fact a
*global* maximizer of its own linear functional.

*Proof.* Let $x^{\ell_j}\to\bar x$. By Prop. 3(3), $x^{\ell_j+1}\to\bar x$
too. $\sigma_{C_\tau}$ is continuous (finite convex on $\mathbb R^{2Nn}$) and
$\langle x^{\ell_j},x^{\ell_j+1}\rangle_{\mathbb R}=\sigma_{C_\tau}(x^{\ell_j})$
by definition of the update; pass to the limit. For the second claim, any
$y\in C_1$ satisfies $\langle x^*,y\rangle_{\mathbb R} \le \|x^*\|\|y\| \le Nn = \|x^*\|^2$
by Cauchy–Schwarz and Prop. 2, with equality at $y=x^*$. $\blacksquare$

(The same Cauchy–Schwarz argument shows any maximal-norm point of *any*
compact convex set is a fixed point of iterated linear optimization —
nothing here is CAZAC-specific except Prop. 2 itself.)

## 3. Theorem A — global convergence to a single point [unconditional]

> **Theorem A.** Fix $\tau \ge 1$. For any $x^0 \ne 0$ and any measurable
> selection $x^{\ell+1} \in \arg\max_{x\in C_\tau}\langle x^\ell,x\rangle_{\mathbb R}$,
> the sequence $(x^\ell)$ has finite length,
> $\sum_\ell \|x^{\ell+1}-x^\ell\| < \infty$, and converges to a single
> fixed point $\bar x \in C_\tau$.

*Proof.* Apply the abstract descent theorem of Attouch, Bolte, and Svaiter
[*Convergence of descent methods for semi-algebraic and tame problems*,
Math. Program. **137** (2013), 91–129, Thm. 2.9] to
$F(x) = -\tfrac12\|x\|^2 + \iota_{C_\tau}(x)$ (negative of our objective,
plus the indicator of $C_\tau$, so minimizing $F$ is maximizing $\|x\|^2/2$
on $C_\tau$):

- **Sufficient decrease** ($a=\tfrac12$): exactly Prop. 3(3).
- **Relative error** ($b=1$): first-order optimality of the argmax step
  gives $x^\ell \in N_{C_\tau}(x^{\ell+1})$ (the normal cone at the new
  iterate). Since $\partial F(x^{\ell+1}) = -x^{\ell+1} + N_{C_\tau}(x^{\ell+1})$
  (sum rule for a smooth function plus an indicator of a convex set), the
  vector $w = x^\ell - x^{\ell+1}$ lies in $\partial F(x^{\ell+1})$, and
  $\|w\| = \|x^{\ell+1}-x^\ell\|$ exactly — the subgradient residual *is*
  the step size. This identity is specific to $f=\tfrac12\|x\|^2$ and is
  the structural fact that makes the whole argument go through: for a
  generic convex objective the relative-error condition is a genuine
  additional hypothesis, but here it is free. (The identical construction
  for a general smooth strongly convex objective, where it holds only as
  a Lipschitz-constant inequality rather than an exact equality, is
  Aktaş–Kroer's Lemma 3.1, arXiv:2505.00221 — our case is the special
  instance where that Lipschitz constant is exactly $1$, so the
  inequality becomes an equality; this is a corollary of their general
  lemma, not an independent discovery.)
- **Compactness/continuity**: iterates lie in compact $C_\tau$; $F$ is
  continuous there.
- **KL property**: $C_\tau$ is defined by polynomial inequalities and $F$ is
  polynomial plus an indicator of a semialgebraic set, hence $F$ is
  semialgebraic, hence it satisfies the Kurdyka–Łojasiewicz (KL) inequality
  [Bolte, Daniilidis, Lewis, Shiota, *Clarke subgradients of stratifiable
  functions*, SIAM J. Optim. **18** (2007)].

Critical points of $F$ are exactly the fixed points of $(\star)$ (Sec. 2).
The Attouch–Bolte–Svaiter theorem then gives finite length and convergence
to a single critical point. $\blacksquare$

The theorem is silent on *which* fixed point the limit is — that is exactly
where Theorem B enters.

### 3.1 A second worked example: the elliptope

The proof of Theorem A uses nothing about $C_\tau$ beyond compactness and
convexity, except at the KL step, which needs semialgebraicity. Stated in
general: for **any** compact convex $C\subset\mathbb R^d$ and the iteration
$x^{\ell+1}\in\arg\max_{x\in C}\langle x^\ell,x\rangle$, sufficient decrease
and the relative-error identity hold unconditionally; compactness gives
continuity. If $C$ is also semialgebraic, KL holds and Theorem A's
conclusion — single-point convergence, not merely a connected limit set —
follows, independent of anything CAZAC-specific.

We exercise this on the **elliptope** $\mathcal E_n=\{X\in\mathbb R^{n\times n}:
X=X^\top,\ X\succeq0,\ X_{ii}=1\ \forall i\}$, the set of correlation
matrices, with iteration $X^{\ell+1}\in\arg\max_{X\in\mathcal E_n}\langle
X^\ell,X\rangle$ ($\langle A,B\rangle=\operatorname{tr}(AB)$). This is
Felzenszwalb–Klivans–Paul's (2021) own worked example — the one case in
their paper where they concede the fixed-point set is a continuum for
$n>3$, and where their own instance (a circle of fixed points on a cone)
stops at set-level convergence.

**[Proved]** $\mathcal E_n$ is compact, convex, and semialgebraic:
boundedness follows from $X\succeq0$, $X_{ii}=1$ giving $|X_{ij}|\le1$ by
Cauchy–Schwarz; $X\succeq0$ for symmetric $X$ is, by Sylvester's criterion,
a finite conjunction of polynomial inequalities on principal minors, and
$X_{ii}=1$ are polynomial equalities — a spectrahedron.

**[Proved]** Sufficient decrease and the relative-error identity transfer
verbatim from Prop. 3(3) and Theorem A's normal-cone argument, replacing
"$C_\tau$" by "$\mathcal E_n$" and $\langle\cdot,\cdot\rangle_{\mathbb R}$ by
$\operatorname{tr}(\cdot\,\cdot)$ — neither source proof used any CAZAC
structure.

**[Proved]** The fixed-point set is a genuine continuum, not a singleton:
$X^*=I_n$ is a fixed point whose optimal face is *all* of $\mathcal E_n$,
since $\operatorname{tr}(I_nX)=\sum_iX_{ii}=n$ identically on $\mathcal E_n$
— every feasible point, including every rank-1 correlation matrix
$vv^\top$ ($v_i=\pm1$), is simultaneously optimal at this direction. This
instantiates, with a machine-exact identity, FKP's own statement that their
$\mathcal L_n$ (their name for $\mathcal E_n$) has infinitely many fixed
points for $n>3$.

**[Verified numerically, $n=5$]** Running the iteration on $\mathcal E_5$
from four starting points ($X^0=I_5$; $I_5$ plus a small random
perturbation renormalized to unit diagonal; two independent random
correlation matrices) using CLARABEL via cvxpy
(`experiments/verify_elliptope_convergence.py` in the companion repository
`cazac-algorithm`): all four trajectories show
$\|X^{\ell+1}-X^\ell\|_F\to0$ (to $\sim10^{-15}$ by iteration 15) and
settle on a single matrix; three of the four escape the fixed-point
continuum entirely and converge to a rank-1 correlation matrix $vv^\top$
(eigenvalues $(0,0,0,0,5)$ to machine precision); the run started at
$X^0=I_5$ is the edge case where the solver's own tie-break returns $I_5$
again — a valid fixed point, not a counterexample. A broader sweep
(`experiments/verify_elliptope_rank_sweep.py`, same repository) of 140
random/Gaussian-perturbed trials across $n\in\{3,4,5,8,10,15,20\}$ finds
**every trajectory converges to rank 1**, no exceptions — still verified
numerically only, not a proof.

**[Proved by direct citation]** FKP's own Theorem 20 settles the
*pointwise* half of the rank-1 question without any new argument here:
the vertices of $\mathcal E_n$ (rank-1 sign matrices) are exactly the
fixed points whose full neighborhood converges to them, and every
non-vertex fixed point admits an explicit escaping curve (their
Proposition 19), so it cannot attract a full neighborhood either. This
complements, rather than overlaps, the single-point-convergence guarantee
below (which fixed point is reached vs. that some single fixed point is
reached). It falls short of settling Q7 as stated for a precise reason:
FKP's Proposition 19 produces, in every neighborhood of a non-vertex
fixed point, *at least one* escaping nearby point — exactly what
"not attractive" requires — but this does not show *every* nearby point
escapes, which "repelling" would additionally require. Whether non-vertex
elliptope points are repelling outright, or could instead be collectively
(not individually) attracting — the way FKP's own unrelated cone example
(§3, Example 5) shows can happen in general for this style of iteration —
is not settled by Theorem 20 either way, so a positive-measure basin for
a non-vertex point is not ruled out.

**Conclusion.** Because $\mathcal E_n$ is compact, convex, and
semialgebraic, Theorem A's general form applies to it directly: the
iteration converges to a single fixed point for every initialization
**despite the fixed-point set being an infinite continuum** — **Proved**,
unconditional. That generic nearby trajectories converge to a single point
rather than merely having a connected limit set, and that they tend to
escape onto a rank-1 extreme point, is **verified numerically only** (140
trials total, $n\in\{3,\dots,20\}$) — not a general proof that every
trajectory on $\mathcal E_n$ converges to a rank-1 point. The pointwise
attractive/repelling dichotomy is **proved** by direct citation (above).
The residual, measure-zero form of the universal claim is filed as open
problem Q7 (below) and remains a **conjecture**, not a theorem.

## 4. Theorem B — the limit need not be CAZAC [unconditional, obstruction]

> **Theorem B.**
> (a) The fixed points of $(\star)$ at $\tau=1$ with every amplitude
> constraint slack are exactly the points whose Fourier transform has
> modulus $\sqrt n$ on some proper subset $K \subsetneq \{0,\dots,n-1\}$ of
> frequencies and vanishes off $K$, with time-domain modulus strictly less
> than $1$ everywhere; such a point has $\|x\|^2=|K|$.
> (b) There exist non-CAZAC fixed points that are the **unique** maximizer
> of their own linear functional over $C_1$ — exposed points, in the
> language of convex geometry. For $n=2$: $x^*=(1,\sqrt2-1)$ is such a
> point; for $N=2$, its diagonal doubling $((1,\sqrt2-1),(1,\sqrt2-1))$ is
> such a point.
> (c) Consequently **no deterministic convergence-to-CAZAC theorem is
> possible**: because the point in (b) is the unique maximizer of its own
> functional, *every* measurable selection rule in $(\star)$ retains it
> exactly once reached. The correct target class of theorem is therefore
> probabilistic over initializations, not deterministic.

*Proof sketch of (b) for $n=2$, $N=1$.* Take
$x^*=(1,t)$ with $t=\sqrt2-1$. One checks directly that for all
$y=(y_0,y_1)\in\mathbb C^2$ with $|y_0|,|y_1|\le 1$ and
$|F(y)(0)|^2+|F(y)(1)|^2 \le 2$,

$$\langle x^*,y\rangle_{\mathbb R} = (1-t)\,\operatorname{Re}y(0) + t\,\operatorname{Re}\widehat y(0) \le (1-t) + t\sqrt2 = 1+t^2 = \|x^*\|^2,$$

with equality iff $y=x^*$; here $\widehat y = F(y)/\sqrt2$ is normalized so
$|\widehat y(0)|\le\sqrt2$ under the coupled constraint. This is a genuine
separating functional exposing $x^*$ as a vertex-like point of $C_1$ even
though $x^*\notin\mathrm{CAZ}(2,1)$. The $N=2$ doubling repeats the
argument coordinatewise. $\blacksquare$ *(Part (a) is a direct KKT
computation on the stationarity system of $(\star)$; omitted here — it is
routine but notation-heavy.)*

The upshot: Theorem A guarantees convergence to *some* fixed point, and
Theorem B shows the fixed-point set genuinely contains non-CAZAC points
that trap every selection rule. The remaining content of "the algorithm
works" is a probabilistic/basin-of-attraction statement (open in general;
see `problems/`), not a deterministic one.

## 5. Theorem C — finite exact termination at regular CAZAC limits [conditional on (R); unconditional in the squarefree Zadoff–Chu case]

Call $x^*\in\mathrm{CAZ}(n,N)$ **regular** — hypothesis (R) — if the only
linear dependency among the gradients of the active constraints at $x^*$ is
the one forced by Parseval's identity (equivalently, the "excess"
dependency space $K$ has $\dim K = 1$ in a precise sense made exact in
Section 5.1 below). Let $\sigma_s>0$ denote the associated smallest
singular value at a regular point.

> **Theorem C.** Near a regular $x^*\in\mathrm{CAZ}(n,N)$:
>
> (a) *Sharp (linear, not merely quadratic) growth:*
> $D(x) \ge \dfrac{\sigma_s}{2\sqrt2}\,\mathrm{dist}(x,\mathrm{CAZ}(n,N))$
> for $x\in C_1$ near $x^*$, where $D(x) = Nn-\|x\|^2$.
>
> (b) *KL exponent $0$:* $\mathrm{dist}(0,\partial F(x)) \ge c_0 := \sigma_s/(8\sqrt2)$
> for all non-CAZAC $x$ in a neighborhood of $x^*$.
>
> (c) *Finite exact arrival:* if the Theorem A limit $\bar x$ is a regular
> CAZAC point, then $x^\ell = \bar x$ **exactly** for all sufficiently large
> $\ell$ — not merely in the limit. The number of "large" steps before
> exact arrival is at most $(Nn-\|x^0\|^2)/c_0^2$.

*Proof idea.*

(a) The key identity is that the objective deficit is *simultaneously* the
$\ell^1$ slack of both constraint families:
$D(x) = \sum_{j,m}(1-|x_j(m)|^2) = \tfrac1n\sum_k(Nn-S_x(k))$, and every
term on both sides is $\ge 0$ for $x\in C_1$. Combined with a
self-contained quantitative Newton-type error bound for the quadratic map
defining $\mathrm{CAZ}(n,N)$ near a point where the constraint gradients
have only the one forced dependency, this $\ell^1$ identity converts into
the stated *linear* lower bound on distance-to-$\mathrm{CAZ}$ — sharper
than the quadratic growth one would expect generically at a non-vertex
critical point.

(b) Follows from (a) via the normal-cone inequality
$\|w\|\,d \ge (F(x)-F(x^*)) - d^2/2$ (with $d=\mathrm{dist}(x,\mathrm{CAZ})$)
applied to any $w\in\partial F(x)$.

(c) Let
$\bar x$ be the Theorem A limit, regular with constants $\rho,\sigma_s,c_0$
as in (a)/(b). Since $x^\ell\to\bar x$ and $\|x^{\ell+1}-x^\ell\|\to0$
(Theorem A's square-summability), fix $L_0$ so that for all $\ell\ge L_0$:
$x^{\ell+1}$ lies in the (b)-neighborhood of $\bar x$, and
$\|x^{\ell+1}-x^\ell\|<c_0$. By the exact identity from Theorem A's proof,
$w^{\ell+1}:=x^\ell-x^{\ell+1}\in\partial F(x^{\ell+1})$ with
$\|w^{\ell+1}\|=\|x^{\ell+1}-x^\ell\|$ **exactly**, at every $\ell$ — no
approximation, no inequality. Suppose toward contradiction
$x^{\ell+1}\notin\mathrm{CAZ}(n,N)$ for some $\ell\ge L_0$. Then
$x^{\ell+1}$ is a non-CAZAC point inside the (b)-neighborhood, so (b) gives
$\mathrm{dist}(0,\partial F(x^{\ell+1}))\ge c_0$. But
$w^{\ell+1}\in\partial F(x^{\ell+1})$ and $\|w^{\ell+1}\|<c_0$ by choice of
$L_0$ — contradiction, since $w^{\ell+1}$ itself witnesses a subgradient of
norm $<c_0$. Hence $x^{\ell+1}\in\mathrm{CAZ}(n,N)$ **exactly**: the
identity is exact and (b)'s bound is a hard floor on every non-CAZAC point
in the neighborhood, so a subgradient below that floor cannot occur unless
the point already lies on $\mathrm{CAZ}(n,N)$. Write
$z:=x^{\ell+1}\in\mathrm{CAZ}(n,N)$. The next step maximizes
$\langle z,y\rangle_{\mathbb R}$ over $y\in C_1$; by Cauchy–Schwarz and
Prop. 2 ($\|z\|=\sqrt{Nn}$ is the global max-norm),
$\langle z,y\rangle_{\mathbb R}\le\|z\|\|y\|\le Nn$ with equality iff
$y=z$ — the argmax is the singleton $\{z\}$, so $x^{\ell+2}=z$, and by
induction $x^m=z$ for all $m>\ell$; in particular $\bar x=z\in\mathrm{CAZ}(n,N)$.
The step-count bound is Theorem A's
$\sum_\ell\|x^{\ell+1}-x^\ell\|^2\le Nn-\|x^0\|^2$ telescoped against
$c_0^2$ per large step. $\blacksquare$

This is a genuine proof by contradiction, not a heuristic squeeze: an exact
identity (equality, from Theorem A) is jointly satisfiable with a hard
lower bound (inequality, from (b)) on the same quantity only at
$\mathrm{dist}=0$. Parts (a), (b), (c) share one status tag because they
rest on exactly the same two inputs — hypothesis (R) at $\bar x$ and the
unconditional exact H2 identity from Theorem A's proof — and (c) adds no
assumption beyond the theorem's own hypothesis "$\bar x\in\mathrm{CAZ}(n,N)$"
plus the already-unconditional singleton-argmax fact.

## 6. Theorem D — regularity (R) at Zadoff–Chu points, all $n$ [unconditional]; a partial result at Björck points

**Regularity is a checkable, number-theoretic condition at Zadoff–Chu
points — proved for all $n$, both parities.** At a Zadoff–Chu tuple, the
dependency space among active constraint gradients is exactly the space of
$d_{\max}$-periodic real vectors, where $d_{\max} = \max\{d : d^2 \mid n\}$
— proved unconditionally for **every** $n\ge1$ (odd and even), all
$N\ge1$, all admissible roots, via an elementary argument (the DFT of a
chirp is a chirp, reducing the dependency count to counting solutions of
$t^2\equiv 0 \pmod n$; no deep analytic number theory is needed). The
even-$n$ case is likewise a complete proof: the even-parity Zadoff–Chu
phase $x_u(m)=\omega_{2n}^{-um^2}$ ($\omega_{2n}=e^{i\pi/n}$) satisfies the
identical self-transform / correlation / Fourier-kernel chain as the odd
case, with $T_m$ (triangular numbers) replaced by $m^2$ and exponents
tracked mod $2n$; the kernel's zero set collapses to the same congruence
$t^2\equiv0\bmod n$, giving $\dim K=d_{\max}$ regardless of parity.
Verified numerically over 43 configurations (13 even $n$ from 2 to 50, up
to 4 roots each; `experiments/verify_even_n_regularity.py`, odd case
`experiments/verify_zc_regularity.py`, both in the companion repository
`cazac-algorithm`). Consequently:

> **Corollary (unconditional finite termination at squarefree $n$, either
> parity).** If $n$ is squarefree (odd or even), hypothesis (R) holds
> automatically at every Zadoff–Chu-orbit CAZAC point, so Theorem C applies
> there with no unverified hypothesis — in particular at every prime $n$.

For non-squarefree $n$, or non-Zadoff–Chu CAZAC points, whether (R) holds
in full is open — see `problems/`. There is, however, genuine partial
progress at one non-Zadoff–Chu family:

**The Björck case (partial, not proved).** For Björck's quadratic-phase
construction $b$ at odd prime $p$ (built from the Legendre symbol $\chi$),
$b$ is not literally self-dual under the DFT (unlike $\chi$ itself and
unlike Zadoff–Chu chirps), but $b$ **is exactly affine** in $\chi$:
$b=A\mathbf1+B\chi+C\delta_0$ ($C=1-A$), and this 3-dimensional space
closes under the DFT with new coefficients — a genuine second instance of
the closure mechanism behind Theorem D, distinct in kind from Zadoff–Chu's
literal self-duality. Using this, and the fact that the dependency space
$K$ is invariant under the quadratic-residue decimation symmetry group
$G$ (order $q=(p-1)/2$), one proves **exactly**: $K$ intersected with the
$G$-trivial-isotypic (3-dimensional, symmetry-invariant) subspace equals
$\mathrm{span}\{\mathbf1\}$ — both residue classes mod 4, exact symbolic
computation. **What is not proved:** whether $K$ meets any of the $q-1$
remaining nontrivial isotypic pieces; this reduces to nonvanishing of $q-1$
Jacobi-sum-type $2\times2$ determinants, confirmed numerically nonzero at
44 primes ($p\in[5,200)$, both classes, $\dim K=1$ throughout, checked to
60-digit precision) but with no uniform lower bound — at $p=71$ the
relevant gap is $\sim10^4\times$ smaller than at neighboring primes, a
genuine (not numerical-artifact) near-degeneracy
(`experiments/verify_bjorck_regularity.py` and
`experiments/verify_bjorck_isotypic.py`, companion repository
`cazac-algorithm`). **The overall regularity
claim at Björck points is therefore still open, not proved** — the
reduction is real and rules out an entire candidate subspace by exact
computation, but the complementary directions are checked only
numerically.

**A reusable schema, and its limits.** The chirp/Björck argument's core
mechanism — a family closed under the DFT up to scalar and reindexing,
plus an algebraic addition law for the phase — reduces a linear-algebra
rank question to a congruence-counting question. It applies to Zadoff–Chu
chirps and (via the affine-closure trick) to Björck's construction. The
Legendre symbol itself has the same closure property classically
(**proved**, standard, newly connected here). It provably does **not**
extend to cubic or higher-degree phase sequences (**proved negative
result**): Weyl sums of degree $\ge3$ lack the exact closed-form
evaluation that quadratic Gauss sums have, so the needed addition law
fails.

**A remark on the $N=1$ vs. $N=2$ gap.** Theorem C's local rate constants
do not distinguish $N=1$ from $N=2$: first-order sharpness at a regular
point decides the local geometry identically in both cases. The
empirically observed difference in success probability between the
single-sequence and coupled formulations is therefore a *global* (basin of
attraction) phenomenon, not a local-rate phenomenon — consistent with
Theorem B's obstruction living at isolated points rather than governing
local curvature.

## 7. What is proved, what is open

| Claim | Status |
|---|---|
| Iteration = generalized power method for $\max_{C_\tau}\|x\|^2$ | Proved (Prop. 1–2) |
| Monotone norm ascent, square-summable increments | Proved (Prop. 3) |
| Limit points are fixed points; CAZAC $\subseteq$ fixed points | Proved (Prop. 4) |
| **Theorem A** — full-sequence convergence to a single fixed point | **Proved, unconditional** |
| Elliptope corollary — semialgebraicity, H1–H3 transfer, single-point convergence despite a fixed-point continuum | **Proved, unconditional** |
| Elliptope corollary — pointwise vertex/non-vertex attractive dichotomy | **Proved**, by direct citation to FKP21 Theorem 20 (no new argument) |
| Elliptope corollary — generic trajectories escape to a rank-1 point | **Numerically verified only** (140 trials, $n\in\{3,\dots,20\}$) |
| Elliptope corollary — every trajectory converges to a rank-1 point, measure-zero form (Q7) | Conjecture |
| **Theorem B** — non-CAZAC fixed points exist and trap every selection rule | **Proved, unconditional** |
| **Theorem C** — sharp growth / KL-exponent-0 / finite exact termination at regular CAZAC limits, including the full proof of part (c) | **Proved under regularity (R)** |
| (R) holds at Zadoff–Chu points, either parity | **Proved unconditionally for all $n$; iff $n$ squarefree** |
| (R) at Björck points, prime $n$ | **Partial** — proved on the symmetry-invariant subspace ($K\cap T=\mathrm{span}\{\mathbf1\}$); open on the complement (one unproved determinant-nonvanishing lemma, numerically verified at 44 primes) |
| (R) at non-squarefree $n$, or other non-Zadoff–Chu CAZAC points | Open |
| Convergence *to* CAZAC with high probability over a Gaussian initialization, for general $n$ | Open — see `problems/` |

See `problems/open-questions.md` for the precise statements and partial
progress on everything marked open, and `../STATUS.md` for the evidence
tier (proved / numerically verified / conjectural) of every claim in this
file.
