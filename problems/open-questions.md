# Open problems

Notation and background: `../theorems/convergence.md`. Problems are ordered
so that progress on an earlier one tends to feed the next.

## P1. Probabilistic convergence to CAZAC

Theorem A guarantees convergence to *some* fixed point of the iteration;
Theorem B shows the fixed-point set contains non-CAZAC points that trap
every deterministic selection rule. What remains is:

> For a Gaussian (or otherwise natural) random initialization, does the
> iteration converge to $\mathrm{CAZ}(n,N)$ with probability $1$ (or with
> probability $\ge 1-c^n$ for some $c<1$)? How does the success probability
> scale with $n$ and $N$?

This is the central open question. It requires two ingredients: a census
of the non-CAZAC fixed points (P2) and a basin-of-attraction / measure
argument bounding how much of the initialization space they can trap (P3).
The empirical picture (single-sequence success decaying from $\approx 1$ at
small $n$ to $\approx 0.3$ at $n=167$, versus coupled two-sequence success
staying at $1.0$ through at least $n=503$) is the target this problem must
explain quantitatively.

## P2. Full census of non-CAZAC fixed points

Theorem B exhibits two families of non-CAZAC fixed points (flat-spectrum
points with slack amplitudes, and exposed points like the $n=2$ example)
and shows they are complete for $n=2$, $N=1$ (exactly four symmetry
classes: CAZAC, the flat-spectrum family, the pure-amplitude point, and the
exposed point). Open:

> Classify all fixed points of $(\star)$ at $\tau=1$ for general $n$ (and
> general $N$). The stationarity system is a KKT eigenvalue-type equation:
> $x^* = (\Lambda + F^*MF)x^*$ for diagonal multiplier matrices $\Lambda,M
> \ge 0$ satisfying complementary slackness — structurally similar to
> discrete prolate/Slepian time-and-band-limiting operators, which may be a
> useful analogy to push on.

A related sharper question: are attracting fixed points of $(\star)$
*always* exactly $\mathrm{CAZ}(n,N)$ (i.e., do the non-CAZAC fixed points
found so far ever attract an open set of initializations, or are they
always measure-zero repellers)? For $n=2$ the known bad point is repelling
along its one genuinely transverse direction (expansion factor
$(1+\sqrt2)^2\approx5.83$ per step) with only the global-phase direction
neutral — consistent with "thin repeller," but this is checked only at
$n=2$.

## P3. Basin-of-attraction measure vs. dimension

The set $\mathrm{CAZ}(n,N)$ has tangent dimension
$\ge (N-1)n+1$ at smooth points (one universal dependency, forced by
Parseval, prevents the amplitude-torus/flat-spectrum intersection from
being transverse). So $\mathrm{CAZ}(n,1)$ is generically just a
one-dimensional phase circle, while $\mathrm{CAZ}(n,2)$ is a manifold whose
dimension grows linearly in $n$. This dimension gap is a natural mechanism
for why coupling ($N\ge2$) should make CAZAC easier to hit than $N=1$, but
it is not yet a probability bound, because the pathological fixed-point
classes themselves are the *same* for $N=1$ and $N=2$ (the gap is not about
which bad points exist, only about how much initialization space they
absorb). Open:

> Turn the dimension gap into an actual measure/basin bound: show the union
> of non-CAZAC basins has Gaussian measure decaying in $n$ (and decaying
> faster for $N=2$ than $N=1$), completing P1.

## P4. Regularity (hypothesis (R)) beyond odd squarefree Zadoff–Chu points

Theorem C requires the regularity hypothesis (R): that the only linear
dependency among active-constraint gradients at a CAZAC point is the one
forced by Parseval. This is proved unconditionally for Zadoff–Chu
points at every squarefree $n$, **both parities**: the even-parity
chirp $x_u(m)=\omega_{2n}^{-um^2}$ satisfies the identical self-transform/
correlation/Fourier-kernel chain as the odd case, giving
$\dim K = \max\{d:d^2\mid n\}$ with no parity dependence). Remaining open:

- **Non-squarefree $n$.** At non-squarefree $n$, regularity genuinely
  fails in the strict sense ($\dim K = \max\{d : d^2\mid n\} > 1$), so
  Theorem C as stated does not apply. The natural fix is a constant-rank
  extension of the sharp-growth argument that works on the *quotient* by
  the excess periodic dependencies (which are exactly the
  $d_{\max}$-periodic real vectors) rather than requiring $\dim K=1$
  outright.
- **Non-Zadoff–Chu CAZAC points (Björck construction, prime $n$) —
  partial progress, not closed.** Björck's quadratic-phase sequence $b$ is
  exactly affine in the Legendre symbol $\chi$ ($b=A\mathbf1+B\chi+C\delta_0$),
  and this 3-dimensional affine family closes under the DFT — a genuine
  second instance of the closure mechanism behind Theorem D. Using this
  plus $G$-equivariance (decimation by quadratic-residue units), one
  proves *exactly* that the dependency space $K$ meets the
  symmetry-invariant subspace $T=\mathrm{span}\{\mathbf1,\chi,\delta_0\}$
  only in $\mathrm{span}\{\mathbf1\}$, both $p\bmod4$ classes. What
  remains open: whether $K$ meets any of the $q-1$ nontrivial isotypic
  pieces of $\mathbb R^p$ under the decimation action ($q=(p-1)/2$) —
  reduces to nonvanishing of $q-1$ Jacobi-sum-type $2\times2$ determinants,
  confirmed numerically nonzero at 44 primes ($p\in[5,200)$, both classes)
  but with no uniform lower bound (at $p=71$ the gap is $\sim10^4\times$
  smaller than at neighboring primes — genuine, not a rounding artifact,
  confirmed at 60-digit precision). **(R) at Björck points is therefore
  still open, not proved.** No argument beyond this has been attempted for
  other non-Zadoff–Chu families or composite non-Zadoff–Chu points.
- **Chirp-argument generalization, more broadly.** The reusable schema
  (closure under DFT up to scalar/reindexing + an algebraic phase addition
  law) extends the Legendre symbol (classical closure fact, proved) and,
  via the affine trick, to Björck's construction (above). It provably does
  *not* extend to cubic or higher-degree phase sequences: Weyl sums of
  degree $\ge3$ lack the exact closed-form evaluation that quadratic Gauss
  sums have, so no variant of the method applies there — a genuine proved
  negative result, not merely an unexplored direction.

## P5. Uniformity of the sharp-growth constant in $n$

Theorem C's constant $\sigma_s$ is proved to scale like $c_N/n$ at
Zadoff–Chu points, but empirically sampled growth ratios show *no* decay
with $n$ over the range checked. Open:

> Is the true (not merely the proved) sharp-growth constant bounded away
> from $0$ uniformly in $n$? If so, the current proof is lossy and a
> sharper argument should exist; if not, the empirical sampling range is
> too small to see the decay yet.

## P6. What does presolve continuation actually buy?

The algorithm as commonly run first solves a "presolve" sequence with
$\tau$ decreasing from a value $>1$ down to $1$, before fixing $\tau=1$ for
the main phase. For $\tau>1$, the maximal-norm set of $C_\tau$ is the
strictly larger flat-spectrum variety (amplitude constraints slack), so
the hypothesis is that presolve first drives the iterate onto (or near)
this easier variety, handing the main phase a head start already close to
the correct fiber. This is a plausible but unproved mechanism. A decisive
experiment: ablate the presolve (cold-start directly at $\tau=1$) and
compare success rates at representative $(n,N)$; if ablation does not hurt
success, the presolve's value is at most computational (fewer main-phase
iterations), not structural, and this problem can be closed as "does not
matter."

## P7. Rate away from CAZAC (large-step regime)

Theorem C pins down the rate *near* a regular CAZAC point (finite exact
termination) but says nothing about the "large step" regime before the
iterate enters that neighborhood. Empirically, the coupled two-sequence
case needs at most a handful of iterations even at $n=503$, suggesting a
much stronger global rate (linear with a contraction factor bounded away
from $1$ uniformly in $n$) than the current proof captures. A concrete
attack: linearize the argmax map at $x^*\in\mathrm{CAZ}(n,N)$ (sensitivity
analysis of the SOCP solution as a function of the objective direction,
valid where strict complementarity and uniqueness hold) and compute the
spectrum of the resulting Jacobian. The eigenvalue-$1$ eigenspace should be
exactly the tangent space of the CAZAC symmetry orbit (global phase,
cyclic shift, modulation); the next-largest eigenvalue would be the rate.

## P8. Elliptope corollary: does every trajectory converge to a rank-1 point?

`theorems/convergence.md` §3.1 applies Theorem A's general (non-CAZAC-
specific) form to the elliptope $\mathcal E_n=\{X\succeq0:X_{ii}=1\}$ —
Felzenszwalb–Klivans–Paul's (2021) own worked example, where they concede a
continuum of fixed points for $n>3$ and their own example stalls at
set-level convergence. Theorem A proves single-point convergence there
unconditionally (semialgebraicity + H1–H3 transfer verbatim; no CAZAC
structure used) — proved, not numerical. What is **not** proved: which
single point the trajectory lands on. Numerical experiments on
$\mathcal E_5$ (4 starting configurations) show 3 of 4 escape the
fixed-point continuum entirely and converge to a rank-1 extreme point
$vv^\top$; the trajectory started exactly at the fixed point $I_5$ stays
there (a valid tie-break, not a counterexample). Conjecture:

> For a.e. (e.g. Gaussian-perturbed) initialization on $\mathcal E_n$, the
> iteration converges to a rank-1 point — i.e. attracting fixed points are
> exactly the rank-1 extreme points, and the higher-rank fixed-point
> continuum (e.g. $I_n$ and its neighbors) is a measure-zero repelling set.

This is structurally the same conjecture as P1 ("attracting fixed points =
extreme points, up to measure zero" is the mechanism P1 needs) in a
second, structurally unrelated domain. If true, it is independent
evidence for P1's general shape; if false, the CAZAC case's extra
structure (the dimension count in P3/Theorem D) may be doing more work
than currently credited. Attack plan: extend the elliptope experiment to
$n\in\{5,10,20\}$ with many random/Gaussian-perturbed starts per $n$, and
histogram the rank of the limit point.
