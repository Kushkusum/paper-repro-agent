# Cross-Paper Comparison

The original project brief suggested an extension: "cross-compare multiple papers claiming the
same result." Finding a genuine pair — not a manufactured one — turned out to be the harder part;
true independent replications of an *identical* numeric claim are rare in the ML literature (which
is, not coincidentally, close to the whole reason this project exists).

One did turn up naturally while researching this benchmark's UCB1 paper, though.

## The shared claim

**Auer, Cesa-Bianchi & Fischer (2002)**, *"Finite-time Analysis of the Multiarmed Bandit Problem"*
— the UCB1 paper already in this benchmark — proves, as its Theorem 1:

```text
E[regret(n)] <= [8 * sum_{i: mu_i < mu*} (ln n) / Delta_i]  +  (1 + pi^2/3) * sum_j Delta_j
```

**Agrawal & Goyal (2012)**, *"Further Optimal Regret Bounds for Thompson Sampling"* (arXiv:1209.3353)
— a paper about a completely different algorithm, published a decade later, with no overlapping
authors — cites, in its own related-work section, the *identical* bound for UCB1 as a point of
comparison against their own Thompson Sampling analysis:

```text
E[R(T)] <= [8 * sum_{i=2}^N 1/Delta_i] * ln T  +  (1 + pi^2/3) * (sum_{i=2}^N Delta_i)
```

Same formula (`n`/`K` relabeled as `T`/`N`, `ln n` factored out of the sum instead of left inside
it). Two independent papers, a decade apart, converge on the same closed-form claim about the same
algorithm — one deriving it, the other citing it as settled.

## The verification

This benchmark's UCB1 entry already empirically verified this exact bound: plain UCB1, distribution
1 (2-armed Bernoulli bandit, means `[0.9, 0.6]`), reduced-scale simulation, measured regret **22.47**
against the bound's value **308.3** at `n = 100,000` — comfortably under, and confirmed genuine
(the code runs an honest simulation, not a formula pass-through; see `LEADERBOARD.md`).

Since both papers state the identical formula, that one empirical run is simultaneously a
reproduction check against Auer et al.'s own theorem *and* an independent-source cross-check
against Agrawal & Goyal's citation of it. No second simulation was needed — the point of the
comparison is that the two papers' claims are the same claim, not two different ones to
reconcile.

## What was tried and rejected first

Before landing on this, three other candidate "second papers" for a cross-comparison were checked
and rejected for the same reason as three earlier benchmark candidates (see the commit history):
their reported numbers either weren't stated precisely in text (figure-only results) or hid an
unspecified constant behind big-O notation (Agrawal & Goyal's *own* Thompson Sampling bounds —
Theorems 1 and 2 in that paper — are exactly this: `O(N/epsilon^2)` and `O(sqrt(N T ln T))` with no
literal constant given anywhere in the paper, so they are not themselves usable as a numeric
reproduction target, only their *citation* of UCB1's fully-explicit bound is).
