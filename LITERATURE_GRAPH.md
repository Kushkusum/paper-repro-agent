# Literature Claim Graph

Extends this project past single-paper reproduction into cross-paper claim comparison at corpus
scale (the "reads literature" part of the merged Autonomous Research Scientist concept: #2 +
#7 + #8). Given a set of papers, it extracts checkable claims from each, links claims across
papers that share a named entity/concept, and runs a small pretrained NLI model (no fine-tuning,
CPU-only) to flag which linked pairs actually agree (entailment) or disagree (contradiction).

Run it: `python literature_graph.py` (extracts claims via the same Groq LLM used elsewhere in
this project, then scores candidate pairs with `cross-encoder/nli-deberta-v3-small`).

## Real findings, on the 6 papers already in this repo

**Two genuine, correct agreements were found automatically:**

> [ENTAILMENT] (0.98) shared entity: 'UCB1'
> - *Finite-time Analysis of the Multiarmed Bandit Problem*: "The regret of UCB1 on distribution 1
>   ([0.9, 0.6], a 2-armed Bernoulli bandit) for n=100,000 plays is at most 8·(ln 100000)/0.3 +
>   (1+π²/3)·0.3."
> - *Further Optimal Regret Bounds for Thompson Sampling*: "The UCB1 algorithm achieves a regret
>   bound of at most [8·Σ(1/Δᵢ)]·ln T + (1+π²/3)·ΣΔⱼ."

This is the identical finding already documented by hand in
[`CROSS_PAPER_COMPARISON.md`](CROSS_PAPER_COMPARISON.md) — the pipeline rediscovered it
automatically, from raw paper text, with no hint about which two papers or which claims to compare.
That's the actual validation this system needed: it found a real, previously-manually-verified
agreement without being told where to look.

**Two contradiction calls remain that are debatable — and that's an honest limitation worth
stating, not hiding:**

> [CONTRADICTION] (1.00) shared entity: 'UCB'
> - *An Empirical Evaluation of Thompson Sampling*: "Thompson sampling achieves state-of-the-art
>   results and outperforms other alternatives like UCB in some cases."
> - *Finite-time Analysis of the Multiarmed Bandit Problem*: "Policy UCB1-TUNED performs
>   substantially better than UCB1 in essentially all experiments."

These aren't actually contradictory — one claim compares Thompson Sampling to UCB in general,
the other compares two *different* UCB variants (UCB1-TUNED vs. plain UCB1) to each other. The
entity link itself is reasonable (both claims are legitimately about UCB-family algorithms, worth
surfacing as related); the NLI model's contradiction call is where it goes wrong. A small
cross-encoder is good at spotting surface-level comparative-language tension ("outperforms X" near
"performs better than Y") without reliably tracking *which specific comparison* each claim is
making. Fixing this would need a stronger NLI model or an LLM-based verification pass on flagged
pairs before reporting them — a real next step, not attempted here.

## A real bug found and fixed along the way

The first end-to-end run also produced two much worse false positives: `Delta_i` (UCB1's
arm-mean gap) linked to `delta` (Count-Min Sketch's failure probability), and `epsilon_n-GREEDY`
(a bandit algorithm name) linked to `epsilon` (Count-Min Sketch's error parameter) — both cases of
generic single-symbol math notation recurring across genuinely unrelated papers with unrelated
meanings. Blocking generic symbols from substring-style matching fixed the obvious cases, but a
second, subtler false positive survived: `Delta_i` normalizes to `delta_i`, which happens to
contain `a_i` (an unrelated Count-Min Sketch symbol) as a literal substring purely by character
coincidence (`del`**`a_i`**), not because they share a prefix or any meaning. Fixed by switching
the entity-linking heuristic from "is one a substring of the other anywhere" to "does one start
with the other" — this still correctly links real abbreviation chains (`UCB` → `UCB1` →
`UCB1-TUNED`, which share a genuine common prefix) while rejecting coincidental mid-string
matches. Both false-positive classes are now regression-tested in `tests/test_claim_graph.py`.

## What this is not (yet)

- Entity resolution is string-heuristic, not semantic (see the limitations above). A production
  version would use embedding similarity or a proper entity-linking model.
- The NLI model is small and imprecise on claims that require tracking exactly *what* is being
  compared, not just detecting comparative language.
- Only run on the 6 papers already in this repo's benchmark — not a general literature-scale claim
  graph yet.
