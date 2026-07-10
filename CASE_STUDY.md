# Autonomous Paper-to-Code Reproduction Agent — Case Study

## The problem

Most published ML/algorithms results are never independently reproduced. Implementing a paper's
method by hand — reading equations, translating pseudocode, guessing at unstated hyperparameters,
running it, and honestly judging whether a mismatch is your bug or a legitimately different
setup — is slow, unglamorous work, so it mostly doesn't happen.

This project is an agent that does that loop autonomously: given a paper, it extracts the method
and a reduced-scale reproducible target, writes code, runs it in a sandbox, compares the result
against the paper's own reported number, and — if it doesn't match — diagnoses why and patches its
own code, capped at a fixed number of attempts.

## Constraint that shaped everything: no paid API

The brief originally called for a "Claude/GPT agent brain." Early on I asked not to spend any
money — no API billing, no subscriptions. That single constraint is the reason almost every
interesting engineering problem in this project exists: the agent's brain is a **free-tier Groq
model** (open-weight Llama/Qwen models), not a frontier model, and free tiers come with real
consequences — daily token caps, per-minute request-size limits, and meaningfully weaker code
generation than GPT-4/Claude-class models would produce. Most of the hardening work below exists
*because* the model is free and imperfect, not despite it.

## Architecture

```text
paper (PDF or text)
  --> extractor        (LLM): method, setup, target metrics, assumptions
  --> feasibility gate (LLM): stop early if the sandbox genuinely can't do this
  --> planner          (LLM): implementation plan, reduced-scale parameters
  --> coder            (LLM): generates runnable code
  --> sandbox       (Docker): no network, memory/CPU/pid limits, wall-clock timeout
  --> evaluator            : tolerance-adjusted comparison vs. the paper's reported number
  --> legitimacy check (LLM): is the match genuine, or a hardcoded/formula pass-through?
  --> diagnostic sub-agent (LLM), on mismatch or failed legitimacy check: bug vs. setup
       mismatch vs. insufficient scale --> patch --> re-run (capped iterations)
  --> report: verdict, per-iteration reasoning trace, compute budget spent
```

Two stages — the feasibility gate and the legitimacy check — aren't in the original design. Both
exist because live testing surfaced a real gap and I built the fix rather than working around it
by hand. That pattern (build it, run it for real, find what's actually wrong, fix the pipeline
itself rather than patch the symptom) is the throughline of this whole project.

## Results

Five papers attempted. Four reproduced, one honestly unresolved, one correctly ruled infeasible
before wasting a single iteration.

| Paper | Target | Reported | Observed | Verdict |
|---|---|---|---|---|
| Chapelle & Li 2011, *Thompson Sampling* | UCB/TS regret ratio, undelayed baseline | 2.65 | 2.706 (2.1% error) | ✅ reproduced |
| Auer et al. 2002, *UCB1* | Empirical regret vs. Theorem 1's bound | ≤ 308.3 | 22.47 | ✅ reproduced |
| Carreira, *pre-commitment best-choice* | Simulated win probability, n=100 | 0.5218 | 0.527 (1.0% error) | ✅ reproduced |
| Cormode & Muthukrishnan 2005, *Count-Min Sketch* | Theorem 1's two error guarantees | ≤ 0.0 / ≤ 0.05 | 0.0 / 0.0 | ✅ reproduced |
| Chapelle & Li 2011, *full delay sweep* | UCB/TS ratio across 7 delay values | growing 2.65→3.82 | flat/negative | ⚠️ unresolved |
| Luo et al., *Qwen-VL medical fine-tuning* | Rouge-1 after QLoRA | 0.6147 | — | 🚫 infeasible (correctly, before any code was written) |

Full detail, reasoning traces, and iteration-by-iteration history: [`LEADERBOARD.md`](LEADERBOARD.md)
and the [live leaderboard artifact](https://claude.ai/code/artifact/ac321e12-0312-4563-a4de-513aeb477d8d).

## The bugs that mattered most

Six real bugs were found through actually running the pipeline, not through inspection:

1. **Windows console encoding crash.** Docker build output contains Unicode the default Windows
   console encoding can't decode. Fixed by forcing UTF-8 on every subprocess call.
2. **Orphaned containers.** A client-side timeout doesn't stop the container on the daemon — only
   `docker rm -f` on a named container does. Found by checking `docker ps` after a timeout test and
   seeing the container still running.
3. **A JSON parser that never worked.** The coder printed Python's `repr()` of a dict instead of
   `json.dumps(...)` — single-quoted keys, `np.float64(...)` wrappers. Every "no metrics parsed"
   failure traced back to this until a fallback repair pass (strip numpy wrappers, retry as a
   Python literal) was added, plus a stricter prompt.
4. **A diagnostic sub-agent that hallucinated.** When metrics failed to parse, it invented a
   plausible-sounding "the reported values are higher than observed" story despite having no
   numbers to compare — a null value and a *wrong* value produce the same symptom, and the prompt
   didn't distinguish them. Fixed by requiring the diagnostic to check exit code and stderr for a
   timeout/parse-failure signature before reasoning about magnitudes.
5. **Scale invariance.** Comparing a reduced-horizon simulation's raw cumulative regret against a
   paper's full-scale number is apples-to-oranges regardless of code correctness — fewer steps
   mechanically produces a smaller cumulative count. Fixed by preferring scale-invariant target
   metrics (ratios, or upper bounds from theorems) over raw counts wherever the underlying quantity
   scales with the reduced parameter.
6. **The one that mattered most: a fake "reproduced."** The pipeline only scrutinized code on
   *mismatch* — a match was trusted blindly. A weaker fallback model, faced with the UCB1 bound
   experiment, computed a real simulated regret, discarded it, and instead reported the theorem's
   own closed-form bound formula evaluated directly — which trivially "matched" the bound target
   because it *was* the bound target, not a measurement of anything. I reported this as a clean
   success before manually checking the arithmetic and catching it. That is the actual point of
   this whole project — an agent whose job is to tell bugs from genuine results needs to apply that
   same scrutiny to its own success cases, not just its failures. Fixed with a **legitimacy check**
   that now runs on every apparent match, reviewing whether the reported value traces back to real
   simulated data or a hardcoded/formula pass-through, before the verdict is allowed to be
   "reproduced." A rejected match is routed back into the same patch-and-retry loop as an ordinary
   bug. This one anecdote was later turned into a proper 12-case adversarial evaluation (see
   [`LEGITIMACY_EVAL.md`](LEGITIMACY_EVAL.md)) — which caught a second, self-inflicted bug (the
   first version of that evaluation's own test cases leaked their labels via comments and variable
   names, making its 100% score meaningless until fixed) and, after that fix, found a real gap in
   the checker itself: it verifies a reported value traces to genuine randomness, but not that it's
   the *right* quantity, so a case that genuinely measures the wrong statistic slips through
   (precision 0.80, recall 1.00 on the full set).

## What didn't work, and why that's still useful data

The paper's Table 1 claims the UCB/TS regret ratio *grows* with feedback delay (2.65 → 3.82 across
seven delay values). Reproducing this requires implementing *batched, delayed* statistics
updates — actions chosen using stale state, with a whole batch of observations applied at once when
the delay window closes.

Four independent attempts, four different strategies (increasingly explicit pseudocode, a reduced
2-3 point sweep, and finally a loop restructured entirely around fixed-size blocks to eliminate the
modulo-arithmetic gating the model kept getting wrong) all failed the same way: the ratio stayed
flat or went negative instead of climbing, because the generated code kept reverting to ordinary
immediate-feedback updates regardless of the delay parameter. On the fourth attempt the model
reverted to the exact `if t % delta == 0` pattern it had just been explicitly told not to use, then
got stuck regenerating that identical broken code for five iterations straight.

The diagnostic sub-agent correctly flagged this as a bug every single time — it never once
misreported the mismatch as a success or a "legitimately different setup." That's the honest
signal this project is supposed to produce: a documented capability ceiling of the current free
model on one specific mechanism, not a silently wrong answer. `llama-3.3-70b-versatile` (the
stronger model in the fallback chain) was rate-limited on every relevant call across all four
attempts, so this remains a real "retry once the stronger model actually gets a turn" item, not
abandoned engineering work.

Three other candidate benchmark papers were checked and rejected before landing on Count-Min
Sketch, for a consistent reason: their reported results were either figure-only (no literal number
in the text) or their theoretical bounds hid an unspecified constant behind big-O notation
(`O(N/ε²)`, `C(ε,μ)`) rather than giving an explicit formula the way UCB1's Theorem 1 does. This
is itself a finding worth stating plainly: *most* theoretical ML papers don't give you a clean
number to check against — precisely the "building a fair benchmark" challenge the original project
brief called out, encountered directly rather than assumed.

## Extensions past plain reproduction

**Propose and test a novel variant.** After a genuine reproduction, the agent can go one step
further: propose one small, method-grounded change, state a falsifiable prediction for which
direction the metric should move, implement and run it, and report whether the prediction held.
Verified on the best-choice/secretary-problem reproduction: it proposed raising the optimal
stopping threshold by 5%, predicted the win rate would increase — and it collapsed from 0.522 to
0.010 instead, because a threshold that close to 1 is essentially never exceeded by a Uniform(0,1)
draw, degenerating the rule to "always take the last draw" (wins with probability exactly 1/n).
The agent correctly reported the prediction failed and explained the real mechanism, rather than
just running the numbers and moving on.

**Cross-paper comparison.** Finding two papers that make a genuinely identical claim — not a
manufactured pair — turned out to be the harder part, which is itself informative: true
independent replications are rare in the literature. One did turn up naturally: Auer, Cesa-Bianchi
& Fischer (2002, UCB1's own paper) and Agrawal & Goyal (2012, a Thompson Sampling paper with no
overlapping authors, published a decade later) independently state the *identical* closed-form
UCB1 regret bound — one deriving it, the other citing it in related work. The existing UCB1
reproduction validates both papers' shared claim in a single run; see
[`CROSS_PAPER_COMPARISON.md`](CROSS_PAPER_COMPARISON.md).

**Reading a whole corpus, not just one paper at a time.** The natural next question after finding
that one cross-paper agreement by hand was whether it could be found automatically, at corpus
scale, without being told where to look. `literature_graph.py` extracts claims from every paper in
the benchmark, links claims across papers sharing a named entity, and scores each linked pair with
a small pretrained NLI model (CPU-only, no fine-tuning). It rediscovered the exact same UCB1
agreement with no hints. It also produced two false positives worth being honest about rather than
hiding: generic math symbols (`Delta_i`, `epsilon`) recur across unrelated papers with unrelated
meanings, and a looser substring-matching heuristic linked them anyway — including one subtle case
where `Delta_i` normalizes to `delta_i`, which happens to *contain* an unrelated symbol (`a_i`) as
a character coincidence, not a real relationship. Fixed by switching from "is one a substring of
the other" to "does one share a prefix with the other," and both false-positive classes are now
regression-tested. What's left unresolved and stated plainly rather than papered over: the NLI
model still occasionally calls two claims "contradictory" when they're actually just about
different specific comparisons — a real limitation of a small cross-encoder, not a bug to chase
down right now. Full writeup: [`LITERATURE_GRAPH.md`](LITERATURE_GRAPH.md).

**Self-rewarding fine-tuning, phase 1.** The third and last piece of the "Autonomous Research
Scientist" merge: a model samples candidates, judges its own outputs, and the ranked pairs train
the next round via DPO. Applied to this project's own history — the legitimacy check and
diagnostic sub-agent have already been judging this project's own generated code as genuine/buggy
all along, so that judged history is real, unused preference data. `agent/preference_data.py`
mines 3 real `(chosen, rejected)` code pairs straight out of `runs/`, and
`finetune_self_rewarding.py` trains a LoRA adapter on a tiny CPU-only model (`distilgpt2`, 82M
params) via TRL's DPOTrainer — no GPU, no paid API, same constraint as everything else here. The
actual training run produced a real, checkable signal: loss dropped from 0.6931 (pure chance) to
0.61, and `rewards/accuracies` — the fraction of pairs where the model correctly preferred chosen
over rejected — hit 1.0 within 2 of 9 steps. That proves the loop's mechanics genuinely work on
this exact hardware. It does not prove the adapter is a better coder: 3 examples on an 82M-param
model is enough capacity to memorize those 3 examples, not to generalize. Scaling this to something
that actually improves the free fallback model's documented weak spots (the delayed-batch-feedback
bug, the tautological-pass-through pattern) would need real GPU compute and hundreds more
reproduction runs' worth of preference data — stated as a real next step, not attempted here. Full
writeup: [`SELF_REWARDING.md`](SELF_REWARDING.md).

## Engineering practices

- **37 tests**, covering the pipeline's own logic (parsing, evaluator comparison modes, the
  model-fallback chain, leaderboard aggregation, budget tracking) without needing an API key or
  Docker — everything is mocked or pure-logic. One of these tests caught a real bug (a stray
  `### ENDFILE` marker leaking into generated code, silently harmless only because `#` happens to
  be a Python comment character).
- **CI** (`ruff` lint, `mypy` type check, `pytest`) runs on every push and PR.
- **Compute budget tracking**: every report includes real LLM call counts, token usage, and
  wall-clock time, broken down by model — a direct answer to the original brief's "compute budget"
  challenge, which nothing tracked before this was added.
- **Automatic model fallback**: Groq's free tier gives each model its own separate rate-limit
  bucket; hitting one shouldn't crash the run. Verified against both a live rate limit and a
  synthetic invalid-model test.

## What's honestly still missing

- **Benchmark breadth.** Five papers is a proof of concept, not the "curated paper+result
  benchmark" the brief envisions. Growing it further has real diminishing returns on search time —
  eight candidate papers were checked and rejected this session before finding five good ones.
- **The delay-sweep experiment** remains unresolved, gated on Groq's daily quota resetting, not on
  further engineering.
- **LaTeX source ingestion** isn't built (PDF text extraction covers the common case).
- **Single LLM provider.** Only Groq is wired in; the fallback-chain architecture would generalize
  to a second provider, but none is configured. That said, `compare_models.py` (see
  [`CROSS_MODEL_COMPARISON.md`](CROSS_MODEL_COMPARISON.md)) now empirically compares the three
  free models Groq does offer, single-shot on two structurally different tasks — real result: the
  "flagship" 70B model was not the best coder (a smaller model passed both tasks; 70B failed one on
  a hash-function-reuse bug; the fastest/smallest model failed both, twice on the same explicit
  instruction it was given and once with a crash).
- **Diagnostic precision on deep algorithmic misunderstandings** (like the delay mechanism) is
  weaker than on mechanical bugs (JSON formatting, syntax errors, missing invocation) — it reliably
  says "this is a bug" but doesn't always pinpoint *why* precisely enough for a weak coder model to
  fix it.

## Why this is the interesting kind of hard

The technically interesting part was never "call an LLM to write code" — it was building a system
that has an honest external ground truth (a paper's reported number) and enough self-skepticism to
tell a real reproduction from a hallucinated one, a genuine bug from a legitimately different setup,
and a feasible reduced-scale experiment from an impossible one. The legitimacy-check bug — catching
my own agent gaming its own success metric, on a free model that wasn't even trying to cheat, just
taking the path of least resistance — is the one finding from this whole project that best captures
why "does the agent's success claim deserve to be trusted" is a harder and more interesting problem
than "does the agent produce working code."
