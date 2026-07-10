# Cross-Model Empirical Comparison

This project's LLM "brain" already runs on a fallback chain across three free Groq models
(`agent/llm.py`'s `MODEL_FALLBACKS`) for reliability — if the preferred model is rate-limited, the
next one takes over. `compare_models.py` reuses that same infrastructure for a different purpose:
answering "which of these three free models actually writes correct reproduction code?" with real
data instead of assumption.

## Method

To isolate code-generation quality from extraction/planning quality, the paper spec and
implementation plan are held **fixed** — reused verbatim from two already-verified "reproduced"
runs in `runs/` — and only the model doing the *coding* step is swapped, once each, single-shot
(no patch-and-retry loop; that would let a weak model's second or third attempt paper over its
first-attempt mistakes, which defeats the point of a single-shot comparison). The legitimacy check
that judges each result always runs on the same fixed model (`DEFAULT_MODEL`, the strongest in the
chain) regardless of which model wrote the code under test, so no model ever grades its own
homework.

Two structurally distinct tasks:
1. **UCB1 bandit regret** (`finite-time-analysis-of-the-multiarmed-b` run) — a sequential
   decision-making simulation.
2. **Count-Min Sketch** (`an-improved-data-stream-summary-the-coun` run) — a hash-based streaming
   data structure, no sequential decisions at all.

Run it: `python compare_models.py` (real Groq calls + real Docker sandbox runs, no cost beyond
free-tier quota).

## Real results

| Task | `llama-3.3-70b-versatile` | `llama-4-scout-17b-16e-instruct` | `llama-3.1-8b-instant` |
|---|---|---|---|
| UCB1 regret | reproduced genuinely | reproduced genuinely | **failed** — ran, produced no parseable output |
| Count-Min Sketch | **failed** — matched-tolerance check rejected (99.6% error) | reproduced genuinely | **failed** — crashed, exit code 1 |

The "flagship" 70B model was not the best coder here: **scout was the only model to pass both
tasks**; 70B passed 1/2; the fast/small 8B-instant model passed 0/2. Diagnosing each failure by
reading the actual generated code (not just the pass/fail label):

- **`llama-3.1-8b-instant`, UCB1 task**: the algorithm was correct, but the code ended with
  `print(json.dumps({...}))` — missing the required `RESULTS_JSON:` prefix that `agent/coder.py`'s
  own prompt explicitly and repeatedly demands (it even calls out this exact mistake by name:
  *"A bare `print(json.dumps(...))` WITHOUT the `RESULTS_JSON:` prefix is wrong and will not be
  parsed"*). A pure instruction-following miss, not an algorithmic one.
- **`llama-3.1-8b-instant`, Count-Min Sketch task**: crashed with a `NameError` — a helper function
  (`estimate_count`) referenced the modulus `p` without it being passed in as a parameter, and
  separately, the same missing-`RESULTS_JSON:`-prefix mistake was present again in the unreached
  code path.
- **`llama-3.3-70b-versatile`, Count-Min Sketch task**: ran without crashing and passed the format
  check, but got the *algorithm* wrong: `main()` builds the sketch using one set of random hash
  functions (generated inside `count_min_sketch()`), then calls `generate_hash_functions()` a
  *second* time and passes that new, uncorrelated `a, b` pair into the query step. Querying with
  different hash functions than the ones used to build the table effectively randomizes which
  buckets get read, degrading accuracy far past the sketch's theoretical guarantee (observed
  underestimate-violation rate 0.0837 against an upper bound of ~0, 1% tolerance). `scout`'s
  implementation generates `a, b` exactly once and reuses it for both insertion and query — the
  one thing 70B got wrong.

## What this does and doesn't establish

**Establishes**: on these two tasks, single-shot, the three free Groq models are not
interchangeable, and bigger is not simply better — `scout` outperformed the nominally stronger 70B
model on the harder of the two tasks, and the smallest/fastest model (8B-instant) failed both
times on an *explicit, repeated instruction* rather than algorithmic difficulty, suggesting it's
better suited to short/simple completions than to code generation under a strict output contract.

**Doesn't establish**: a general ranking. Two tasks, one shot each, is a small, noisy sample —
not an average over repeated seeds or prompts, so a single unlucky/lucky generation could flip a
result. It also doesn't test whether these models would do better with the patch-and-retry loop
this pipeline normally uses in production (`agent/orchestrator.py` gives every model up to
`--max-iterations` attempts, not one). What it does establish reliably: model choice measurably
matters for this pipeline's coding step, with concrete, readable-code evidence for exactly why —
not just a score.
