# Self-Rewarding Fine-Tuning Loop

The third piece of the "Autonomous Research Scientist" merge (#2 + #7 + #8 from the AI Systems
Catalog). The idea from #7: a model samples candidates, judges its own outputs, and the ranked
pairs train the next round via DPO. Applied here to this project's own history rather than a
generic domain: the reproduction agent has been generating code, and its own legitimacy checks and
diagnostic sub-agent have already been judging that code as genuine/buggy for months of run
history — that judged history is real preference data, sitting unused in `runs/`.

Run it: `python finetune_self_rewarding.py` (CPU only, no GPU, no paid API).

## What this actually proves, and what it doesn't

**Proves**: the mechanics of a self-rewarding loop work end-to-end on this exact machine, no GPU,
no paid API. `agent/preference_data.py` mines 3 real `(chosen, rejected)` code pairs directly out
of this project's own `runs/` directory — two from the same run (an early buggy/rejected iteration
paired with a later genuinely-passing one for the *same* task) and one across two different runs
targeting the same experiment (the UCB1 legitimacy-check story: one run's code ran a real
simulation, the other computed the theorem's own bound formula and reported that — caught as
non-genuine at the time, now repurposed as a labeled negative example). `finetune_self_rewarding.py`
formats these as a DPO dataset and trains a LoRA adapter on `distilgpt2` (82M params) via TRL.

Real numbers from the actual run: loss dropped from 0.6931 (the DPO loss at zero preference
signal — pure chance) to as low as 0.61, and `rewards/accuracies` — the fraction of pairs where the
model's implicit reward correctly ranks chosen above rejected — reached **1.0** after 2 of 9 total
steps, with the reward margin growing from 0.006 to 0.126 over training. That's a real, measurable
signal that the loss function, reward computation, and backprop are all doing what DPO is supposed
to do — not just "the script didn't crash."

**Doesn't prove**: that the resulting adapter is a better coder. Three preference pairs on an
82M-parameter model is enough capacity to memorize those three examples, not to learn a
generalizable "write correct simulation code, not tautological pass-throughs" policy. That would
need what every other GPU-bound idea in this project has needed and not had: real compute (this
laptop has no GPU, confirmed repeatedly via `agent/feasibility.py` throughout this project) and
far more than 3 labeled examples — realistically, hundreds of reproduction attempts' worth of
history, which this project doesn't have yet.

## Why the preference pairs are honest, not synthetic

Every `(chosen, rejected)` pair here is real code this project's own coder sub-agent actually
generated during an actual reproduction attempt, judged by the actual legitimacy check or
diagnostic sub-agent that decided its fate at the time — not hand-written or synthetically
constructed for this exercise. The cross-run pair is the one place a simplification was made
explicit rather than hidden: pairing code from two *different* runs assumes they're comparable
because they targeted the same experiment, which the run metadata alone can't strictly verify
(different runs can have slightly different focus-hint wording) — so it's listed as a manually
curated exception in `agent/preference_data.py`, not inferred automatically.

## What would come next if this were pursued further

- Every future reproduction run adds more (chosen, rejected) pairs automatically — the dataset
  grows by itself as the benchmark grows, with no extra labeling effort.
- A GPU environment (e.g. free-tier Colab) would allow a real base model (e.g. a 1-3B coding model)
  and enough preference pairs to plausibly move the needle on the free fallback model's actual
  weak spots documented elsewhere in this project — the delayed-batch-feedback implementation bug,
  the tautological-pass-through pattern, the `main()`-never-called bug.
- None of that is attempted here. This phase's job was to prove the loop's mechanics are real and
  runnable on exactly the hardware this whole project has been built on, and it does that.
