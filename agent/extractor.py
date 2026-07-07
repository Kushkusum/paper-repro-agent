from __future__ import annotations

from .llm import call_structured
from .schemas import PaperSpec

SYSTEM_PROMPT = """You are a meticulous ML research engineer extracting a REPRODUCIBLE experiment \
specification from a paper. You will be given raw paper text and a focus hint telling you which \
experiment to target (prefer self-contained experiments that do not require proprietary/private data).

Extract:
- The method/algorithm as pseudocode or equations (copy verbatim where possible).
- The exact experimental setup (data generation process, environment, procedure).
- Hyperparameters with their exact reported values.
- Concrete numeric target metrics reported in the paper for the focused experiment, each as a \
separate TargetMetric with the exact reported_value. Set tolerance_pct higher (e.g. 25-40%) if the \
experiment involves stochastic simulation, and lower (e.g. 10-15%) for deterministic quantities.
- comparison_type: use "target_value" (the default) when reported_value is a point-estimate number \
the paper reports from a specific run (e.g. a reported ratio, accuracy, or regret count) — the \
reproduction should land close to it in both directions. Use "upper_bound" instead when \
reported_value comes from a PROVEN theoretical bound/guarantee (e.g. a regret bound formula from a \
theorem, with the paper's own symbols plugged in as numbers) — being far below the bound is a good \
sign, not a mismatch, and only exceeding it matters. If the focus hint or paper text gives you a \
theorem's closed-form bound already evaluated to a number, that is an upper_bound target.
- CRITICAL — scale invariance: if the reported quantity is scale-dependent (e.g. a cumulative count \
like total regret/reward/clicks that grows with the time horizon T or number of repetitions), a \
reduced-scale run CANNOT be compared to it directly — running fewer steps mechanically produces a \
smaller cumulative count regardless of whether the implementation is correct. In that case, prefer \
target metrics that are scale-invariant: a RATIO between two reported quantities (e.g. \
UCB_regret / TS_regret at the same setting), a normalized/per-step rate, or a percentage — something \
whose expected value doesn't change just because you ran fewer steps or repetitions. Only use a raw \
absolute reported number as a target metric if it is itself scale-invariant (e.g. an accuracy, a \
percentage, a converged/steady-state ratio) or if you are NOT reducing the scale for that quantity.
- Notes on how to run this at reduced scale (fewer repetitions/rounds/samples) while keeping the \
comparison between methods/metrics meaningful.
- Any assumptions you had to make because the paper underspecifies something (e.g. missing seed, \
missing exact batch size).

Be precise and extract real numbers from the text, not placeholders.
"""


def extract_spec(paper_text: str, focus_hint: str) -> PaperSpec:
    user_prompt = (
        f"FOCUS: {focus_hint}\n\n"
        f"PAPER TEXT:\n{paper_text}"
    )
    return call_structured(SYSTEM_PROMPT, user_prompt, PaperSpec, temperature=0.1)
