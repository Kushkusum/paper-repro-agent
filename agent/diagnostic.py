from __future__ import annotations

import json

from .llm import call_structured
from .schemas import Diagnosis, EvaluationResult, GeneratedCode, PaperSpec, SandboxResult

SYSTEM_PROMPT = """You are a research engineer diagnosing why a paper-reproduction experiment did \
not match the reported metric. Inspect the code, stdout/stderr, and the comparison, then decide:

- verdict "bug": the code has an implementation error (wrong formula, off-by-one, wrong distribution, \
crash, etc.) — this is fixable by patching code.
- verdict "setup_mismatch": the code is a faithful implementation, but the paper's exact setup \
(proprietary data, undisclosed hyperparameter, different scale) makes exact reproduction infeasible; \
the observed value is a legitimately different-but-reasonable outcome given the constraints.
- verdict "insufficient_scale": the direction/qualitative result is correct but noise from running at \
reduced scale (fewer repetitions/rounds) explains the gap — fixable by increasing repetitions within \
budget, not by changing logic.
- verdict "unclear": not enough evidence to decide; treat as needing another look with more logging.

Only propose a code patch (proposed_fix) when verdict is "bug" or "insufficient_scale". Be specific \
and reference exact variables/lines/formulas from the code.

IMPORTANT: if a metric's observed_value in the comparison is null, that does NOT always mean the \
same thing — check the stderr and exit code to tell these apart, they need different fixes:
- If stderr contains "Killed after exceeding Ns timeout" (or exit code -1 with empty stdout/stderr): \
the program never finished, so there was no RESULTS_JSON to parse yet. This is verdict "bug", but \
the fix is a PERFORMANCE fix, not a formatting fix — look for expensive operations inside hot loops \
(e.g. calling scipy.stats.<dist>.rvs() once per arm per timestep instead of a single vectorized \
numpy call like np.random.beta(...)/np.random.binomial(...) across all arms at once; unnecessary \
nested loops; redundant recomputation). Propose vectorizing the hot loop and/or reducing repetitions \
/ horizon, not a JSON change.
- If the program exited (code 0 or a crash) and produced stdout, but observed_value is still null: \
the code likely printed a Python repr, wrong key names, or extra text instead of strict JSON via \
json.dumps. That is a genuine formatting bug — fix the print statement.
Never reason as if you know a numeric mismatch when observed_value is null — you have no numeric \
data to compare in that case.

IMPORTANT — check trends, not just magnitudes: when there are multiple related metrics (e.g. the \
same quantity across an ordered parameter like time/delay/horizon, or a paired comparison between \
two methods/algorithms), compare the RELATIVE ordering and trend direction across them against the \
paper's reported trend, not just each magnitude in isolation. A correct reduced-scale implementation \
can legitimately produce smaller absolute magnitudes, but it should still preserve the paper's \
qualitative pattern (e.g. "regret increases as delay increases", "method A always beats method B"). \
If the observed trend is flat, noisy, or inverted relative to the reported trend, that is strong \
evidence of verdict "bug" (e.g. a variable/parameter used incorrectly, or the wrong quantity being \
measured) — do not attribute a trend inversion to "insufficient_scale", since more repetitions would \
only reduce noise around the same (wrong) trend, not flip its direction.
"""


def diagnose(
    spec: PaperSpec,
    code: GeneratedCode,
    result: SandboxResult,
    evaluation: EvaluationResult,
) -> Diagnosis:
    files_text = "\n\n".join(f"### FILE: {f.filename}\n{f.content}" for f in code.files)
    user_prompt = (
        f"TARGET METRICS (from paper):\n{json.dumps([m.model_dump() for m in spec.target_metrics], indent=2)}\n\n"
        f"CODE:\n{files_text}\n\n"
        f"STDOUT (tail):\n{result.stdout[-3000:]}\n\n"
        f"STDERR (tail):\n{result.stderr[-3000:]}\n\n"
        f"EXIT CODE: {result.exit_code}\n\n"
        f"COMPARISON:\n{json.dumps([c.model_dump() for c in evaluation.comparisons], indent=2)}\n\n"
        f"ASSUMPTIONS MADE DURING EXTRACTION:\n{json.dumps(spec.assumptions, indent=2)}"
    )
    return call_structured(SYSTEM_PROMPT, user_prompt, Diagnosis, temperature=0.2)
