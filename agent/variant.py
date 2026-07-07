from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from . import sandbox
from .coder import _generate_with_retry
from .llm import call_structured
from .schemas import GeneratedCode, PaperSpec, VariantProposal, VariantResult


class _VariantAnalysis(BaseModel):
    prediction_held: bool
    analysis: str = Field(description="Cite the specific before/after numbers")

PROPOSE_SYSTEM_PROMPT = """You are a research engineer who just successfully reproduced a paper's \
method at reduced scale. Now propose ONE small, well-motivated variant of the method to test —
not a random tweak, but a change grounded in the method's own logic that a thoughtful researcher
would actually wonder about (e.g. a different prior/parameter value, a different exploration
schedule, a modified update rule).

Your prediction must be concrete and falsifiable: state which direction the ALREADY-MEASURED \
metric(s) should move (increase/decrease/no meaningful change) if your hypothesis about the \
method is correct. Do not propose something that would require new metrics — it must be checkable \
against the same metric(s) already being measured.
"""

IMPLEMENT_SYSTEM_PROMPT = (
    "You are a precise research engineer modifying existing reproduction code to test one variant."
)

VARIANT_FORMAT_INSTRUCTIONS = """Output the COMPLETE modified code (all files, in full) implementing \
the described variant, in this exact delimited format, nothing else:

### FILE: <filename>
<full file content, no markdown fences>
### ENDFILE

### ENTRYPOINT: <filename to execute>
### RUN_COMMAND: <shell command to run inside the sandbox>

Rules:
- Keep everything about the original code the same EXCEPT the one specific change described.
- The output contract is unchanged: last line of stdout must be `RESULTS_JSON: {...}` with the
  SAME metric names as before (so the variant's results are directly comparable to the baseline).
- Every file that defines main() must also call it.
- Only numpy and scipy, no internet, finish in well under a minute.
"""

ANALYZE_SYSTEM_PROMPT = """You are a skeptical research engineer comparing a variant's measured \
results against its own predicted effect. Determine whether the prediction actually held, citing \
the specific before/after numbers. Do not be generous — a prediction of "decrease" that resulted in \
a negligible or opposite change did not hold.
"""


def propose_variant(spec: PaperSpec, code: GeneratedCode, baseline_metrics: dict[str, float]) -> VariantProposal:
    files_text = "\n\n".join(f"### FILE: {f.filename}\n{f.content}" for f in code.files)
    user_prompt = (
        f"METHOD: {spec.method_summary}\n\n"
        f"WORKING CODE (already reproduces the paper's reported result):\n{files_text}\n\n"
        f"BASELINE MEASURED METRICS: {json.dumps(baseline_metrics, indent=2)}"
    )
    return call_structured(PROPOSE_SYSTEM_PROMPT, user_prompt, VariantProposal, temperature=0.4)


def implement_variant(code: GeneratedCode, proposal: VariantProposal) -> GeneratedCode:
    files_text = "\n\n".join(f"### FILE: {f.filename}\n{f.content}\n### ENDFILE" for f in code.files)
    user_prompt = (
        f"ORIGINAL WORKING CODE:\n{files_text}\n\n"
        f"VARIANT TO IMPLEMENT: {proposal.description}\n"
        f"MOTIVATION: {proposal.motivation}\n\n"
        f"{VARIANT_FORMAT_INSTRUCTIONS}"
    )
    return _generate_with_retry(user_prompt)


def analyze_variant(
    proposal: VariantProposal, baseline_metrics: dict[str, float], variant_metrics: dict[str, float]
) -> tuple[bool, str]:
    user_prompt = (
        f"PREDICTION: the metric(s) should {proposal.predicted_direction} — {proposal.predicted_effect}\n\n"
        f"BASELINE METRICS: {json.dumps(baseline_metrics, indent=2)}\n"
        f"VARIANT METRICS: {json.dumps(variant_metrics, indent=2)}"
    )
    result = call_structured(ANALYZE_SYSTEM_PROMPT, user_prompt, _VariantAnalysis, temperature=0.1)
    return result.prediction_held, result.analysis


def run_variant_experiment(
    spec: PaperSpec,
    baseline_code: GeneratedCode,
    baseline_metrics: dict[str, float],
    run_dir: Path,
) -> VariantResult:
    proposal = propose_variant(spec, baseline_code, baseline_metrics)
    variant_code = implement_variant(baseline_code, proposal)
    result = sandbox.run_code(variant_code, run_dir / "variant")
    prediction_held, analysis = analyze_variant(proposal, baseline_metrics, result.parsed_metrics)
    return VariantResult(
        proposal=proposal,
        baseline_metrics=baseline_metrics,
        variant_metrics=result.parsed_metrics,
        prediction_held=prediction_held,
        analysis=analysis,
    )
