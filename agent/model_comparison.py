"""Cross-model empirical comparison: which free Groq model is actually best at the coding step
of this pipeline?

Reuses the existing model-fallback infrastructure (agent/llm.py's MODEL_FALLBACKS) but for a
different purpose: instead of falling back on failure, it holds the paper spec and implementation
plan fixed (taken from an already-verified "reproduced" run) and swaps out ONLY which model writes
the code, once each, single-shot (no patch-and-retry loop). That isolates the comparison to code
generation quality rather than confounding it with extraction/planning differences too.

The legitimacy check is always run with the same fixed judge model (agent/llm.py's DEFAULT_MODEL,
the strongest model in the chain) regardless of which model wrote the code under test -- so no
model ever grades its own homework.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from . import coder, llm, sandbox
from .evaluator import evaluate
from .legitimacy import verify_legitimacy
from .schemas import ImplementationPlan, PaperSpec


class ModelComparisonResult(BaseModel):
    model: str
    generated_ok: bool
    generation_error: str | None = None
    generation_tokens: int = 0
    generation_wall_time_sec: float = 0.0
    sandbox_success: bool = False
    exit_code: int | None = None
    observed_metrics: dict[str, float] = {}
    all_within_tolerance: bool = False
    max_relative_error_pct: float | None = None
    legitimate: bool | None = None
    legitimacy_reasoning: str | None = None


class ModelComparisonReport(BaseModel):
    paper_title: str
    target_metrics: list[str]
    judge_model: str
    results: list[ModelComparisonResult]


def load_fixed_spec_and_plan(run_dir: Path) -> tuple[PaperSpec, ImplementationPlan]:
    spec = PaperSpec.model_validate(json.loads((run_dir / "spec.json").read_text(encoding="utf-8")))
    plan = ImplementationPlan.model_validate(json.loads((run_dir / "plan.json").read_text(encoding="utf-8")))
    return spec, plan


def run_comparison(
    spec: PaperSpec,
    plan: ImplementationPlan,
    models: list[str],
    workdir_root: Path,
    paper_title: str,
) -> ModelComparisonReport:
    results = []
    for model in models:
        llm.reset_budget()
        result = ModelComparisonResult(model=model, generated_ok=False)
        try:
            code = coder.generate_code(plan, spec, model=model)
            result.generated_ok = True
        except Exception as e:
            result.generation_error = str(e)
            results.append(result)
            continue

        budget = llm.get_budget_summary()
        result.generation_tokens = budget.total_tokens
        result.generation_wall_time_sec = budget.total_wall_time_sec

        sandbox_result = sandbox.run_code(code, workdir_root / model.replace("/", "_"))
        result.sandbox_success = sandbox_result.success
        result.exit_code = sandbox_result.exit_code
        result.observed_metrics = sandbox_result.parsed_metrics

        evaluation = evaluate(spec, sandbox_result)
        result.all_within_tolerance = evaluation.all_within_tolerance
        errors = [c.relative_error_pct for c in evaluation.comparisons if c.relative_error_pct is not None]
        result.max_relative_error_pct = max(errors) if errors else None

        if evaluation.all_within_tolerance:
            legitimacy = verify_legitimacy(spec, code, sandbox_result)
            result.legitimate = legitimacy.genuine
            result.legitimacy_reasoning = legitimacy.reasoning

        results.append(result)

    return ModelComparisonReport(
        paper_title=paper_title,
        target_metrics=[m.name for m in spec.target_metrics],
        judge_model=llm.DEFAULT_MODEL,
        results=results,
    )
