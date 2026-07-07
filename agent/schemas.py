from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TargetMetric(BaseModel):
    name: str = Field(description="Short identifier, e.g. 'ucb_regret_delay_1'")
    description: str = Field(description="What this number means in the paper")
    reported_value: float
    unit: str = Field(default="", description="e.g. '%', 'regret count', 'accuracy'")
    tolerance_pct: float = Field(
        default=20.0,
        description="Allowed relative deviation (%) before flagging as a mismatch, "
        "accounting for reduced-scale reproduction and stochastic variance",
    )
    comparison_type: Literal["target_value", "upper_bound"] = Field(
        default="target_value",
        description="'target_value': observed must be within tolerance_pct of reported_value "
        "(use for point-estimate reported numbers, e.g. a reported regret ratio). "
        "'upper_bound': observed must not exceed reported_value by more than tolerance_pct "
        "(use for a proven theoretical bound/guarantee from the paper, e.g. a regret bound "
        "formula) — being far below the bound is fine and not a mismatch.",
    )


class PaperSpec(BaseModel):
    title: str
    method_summary: str = Field(description="Plain-language summary of the algorithm/method")
    pseudocode_or_equations: str = Field(
        description="Extracted pseudocode/equations relevant to the target experiment, verbatim if possible"
    )
    experimental_setup: str = Field(
        description="Data, environment, and procedure needed to reproduce the target experiment"
    )
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    reduced_scale_notes: str = Field(
        description="How to scale the experiment down (fewer trials/repetitions/rounds) while keeping "
        "the comparison meaningful"
    )
    target_metrics: list[TargetMetric]
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions the extractor had to make due to underspecification"
    )

    @model_validator(mode="after")
    def _unique_metric_names(self) -> "PaperSpec":
        names = [m.name for m in self.target_metrics]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"target_metrics names must be unique, but these repeat: {dupes}. "
                "Give each metric a distinct name, e.g. by suffixing the varying parameter "
                "(ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_3, ...)."
            )
        return self


class ImplementationPlan(BaseModel):
    approach: str = Field(description="High-level plan for implementing the method")
    modules: list[str] = Field(description="Files/functions to write")
    output_contract: str = Field(
        description="Exact machine-readable format the code must print/write so results can be parsed, "
        "e.g. a final line 'RESULTS_JSON: {...}'"
    )
    reduced_scale_params: dict[str, str] = Field(
        description="Concrete reduced-scale values to use (trial counts, repetitions, etc.)"
    )


class CodeArtifact(BaseModel):
    filename: str
    content: str


class GeneratedCode(BaseModel):
    files: list[CodeArtifact]
    entrypoint: str = Field(description="Filename to execute, e.g. 'run.py'")
    run_command: str = Field(description="Command to run inside the sandbox, e.g. 'python run.py'")


class SandboxResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    parsed_metrics: dict[str, float] = Field(default_factory=dict)
    duration_sec: float = 0.0


class MetricComparison(BaseModel):
    name: str
    reported_value: float
    observed_value: float | None
    tolerance_pct: float
    within_tolerance: bool
    relative_error_pct: float | None


class EvaluationResult(BaseModel):
    all_within_tolerance: bool
    comparisons: list[MetricComparison]


class Diagnosis(BaseModel):
    verdict: str = Field(description="One of: 'bug', 'setup_mismatch', 'insufficient_scale', 'unclear'")
    reasoning: str
    proposed_fix: str = Field(description="Concrete description of the patch to apply")


class IterationRecord(BaseModel):
    iteration: int
    sandbox_result: SandboxResult
    evaluation: EvaluationResult
    diagnosis: Diagnosis | None = None


class ReproductionReport(BaseModel):
    paper_title: str
    spec: PaperSpec
    plan: ImplementationPlan
    iterations: list[IterationRecord]
    final_verdict: str
    final_reasoning: str
