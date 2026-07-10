import json
from pathlib import Path

from agent.model_comparison import ModelComparisonResult, load_fixed_spec_and_plan
from agent.schemas import ImplementationPlan, PaperSpec, TargetMetric


def _write_fixed_run(run_dir: Path) -> None:
    spec = PaperSpec(
        title="t",
        method_summary="m",
        pseudocode_or_equations="p",
        experimental_setup="e",
        reduced_scale_notes="r",
        target_metrics=[TargetMetric(name="m", description="d", reported_value=1.0)],
    )
    plan = ImplementationPlan(
        approach="a", modules=["run.py"], output_contract="RESULTS_JSON: {}", reduced_scale_params={}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "plan.json").write_text(json.dumps(plan.model_dump()), encoding="utf-8")


def test_loads_fixed_spec_and_plan_from_a_run_dir(tmp_path: Path):
    run_dir = tmp_path / "some-run"
    _write_fixed_run(run_dir)

    spec, plan = load_fixed_spec_and_plan(run_dir)

    assert spec.title == "t"
    assert plan.approach == "a"


def test_result_defaults_are_independent_between_instances():
    a = ModelComparisonResult(model="model-a", generated_ok=True)
    b = ModelComparisonResult(model="model-b", generated_ok=True)

    a.observed_metrics["x"] = 1.0

    assert b.observed_metrics == {}
