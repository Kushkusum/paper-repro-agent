from agent.evaluator import evaluate
from agent.schemas import PaperSpec, SandboxResult, TargetMetric


def _spec(metric: TargetMetric) -> PaperSpec:
    return PaperSpec(
        title="t",
        method_summary="m",
        pseudocode_or_equations="p",
        experimental_setup="e",
        reduced_scale_notes="r",
        target_metrics=[metric],
    )


def _result(value: float | None) -> SandboxResult:
    metrics = {} if value is None else {"m": value}
    return SandboxResult(success=True, stdout="", stderr="", exit_code=0, parsed_metrics=metrics)


def test_target_value_within_tolerance_passes():
    spec = _spec(TargetMetric(name="m", description="d", reported_value=2.65, tolerance_pct=20))
    ev = evaluate(spec, _result(2.70))
    assert ev.all_within_tolerance
    assert ev.comparisons[0].within_tolerance


def test_target_value_outside_tolerance_fails():
    spec = _spec(TargetMetric(name="m", description="d", reported_value=2.65, tolerance_pct=5))
    ev = evaluate(spec, _result(10.0))
    assert not ev.all_within_tolerance
    assert not ev.comparisons[0].within_tolerance


def test_missing_metric_is_null_and_fails():
    spec = _spec(TargetMetric(name="m", description="d", reported_value=2.65))
    ev = evaluate(spec, _result(None))
    assert not ev.all_within_tolerance
    c = ev.comparisons[0]
    assert c.observed_value is None
    assert c.relative_error_pct is None
    assert not c.within_tolerance


def test_upper_bound_far_below_passes():
    spec = _spec(
        TargetMetric(name="m", description="d", reported_value=308.3, tolerance_pct=25, comparison_type="upper_bound")
    )
    ev = evaluate(spec, _result(23.16))
    assert ev.all_within_tolerance


def test_upper_bound_exceeding_fails():
    spec = _spec(
        TargetMetric(name="m", description="d", reported_value=308.3, tolerance_pct=25, comparison_type="upper_bound")
    )
    ev = evaluate(spec, _result(400.0))
    assert not ev.all_within_tolerance


def test_upper_bound_slightly_over_within_tolerance_passes():
    # 25% tolerance means up to 308.3 * 1.25 is still acceptable.
    spec = _spec(
        TargetMetric(name="m", description="d", reported_value=308.3, tolerance_pct=25, comparison_type="upper_bound")
    )
    ev = evaluate(spec, _result(320.0))
    assert ev.all_within_tolerance


def test_target_value_below_and_above_both_flagged_equally():
    spec = _spec(TargetMetric(name="m", description="d", reported_value=100.0, tolerance_pct=10))
    below = evaluate(spec, _result(50.0)).comparisons[0]
    above = evaluate(spec, _result(150.0)).comparisons[0]
    assert not below.within_tolerance
    assert not above.within_tolerance
