import pytest
from pydantic import ValidationError

from agent.schemas import PaperSpec, TargetMetric


def _base_kwargs(**overrides):
    kwargs = dict(
        title="t",
        method_summary="m",
        pseudocode_or_equations="p",
        experimental_setup="e",
        reduced_scale_notes="r",
        target_metrics=[],
    )
    kwargs.update(overrides)
    return kwargs


def test_duplicate_metric_names_rejected():
    dup = [
        TargetMetric(name="ratio", description="d1", reported_value=1.0),
        TargetMetric(name="ratio", description="d2", reported_value=2.0),
    ]
    with pytest.raises(ValidationError, match="must be unique"):
        PaperSpec(**_base_kwargs(target_metrics=dup))


def test_unique_metric_names_accepted():
    metrics = [
        TargetMetric(name="ratio_1", description="d1", reported_value=1.0),
        TargetMetric(name="ratio_2", description="d2", reported_value=2.0),
    ]
    spec = PaperSpec(**_base_kwargs(target_metrics=metrics))
    assert len(spec.target_metrics) == 2


def test_comparison_type_defaults_to_target_value():
    tm = TargetMetric(name="m", description="d", reported_value=1.0)
    assert tm.comparison_type == "target_value"


def test_comparison_type_rejects_invalid_literal():
    with pytest.raises(ValidationError):
        TargetMetric(name="m", description="d", reported_value=1.0, comparison_type="not_a_real_mode")
