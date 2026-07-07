from __future__ import annotations

from .schemas import EvaluationResult, MetricComparison, PaperSpec, SandboxResult


def evaluate(spec: PaperSpec, result: SandboxResult) -> EvaluationResult:
    comparisons = []
    all_ok = True
    for tm in spec.target_metrics:
        observed = result.parsed_metrics.get(tm.name)
        if observed is None:
            comparisons.append(
                MetricComparison(
                    name=tm.name,
                    reported_value=tm.reported_value,
                    observed_value=None,
                    tolerance_pct=tm.tolerance_pct,
                    within_tolerance=False,
                    relative_error_pct=None,
                )
            )
            all_ok = False
            continue

        if tm.reported_value == 0:
            rel_err = abs(observed) * 100
        else:
            rel_err = (observed - tm.reported_value) / abs(tm.reported_value) * 100

        if tm.comparison_type == "upper_bound":
            # Below the bound is always fine; only exceeding it (beyond tolerance) is a mismatch.
            within = rel_err <= tm.tolerance_pct
            rel_err = abs(rel_err)
        else:
            rel_err = abs(rel_err)
            within = rel_err <= tm.tolerance_pct
        all_ok = all_ok and within
        comparisons.append(
            MetricComparison(
                name=tm.name,
                reported_value=tm.reported_value,
                observed_value=observed,
                tolerance_pct=tm.tolerance_pct,
                within_tolerance=within,
                relative_error_pct=rel_err,
            )
        )
    return EvaluationResult(all_within_tolerance=all_ok, comparisons=comparisons)
