from pathlib import Path

from agent.preference_data import same_run_pairs
from agent.schemas import (
    Diagnosis,
    EvaluationResult,
    IterationRecord,
    LegitimacyCheck,
    MetricComparison,
    PaperSpec,
    ReproductionReport,
    SandboxResult,
    TargetMetric,
)


def _iteration(
    iteration: int, exit_code: int, ok: bool, genuine: bool | None = None, has_diagnosis: bool = True
) -> IterationRecord:
    comparisons = [
        MetricComparison(
            name="m", reported_value=1.0, observed_value=1.0 if ok else 5.0, tolerance_pct=20,
            within_tolerance=ok, relative_error_pct=0.0 if ok else 400.0,
        )
    ]
    needs_diagnosis = (has_diagnosis and not ok) or genuine is False
    diagnosis = Diagnosis(verdict="bug", reasoning="r", proposed_fix="f") if needs_diagnosis else None
    legitimacy = LegitimacyCheck(genuine=genuine, reasoning="r") if genuine is not None else None
    return IterationRecord(
        iteration=iteration,
        sandbox_result=SandboxResult(success=ok, stdout="", stderr="", exit_code=exit_code, parsed_metrics={}),
        evaluation=EvaluationResult(all_within_tolerance=ok, comparisons=comparisons),
        diagnosis=diagnosis,
        legitimacy=legitimacy,
    )


def _write_run(run_dir: Path, iterations: list[IterationRecord], write_code: bool = True) -> None:
    spec = PaperSpec(
        title="t", method_summary="m", pseudocode_or_equations="p", experimental_setup="e",
        reduced_scale_notes="r", target_metrics=[TargetMetric(name="m", description="d", reported_value=1.0)],
    )
    report = ReproductionReport(
        paper_title="Test Paper", spec=spec, iterations=iterations,
        final_verdict="reproduced", final_reasoning="because",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if write_code:
        for it in iterations:
            iter_dir = run_dir / f"iteration_{it.iteration}"
            iter_dir.mkdir(exist_ok=True)
            (iter_dir / "run.py").write_text(f"# code for iteration {it.iteration}\n", encoding="utf-8")


def test_finds_bad_then_good_pair_in_same_run(tmp_path: Path):
    iterations = [
        _iteration(1, exit_code=0, ok=False),
        _iteration(2, exit_code=0, ok=True),
    ]
    _write_run(tmp_path / "run-a", iterations)

    pairs = same_run_pairs(tmp_path)

    assert len(pairs) == 1
    assert "iteration 1" in pairs[0].note
    assert "code for iteration 2" in pairs[0].chosen
    assert "code for iteration 1" in pairs[0].rejected


def test_legitimacy_false_counts_as_bad_even_if_metrics_matched(tmp_path: Path):
    iterations = [
        _iteration(1, exit_code=0, ok=True, genuine=False),  # the tautological-pass-through case
        _iteration(2, exit_code=0, ok=True, genuine=True),
    ]
    _write_run(tmp_path / "run-b", iterations)

    pairs = same_run_pairs(tmp_path)

    assert len(pairs) == 1
    assert "code for iteration 2" in pairs[0].chosen
    assert "code for iteration 1" in pairs[0].rejected


def test_no_pair_when_run_never_recovers(tmp_path: Path):
    iterations = [_iteration(1, exit_code=0, ok=False), _iteration(2, exit_code=0, ok=False)]
    _write_run(tmp_path / "run-c", iterations)

    assert same_run_pairs(tmp_path) == []


def test_no_pair_when_run_is_good_from_the_start(tmp_path: Path):
    iterations = [_iteration(1, exit_code=0, ok=True)]
    _write_run(tmp_path / "run-d", iterations)

    assert same_run_pairs(tmp_path) == []


def test_skips_runs_missing_code_files(tmp_path: Path):
    iterations = [_iteration(1, exit_code=0, ok=False), _iteration(2, exit_code=0, ok=True)]
    _write_run(tmp_path / "run-e", iterations, write_code=False)

    assert same_run_pairs(tmp_path) == []


def test_skips_directories_without_a_report(tmp_path: Path):
    (tmp_path / "incomplete_run").mkdir()
    assert same_run_pairs(tmp_path) == []
