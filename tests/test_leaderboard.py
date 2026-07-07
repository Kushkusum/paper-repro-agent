from pathlib import Path

from agent.leaderboard import latest_per_paper, scan_runs
from agent.schemas import (
    EvaluationResult,
    IterationRecord,
    MetricComparison,
    PaperSpec,
    ReproductionReport,
    SandboxResult,
    TargetMetric,
)


def _write_report(run_dir: Path, paper_title: str, metric_name: str, verdict: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = PaperSpec(
        title=paper_title,
        method_summary="m",
        pseudocode_or_equations="p",
        experimental_setup="e",
        reduced_scale_notes="r",
        target_metrics=[TargetMetric(name=metric_name, description="d", reported_value=1.0)],
    )
    comparisons = [
        MetricComparison(
            name=metric_name, reported_value=1.0, observed_value=1.0, tolerance_pct=20, within_tolerance=True,
            relative_error_pct=0.0,
        )
    ]
    sandbox_result = SandboxResult(
        success=True, stdout="", stderr="", exit_code=0, parsed_metrics={metric_name: 1.0}
    )
    iteration = IterationRecord(
        iteration=1,
        sandbox_result=sandbox_result,
        evaluation=EvaluationResult(all_within_tolerance=True, comparisons=comparisons),
    )
    report = ReproductionReport(
        paper_title=paper_title,
        spec=spec,
        iterations=[iteration],
        final_verdict=verdict,
        final_reasoning="because",
    )
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")


def test_scan_runs_reads_all_reports(tmp_path: Path):
    _write_report(tmp_path / "20260101-000000_paper-a", "Paper A", "metric_1", "reproduced")
    _write_report(tmp_path / "20260101-000001_paper-b", "Paper B", "metric_2", "reproduced")
    summaries = scan_runs(tmp_path)
    assert len(summaries) == 2
    assert {s.paper_title for s in summaries} == {"Paper A", "Paper B"}


def test_scan_runs_skips_dirs_without_report(tmp_path: Path):
    (tmp_path / "incomplete_run").mkdir()
    _write_report(tmp_path / "20260101-000000_paper-a", "Paper A", "metric_1", "reproduced")
    summaries = scan_runs(tmp_path)
    assert len(summaries) == 1


def test_different_experiments_on_same_paper_are_not_collapsed(tmp_path: Path):
    # The real bug found in this session: two different experiments on the same paper
    # (different target metrics) must not collapse into a single leaderboard row.
    _write_report(tmp_path / "20260101-000000_paper-a", "Paper A", "easy_metric", "reproduced")
    _write_report(tmp_path / "20260101-000001_paper-a", "Paper A", "hard_metric", "unresolved_after_max_iterations")
    summaries = latest_per_paper(scan_runs(tmp_path))
    assert len(summaries) == 2
    verdicts = {s.experiment_key: s.final_verdict for s in summaries}
    assert verdicts[("easy_metric",)] == "reproduced"
    assert verdicts[("hard_metric",)] == "unresolved_after_max_iterations"


def test_repeated_attempts_at_same_experiment_keep_latest_only(tmp_path: Path):
    _write_report(tmp_path / "20260101-000000_paper-a", "Paper A", "metric_1", "unresolved_after_max_iterations")
    _write_report(tmp_path / "20260101-000001_paper-a", "Paper A", "metric_1", "reproduced")
    summaries = latest_per_paper(scan_runs(tmp_path))
    assert len(summaries) == 1
    assert summaries[0].final_verdict == "reproduced"
