from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .schemas import MetricComparison, ReproductionReport

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class RunSummary:
    run_id: str
    paper_title: str
    experiment_key: tuple[str, ...]
    final_verdict: str
    final_reasoning: str
    num_iterations: int
    comparisons: list[MetricComparison] = field(default_factory=list)


def _load_report(run_dir: Path) -> ReproductionReport | None:
    path = run_dir / "report.json"
    if not path.exists():
        return None
    return ReproductionReport.model_validate_json(path.read_text(encoding="utf-8"))


def scan_runs(runs_dir: Path = RUNS_DIR) -> list[RunSummary]:
    summaries = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        report = _load_report(run_dir)
        if report is None:
            continue
        last_comparisons = report.iterations[-1].evaluation.comparisons if report.iterations else []
        # Two runs of the *same* paper can target genuinely different experiments (e.g. an
        # easier baseline vs. a harder full sweep) — group by paper + which target metrics
        # were being reproduced, not by paper title alone, or a harder later attempt would
        # silently mask an earlier successful one (or vice versa) in the "latest" view.
        experiment_key = tuple(sorted(m.name for m in report.spec.target_metrics))
        summaries.append(
            RunSummary(
                run_id=run_dir.name,
                paper_title=report.paper_title,
                experiment_key=experiment_key,
                final_verdict=report.final_verdict,
                final_reasoning=report.final_reasoning,
                num_iterations=len(report.iterations),
                comparisons=last_comparisons,
            )
        )
    return summaries


def latest_per_paper(summaries: list[RunSummary]) -> list[RunSummary]:
    # run_id is a sortable "YYYYMMDD-HHMMSS_slug" timestamp prefix, so within each distinct
    # (paper, experiment) group, the last entry after the input's chronological sort is the
    # most recent attempt at that specific experiment.
    latest: dict[tuple[str, tuple[str, ...]], RunSummary] = {}
    for s in summaries:
        latest[(s.paper_title, s.experiment_key)] = s
    return sorted(latest.values(), key=lambda s: (s.paper_title, s.run_id))


_VERDICT_BADGE = {
    "reproduced": "✅ reproduced",
    "unresolved_after_max_iterations": "⚠️ unresolved",
    "setup_mismatch": "🔍 setup mismatch",
    "insufficient_scale": "🔍 insufficient scale",
    "error_during_iteration": "💥 error",
    "infeasible_needs_gpu": "🚫 infeasible (GPU)",
    "infeasible_private_data": "🚫 infeasible (private data)",
    "infeasible_needs_internet": "🚫 infeasible (needs internet)",
    "infeasible_unsupported_framework": "🚫 infeasible (framework)",
    "infeasible_other": "🚫 infeasible",
}


def render_markdown(summaries: list[RunSummary]) -> str:
    lines = [
        "# Reproducibility Leaderboard",
        "",
        f"{len(summaries)} experiment(s) attempted across "
        f"{len({s.paper_title for s in summaries})} paper(s).",
        "",
        "| Paper | Verdict | Metrics (reported vs observed) | Iterations |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        badge = _VERDICT_BADGE.get(s.final_verdict, s.final_verdict)
        if s.comparisons:
            metrics_cell = "<br>".join(
                f"`{c.name}`: {c.reported_value:.4g} vs {c.observed_value:.4g}"
                if c.observed_value is not None
                else f"`{c.name}`: {c.reported_value:.4g} vs N/A"
                for c in s.comparisons
            )
        else:
            metrics_cell = "—"
        n = len(s.experiment_key)
        experiment_label = f"{n} metric{'s' if n != 1 else ''}: {', '.join(s.experiment_key)}"
        lines.append(f"| {s.paper_title}<br><sub>{experiment_label}</sub> | {badge} | {metrics_cell} | {s.num_iterations} |")

    lines += ["", "## Details", ""]
    for s in summaries:
        n = len(s.experiment_key)
        experiment_label = f"{n} metric{'s' if n != 1 else ''}: {', '.join(s.experiment_key)}"
        lines.append(f"### {s.paper_title} — {experiment_label}")
        lines.append("")
        lines.append(f"**Verdict:** {_VERDICT_BADGE.get(s.final_verdict, s.final_verdict)} (`{s.run_id}`)")
        lines.append("")
        lines.append(s.final_reasoning)
        lines.append("")
    return "\n".join(lines)


def build_leaderboard(runs_dir: Path = RUNS_DIR) -> str:
    summaries = latest_per_paper(scan_runs(runs_dir))
    return render_markdown(summaries)
