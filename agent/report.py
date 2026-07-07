from __future__ import annotations

from pathlib import Path

from .schemas import ReproductionReport


def render_markdown(report: ReproductionReport) -> str:
    lines = [
        f"# Reproduction Report: {report.paper_title}",
        "",
        f"**Final verdict:** {report.final_verdict}",
        "",
        report.final_reasoning,
        "",
        "## Extracted Spec",
        "",
        f"**Method:** {report.spec.method_summary}",
        "",
        f"**Reduced-scale notes:** {report.spec.reduced_scale_notes}",
        "",
        "**Assumptions made:**",
    ]
    for a in report.spec.assumptions:
        lines.append(f"- {a}")

    lines += ["", "## Implementation Plan", "", report.plan.approach, ""]

    lines += ["## Iterations", ""]
    for it in report.iterations:
        lines.append(f"### Iteration {it.iteration}")
        lines.append("")
        lines.append(f"- exit_code: {it.sandbox_result.exit_code}, duration: {it.sandbox_result.duration_sec:.1f}s")
        lines.append(f"- all_within_tolerance: {it.evaluation.all_within_tolerance}")
        lines.append("")
        lines.append("| metric | reported | observed | rel_error% | tolerance% | ok |")
        lines.append("|---|---|---|---|---|---|")
        for c in it.evaluation.comparisons:
            obs = f"{c.observed_value:.4g}" if c.observed_value is not None else "N/A"
            err = f"{c.relative_error_pct:.1f}" if c.relative_error_pct is not None else "N/A"
            lines.append(f"| {c.name} | {c.reported_value:.4g} | {obs} | {err} | {c.tolerance_pct} | {c.within_tolerance} |")
        if it.diagnosis:
            lines.append("")
            lines.append(f"**Diagnosis ({it.diagnosis.verdict}):** {it.diagnosis.reasoning}")
            lines.append("")
            lines.append(f"**Proposed fix:** {it.diagnosis.proposed_fix}")
        lines.append("")

    return "\n".join(lines)


def write_report(report: ReproductionReport, run_dir: Path) -> Path:
    md = render_markdown(report)
    path = run_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
