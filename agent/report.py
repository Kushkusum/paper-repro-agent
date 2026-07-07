from __future__ import annotations

from pathlib import Path

from .schemas import BudgetSummary, ReproductionReport


def _render_budget(budget: BudgetSummary) -> list[str]:
    lines = [
        "## Compute Budget",
        "",
        f"{budget.total_calls} LLM call(s), {budget.total_tokens:,} total tokens "
        f"({budget.total_prompt_tokens:,} prompt + {budget.total_completion_tokens:,} completion), "
        f"{budget.total_wall_time_sec:.1f}s of LLM wall-clock time.",
        "",
    ]
    if budget.calls_by_model:
        lines += ["| model | calls | tokens |", "|---|---|---|"]
        for model, calls in budget.calls_by_model.items():
            lines.append(f"| {model} | {calls} | {budget.tokens_by_model.get(model, 0):,} |")
        lines.append("")
    return lines


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

    if report.feasibility is not None:
        lines += [
            "",
            "## Feasibility Assessment",
            "",
            f"**Verdict:** {report.feasibility.verdict}",
            "",
            report.feasibility.reasoning,
        ]

    if report.plan is None:
        if report.budget is not None:
            lines += [""] + _render_budget(report.budget)
        return "\n".join(lines)

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
        if it.legitimacy:
            lines.append("")
            lines.append(f"**Legitimacy check (genuine={it.legitimacy.genuine}):** {it.legitimacy.reasoning}")
        if it.diagnosis:
            lines.append("")
            lines.append(f"**Diagnosis ({it.diagnosis.verdict}):** {it.diagnosis.reasoning}")
            lines.append("")
            lines.append(f"**Proposed fix:** {it.diagnosis.proposed_fix}")
        lines.append("")

    if report.variant is not None:
        v = report.variant
        lines += [
            "## Proposed Variant",
            "",
            f"**Change:** {v.proposal.description}",
            "",
            f"**Motivation:** {v.proposal.motivation}",
            "",
            f"**Prediction:** {v.proposal.predicted_direction} — {v.proposal.predicted_effect}",
            "",
            "| metric | baseline | variant |",
            "|---|---|---|",
        ]
        for name in v.baseline_metrics:
            base_val = v.baseline_metrics.get(name)
            var_val = v.variant_metrics.get(name)
            var_str = f"{var_val:.4g}" if var_val is not None else "N/A"
            lines.append(f"| {name} | {base_val:.4g} | {var_str} |")
        lines += [
            "",
            f"**Prediction held:** {v.prediction_held}",
            "",
            v.analysis,
            "",
        ]

    if report.budget is not None:
        lines += _render_budget(report.budget)

    return "\n".join(lines)


def write_report(report: ReproductionReport, run_dir: Path) -> Path:
    md = render_markdown(report)
    path = run_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
