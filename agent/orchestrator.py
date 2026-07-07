from __future__ import annotations

import datetime
import re
from pathlib import Path

from . import coder, llm, sandbox
from .diagnostic import diagnose
from .evaluator import evaluate
from .extractor import extract_spec
from .feasibility import assess_feasibility
from .legitimacy import verify_legitimacy
from .pdf_ingest import load_paper_text
from .planner import draft_plan
from .report import write_report
from .schemas import Diagnosis, FeasibilityAssessment, IterationRecord, ReproductionReport, VariantResult
from .variant import run_variant_experiment

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "run"


def run_pipeline(
    paper_path: Path,
    focus_hint: str,
    paper_title: str,
    max_iterations: int = 4,
    log=print,
    propose_variant: bool = False,
) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{_slugify(paper_title)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    llm.reset_budget()
    feasibility: FeasibilityAssessment | None = None

    try:
        log(f"Reading {paper_path.name}...")
        paper_text = load_paper_text(paper_path)

        log("[1/6] Extracting structured spec from paper...")
        spec = extract_spec(paper_text, focus_hint)
        (run_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        log(f"  -> {len(spec.target_metrics)} target metric(s): {[m.name for m in spec.target_metrics]}")

        log("[2/6] Assessing feasibility within sandbox constraints...")
        feasibility = assess_feasibility(spec)
        (run_dir / "feasibility.json").write_text(feasibility.model_dump_json(indent=2), encoding="utf-8")
        log(f"  -> verdict={feasibility.verdict}")
        if not feasibility.feasible:
            log("  -> infeasible: stopping before planning/coding to avoid wasted iterations.")
            report = ReproductionReport(
                paper_title=paper_title,
                spec=spec,
                feasibility=feasibility,
                final_verdict=feasibility.verdict,
                final_reasoning=feasibility.reasoning,
                budget=llm.get_budget_summary(),
            )
            report_path = write_report(report, run_dir)
            log(f"Done. Report: {report_path}")
            return report_path

        log("[3/6] Drafting implementation plan...")
        plan = draft_plan(spec)
        (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        log("[4/6] Generating code...")
        code = coder.generate_code(plan, spec)
    except Exception as e:
        log(f"Aborted before any sandbox iteration ran (extraction/feasibility/planning/coding stage): {e}")
        raise

    iterations: list[IterationRecord] = []
    final_verdict = "unknown"
    final_reasoning = ""
    winning_metrics: dict[str, float] = {}

    for i in range(1, max_iterations + 1):
        try:
            log(f"[5/6] Iteration {i}/{max_iterations}: running in sandbox...")
            iter_dir = run_dir / f"iteration_{i}"
            result = sandbox.run_code(code, iter_dir)
            log(f"  -> exit_code={result.exit_code}, metrics={result.parsed_metrics}")

            evaluation = evaluate(spec, result)
            log(f"  -> all_within_tolerance={evaluation.all_within_tolerance}")

            if evaluation.all_within_tolerance:
                log("  -> metrics match; verifying the match is genuine (not a hardcoded/formula pass-through)...")
                legitimacy = verify_legitimacy(spec, code, result)
                log(f"  -> genuine={legitimacy.genuine}")
                if legitimacy.genuine:
                    iterations.append(
                        IterationRecord(
                            iteration=i, sandbox_result=result, evaluation=evaluation, legitimacy=legitimacy
                        )
                    )
                    final_verdict = "reproduced"
                    final_reasoning = (
                        f"All {len(spec.target_metrics)} target metric(s) fell within tolerance on "
                        f"iteration {i}, and the match was verified as a genuine measurement, not a "
                        "hardcoded/formula pass-through."
                    )
                    winning_metrics = result.parsed_metrics
                    break

                log("  -> match rejected as non-genuine; treating as a bug and retrying...")
                diagnosis = Diagnosis(
                    verdict="bug",
                    reasoning=legitimacy.reasoning,
                    proposed_fix="Compute the reported metric from the actual simulated data (the "
                    "quantity produced by running the random/simulated process), not from a formula "
                    "or the target value evaluated directly.",
                )
                iterations.append(
                    IterationRecord(
                        iteration=i,
                        sandbox_result=result,
                        evaluation=evaluation,
                        diagnosis=diagnosis,
                        legitimacy=legitimacy,
                    )
                )
                if i == max_iterations:
                    final_verdict = "unresolved_after_max_iterations"
                    final_reasoning = (
                        f"Reached the cap of {max_iterations} iterations; the last apparent match was "
                        f"rejected as non-genuine: {legitimacy.reasoning}"
                    )
                    break
                log("  -> patching code and retrying...")
                code = coder.revise_code(plan, spec, code, result, diagnosis)
                continue

            log("  -> mismatch detected, running diagnostic sub-agent...")
            diagnosis = diagnose(spec, code, result, evaluation)
            log(f"  -> diagnosis verdict={diagnosis.verdict}")
            iterations.append(
                IterationRecord(iteration=i, sandbox_result=result, evaluation=evaluation, diagnosis=diagnosis)
            )

            if diagnosis.verdict not in ("bug", "insufficient_scale"):
                final_verdict = diagnosis.verdict
                final_reasoning = diagnosis.reasoning
                break

            if i == max_iterations:
                final_verdict = "unresolved_after_max_iterations"
                final_reasoning = (
                    f"Reached the cap of {max_iterations} iterations without matching all target metrics. "
                    f"Last diagnosis: {diagnosis.reasoning}"
                )
                break

            log("  -> patching code and retrying...")
            code = coder.revise_code(plan, spec, code, result, diagnosis)
        except Exception as e:  # e.g. LLM API/rate-limit errors mid-loop
            log(f"  -> iteration {i} aborted by an unexpected error: {e}")
            final_verdict = "error_during_iteration"
            final_reasoning = (
                f"Iteration {i} could not complete due to an error (e.g. API/rate limit): {e}. "
                f"{len(iterations)} completed iteration(s) are recorded below."
            )
            break

    variant: VariantResult | None = None
    if propose_variant and final_verdict == "reproduced":
        try:
            log("[extra] Proposing and testing a novel variant of the reproduced method...")
            variant = run_variant_experiment(spec, code, winning_metrics, run_dir)
            log(f"  -> prediction_held={variant.prediction_held}")
        except Exception as e:
            log(f"  -> variant experiment failed (non-fatal, main reproduction result stands): {e}")

    log("[6/6] Writing report...")

    report = ReproductionReport(
        paper_title=paper_title,
        spec=spec,
        feasibility=feasibility,
        plan=plan,
        iterations=iterations,
        final_verdict=final_verdict,
        final_reasoning=final_reasoning,
        variant=variant,
        budget=llm.get_budget_summary(),
    )
    report_path = write_report(report, run_dir)
    log(f"Done. Report: {report_path}")
    return report_path
