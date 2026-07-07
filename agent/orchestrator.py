from __future__ import annotations

import datetime
import re
from pathlib import Path

from . import coder, sandbox
from .diagnostic import diagnose
from .evaluator import evaluate
from .extractor import extract_spec
from .planner import draft_plan
from .schemas import IterationRecord, ReproductionReport

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
) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{_slugify(paper_title)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    paper_text = paper_path.read_text(encoding="utf-8")

    try:
        log("[1/5] Extracting structured spec from paper...")
        spec = extract_spec(paper_text, focus_hint)
        (run_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        log(f"  -> {len(spec.target_metrics)} target metric(s): {[m.name for m in spec.target_metrics]}")

        log("[2/5] Drafting implementation plan...")
        plan = draft_plan(spec)
        (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        log("[3/5] Generating code...")
        code = coder.generate_code(plan, spec)
    except Exception as e:
        log(f"Aborted before any sandbox iteration ran (extraction/planning/coding stage): {e}")
        raise

    iterations: list[IterationRecord] = []
    final_verdict = "unknown"
    final_reasoning = ""

    for i in range(1, max_iterations + 1):
        try:
            log(f"[4/5] Iteration {i}/{max_iterations}: running in sandbox...")
            iter_dir = run_dir / f"iteration_{i}"
            result = sandbox.run_code(code, iter_dir)
            log(f"  -> exit_code={result.exit_code}, metrics={result.parsed_metrics}")

            evaluation = evaluate(spec, result)
            log(f"  -> all_within_tolerance={evaluation.all_within_tolerance}")

            if evaluation.all_within_tolerance:
                iterations.append(IterationRecord(iteration=i, sandbox_result=result, evaluation=evaluation))
                final_verdict = "reproduced"
                final_reasoning = (
                    f"All {len(spec.target_metrics)} target metric(s) fell within tolerance on iteration {i}."
                )
                break

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

    log("[5/5] Writing report...")
    from .report import write_report

    report = ReproductionReport(
        paper_title=paper_title,
        spec=spec,
        plan=plan,
        iterations=iterations,
        final_verdict=final_verdict,
        final_reasoning=final_reasoning,
    )
    report_path = write_report(report, run_dir)
    log(f"Done. Report: {report_path}")
    return report_path
