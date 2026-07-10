from __future__ import annotations

from pathlib import Path

from agent.llm import MODEL_FALLBACKS
from agent.model_comparison import ModelComparisonResult, load_fixed_spec_and_plan, run_comparison

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "runs" / "_model_comparison"

# Two structurally distinct, already-verified "reproduced" runs, reused as fixed spec+plan pairs
# so the comparison isolates code-generation quality rather than extraction/planning differences.
FIXED_RUN_DIRS = [
    REPO_ROOT / "runs" / "20260707-225140_finite-time-analysis-of-the-multiarmed-b",  # UCB1 bandit regret
    REPO_ROOT / "runs" / "20260707-235638_an-improved-data-stream-summary-the-coun",  # Count-Min Sketch
]


def _describe(r: ModelComparisonResult) -> str:
    if not r.generated_ok:
        return f"FAILED to generate code: {r.generation_error}"
    if not r.sandbox_success:
        return f"code ran but exit_code={r.exit_code}, no parseable metrics ({r.observed_metrics})"
    if not r.all_within_tolerance:
        err = f"{r.max_relative_error_pct:.1f}%" if r.max_relative_error_pct is not None else "n/a (metric missing)"
        return f"ran, metrics={r.observed_metrics}, but outside tolerance (max err {err})"
    if r.legitimate is False:
        return f"matched tolerance but REJECTED as non-genuine: {r.legitimacy_reasoning}"
    err = f"{r.max_relative_error_pct:.1f}%" if r.max_relative_error_pct is not None else "n/a"
    return f"reproduced genuinely: metrics={r.observed_metrics}, max err {err}"


def run() -> None:
    for fixed_run_dir in FIXED_RUN_DIRS:
        spec, plan = load_fixed_spec_and_plan(fixed_run_dir)
        print("=" * 70)
        print(f"Fixed spec/plan reused from {fixed_run_dir.name}")
        print(f"Target metric(s): {[m.name for m in spec.target_metrics]}\n")
        print(f"Comparing models (single-shot, no patch-retry): {MODEL_FALLBACKS}\n")

        out_dir = OUT_DIR / fixed_run_dir.name
        report = run_comparison(spec, plan, MODEL_FALLBACKS, out_dir, paper_title=spec.title)

        for r in report.results:
            print(f"--- {r.model} ---")
            print(f"  {_describe(r)}")
            print(f"  generation: {r.generation_tokens} tokens, {r.generation_wall_time_sec:.1f}s\n")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.json"
        out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {out_path}\n")


if __name__ == "__main__":
    run()
