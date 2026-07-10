from __future__ import annotations

from pathlib import Path

from agent.legitimacy_eval import run_evaluation

OUT_PATH = Path(__file__).resolve().parent / "runs" / "_legitimacy_eval" / "report.json"


def run() -> None:
    print("Running adversarial evaluation of the legitimacy check (real LLM calls, one per case)...\n")
    report = run_evaluation()

    for o in report.outcomes:
        mark = "OK " if o.correct else "MISS"
        print(f"[{mark}] {o.case_name}  (expected genuine={o.expected_genuine}, got {o.predicted_genuine})")
        print(f"       {o.category}")
        if not o.correct:
            print(f"       reasoning: {o.reasoning[:300]}")
        print()

    print("=" * 70)
    print(f"True positives  (genuine correctly passed):     {report.true_positives}")
    print(f"True negatives  (cheating correctly caught):     {report.true_negatives}")
    print(f"False positives (cheating passed as genuine):    {report.false_positives}  <-- the dangerous failure mode")
    print(f"False negatives (genuine wrongly rejected):      {report.false_negatives}")
    print(f"Precision: {report.precision:.2f}  Recall: {report.recall:.2f}  Accuracy: {report.accuracy:.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    run()
