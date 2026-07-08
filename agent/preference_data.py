from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class CrossRunPairSpec(BaseModel):
    task: str
    chosen_run: str
    chosen_iteration: int
    rejected_run: str
    rejected_iteration: int
    note: str


# Manually curated cross-run pair: these are two DIFFERENT runs targeting the same experiment
# (UCB1's regret bound), so their code isn't from an identical prompt the way same-run pairs
# are -- but it's the cleanest example this project produced: one run's code computed the
# theorem's bound formula directly and reported that (caught as non-genuine by the legitimacy
# check); the other actually ran the simulation. Listed explicitly rather than inferred, since
# "same experiment" isn't something the run metadata alone can safely establish.
CROSS_RUN_PAIRS = [
    CrossRunPairSpec(
        task="Verify Theorem 1's UCB1 regret bound empirically on a 2-armed Bernoulli bandit.",
        chosen_run="20260707-225140_finite-time-analysis-of-the-multiarmed-b",
        chosen_iteration=1,
        rejected_run="20260707-220136_finite-time-analysis-of-the-multiarmed-b",
        rejected_iteration=1,
        note="chosen actually runs UCB1 and measures regret; rejected computes the theorem's "
        "own bound formula and reports that instead -- caught by the legitimacy check.",
    )
]


class PreferencePair(BaseModel):
    task: str
    chosen: str
    rejected: str
    note: str


def _read_iteration_code(run_dir: Path, iteration: int) -> str | None:
    iter_dir = run_dir / f"iteration_{iteration}"
    if not iter_dir.exists():
        return None
    py_files = sorted(iter_dir.glob("*.py"))
    if not py_files:
        return None
    return "\n\n".join(f.read_text(encoding="utf-8") for f in py_files)


def _load_report(run_dir: Path) -> dict | None:
    path = run_dir / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _iteration_is_good(iteration: dict) -> bool:
    if not iteration["evaluation"]["all_within_tolerance"]:
        return False
    legitimacy = iteration.get("legitimacy")
    return legitimacy is None or legitimacy["genuine"]


def _iteration_is_bad(iteration: dict) -> bool:
    if iteration["sandbox_result"]["exit_code"] != 0:
        return True
    if not iteration["evaluation"]["all_within_tolerance"]:
        return True
    legitimacy = iteration.get("legitimacy")
    return legitimacy is not None and not legitimacy["genuine"]


def same_run_pairs(runs_dir: Path = RUNS_DIR) -> list[PreferencePair]:
    """Within a single run, pair the first 'bad' iteration's code against the first later
    'good' iteration's code -- same underlying task (same paper spec, same target metrics),
    genuinely different code, a real before/after."""
    pairs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        report = _load_report(run_dir)
        if report is None:
            continue
        iterations = report.get("iterations", [])
        bad_idx = next((i for i, it in enumerate(iterations) if _iteration_is_bad(it)), None)
        if bad_idx is None:
            continue
        good_idx = next(
            (i for i in range(bad_idx + 1, len(iterations)) if _iteration_is_good(iterations[i])), None
        )
        if good_idx is None:
            continue

        rejected_code = _read_iteration_code(run_dir, iterations[bad_idx]["iteration"])
        chosen_code = _read_iteration_code(run_dir, iterations[good_idx]["iteration"])
        if not rejected_code or not chosen_code:
            continue

        pairs.append(
            PreferencePair(
                task=f"{report['paper_title']}: {', '.join(m['name'] for m in report['spec']['target_metrics'])}",
                chosen=chosen_code,
                rejected=rejected_code,
                note=f"same run ({run_dir.name}), iteration {iterations[bad_idx]['iteration']} -> "
                f"{iterations[good_idx]['iteration']}",
            )
        )
    return pairs


def cross_run_pairs(runs_dir: Path = RUNS_DIR) -> list[PreferencePair]:
    pairs = []
    for spec in CROSS_RUN_PAIRS:
        chosen = _read_iteration_code(runs_dir / spec.chosen_run, spec.chosen_iteration)
        rejected = _read_iteration_code(runs_dir / spec.rejected_run, spec.rejected_iteration)
        if not chosen or not rejected:
            continue
        pairs.append(PreferencePair(task=spec.task, chosen=chosen, rejected=rejected, note=spec.note))
    return pairs


def collect_preference_pairs(runs_dir: Path = RUNS_DIR) -> list[PreferencePair]:
    return same_run_pairs(runs_dir) + cross_run_pairs(runs_dir)
