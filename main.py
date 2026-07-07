from __future__ import annotations

import argparse
from pathlib import Path

from agent.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous paper-to-code reproduction agent")
    parser.add_argument("paper", type=Path, help="Path to the paper: a .pdf or a raw .txt file")
    parser.add_argument(
        "--focus",
        required=True,
        help="Which experiment in the paper to reproduce (self-contained, no proprietary data)",
    )
    parser.add_argument("--title", required=True, help="Paper title, for the report header")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument(
        "--propose-variant",
        action="store_true",
        help="After a genuine reproduction, propose and test one small variant of the method "
        "and check whether its predicted effect actually held",
    )
    args = parser.parse_args()

    run_pipeline(
        args.paper,
        args.focus,
        args.title,
        max_iterations=args.max_iterations,
        propose_variant=args.propose_variant,
    )


if __name__ == "__main__":
    main()
