from __future__ import annotations

import argparse
from pathlib import Path

from agent.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous paper-to-code reproduction agent")
    parser.add_argument("paper", type=Path, help="Path to a raw text file of the paper")
    parser.add_argument(
        "--focus",
        required=True,
        help="Which experiment in the paper to reproduce (self-contained, no proprietary data)",
    )
    parser.add_argument("--title", required=True, help="Paper title, for the report header")
    parser.add_argument("--max-iterations", type=int, default=4)
    args = parser.parse_args()

    run_pipeline(args.paper, args.focus, args.title, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
