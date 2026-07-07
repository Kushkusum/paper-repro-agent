from __future__ import annotations

from pathlib import Path

from agent.leaderboard import build_leaderboard

if __name__ == "__main__":
    md = build_leaderboard()
    out_path = Path(__file__).resolve().parent / "LEADERBOARD.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print(md)
