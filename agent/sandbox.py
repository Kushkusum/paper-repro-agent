from __future__ import annotations

import ast
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

from .schemas import GeneratedCode, SandboxResult

IMAGE_TAG = "paper-repro-sandbox:latest"
SANDBOX_DIR = Path(__file__).resolve().parent.parent / "sandbox"


def ensure_image() -> None:
    check = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check.returncode == 0:
        return
    build = subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, str(SANDBOX_DIR)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if build.returncode != 0:
        raise RuntimeError(f"Failed to build sandbox image:\n{build.stdout}\n{build.stderr}")


def _to_mount_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def _extract_results(stdout: str) -> dict[str, float]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("RESULTS_JSON:"):
            payload = line[len("RESULTS_JSON:") :].strip()
            try:
                data = json.loads(payload)
                return {k: float(v) for k, v in data.items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            # Fallback: some models emit a Python repr (single quotes, numpy scalar
            # wrappers like np.float64(...)) instead of real JSON. Strip the wrappers
            # and parse as a Python literal instead of failing outright.
            repaired = re.sub(r"np\.\w+\(([^()]+)\)", r"\1", payload)
            try:
                data = ast.literal_eval(repaired)
                return {k: float(v) for k, v in data.items()}
            except (ValueError, SyntaxError, TypeError):
                return {}
    return {}


def run_code(
    code: GeneratedCode,
    workdir: Path,
    timeout_sec: int = 90,
    memory_limit: str = "512m",
    cpus: str = "1",
) -> SandboxResult:
    ensure_image()
    workdir.mkdir(parents=True, exist_ok=True)
    for f in code.files:
        target = workdir / f.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")

    mount = _to_mount_path(workdir)
    container_name = f"paper-repro-{uuid.uuid4().hex[:12]}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        f"--memory={memory_limit}",
        f"--cpus={cpus}",
        "--pids-limit",
        "128",
        "-v",
        f"{mount}:/work",
        "-w",
        "/work",
        IMAGE_TAG,
        "sh",
        "-c",
        code.run_command,
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        duration = time.monotonic() - start
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        duration = time.monotonic() - start
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + \
            f"\n[sandbox] Killed after exceeding {timeout_sec}s timeout"
        exit_code = -1
        # subprocess.run only kills the local `docker run` client; the container itself
        # keeps running on the daemon (--rm alone won't clean it up on a client-side kill).
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    metrics = _extract_results(stdout)
    return SandboxResult(
        success=(exit_code == 0 and bool(metrics)),
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        parsed_metrics=metrics,
        duration_sec=duration,
    )
