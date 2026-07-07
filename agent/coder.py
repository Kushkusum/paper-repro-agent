from __future__ import annotations

import json
import re

from .llm import call_text
from .schemas import CodeArtifact, Diagnosis, GeneratedCode, ImplementationPlan, PaperSpec, SandboxResult

FORMAT_INSTRUCTIONS = """Output ONLY code files in this exact delimited format, nothing else:

### FILE: <filename>
<full file content, no markdown fences>
### ENDFILE

(repeat ### FILE / ### ENDFILE blocks for each file)

### ENTRYPOINT: <filename to execute>
### RUN_COMMAND: <shell command to run inside the sandbox>

Rules:
- Only use numpy and scipy (already installed). No other third-party packages, no internet access.
- The program must finish in well under a minute on a single CPU core.
- Performance: avoid calling scipy.stats.<dist>.rvs() (or similar) once per item inside a per-timestep \
Python loop — its per-call overhead is huge at millions of calls. Prefer numpy's vectorized \
generators instead (e.g. np.random.beta(a_array, b_array) or np.random.binomial(1, p_array) sample \
all arms/items at once). If a loop over timesteps is unavoidable, keep the per-timestep work O(1) \
vectorized numpy operations, not a Python-level loop over arms calling scipy.
- If the setup involves a delay/batch parameter (e.g. feedback only arriving every N steps): model \
it literally — accumulate observations each step, but only update the decision-making statistics \
(posterior parameters, confidence bounds, etc.) once every N steps using the batch accumulated since \
the last update; actions in between must be chosen using the stale, not-yet-updated statistics. A \
delay parameter that is computed but never actually changes what data the decision rule sees (e.g. \
`if t % delay == 0: pass`) does not implement delay at all — it must gate when statistics update.
- The program's LAST line of stdout must be exactly `RESULTS_JSON: {...}` — a single-line, \
strictly-valid JSON object (double-quoted keys, no trailing commas, no Python-isms) mapping each \
target metric name to a numeric value. You MUST build it with `json.dumps(...)` — never with \
`print(dict)`, f-string interpolation of a dict, or str()/repr(). Cast every value to a plain \
Python `float(...)` first — numpy scalar types (np.float64, np.int64, etc.) do not serialize to \
valid JSON and will break parsing.
- Seed the RNG exactly ONCE, at the very top of main(), before the repetitions loop — never re-seed \
inside a function that gets called per-repetition or per-algorithm, or every "repetition" will replay \
the identical trajectory and the average will not reduce noise at all (a very common, easy-to-miss \
bug). Repetitions must each draw genuinely different random samples.
- If an entity's underlying value/identity changes mid-run (e.g. an arm "retires" and is replaced), \
reset ALL of that entity's accumulated statistics (counts, success/failure totals, confidence-bound \
state) at that moment — otherwise it keeps stale history from before the change, which silently \
breaks the intended experiment.
- Do not print anything after the RESULTS_JSON line.
"""

SYSTEM_PROMPT = "You are a precise research engineer writing self-contained Python reproduction code."


def _strip_code_fence(content: str) -> str:
    content = content.strip("\n")
    fence = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", content, re.DOTALL)
    if fence:
        return fence.group(1).strip("\n")
    # Some models only wrap the opening fence (or leave a stray closing one) rather
    # than matching pairs — strip either edge independently as a fallback.
    content = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n", "", content)
    content = re.sub(r"\n```\s*$", "", content)
    return content


def _parse_generated_code(raw: str) -> GeneratedCode:
    # Split on "### FILE:" headers rather than requiring a paired "### ENDFILE" — models
    # sometimes close a file with a bare markdown fence or omit ENDFILE entirely, and
    # requiring the exact closing marker made parsing fail even when the content was fine.
    headers = list(re.finditer(r"###\s*FILE:\s*(\S+)\s*\n", raw))
    files = []
    for i, h in enumerate(headers):
        filename = h.group(1).strip()
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        content = raw[start:end]
        content = re.sub(r"###\s*ENDFILE\s*$", "", content, flags=re.DOTALL).strip("\n")
        # Truncate at the metadata footer if it landed inside the last file's span.
        content = re.split(r"\n###\s*ENTRYPOINT:", content)[0]
        content = _strip_code_fence(content.strip("\n"))
        files.append(CodeArtifact(filename=filename, content=content))
    if not files:
        raise ValueError(f"No ### FILE blocks found in model output:\n{raw[:2000]}")

    entry_m = re.search(r"###\s*ENTRYPOINT:\s*(\S+)", raw)
    cmd_m = re.search(r"###\s*RUN_COMMAND:\s*(.+)", raw)
    entrypoint = entry_m.group(1).strip() if entry_m else files[0].filename
    run_command = cmd_m.group(1).strip() if cmd_m else f"python {entrypoint}"
    return GeneratedCode(files=files, entrypoint=entrypoint, run_command=run_command)


CODE_MAX_TOKENS = 6000


def _generate_with_retry(user_prompt: str, max_retries: int = 3) -> GeneratedCode:
    messages_context = user_prompt
    last_error = ""
    for attempt in range(max_retries):
        raw = call_text(SYSTEM_PROMPT, messages_context, temperature=0.2, max_tokens=CODE_MAX_TOKENS)
        try:
            return _parse_generated_code(raw)
        except ValueError as e:
            last_error = str(e)
            # Likely truncated (hit max_tokens) or malformed delimiters — ask for a
            # corrected, complete retry rather than crashing the whole pipeline run.
            messages_context = (
                f"{user_prompt}\n\n"
                f"Your previous response was invalid/incomplete: {last_error[:500]}\n"
                "Output the COMPLETE files again from scratch, strictly following the delimited "
                "format. Keep it concise enough to fit in one response."
            )
    raise RuntimeError(f"Failed to get valid generated code after {max_retries} attempts: {last_error}")


def generate_code(plan: ImplementationPlan, spec: PaperSpec) -> GeneratedCode:
    user_prompt = (
        f"PAPER SPEC:\n{json.dumps(spec.model_dump(), indent=2)}\n\n"
        f"IMPLEMENTATION PLAN:\n{json.dumps(plan.model_dump(), indent=2)}\n\n"
        f"{FORMAT_INSTRUCTIONS}"
    )
    return _generate_with_retry(user_prompt)


def revise_code(
    plan: ImplementationPlan,
    spec: PaperSpec,
    previous_code: GeneratedCode,
    sandbox_result: SandboxResult,
    diagnosis: Diagnosis,
) -> GeneratedCode:
    prev_files = "\n\n".join(f"### FILE: {f.filename}\n{f.content}\n### ENDFILE" for f in previous_code.files)
    user_prompt = (
        f"PAPER SPEC:\n{json.dumps(spec.model_dump(), indent=2)}\n\n"
        f"IMPLEMENTATION PLAN:\n{json.dumps(plan.model_dump(), indent=2)}\n\n"
        f"PREVIOUS CODE:\n{prev_files}\n\n"
        f"PREVIOUS RUN — stdout (tail):\n{sandbox_result.stdout[-2000:]}\n\n"
        f"PREVIOUS RUN — stderr (tail):\n{sandbox_result.stderr[-2000:]}\n\n"
        f"DIAGNOSIS:\nverdict={diagnosis.verdict}\nreasoning={diagnosis.reasoning}\n"
        f"proposed_fix={diagnosis.proposed_fix}\n\n"
        "Rewrite the full code (all files) with the fix applied.\n\n"
        f"{FORMAT_INSTRUCTIONS}"
    )
    return _generate_with_retry(user_prompt)
