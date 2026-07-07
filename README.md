# Paper Repro Agent

Autonomous agent that reproduces a paper's reported experimental result at reduced scale:

```
paper text --> extractor (LLM) --> PaperSpec (method, setup, target metrics)
            --> planner (LLM)   --> ImplementationPlan
            --> coder (LLM)     --> GeneratedCode
            --> sandbox (Docker, no network) --> SandboxResult
            --> evaluator       --> EvaluationResult (tolerance-adjusted)
            --> if mismatch: diagnostic (LLM) --> Diagnosis --> coder.revise_code --> re-run
              (capped at --max-iterations)
            --> report.md / report.json
```

The LLM "brain" is a free-tier hosted model via [Groq](https://console.groq.com) (open-weight models
like Llama 3.3 70B), so there is no per-token billing beyond Groq's free tier.

Groq's free tier gives each model its own separate rate-limit bucket (tokens/minute and
tokens/day). If the preferred model (`GROQ_MODEL`, default `llama-3.3-70b-versatile`) is rate-limited
or a request is too large for its per-minute cap, `agent/llm.py` automatically falls back to the next
model in `GROQ_MODEL_FALLBACKS` (comma-separated env var; defaults to `llama-3.3-70b-versatile,
meta-llama/llama-4-scout-17b-16e-instruct, llama-3.1-8b-instant`) rather than crashing the run.

## Setup

1. `pip install -r requirements.txt`
2. Sign up at https://console.groq.com, create an API key.
3. Copy `.env.example` to `.env` and paste your key into `GROQ_API_KEY`.
4. Make sure Docker Desktop is running (used as the sandbox — no network access, memory/CPU/pid
   limits, and containers are force-removed on timeout).

## Run

```
python main.py papers/thompson_sampling_raw.txt \
  --title "An Empirical Evaluation of Thompson Sampling" \
  --focus "Section 3, Table 1: regret of Thompson Sampling vs UCB on a Bernoulli bandit under varying feedback delay" \
  --max-iterations 4
```

Output goes to `runs/<timestamp>_<slug>/`:
- `spec.json` — extracted method/setup/target metrics
- `plan.json` — implementation plan
- `iteration_N/` — generated code + sandbox workdir for each attempt
- `report.md` / `report.json` — final reproducibility report with per-iteration reasoning trace

## Choosing a focus

Point `--focus` at a **self-contained** experiment (pure simulation, or data described in the paper
itself) rather than one relying on proprietary/private data the paper can't hand you — the agent
can't reproduce what it can't obtain. For the included Thompson Sampling paper, Section 3's Bernoulli
bandit simulations (Figure 1, Table 1) are self-contained; Sections 4-5 use private Yahoo! ad/news
click data and are not reproducible from the paper alone.

## Testing

`pip install -r requirements-dev.txt && python -m pytest tests/ -v`

Covers the pipeline's own logic (parsing, evaluator comparison modes, model-fallback chain,
leaderboard aggregation) without needing an API key or Docker — all LLM/sandbox calls are mocked or
avoided. Not covered by tests: anything that requires a live Groq call or a real Docker container
(extraction, planning, code generation quality, actual sandboxed execution) — those are only
verified by running the pipeline end-to-end on real papers.

## Safety notes

- Sandbox runs with `--network none`, memory/CPU/pid limits, a wall-clock timeout, and a named
  container that's force-removed if the client-side timeout fires (Docker doesn't auto-clean
  `--rm` containers when the *client* is killed, only when the container process itself exits).
- Iterations are capped (`--max-iterations`) to avoid infinite retry loops; if the diagnostic
  sub-agent concludes a mismatch is a `setup_mismatch` (not a bug), the loop stops early rather than
  retrying indefinitely.
