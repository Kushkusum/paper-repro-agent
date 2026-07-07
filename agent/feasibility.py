from __future__ import annotations

import json

from .llm import call_structured
from .schemas import FeasibilityAssessment, PaperSpec

SANDBOX_DESCRIPTION = """The reproduction will run inside a Docker sandbox with these hard constraints:
- No GPU — single CPU core only.
- No network access at all (--network none) — nothing can be downloaded: no pretrained model
  weights, no datasets, no pip packages beyond what's already installed.
- Only numpy and scipy are installed. No PyTorch, TensorFlow, JAX, transformers, or any other
  ML/deep-learning framework.
- A wall-clock budget of well under a minute per run.
- Whatever data the experiment needs must either be (a) generated synthetically in code (e.g. a
  simulated environment, random data with known distributions), or (b) small, public, and already
  embedded in the paper's text/tables — it cannot be fetched from anywhere at runtime."""

SYSTEM_PROMPT = f"""You are assessing whether a paper's target experiment can be reduced-scale \
reproduced inside a specific sandbox, BEFORE any planning or coding is attempted — this saves \
wasted iterations on something that can never succeed regardless of how it's implemented.

{SANDBOX_DESCRIPTION}

Judge feasibility from the extracted spec:
- infeasible_needs_gpu: the method requires training/fine-tuning/running a model whose size or \
compute demands make CPU-only execution impractical even at a token reduced scale (e.g. fine-tuning \
a multi-billion-parameter model, training a large deep network from scratch to convergence).
- infeasible_private_data: the experiment depends on a specific dataset that is proprietary, \
unreleased, or otherwise not obtainable (and not small enough to be synthesized from the paper's \
own text/tables).
- infeasible_needs_internet: the method requires downloading something at runtime (pretrained \
weights, an API call to an external service, a dataset from a URL) that the no-network sandbox \
cannot fetch.
- infeasible_unsupported_framework: the method fundamentally requires a deep-learning framework \
(e.g. defining and training a neural network with backprop) that isn't numpy/scipy-expressible in \
a reasonable amount of code.
- infeasible_other: some other hard blocker not covered above — explain it clearly.
- feasible: none of the above apply; a reduced-scale (fewer steps/samples/repetitions) version of \
the experiment can genuinely be implemented and run with just numpy/scipy, no GPU, no network, \
within the time budget.

Be honest and specific. A simulation, a classical algorithm, a statistical test, or a small \
closed-form/synthetic-data experiment is normally feasible. Fine-tuning or training a real neural \
network of any meaningful size, or anything requiring a proprietary dataset or an internet \
download, is normally not.
"""


def assess_feasibility(spec: PaperSpec) -> FeasibilityAssessment:
    user_prompt = f"PAPER SPEC:\n{json.dumps(spec.model_dump(), indent=2)}"
    return call_structured(SYSTEM_PROMPT, user_prompt, FeasibilityAssessment, temperature=0.1)
