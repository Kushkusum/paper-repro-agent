from __future__ import annotations

import json

from .llm import call_structured
from .schemas import ImplementationPlan, PaperSpec

SYSTEM_PROMPT = """You are a research engineer planning a small, self-contained Python reproduction \
of a paper's experiment. The code will run in a sandboxed container with only numpy/scipy available \
(no internet, no GPU, no other packages) and a strict wall-clock budget of a few minutes.

Produce a concrete implementation plan:
- approach: how you'll structure the simulation/algorithm.
- modules: filenames you'll write (usually just one or two files is enough).
- output_contract: the code MUST print a final line of the exact form \
`RESULTS_JSON: {"metric_name": value, ...}` using the SAME metric names as the target metrics, \
as the very last line of stdout, with a valid single-line JSON object.
- reduced_scale_params: concrete numbers (repetitions, horizon length, etc.) that are small enough \
to run in well under a minute on a single CPU core, while still being large enough for the \
comparison to be meaningful (e.g. hundreds to low thousands of repetitions/rounds rather than the \
paper's full scale of millions).
"""


def draft_plan(spec: PaperSpec) -> ImplementationPlan:
    user_prompt = f"PAPER SPEC:\n{json.dumps(spec.model_dump(), indent=2)}"
    return call_structured(SYSTEM_PROMPT, user_prompt, ImplementationPlan, temperature=0.2)
