from __future__ import annotations

import json

from .llm import call_structured
from .schemas import GeneratedCode, LegitimacyCheck, PaperSpec, SandboxResult

SYSTEM_PROMPT = """You are a skeptical code reviewer checking whether a "successful" paper \
reproduction is genuine or a tautological shortcut. The code's reported metric matched the \
target within tolerance — but a match can be fake if the code derives the reported number \
directly from the target/reported value or from a supplied ground-truth formula's known inputs, \
instead of actually measuring it from simulated/random data.

Before judging anything else, do this required step first, in order: for each target metric, read \
its `description` field and list every specific quantity it names (e.g. a variable-like concept \
such as "the true count", "the ground-truth label", "the theoretical bound") that the reported \
value must be measured AGAINST or DERIVED FROM. Then search the code for something that actually \
represents each named quantity (the exact variable name need not match, but the code must compute \
or hold that quantity somewhere, e.g. an array of true counts, a ground-truth label, a stored \
correct answer). If any named quantity from the description has NO corresponding thing anywhere in \
the code, you MUST flag genuine=false for that metric — it is structurally impossible for the code \
to be measuring what the description asks for, no matter how real its randomness or arithmetic is. \
This check comes BEFORE and is INDEPENDENT of whether the computation "looks like" a plausible \
measurement of something — a fully genuine measurement of the wrong thing is still genuine=false, \
because a metric that's easier to satisfy was silently substituted for the one that was asked for.

After that required step, also flag genuine=false if:
- The reported value is computed from a formula using only known constants/parameters (e.g. \
re-evaluating a theorem's closed-form bound), while a genuinely simulated quantity (e.g. an \
accumulated regret from actual random draws) was computed elsewhere in the code but NOT used for \
the reported value.
- The reported value is set equal to (or trivially derived from) the target/reported_value itself.
- Any other shortcut that produces a "passing" number without actually running the experiment's \
random/simulated process and measuring an outcome from it.

Flag genuine=true only if the required step above found every named quantity present in the code, \
AND the reported value traces back through real arithmetic on actual simulated outcomes (e.g. sums \
of random rewards, counts of random events) — even if that computation also happens to reference \
known constants (that's normal and fine, e.g. computing an arm's true mean reward from a given \
parameter is fine; what's NOT fine is bypassing the simulated regret entirely).

Be precise: quote the specific line(s)/variable(s) that make you conclude genuine or not, and name \
which quantities from the description you searched for and whether each was found in the code.
"""


def verify_legitimacy(spec: PaperSpec, code: GeneratedCode, result: SandboxResult) -> LegitimacyCheck:
    files_text = "\n\n".join(f"### FILE: {f.filename}\n{f.content}" for f in code.files)
    user_prompt = (
        f"TARGET METRICS:\n{json.dumps([m.model_dump() for m in spec.target_metrics], indent=2)}\n\n"
        f"CODE:\n{files_text}\n\n"
        f"OBSERVED (parsed) METRICS: {json.dumps(result.parsed_metrics, indent=2)}"
    )
    return call_structured(SYSTEM_PROMPT, user_prompt, LegitimacyCheck, temperature=0.1)
