from __future__ import annotations

from .llm import call_structured
from .schemas import PaperClaims

SYSTEM_PROMPT = """You are a research analyst extracting checkable factual claims from a paper, for \
the purpose of comparing claims ACROSS papers to find where they agree or disagree.

Extract 3-8 claims. A good claim:
- Is self-contained: someone who hasn't read the paper can understand it on its own.
- Is specific enough to potentially agree or disagree with another paper's claim about the same
  algorithm/method/quantity -- not a vague summary like "this method works well."
- Names the concrete entities involved (algorithm names, problem names, specific bounds or
  quantities) so it can be linked to related claims from other papers.

Prefer claims about: which method outperforms which other method and under what condition; a
specific numeric bound, guarantee, or result; a claimed limitation of a method. Avoid claims that
are just restating the paper's title or abstract framing without technical content.
"""


def extract_claims(paper_title: str, paper_text: str) -> PaperClaims:
    user_prompt = f"PAPER TITLE: {paper_title}\n\nPAPER TEXT:\n{paper_text}"
    result = call_structured(SYSTEM_PROMPT, user_prompt, PaperClaims, temperature=0.2)
    result.paper_title = paper_title
    return result
