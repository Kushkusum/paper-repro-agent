from __future__ import annotations

import json
from pathlib import Path

from agent.claim_graph import build_claim_graph, score_relations
from agent.claims import extract_claims
from agent.schemas import PaperClaims

PAPERS_DIR = Path(__file__).resolve().parent / "papers"
OUT_DIR = Path(__file__).resolve().parent / "runs" / "_literature_graph"

# (raw text filename, human title) -- kept explicit rather than guessed from the filename.
PAPERS = [
    ("thompson_sampling_raw.txt", "An Empirical Evaluation of Thompson Sampling"),
    ("ucb1_multiarmed_bandit_raw.txt", "Finite-time Analysis of the Multiarmed Bandit Problem"),
    ("agrawal_goyal_ts_bound_raw.txt", "Further Optimal Regret Bounds for Thompson Sampling"),
    ("precommitment_best_choice_raw.txt", "The pre-commitment best-choice problem: exact finite-n formulas"),
    ("count_min_sketch_raw.txt", "An Improved Data Stream Summary: The Count-Min Sketch and its Applications"),
    ("qwen_vl_medical_raw.txt", "Enhanced Qwen-VL 7B Model via Instruction Finetuning on Chinese Medical Dataset"),
]


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_claims: list[PaperClaims] = []

    for filename, title in PAPERS:
        text = (PAPERS_DIR / filename).read_text(encoding="utf-8")
        print(f"Extracting claims: {title}...")
        claims = extract_claims(title, text)
        print(f"  -> {len(claims.claims)} claim(s)")
        for c in claims.claims:
            print(f"     - {c.text}  [{', '.join(c.entities)}]")
        all_claims.append(claims)

    (OUT_DIR / "claims.json").write_text(
        json.dumps([pc.model_dump() for pc in all_claims], indent=2), encoding="utf-8"
    )

    print("\nBuilding entity-linked claim graph...")
    graph = build_claim_graph(all_claims)
    print(f"  -> {graph.number_of_nodes()} claim nodes, {graph.number_of_edges()} candidate cross-paper links")

    print("\nScoring candidate pairs with NLI model (first run downloads ~140MB)...")
    relations = score_relations(graph)
    print(f"  -> {len(relations)} confident entailment/contradiction relation(s) found\n")

    for r in relations:
        print(f"[{r.label.upper()}] ({r.confidence:.2f}) shared entity: {r.shared_entity!r}")
        print(f"  {r.paper_a}:\n    {r.claim_a}")
        print(f"  {r.paper_b}:\n    {r.claim_b}\n")

    (OUT_DIR / "relations.json").write_text(
        json.dumps([r.model_dump() for r in relations], indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUT_DIR / 'claims.json'} and {OUT_DIR / 'relations.json'}")


if __name__ == "__main__":
    run()
