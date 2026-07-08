from __future__ import annotations

import networkx as nx

from .contradiction import classify_pair
from .schemas import ClaimRelation, PaperClaims


# Generic single-symbol math notation (Greek letters, single-letter variable names) recurs
# across completely unrelated papers with unrelated meanings -- e.g. "delta" is both UCB1's
# arm-mean-gap and Count-Min Sketch's failure probability. Substring matching on these produces
# confident-looking but meaningless cross-paper links, so they may only match exactly, never as
# a substring of a more specific term (e.g. "delta" must not link to "delta_i").
_GENERIC_SYMBOLS = {
    "delta",
    "epsilon",
    "alpha",
    "beta",
    "gamma",
    "theta",
    "lambda",
    "sigma",
    "mu",
    "n",
    "k",
    "t",
    "x",
    "y",
}


def _normalize_entity(entity: str) -> str:
    return entity.strip().lower()


def _entities_match(a: str, b: str) -> bool:
    a, b = _normalize_entity(a), _normalize_entity(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in _GENERIC_SYMBOLS or b in _GENERIC_SYMBOLS:
        return False  # generic symbols only ever match exactly, never as a substring
    # Prefix match (e.g. "UCB1" / "UCB1-TUNED", "UCB" / "UCB1") rather than substring-anywhere --
    # a real entity-resolution system would do better, but this is enough to catch the common
    # abbreviation-shares-a-prefix pattern without the false positives arbitrary substring
    # matching produces (e.g. "Delta_i" normalizes to "delta_i", which contains "a_i" -- an
    # unrelated Count-Min Sketch symbol -- purely by character coincidence, not prefix).
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 3 and longer.startswith(shorter)


def build_claim_graph(all_claims: list[PaperClaims]) -> nx.Graph:
    """Build a graph where each node is (paper_title, claim_index) and edges connect claims
    from DIFFERENT papers that share a named entity/concept."""
    graph = nx.Graph()
    nodes: list[tuple[str, int]] = []
    for pc in all_claims:
        for i in range(len(pc.claims)):
            node = (pc.paper_title, i)
            nodes.append(node)
            graph.add_node(node, claim=pc.claims[i], paper_title=pc.paper_title)

    for idx_a, node_a in enumerate(nodes):
        claim_a = graph.nodes[node_a]["claim"]
        for node_b in nodes[idx_a + 1 :]:
            if node_a[0] == node_b[0]:
                continue  # only cross-paper relations are interesting here
            claim_b = graph.nodes[node_b]["claim"]
            for ea in claim_a.entities:
                for eb in claim_b.entities:
                    if _entities_match(ea, eb):
                        graph.add_edge(node_a, node_b, shared_entity=ea)
                        break
                else:
                    continue
                break
    return graph


def score_relations(graph: nx.Graph, min_confidence: float = 0.6) -> list[ClaimRelation]:
    """Run NLI scoring on every entity-linked cross-paper claim pair in the graph, keeping only
    confident entailment/contradiction calls (neutral pairs aren't interesting to report)."""
    relations = []
    for node_a, node_b, data in graph.edges(data=True):
        claim_a = graph.nodes[node_a]["claim"]
        claim_b = graph.nodes[node_b]["claim"]
        label, confidence = classify_pair(claim_a.text, claim_b.text)
        if label == "neutral" or confidence < min_confidence:
            continue
        relations.append(
            ClaimRelation(
                paper_a=node_a[0],
                claim_a=claim_a.text,
                paper_b=node_b[0],
                claim_b=claim_b.text,
                shared_entity=data["shared_entity"],
                label=label,
                confidence=confidence,
            )
        )
    return relations
