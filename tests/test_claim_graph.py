import agent.claim_graph as claim_graph
from agent.claim_graph import _entities_match, build_claim_graph, score_relations
from agent.schemas import Claim, PaperClaims


def _paper(title: str, claims: list[tuple[str, list[str]]]) -> PaperClaims:
    return PaperClaims(
        paper_title=title,
        claims=[Claim(text=text, entities=entities, evidence_quote="q") for text, entities in claims],
    )


def test_entities_match_exact_case_insensitive():
    assert _entities_match("UCB1", "ucb1")
    assert _entities_match("  Thompson Sampling ", "thompson sampling")


def test_entities_match_shared_prefix():
    assert _entities_match("UCB1", "UCB")
    assert _entities_match("UCB1", "UCB1-TUNED")
    assert not _entities_match("UCB1", "a")  # too short to count


def test_entities_do_not_match_on_coincidental_mid_string_substring():
    # The real false positive found in the first end-to-end run: "Delta_i" normalizes to
    # "delta_i", which contains "a_i" (an unrelated Count-Min Sketch symbol) purely by
    # character coincidence in the middle of the string, not because they share a meaning
    # or even a prefix.
    assert not _entities_match("Delta_i", "a_i")


def test_generic_math_symbols_only_match_exactly():
    # UCB1's arm-mean-gap "Delta_i" is not the same concept as Count-Min Sketch's failure
    # probability "delta", even though one is a prefix-adjacent substring of the other.
    assert not _entities_match("Delta_i", "delta")
    assert not _entities_match("epsilon_n-GREEDY", "epsilon")
    assert _entities_match("delta", "delta")  # exact match on a generic symbol is still fine


def test_entities_no_match_unrelated():
    assert not _entities_match("UCB1", "Count-Min Sketch")


def test_graph_links_claims_sharing_an_entity_across_papers():
    paper_a = _paper("Paper A", [("UCB1 has logarithmic regret.", ["UCB1"])])
    paper_b = _paper("Paper B", [("UCB1's bound is 8*sum(1/delta)*ln(n).", ["UCB1"])])
    graph = build_claim_graph([paper_a, paper_b])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


def test_graph_does_not_link_claims_within_the_same_paper():
    paper_a = _paper(
        "Paper A",
        [
            ("UCB1 has logarithmic regret.", ["UCB1"]),
            ("UCB1 is deterministic.", ["UCB1"]),
        ],
    )
    graph = build_claim_graph([paper_a])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 0  # same-paper claims never link, regardless of shared entities


def test_graph_does_not_link_unrelated_entities():
    paper_a = _paper("Paper A", [("UCB1 has logarithmic regret.", ["UCB1"])])
    paper_b = _paper("Paper B", [("The sketch never underestimates.", ["Count-Min Sketch"])])
    graph = build_claim_graph([paper_a, paper_b])

    assert graph.number_of_edges() == 0


def test_score_relations_filters_out_neutral_and_low_confidence(monkeypatch):
    paper_a = _paper("Paper A", [("Claim one.", ["X"])])
    paper_b = _paper("Paper B", [("Claim two.", ["X"])])
    graph = build_claim_graph([paper_a, paper_b])
    assert graph.number_of_edges() == 1

    monkeypatch.setattr(claim_graph, "classify_pair", lambda a, b: ("neutral", 0.99))
    assert score_relations(graph) == []

    monkeypatch.setattr(claim_graph, "classify_pair", lambda a, b: ("contradiction", 0.4))
    assert score_relations(graph, min_confidence=0.6) == []

    monkeypatch.setattr(claim_graph, "classify_pair", lambda a, b: ("contradiction", 0.9))
    relations = score_relations(graph, min_confidence=0.6)
    assert len(relations) == 1
    assert relations[0].label == "contradiction"
    assert relations[0].confidence == 0.9
