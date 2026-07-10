from agent.legitimacy_eval import build_eval_cases


def test_builds_eleven_cases_with_expected_split():
    cases = build_eval_cases()

    assert len(cases) == 11
    genuine = [c for c in cases if c.expected_genuine]
    cheating = [c for c in cases if not c.expected_genuine]
    assert len(genuine) == 4
    assert len(cheating) == 7


def test_case_names_are_unique():
    cases = build_eval_cases()
    names = [c.name for c in cases]
    assert len(names) == len(set(names))


def test_every_case_has_one_code_file_and_a_matching_result():
    for case in build_eval_cases():
        assert len(case.code.files) == 1
        assert case.code.entrypoint == case.code.files[0].filename
        metric_names = {m.name for m in case.spec.target_metrics}
        assert set(case.result.parsed_metrics.keys()) == metric_names


def test_no_case_leaks_its_own_label_via_source_text():
    tells = [
        "cheat",
        "fake",
        "hardcod",
        "shortcut",
        "fudge",
        "unused",
        "discard",
        "silently",
        "not actually",
        "looks like",
        "pass-through",
        "passthrough",
    ]
    for case in build_eval_cases():
        source = case.code.files[0].content.lower()
        for tell in tells:
            assert tell not in source, f"{case.name} leaks tell word {tell!r} in its own code"
