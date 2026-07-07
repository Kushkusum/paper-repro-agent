from agent.sandbox import _extract_results


def test_extracts_valid_json():
    stdout = 'some log line\nRESULTS_JSON: {"a": 1.5, "b": 2.0}\n'
    assert _extract_results(stdout) == {"a": 1.5, "b": 2.0}


def test_extracts_from_last_matching_line_only():
    stdout = 'RESULTS_JSON: {"a": 1.0}\nmore output\nRESULTS_JSON: {"a": 2.0}\n'
    assert _extract_results(stdout) == {"a": 2.0}


def test_repairs_python_repr_with_numpy_scalars():
    # The exact real-world bug: print('RESULTS_JSON:', results) with np.float64 wrappers
    # and single-quoted keys instead of json.dumps(...).
    stdout = "RESULTS_JSON: {'a': np.float64(1.5), 'b': np.float64(2.0)}\n"
    assert _extract_results(stdout) == {"a": 1.5, "b": 2.0}


def test_no_results_line_returns_empty():
    assert _extract_results("no marker here at all\njust some text\n") == {}


def test_empty_stdout_returns_empty():
    assert _extract_results("") == {}


def test_garbage_after_marker_returns_empty():
    assert _extract_results("RESULTS_JSON: not json and not python either {{{\n") == {}
