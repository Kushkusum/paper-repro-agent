import pytest

from agent.coder import _parse_generated_code, _strip_code_fence


def test_parses_basic_file_block():
    raw = """### FILE: run.py
print("hello")
### ENDFILE
### ENTRYPOINT: run.py
### RUN_COMMAND: python run.py
"""
    code = _parse_generated_code(raw)
    assert len(code.files) == 1
    assert code.files[0].filename == "run.py"
    assert 'print("hello")' in code.files[0].content
    assert code.entrypoint == "run.py"
    assert code.run_command == "python run.py"


def test_missing_endfile_marker_still_parses_via_next_file_header():
    # Some models drop the ### ENDFILE marker entirely and close with a bare fence instead.
    raw = """### FILE: run.py
```python
print("hi")
```
### ENTRYPOINT: run.py
### RUN_COMMAND: python run.py
"""
    code = _parse_generated_code(raw)
    assert code.files[0].content.strip() == 'print("hi")'


def test_multiple_files_split_correctly():
    raw = """### FILE: a.py
import b
### ENDFILE

### FILE: b.py
X = 1
### ENDFILE
### ENTRYPOINT: a.py
### RUN_COMMAND: python a.py
"""
    code = _parse_generated_code(raw)
    assert [f.filename for f in code.files] == ["a.py", "b.py"]
    assert code.files[0].content.strip() == "import b"
    assert code.files[1].content.strip() == "X = 1"


def test_no_file_blocks_raises():
    with pytest.raises(ValueError, match="No ### FILE blocks"):
        _parse_generated_code("just some prose, no delimiters at all")


def test_main_defined_but_never_called_is_rejected():
    raw = """### FILE: run.py
def main():
    print("RESULTS_JSON: {}")
### ENDFILE
### ENTRYPOINT: run.py
### RUN_COMMAND: python run.py
"""
    with pytest.raises(ValueError, match="never calls it"):
        _parse_generated_code(raw)


def test_main_called_via_dunder_guard_is_accepted():
    raw = """### FILE: run.py
def main():
    print("RESULTS_JSON: {}")

if __name__ == "__main__":
    main()
### ENDFILE
### ENTRYPOINT: run.py
### RUN_COMMAND: python run.py
"""
    code = _parse_generated_code(raw)
    assert "__main__" in code.files[0].content


def test_main_called_directly_is_accepted():
    raw = """### FILE: run.py
def main():
    print("RESULTS_JSON: {}")

main()
### ENDFILE
### ENTRYPOINT: run.py
### RUN_COMMAND: python run.py
"""
    code = _parse_generated_code(raw)  # should not raise
    assert code.files[0].filename == "run.py"


def test_strip_code_fence_removes_matched_pair():
    content = '```python\nx = 1\n```'
    assert _strip_code_fence(content) == "x = 1"


def test_strip_code_fence_noop_when_no_fence():
    content = "x = 1"
    assert _strip_code_fence(content) == "x = 1"
