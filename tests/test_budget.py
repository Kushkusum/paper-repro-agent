import agent.llm as llm


def test_reset_budget_clears_log():
    llm._call_log.append({"model": "m", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "wall_time_sec": 0.1})
    llm.reset_budget()
    summary = llm.get_budget_summary()
    assert summary.total_calls == 0
    assert summary.total_tokens == 0


def test_budget_summary_aggregates_across_calls():
    llm.reset_budget()
    llm._call_log.append(
        {"model": "model-a", "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "wall_time_sec": 0.5}
    )
    llm._call_log.append(
        {"model": "model-a", "prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60, "wall_time_sec": 0.3}
    )
    llm._call_log.append(
        {"model": "model-b", "prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240, "wall_time_sec": 1.0}
    )

    summary = llm.get_budget_summary()

    assert summary.total_calls == 3
    assert summary.total_prompt_tokens == 350
    assert summary.total_completion_tokens == 70
    assert summary.total_tokens == 420
    assert summary.total_wall_time_sec == 1.8
    assert summary.calls_by_model == {"model-a": 2, "model-b": 1}
    assert summary.tokens_by_model == {"model-a": 180, "model-b": 240}


def test_create_completion_records_usage(monkeypatch):
    llm.reset_budget()
    monkeypatch.setattr(llm, "MODEL_FALLBACKS", ["model-a"])

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _Msg:
        content = "OK"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _FakeCompletions:
        def create(self, model, **kwargs):
            return _Resp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(llm, "client", lambda: _FakeClient())

    llm.call_text("sys", "user", model="model-a")

    summary = llm.get_budget_summary()
    assert summary.total_calls == 1
    assert summary.total_tokens == 15
    assert summary.calls_by_model == {"model-a": 1}
