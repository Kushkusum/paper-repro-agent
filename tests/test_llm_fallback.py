import httpx
import pytest

import agent.llm as llm


def _fake_rate_limit_error(model: str):
    import groq

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request, json={"error": {"message": "rate limited"}})
    return groq.RateLimitError(f"rate limited on {model}", response=response, body=None)


class _FakeCompletions:
    def __init__(self, fail_models: set[str], response_content: str):
        self.fail_models = fail_models
        self.response_content = response_content
        self.calls: list[str] = []

    def create(self, model: str, **kwargs):
        self.calls.append(model)
        if model in self.fail_models:
            raise _fake_rate_limit_error(model)

        class _Msg:
            content = self.response_content

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions):
        self.chat = _FakeChat(completions)


@pytest.fixture(autouse=True)
def _restore_fallback_list():
    original = llm.MODEL_FALLBACKS
    yield
    llm.MODEL_FALLBACKS = original


def test_falls_back_to_next_model_on_rate_limit(monkeypatch):
    llm.MODEL_FALLBACKS = ["model-a", "model-b", "model-c"]
    fake = _FakeCompletions(fail_models={"model-a"}, response_content="OK")
    monkeypatch.setattr(llm, "client", lambda: _FakeClient(fake))

    result = llm.call_text("sys", "user", model="model-a")

    assert result == "OK"
    assert fake.calls == ["model-a", "model-b"]  # stopped at the first success


def test_uses_preferred_model_first_when_it_works(monkeypatch):
    llm.MODEL_FALLBACKS = ["model-a", "model-b"]
    fake = _FakeCompletions(fail_models=set(), response_content="OK")
    monkeypatch.setattr(llm, "client", lambda: _FakeClient(fake))

    llm.call_text("sys", "user", model="model-a")

    assert fake.calls == ["model-a"]


def test_raises_when_all_models_in_chain_fail(monkeypatch):
    llm.MODEL_FALLBACKS = ["model-a", "model-b"]
    fake = _FakeCompletions(fail_models={"model-a", "model-b"}, response_content="unused")
    monkeypatch.setattr(llm, "client", lambda: _FakeClient(fake))

    with pytest.raises(RuntimeError, match="All models in fallback chain failed"):
        llm.call_text("sys", "user", model="model-a")


def test_preferred_model_not_duplicated_in_chain(monkeypatch):
    # model-a is both the preferred model and already in MODEL_FALLBACKS -- must not be tried twice.
    llm.MODEL_FALLBACKS = ["model-a", "model-b"]
    fake = _FakeCompletions(fail_models={"model-a"}, response_content="OK")
    monkeypatch.setattr(llm, "client", lambda: _FakeClient(fake))

    llm.call_text("sys", "user", model="model-a")

    assert fake.calls == ["model-a", "model-b"]
