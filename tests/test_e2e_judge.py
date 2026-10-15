"""Optional judge (Req 4): off by default, cached, capped, unparsable or failed is skip."""

from types import SimpleNamespace

from eval.e2e.judge import Judge, from_env


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_off_without_env_and_never_uses_the_app_key(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "app-key")
    assert from_env(10) is None


def test_verdicts_are_cached(tmp_path):
    client = FakeClient('{"supported": false, "unsupported_claims": ["room 9"]}')
    judge = Judge(client, "m", 5, cache=tmp_path / "c.jsonl")
    first = judge.check("Room 9 [d].", ["Room 12."])
    assert first.status == "fail" and first.reason == "room 9"
    again = Judge(client, "m", 5, cache=tmp_path / "c.jsonl").check("Room 9 [d].", ["Room 12."])
    assert again == first and client.calls == 1


def test_cap_and_unparsable(tmp_path):
    judge = Judge(FakeClient("looks fine to me"), "m", 1, cache=tmp_path / "c.jsonl")
    assert judge.check("a", ["p"]).reason == "unparsable verdict"
    assert judge.check("b", ["p"]).reason == "judge budget spent"


def test_failed_call_is_skip_and_not_cached(tmp_path):
    client = FakeClient('{"supported": true}')

    def boom(**kwargs):
        raise TimeoutError("slow")

    ok = client.chat.completions.create
    client.chat.completions.create = boom
    judge = Judge(client, "m", 5, cache=tmp_path / "c.jsonl")
    assert judge.check("a", ["p"]).reason == "judge error: TimeoutError"
    client.chat.completions.create = ok
    assert judge.check("a", ["p"]).status == "pass"
