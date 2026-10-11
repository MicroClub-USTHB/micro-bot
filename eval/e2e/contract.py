"""API contract suite (Req 7.2) and the probe helper shared by the system suites.

Each probe is one ``CaseResult`` whose checks compare the expected status, error code and
``llm_calls`` with what came back; error bodies must have the JSON error shape.
"""

from collections.abc import Sequence

import httpx

from eval.e2e.score import CaseResult, Check


def json_body(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return None


def check(name: str, ok: bool, reason: str) -> Check:
    return Check(name, "pass" if ok else "fail", "" if ok else reason)


def expect(
    id: str,
    suite: str,
    response: httpx.Response,
    status: int | Sequence[int],
    code: str | None = None,
    calls: int | None = None,
) -> CaseResult:
    statuses = (status,) if isinstance(status, int) else tuple(status)
    body = json_body(response)
    got = response.status_code
    checks = [check("status", got in statuses, f"got {got}, want {statuses}")]
    if got >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        shaped = isinstance(error, dict) and {"code", "message"} <= set(error)
        checks.append(check("error_shape", shaped, str(body)[:120]))
        if code is not None and shaped:
            checks.append(check("error_code", error["code"] == code, f"got {error['code']}"))
    if calls is not None and isinstance(body, dict) and "usage" in body:
        used = body["usage"]["llm_calls"]
        checks.append(check("llm_calls", used == calls, f"got {used}, want {calls}"))
    return CaseResult(id, suite, f"http {response.status_code}", checks)


async def contract(driver) -> list[CaseResult]:
    member, admin = driver.headers("member"), driver.headers(admin=True)
    chat = "/api/v1/chat"
    msg = {"message": "Quand ferment les inscriptions au hackathon ?"}
    probes = [
        ("chat-public", "POST", chat, {"json": msg}, 200, None),
        ("chat-member", "POST", chat, {"json": msg, "headers": member}, 200, None),
        (
            "chat-bad-token",
            "POST",
            chat,
            {"json": msg, "headers": {"Authorization": "Bearer nope"}},
            401,
            "unauthorized",
        ),
        (
            "chat-basic-scheme",
            "POST",
            chat,
            {"json": msg, "headers": {"Authorization": "Basic eA=="}},
            401,
            "unauthorized",
        ),
        ("chat-2000-chars", "POST", chat, {"json": {"message": "a" * 2000}}, 200, None),
        (
            "chat-2001-chars",
            "POST",
            chat,
            {"json": {"message": "a" * 2001}},
            422,
            "validation_error",
        ),
        ("chat-missing-message", "POST", chat, {"json": {}}, 422, None),
        (
            "chat-malformed-json",
            "POST",
            chat,
            {"content": b"{not json", "headers": {"Content-Type": "application/json"}},
            422,
            None,
        ),
        ("chat-empty-message", "POST", chat, {"json": {"message": ""}}, (200, 422), None),
        ("chat-whitespace-message", "POST", chat, {"json": {"message": "   "}}, (200, 422), None),
        ("docs-list-no-key", "GET", "/api/v1/documents", {}, 403, "forbidden"),
        (
            "docs-list-bad-key",
            "GET",
            "/api/v1/documents",
            {"headers": {"X-Admin-Key": "nope"}},
            403,
            "forbidden",
        ),
        (
            "docs-list-member-token",
            "GET",
            "/api/v1/documents",
            {"headers": member},
            403,
            "forbidden",
        ),
        ("docs-list-admin", "GET", "/api/v1/documents", {"headers": admin}, 200, None),
        ("docs-delete-no-key", "DELETE", "/api/v1/documents/x", {}, 403, "forbidden"),
        (
            "docs-upload-no-key",
            "POST",
            "/api/v1/documents",
            {"files": {"file": ("a.md", b"# A\n\ntext")}, "data": {"visibility": "public"}},
            403,
            "forbidden",
        ),
        (
            "docs-upload-bad-visibility",
            "POST",
            "/api/v1/documents",
            {
                "files": {"file": ("a.md", b"# A\n\ntext")},
                "data": {"visibility": "secret"},
                "headers": admin,
            },
            422,
            None,
        ),
        (
            "feedback-ok",
            "POST",
            "/api/v1/feedback",
            {"json": {"conversation_id": "c_x", "rating": "up"}},
            200,
            None,
        ),
        (
            "feedback-bad-rating",
            "POST",
            "/api/v1/feedback",
            {"json": {"conversation_id": "c_x", "rating": "meh"}},
            422,
            None,
        ),
        ("health", "GET", "/api/v1/health", {}, 200, None),
        ("unknown-route", "GET", "/api/v1/nope", {}, 404, "not_found"),
        ("wrong-method", "GET", chat, {}, 405, "method_not_allowed"),
    ]
    results = []
    for id, method, path, kwargs, status, code in probes:
        response = await driver.request(method, path, **kwargs)
        results.append(expect(f"contract-{id}", "contract", response, status, code))
    return results
