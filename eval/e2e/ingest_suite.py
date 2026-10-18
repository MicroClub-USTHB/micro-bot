"""Ingest suite over the documents API (Req 12.3).

One probe document goes through upload, identical re-upload, new content under the same topic,
an expiry date and deletion; after each step retrieval must see exactly the current
version. Then every kind of bad upload must give its typed JSON error. The corpus version is
read from the in-process app; over HTTP only the listing is compared.
"""

import os
from pathlib import Path

from eval.e2e.cases import Case
from eval.e2e.contract import check, expect, json_body
from eval.e2e.score import CaseResult

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
TOPIC = "eval-ingest-probe"
QUESTION = "What is the eval probe room code?"


def _doc(code: str) -> bytes:
    return f"# Eval ingest probe\n\nThe eval probe room code is {code}.\n".encode()


async def _found(driver, needle: str) -> bool:
    case = Case(id="probe", question=QUESTION, lang="en", expect="any")
    if hasattr(driver, "retrieve"):
        retrieval, _ = await driver.retrieve(case)
        return any(needle in c.text for c in retrieval.chunks)
    body = (await driver.chat(case)).body
    return isinstance(body, dict) and any(needle in s["snippet"] for s in body.get("sources", []))


async def _version(driver) -> str | None:
    documents = getattr(driver, "documents", None)
    return await documents.corpus_version() if documents else None


async def ingest(driver) -> list[CaseResult]:
    admin = driver.headers(admin=True)

    async def upload(name: str, data: bytes, **form):
        files = {"file": (name, data)}
        fields = {"topic": TOPIC} | form
        return await driver.request(
            "POST", "/api/v1/documents", files=files, data=fields, headers=admin
        )

    results = []

    def step(id: str, response, status, *checks, code=None) -> None:
        result = expect(f"ingest-{id}", "ingest", response, status, code)
        result.checks.extend(checks)
        results.append(result)

    v0 = await _version(driver)
    first = await upload("probe.md", _doc("ZEBRA-4417"))
    doc_id = (json_body(first) or {}).get("doc_id")
    v1 = await _version(driver)
    step(
        "new",
        first,
        200,
        check("version_changed", v1 != v0 or v1 is None, "corpus_version unchanged"),
        check("retrieved", await _found(driver, "ZEBRA-4417"), "new text not found"),
    )

    again = await upload("probe.md", _doc("ZEBRA-4417"))
    step(
        "duplicate",
        again,
        200,
        check("same_doc_id", (json_body(again) or {}).get("doc_id") == doc_id, "doc_id changed"),
        check("version_kept", await _version(driver) == v1, "corpus_version changed"),
    )

    replaced = await upload("probe.md", _doc("ZEBRA-9001"))
    step(
        "replaced",
        replaced,
        200,
        check("same_doc_id", (json_body(replaced) or {}).get("doc_id") == doc_id, "new doc_id"),
        check("new_text", await _found(driver, "ZEBRA-9001"), "new text not found"),
        check("old_text_gone", not await _found(driver, "ZEBRA-4417"), "old text"),
    )

    expiring = await upload("probe.md", _doc("ZEBRA-9001"), expires_at="2099-12-31T00:00:00Z")
    listing = json_body(await driver.request("GET", "/api/v1/documents", headers=admin)) or []
    entry = next((d for d in listing if d.get("doc_id") == doc_id), {})
    step(
        "expires-at-kept",
        expiring,
        200,
        check("expires_at", entry.get("expires_at") is not None, "expires_at dropped"),
    )

    deleted = await driver.request("DELETE", f"/api/v1/documents/{doc_id}", headers=admin)
    listing = json_body(await driver.request("GET", "/api/v1/documents", headers=admin)) or []
    step(
        "delete",
        deleted,
        200,
        check("ok", json_body(deleted) == {"ok": True}, str(json_body(deleted))),
        check("unlisted", all(d.get("doc_id") != doc_id for d in listing), "still listed"),
        check("not_retrieved", not await _found(driver, "ZEBRA-9001"), "retrieved"),
    )
    again = await driver.request("DELETE", f"/api/v1/documents/{doc_id}", headers=admin)
    step(
        "delete-again",
        again,
        (200, 404),
        check("not_ok", json_body(again) != {"ok": True}, "deleting twice says ok"),
    )

    bad = [
        ("empty", "empty.md", b"", 422, "empty_document"),
        ("whitespace-only", "blank.txt", b"   \n\n  ", 422, "empty_document"),
        ("corrupt-pdf", "corrupt.pdf", (FIXTURES / "corrupt.pdf").read_bytes(), 422, "parse_error"),
        ("junk-named-pdf", "junk.pdf", os.urandom(4096), 415, "unsupported_file_type"),
        ("executable", "tool.exe", b"MZ\x90\x00" + os.urandom(2048), 415, "unsupported_file_type"),
        ("oversize", "big.txt", b"a " * (13 * 1024 * 1024), 413, "file_too_large"),
    ]
    for id, name, data, status, code in bad:
        step(id, await upload(name, data), status, code=code)

    for id, name in [
        ("scan-png", "notice.png"),
        ("docx", "doc.docx"),
        ("cp1256", "arabic_cp1256.txt"),
    ]:
        response = await upload(name, (FIXTURES / name).read_bytes())
        step(id, response, 200)
        created = (json_body(response) or {}).get("doc_id")
        if created:
            await driver.request("DELETE", f"/api/v1/documents/{created}", headers=admin)
    return results
