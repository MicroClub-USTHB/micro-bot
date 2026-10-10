import httpx
import pytest
from structlog.testing import capture_logs

from mcclub_rag.ingest.errors import EmptyDocument, FetchError, InvalidInput, UrlNotAllowed
from mcclub_rag.ingest.preprocess import preprocess_url
from tests.test_web import ARTICLE


async def resolve(host, port):
    return ["93.184.216.34"]


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _html(body: bytes):
    return lambda request: httpx.Response(200, content=body, headers={"content-type": "text/html"})


async def test_html_article(ocr_settings):
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/mars?ref=home"})
        return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})

    doc = await preprocess_url(
        "https://club.example/old",
        visibility="public",
        settings=ocr_settings,
        client=_client(handler),
        resolve=resolve,
    )
    assert doc.source_type == "web"
    assert doc.source == "https://club.example/mars?ref=home"
    assert doc.mime_type == "text/html"
    assert doc.language == "fr"
    assert "salle B12" in doc.text
    assert "Programme de mars" in doc.title
    assert doc.parser.startswith("trafilatura")


async def test_pdf_url_goes_through_file_parser(ocr_settings, fixture_bytes):
    pdf = fixture_bytes("text.pdf")

    def handler(request):
        return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

    doc = await preprocess_url(
        "https://club.example/files/handbook.pdf",
        visibility="members",
        settings=ocr_settings,
        client=_client(handler),
        resolve=resolve,
    )
    assert doc.source_type == "web"
    assert doc.mime_type == "application/pdf"
    assert doc.page_count == 4
    assert doc.parser.startswith("xberg")
    import hashlib

    assert doc.content_hash == hashlib.sha256(pdf).hexdigest()


async def test_plain_text_url_without_extension(ocr_settings):
    def handler(request):
        return httpx.Response(
            200,
            content=b"The club office is open every weekday morning.",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    doc = await preprocess_url(
        "https://club.example/hours",
        visibility="public",
        settings=ocr_settings,
        client=_client(handler),
        resolve=resolve,
    )
    assert doc.mime_type == "text/plain" and "weekday" in doc.text


async def test_html_hash_ignores_boilerplate_changes(ocr_settings):
    other = ARTICLE.replace(b"Accueil", b"Home page").replace(b"cookies", b"traceurs")
    docs = []
    for body in (ARTICLE, other):
        docs.append(
            await preprocess_url(
                "https://club.example/mars",
                visibility="public",
                settings=ocr_settings,
                client=_client(_html(body)),
                resolve=resolve,
            )
        )
    assert docs[0].content_hash == docs[1].content_hash


async def test_query_string_never_logged(ocr_settings):
    with capture_logs() as logs:
        await preprocess_url(
            "https://club.example/mars?token=s3cret#frag",
            visibility="public",
            settings=ocr_settings,
            client=_client(_html(ARTICLE)),
            resolve=resolve,
        )
    assert "s3cret" not in repr(logs)
    [event] = [e for e in logs if e["event"] == "document_preprocessed"]
    assert event["source"] == "https://club.example/mars"


async def test_errors_propagate_and_are_logged(ocr_settings):
    with capture_logs() as logs, pytest.raises(FetchError, match="HTTP 404"):
        await preprocess_url(
            "https://club.example/missing?key=abc",
            visibility="public",
            settings=ocr_settings,
            client=_client(lambda r: httpx.Response(404)),
            resolve=resolve,
        )
    [event] = [e for e in logs if e["event"] == "document_rejected"]
    assert event["error_code"] == "fetch_error"
    assert "abc" not in repr(logs)


async def test_internal_url_rejected(ocr_settings):
    with pytest.raises(UrlNotAllowed):
        await preprocess_url(
            "http://127.0.0.1:6333/collections", visibility="public", settings=ocr_settings
        )


async def test_empty_page(ocr_settings):
    page = b"<html><body><nav><a href='/'>Home</a></nav></body></html>"
    with pytest.raises(EmptyDocument):
        await preprocess_url(
            "https://club.example/",
            visibility="public",
            settings=ocr_settings,
            client=_client(_html(page)),
            resolve=resolve,
        )


@pytest.mark.parametrize(("url", "visibility"), [("", "public"), ("https://club.example", "x")])
async def test_invalid_input(ocr_settings, url, visibility):
    with pytest.raises(InvalidInput):
        await preprocess_url(url, visibility=visibility, settings=ocr_settings)
