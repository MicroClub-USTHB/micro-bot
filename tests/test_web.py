from pathlib import Path

import httpx
import pytest

from mcclub_rag.ingest.errors import EmptyDocument, FetchError, UrlNotAllowed
from mcclub_rag.ingest.preprocess import preprocess_url
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.ingest.web import extract_html, fetch, prepare_html

PUBLIC = {
    "club.example": "93.184.216.34",
    "cdn.example": "93.184.216.35",
    "evil.example": "10.0.0.1",
}


async def fake_resolve(host, port):
    return [PUBLIC.get(host, host)]


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def settings():
    return IngestSettings(web_max_bytes=1000, web_max_redirects=5)


async def test_fetch_html(settings):
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(
            200, html="<p>hi</p>", headers={"content-type": "text/html; charset=utf-8"}
        )

    res = await fetch(
        "https://club.example/a", settings, client=_client(handler), resolve=fake_resolve
    )
    assert res.final_url == "https://club.example/a"
    assert res.content_type == "text/html"
    assert res.data == b"<p>hi</p>"
    assert seen["ua"] == settings.web_user_agent


async def test_redirect_sets_final_url_and_rechecks_each_hop(settings):
    checked = []

    async def resolve(host, port):
        checked.append(host)
        return await fake_resolve(host, port)

    def handler(request):
        if request.url.host == "club.example":
            return httpx.Response(301, headers={"location": "https://cdn.example/final"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    res = await fetch(
        "https://club.example/start", settings, client=_client(handler), resolve=resolve
    )
    assert res.final_url == "https://cdn.example/final"
    assert checked == ["club.example", "cdn.example"]


async def test_relative_redirect_is_resolved(settings):
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"})
        return httpx.Response(200, text="ok")

    res = await fetch(
        "https://club.example/old", settings, client=_client(handler), resolve=fake_resolve
    )
    assert res.final_url == "https://club.example/new"


@pytest.mark.parametrize(
    "target", ["http://127.0.0.1/admin", "https://evil.example/", "file:///etc/passwd"]
)
async def test_redirect_to_internal_address_blocked(settings, target):
    def handler(request):
        return httpx.Response(302, headers={"location": target})

    with pytest.raises(UrlNotAllowed):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_too_many_redirects(settings):
    def handler(request):
        n = int(request.url.path.strip("/") or 0)
        return httpx.Response(302, headers={"location": f"/{n + 1}"})

    with pytest.raises(FetchError, match="too many redirects"):
        await fetch(
            "https://club.example/0", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_redirect_without_location(settings):
    with pytest.raises(FetchError, match="Location"):
        await fetch(
            "https://club.example/",
            settings,
            client=_client(lambda r: httpx.Response(302)),
            resolve=fake_resolve,
        )


@pytest.mark.parametrize("status", [404, 500, 403])
async def test_http_error_status(settings, status):
    with pytest.raises(FetchError, match=f"HTTP {status}"):
        await fetch(
            "https://club.example/",
            settings,
            client=_client(lambda r: httpx.Response(status)),
            resolve=fake_resolve,
        )


async def test_timeout(settings):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(FetchError, match="timeout"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_connection_error(settings):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(FetchError, match="request failed"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_declared_length_over_cap_rejected_before_body(settings):
    def handler(request):
        return httpx.Response(200, headers={"content-length": "5000"}, content=b"x")

    with pytest.raises(FetchError, match="too large"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_streamed_body_over_cap_aborted(settings):
    async def chunks():
        for _ in range(50):
            yield b"x" * 100

    def handler(request):
        return httpx.Response(200, content=chunks())  # chunked, no Content-Length

    with pytest.raises(FetchError, match="too large"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_initial_url_checked(settings):
    with pytest.raises(UrlNotAllowed):
        await fetch(
            "http://127.0.0.1/",
            settings,
            client=_client(lambda r: httpx.Response(200)),
            resolve=fake_resolve,
        )


# --- extract_html ------------------------------------------------------------------

ARTICLE = """<html lang="fr"><head><title>Programme de mars - Club MC</title></head><body>
<nav><a href="/">Accueil</a> | <a href="/contact">Contact</a> | <a href="/login">Connexion</a></nav>
<div class="cookie-banner">Nous utilisons des cookies pour améliorer votre expérience.</div>
<article>
<h1>Programme de mars</h1>
<p>La réunion mensuelle du club aura lieu le lundi 3 mars à dix-huit heures dans la salle B12.
Tous les membres sont invités à y participer et à présenter leur carte d'adhérent.</p>
<h2>Tarifs</h2>
<p>Les cotisations restent inchangées cette année pour tous les membres du club.</p>
<table><tr><th>Type</th><th>Prix</th></tr><tr><td>Membre</td><td>50 DT</td></tr>
<tr><td>Étudiant</td><td>30 DT</td></tr></table>
</article>
<footer>© 2026 Club MC - Tous droits réservés - Mentions légales</footer>
</body></html>""".encode()


def test_extract_html_keeps_article_drops_boilerplate():
    raw = extract_html(ARTICLE, "https://club.example/mars")
    assert raw.kind == "html"
    assert raw.parser.startswith("trafilatura ")
    assert "# Programme de mars" in raw.markdown
    assert "## Tarifs" in raw.markdown
    assert "salle B12" in raw.markdown
    assert "| Membre | 50 DT |" in raw.markdown.replace("| \n", "|\n")
    for boilerplate in ["Accueil", "Connexion", "cookies", "Tous droits réservés"]:
        assert boilerplate not in raw.markdown
    assert raw.metadata_title and "Programme de mars" in raw.metadata_title


@pytest.mark.parametrize(
    "page",
    [
        b"<html><body><nav><a href='/'>Home</a> | <a href='/c'>Contact</a></nav>"
        b"<div class='cookie'>We use cookies.</div>"
        b"<footer>(c) 2026 Club MC. All rights reserved.</footer></body></html>",
        # trafilatura's fallback returns just "Home" here
        b"<html><body><nav><a href='/'>Home</a></nav><footer>(c) 2026</footer></body></html>",
        b"<html><body></body></html>",
    ],
)
def test_extract_html_without_main_content_is_empty(page):
    with pytest.raises(EmptyDocument):
        extract_html(page, "https://club.example/empty")


# --- prepare_html: patterns trafilatura handles badly ---------------------------

MCOLI_URL = "https://mcoli-ui.microclub.info/docs/introduction"
MCOLI_HTML = (Path(__file__).parent / "fixtures" / "web" / "mcoli_introduction.html").read_bytes()


def _md(html: bytes, url: str = "https://club.example/page") -> str:
    return extract_html(html, url).markdown


def test_mcoli_docs_page_regression():
    """Real docs page (Next.js + Fumadocs). Before the fix: no h2, no install commands."""
    raw = extract_html(MCOLI_HTML, MCOLI_URL)
    md = raw.markdown
    assert raw.metadata_title and "Introduction" in raw.metadata_title
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Own Your Code, Master Your UI",
        "## Get Started",
        "## Why mcoli-ui?",
        "## Contributing",
        "## Connect With Us",
    ]
    for line in [
        "- npm: npx mcoli-ui@latest init && npx mcoli-ui@latest add mc-button",
        "- pnpm: pnpm dlx mcoli-ui@latest init && pnpm dlx mcoli-ui@latest add mc-button",
        "- yarn: yarn dlx mcoli-ui@latest init && yarn dlx mcoli-ui@latest add mc-button",
        "- bun: bunx mcoli-ui@latest init && bunx mcoli-ui@latest add mc-button",
        "- GitHub Repository (https://github.com/MicroClub-USTHB/mcoli-ui)",
        "- Official Website (https://microclub.info)",
    ]:
        assert line in md
    assert "New to mcoli-ui?" in md  # callout
    assert "Installation (https://mcoli-ui.microclub.info/docs/installation)" in md  # relative link
    for boilerplate in ["Search docs", "Toggle Color Palette", "On this page", "Theming"]:
        assert boilerplate not in md


async def test_mcoli_docs_page_sections_end_to_end(ocr_settings):
    async def resolve(host, port):
        return ["93.184.216.34"]

    def handler(request):
        return httpx.Response(200, content=MCOLI_HTML, headers={"content-type": "text/html"})

    doc = await preprocess_url(
        MCOLI_URL,
        visibility="public",
        settings=ocr_settings,
        client=_client(handler),
        resolve=resolve,
    )
    assert [s.heading for s in doc.sections] == [
        "Introduction",
        "Own Your Code, Master Your UI",
        "Get Started",
        "Why mcoli-ui?",
        "Contributing",
        "Connect With Us",
    ]
    assert doc.language == "en"
    get_started = doc.sections[2].text
    assert "npm: npx mcoli-ui@latest init" in get_started


PAGE = b"""<html><head><title>T</title></head><body>
<nav><a href="/">Home</a> <a href="/about">About the club</a></nav>
<article>
<h1><a href="#top">Club guide</a></h1>
<p>The club meets weekly in room B12 and welcomes every student who wants to learn.</p>
<h2 id="links"><a href="#links">Useful links</a></h2>
<p>Read the <a href="/rules">club rules</a>, write to
<a href="mailto:office@club.example">the office</a>,
or open <a href="https://club.example/docs">https://club.example/docs</a> directly.</p>
<div><div role="tablist"><button role="tab">Linux</button><button role="tab">Windows</button></div>
<div role="tabpanel">sudo apt install club-tool</div>
<div role="tabpanel">winget install club-tool</div></div>
<p>All members can borrow books for two weeks and renew them once at the desk.</p>
</article></body></html>"""


def test_heading_permalinks_unwrapped():
    md = _md(PAGE)
    assert "# Club guide" in md
    assert "## Useful links" in md
    assert "#links" not in md and "#top" not in md


def test_content_links_inlined_with_absolute_urls():
    md = _md(PAGE)
    assert "club rules (https://club.example/rules)" in md
    assert "the office (office@club.example)" in md
    # label already equal to the URL: not duplicated
    assert "https://club.example/docs (https://club.example/docs)" not in md
    assert "https://club.example/docs" in md


def test_navigation_links_untouched_and_dropped():
    md = _md(PAGE)
    assert "About the club" not in md
    assert "(https://club.example/about)" not in md


def test_tabs_become_labelled_items():
    md = _md(PAGE)
    assert "- Linux: sudo apt install club-tool" in md
    assert "- Windows: winget install club-tool" in md


def test_tab_panels_without_labels():
    html = PAGE.replace(
        b'<button role="tab">Linux</button><button role="tab">Windows</button>', b""
    )
    md = _md(html)
    assert "- sudo apt install club-tool" in md


def test_prepare_html_leaves_plain_pages_alone():
    plain = b"<html><body><p>Just a paragraph with <b>bold</b> text.</p></body></html>"
    tree = prepare_html(plain, "https://club.example/")
    assert tree.text_content() == "Just a paragraph with bold text."


def test_prepare_html_unparsable_is_empty():
    with pytest.raises(EmptyDocument):
        prepare_html(b"", "https://club.example/")
