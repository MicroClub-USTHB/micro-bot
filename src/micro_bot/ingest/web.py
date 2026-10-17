"""Web page fetching (httpx) and main-content extraction (trafilatura), Req 4.

Redirects are followed manually so every hop goes through ``check_url``. The body is
streamed and capped; ``aiter_bytes`` decompresses, so the cap also bounds gzip bombs.
"""

import asyncio
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx
import lxml.etree
import lxml.html
import trafilatura

from micro_bot.ingest.errors import EmptyDocument, FetchError
from micro_bot.ingest.models import RawExtraction
from micro_bot.ingest.settings import IngestSettings
from micro_bot.ingest.url_guard import Resolver, check_url, default_resolver

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
# trafilatura falls back to whatever text it finds (e.g. a lone "Home" link) on pages with no
# article; fewer letters than this cannot answer anything.
_MIN_CONTENT_LETTERS = 20


@dataclass(frozen=True)
class FetchedResource:
    final_url: str
    content_type: str  # lowercased, parameters stripped ("text/html")
    data: bytes


def _too_large(settings: IngestSettings) -> FetchError:
    return FetchError(f"response too large (limit {settings.web_max_bytes} bytes)")


async def _fetch(
    url: str, settings: IngestSettings, http: httpx.AsyncClient, resolve: Resolver
) -> FetchedResource:
    headers = {"User-Agent": settings.web_user_agent}
    current = url
    for _ in range(settings.web_max_redirects + 1):
        await check_url(current, resolve)
        try:
            async with http.stream("GET", current, headers=headers, follow_redirects=False) as resp:
                if resp.status_code in _REDIRECT_STATUSES:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError(
                            f"HTTP {resp.status_code} redirect without Location header"
                        )
                    current = str(resp.url.join(location))
                    continue
                if not resp.is_success:
                    raise FetchError(f"HTTP {resp.status_code}")
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > settings.web_max_bytes:
                    raise _too_large(settings)
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > settings.web_max_bytes:
                        raise _too_large(settings)
                content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                return FetchedResource(str(resp.url), content_type, bytes(body))
        except httpx.TimeoutException as exc:
            raise FetchError("timeout") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"request failed: {type(exc).__name__}") from exc
    raise FetchError(f"too many redirects (limit {settings.web_max_redirects})")


async def fetch(
    url: str,
    settings: IngestSettings,
    *,
    client: httpx.AsyncClient | None = None,
    resolve: Resolver = default_resolver,
) -> FetchedResource:
    http = client or httpx.AsyncClient(timeout=settings.web_timeout_s)
    try:
        # httpx timeouts are per operation; this bounds a slow-drip server overall.
        async with asyncio.timeout(settings.web_timeout_s * 2):
            return await _fetch(url, settings, http, resolve)
    except TimeoutError as exc:
        raise FetchError("timeout") from exc
    finally:
        if client is None:
            await http.aclose()


# Main-content containers in which links are rewritten as "text (url)".
_CONTENT_LINKS = "//article//a[@href] | //main//a[@href] | //*[@role='main']//a[@href]"
_HEADING_LINKS = "//h1//a | //h2//a | //h3//a | //h4//a | //h5//a | //h6//a"
_URL_SCHEMES = ("http://", "https://", "mailto:")

# Share buttons (eval-fixes Req 15): host suffix -> path prefixes ("" = any path).
_SHARE_URLS = {
    "facebook.com": ("/sharer", "/share", "/dialog/share"),
    "twitter.com": ("/intent", "/share"),
    "x.com": ("/intent", "/share"),
    "linkedin.com": ("/share", "/cws/share"),
    "wa.me": ("",),
    "whatsapp.com": ("/send",),
    "t.me": ("/share",),
    "telegram.me": ("/share",),
    "reddit.com": ("/submit",),
    "pinterest.com": ("/pin/create",),
}
_SHARE_MARKERS = ("share", "partag", "مشاركة")  # in a link's class or aria-label
_SHARE_WORDS = frozenset(
    "share sharing partager partagez partage tweet tweeter print imprimer email e mail courriel "
    "facebook twitter x linkedin whatsapp telegram reddit pinterest on sur via this ce cette page "
    "article مشاركة شارك".split()
)
_SHARE_STOP = {"article", "main", "body", "html"}  # never climb out of the content
_WORDS = re.compile(r"\w+")


def _is_share_link(link: lxml.html.HtmlElement) -> bool:
    href = link.get("href", "").strip()
    if href.startswith("mailto:?"):  # no recipient: "send this page by email"
        return True
    marks = f"{link.get('class', '')} {link.get('aria-label', '')}".casefold()
    if any(marker in marks for marker in _SHARE_MARKERS):
        return True
    parts = urlsplit(href)
    host = (parts.hostname or "").removeprefix("www.")
    for suffix, paths in _SHARE_URLS.items():
        if host == suffix or host.endswith("." + suffix):
            return any(parts.path.startswith(prefix) for prefix in paths)
    return False


def _share_only(element: lxml.html.HtmlElement) -> bool:
    """Every link is a share link and the rest of the text is share words ("Partager :")."""
    links = element.xpath(".//a")
    if not links or not all(_is_share_link(a) for a in links):
        return False
    text = element.text_content()
    for link in links:
        text = text.replace(link.text_content(), " ", 1)
    return all(word in _SHARE_WORDS for word in _WORDS.findall(text.casefold()))


def _drop_share_blocks(tree: lxml.html.HtmlElement) -> None:
    """Remove share bars: the largest share-only block around each share link.

    A share link inside a sentence ("voir aussi Twitter") is kept: its paragraph has other text.
    Without this, a page holding only a title and a share bar passed the empty check.
    """
    for link in tree.xpath("//a[@href]"):
        if not _is_share_link(link) or link.getparent() is None:
            continue
        block = None
        node = link.getparent()
        while node is not None and node.tag not in _SHARE_STOP and _share_only(node):
            block, node = node, node.getparent()
        if block is not None and block.getparent() is not None:
            block.getparent().remove(block)


def _flatten_tabs(tree: lxml.html.HtmlElement) -> None:
    """Turn tab widgets (npm / pnpm / yarn ...) into a list of "label: content" items.

    trafilatura drops role="tabpanel" blocks, which is where docs sites put install commands.
    """
    for tablist in tree.xpath("//*[@role='tablist']"):
        parent = tablist.getparent()
        if parent is None:
            continue
        labels = [tab.text_content().strip() for tab in tablist.xpath(".//*[@role='tab']")]
        panels = parent.xpath("./*[@role='tabpanel']")
        items = lxml.html.Element("ul")
        for index, panel in enumerate(panels):
            content = " ".join(panel.text_content().split())
            if not content:
                continue
            label = labels[index] if index < len(labels) and labels[index] else ""
            item = lxml.html.Element("li")
            item.text = f"{label}: {content}" if label else content
            items.append(item)
        for panel in panels:
            parent.remove(panel)
        tablist.addprevious(items)
        parent.remove(tablist)


def _inline_links(tree: lxml.html.HtmlElement, base_url: str) -> None:
    """Inside the main content, replace links with "text (absolute url)".

    Keeps URLs in the text, and stops trafilatura discarding link-only lists
    ("GitHub Repository", "Official Website") as navigation. Navigation outside
    <article>/<main> is left alone so it is still recognised as boilerplate.
    """
    for link in tree.xpath(_CONTENT_LINKS):
        href = urljoin(base_url, link.get("href", "").strip())
        label = " ".join(link.text_content().split())
        target = href.removeprefix("mailto:")
        if href.startswith(_URL_SCHEMES) and label and target not in label:
            link.tail = f" ({target}){link.tail or ''}"
        link.drop_tag()


def prepare_html(data: bytes, base_url: str) -> lxml.html.HtmlElement:
    """Fix the HTML patterns trafilatura handles badly, before extraction."""
    try:
        tree = lxml.html.fromstring(data)
    except (lxml.etree.ParserError, ValueError) as exc:
        raise EmptyDocument("page has no parsable HTML") from exc
    _drop_share_blocks(tree)  # first: share buttons with "#" hrefs lose their link below
    # Heading permalinks ("<h2><a href=#x>Title</a></h2>") make trafilatura treat the
    # heading as a link; same-page anchors carry no information.
    for link in tree.xpath(_HEADING_LINKS + " | //a[starts-with(@href, '#')]"):
        link.drop_tag()
    _flatten_tabs(tree)
    _inline_links(tree, base_url)
    return tree


def _body_letters(markdown: str) -> int:
    """Letters outside headings: a page that is only its title has no content."""
    lines = (line for line in markdown.splitlines() if not line.lstrip().startswith("#"))
    return sum(ch.isalpha() for line in lines for ch in line)


def extract_html(data: bytes, final_url: str) -> RawExtraction:
    """Main content as Markdown (headings and tables kept). Blocking: run in a thread."""
    markdown = trafilatura.extract(
        prepare_html(data, final_url),
        url=final_url,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        include_images=False,
        include_links=False,  # URLs are already inlined by prepare_html
        # Strict mode keeps nav/cookie-only pages out. It used to drop permalinked headings
        # too; prepare_html unwraps those links first, so headings now survive.
        favor_precision=True,
    )
    if not markdown or _body_letters(markdown) < _MIN_CONTENT_LETTERS:
        raise EmptyDocument("no main content found on the page")
    metadata = trafilatura.extract_metadata(data)
    title = (metadata.title or "").strip() if metadata else ""
    return RawExtraction(
        kind="html",
        mime_type="text/html",
        parser=f"trafilatura {trafilatura.__version__}",
        markdown=markdown,
        metadata_title=title or None,
    )
