"""SSRF protection for admin-submitted URLs (Req 4.5).

Only http(s), no credentials, and every address the host resolves to must be public.
Called for the initial URL and again for every redirect hop.

Known residual risk: DNS rebinding between this check and httpx's own resolution (TOCTOU).
Closing it needs a transport that connects to the vetted IP; recorded in EDGE_CASES.md.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from mcclub_rag.ingest.errors import FetchError, UrlNotAllowed

Resolver = Callable[[str, int], Awaitable[list[str]]]

_DEFAULT_PORTS = {"http": 80, "https": 443}


async def default_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def check_url(url: str, resolve: Resolver = default_resolver) -> None:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise UrlNotAllowed(f"URL scheme not allowed: {scheme or 'missing'} (use http or https)")
    if parts.username is not None or parts.password is not None:
        raise UrlNotAllowed("credentials in URLs are not allowed")
    host = parts.hostname
    if not host:
        raise UrlNotAllowed("URL has no host")
    try:
        port = parts.port or _DEFAULT_PORTS[scheme]
    except ValueError as exc:
        raise UrlNotAllowed("URL has an invalid port") from exc

    try:
        addresses = await resolve(host, port)
    except OSError as exc:  # socket.gaierror is an OSError
        raise FetchError(f"cannot resolve host {host}") from exc
    if not addresses:
        raise FetchError(f"cannot resolve host {host}")
    if not all(_is_public(address) for address in addresses):
        raise UrlNotAllowed(f"host {host} resolves to a non-public address")
