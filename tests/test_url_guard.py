import socket

import pytest

from mcclub_rag.ingest.errors import FetchError, UrlNotAllowed
from mcclub_rag.ingest.url_guard import check_url


def resolver(*addresses):
    calls = []

    async def resolve(host, port):
        calls.append((host, port))
        return list(addresses)

    resolve.calls = calls
    return resolve


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.1",  # carrier-grade NAT
        "0.0.0.0",
        "224.0.0.1",  # multicast
        "::1",
        "fc00::1",
        "fe80::1",
        "fe80::1%eth0",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:10.0.0.1",
    ],
)
async def test_internal_addresses_rejected(address):
    with pytest.raises(UrlNotAllowed, match="non-public"):
        await check_url("https://club.example/page", resolve=resolver(address))


async def test_any_internal_address_among_several_rejects():
    with pytest.raises(UrlNotAllowed):
        await check_url("https://club.example/", resolve=resolver("93.184.216.34", "10.0.0.1"))


async def test_public_address_passes_and_uses_default_port():
    resolve = resolver("93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")
    await check_url("https://club.example/page?x=1", resolve=resolve)
    assert resolve.calls == [("club.example", 443)]


async def test_explicit_port_passed_to_resolver():
    resolve = resolver("93.184.216.34")
    await check_url("http://club.example:8080/", resolve=resolve)
    assert resolve.calls == [("club.example", 8080)]


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("file:///etc/passwd", "scheme"),
        ("ftp://club.example/file", "scheme"),
        ("gopher://club.example/", "scheme"),
        ("club.example/page", "scheme"),
        ("http://user:pw@club.example/", "credentials"),
        ("http://user@club.example/", "credentials"),
        ("http:///nohost", "host"),
        ("http://club.example:99999/", "port"),
    ],
)
async def test_malformed_or_forbidden_urls(url, reason):
    with pytest.raises(UrlNotAllowed, match=reason):
        await check_url(url, resolve=resolver("93.184.216.34"))


async def test_unresolvable_host_is_fetch_error():
    async def failing(host, port):
        raise socket.gaierror("Name or service not known")

    with pytest.raises(FetchError, match="resolve") as exc:
        await check_url("https://does-not-exist.example/", resolve=failing)
    assert not isinstance(exc.value, UrlNotAllowed)


async def test_default_resolver_blocks_literal_loopback():
    # literal IPs resolve without any network access
    with pytest.raises(UrlNotAllowed):
        await check_url("http://127.0.0.1:8000/admin")
    with pytest.raises(UrlNotAllowed):
        await check_url("http://[::1]/")
