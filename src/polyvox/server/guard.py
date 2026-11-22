"""Requests from other web pages are refused.

The API has no login: it trusts whoever can reach it. Two ways a page on another site could
still use it through the user's browser: a form or ``fetch(..., {mode: "no-cors"})`` posted to
localhost (the browser sends it, it just can't read the answer), and DNS rebinding (the page's
own host name re-pointed at 127.0.0.1, so it can even read the answers). The first carries the
other site's ``Origin``, the second its host name in ``Host``.
"""

from __future__ import annotations

import ipaddress
import json
from urllib.parse import urlsplit

# Host names that always mean this machine; IP addresses are always fine (rebinding needs a name).
LOCAL_HOSTS = {"localhost"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def host_name(host: str) -> str:
    """'name:port' / '[::1]:port' / 'name' -> the name, lower case."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:].partition("]")[0]
    return host.rpartition(":")[0] if host.count(":") == 1 else host


def allowed_host(host: str, extra: tuple[str, ...] = ()) -> bool:
    name = host_name(host)
    if not name:
        return False
    if name in LOCAL_HOSTS or name.endswith(".localhost") or name in extra:
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def same_origin(origin: str, host: str) -> bool:
    """``origin`` (scheme://host[:port]) names the server the request went to."""
    parts = urlsplit(origin)
    return parts.scheme in ("http", "https") and parts.netloc.lower() == host.strip().lower()


class RequestGuard:
    """ASGI middleware: 403 for a Host this server doesn't answer to, and for a changing
    request (not GET / HEAD / OPTIONS) sent by a page of another origin."""

    def __init__(self, app, allowed_hosts: tuple[str, ...] = ()) -> None:
        self.app = app
        self.allowed_hosts = tuple(h.lower() for h in allowed_hosts)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            host = headers.get("host", "")
            origin = headers.get("origin")
            if not allowed_host(host, self.allowed_hosts):
                return await _refuse(send, f"Host '{host}' isn't this server's (add it to POLYVOX_ALLOWED_HOSTS).")
            if scope["method"] not in SAFE_METHODS and origin is not None and not same_origin(origin, host):
                return await _refuse(send, "Requests from other web pages are refused.")
        await self.app(scope, receive, send)


async def _refuse(send, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 403,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
