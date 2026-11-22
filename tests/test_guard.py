from fastapi import FastAPI
from fastapi.testclient import TestClient

from polyvox.server.guard import RequestGuard, allowed_host, host_name, same_origin


def client(allowed_hosts=()) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestGuard, allowed_hosts=allowed_hosts)

    @app.get("/api/thing")
    def get_thing():
        return {"ok": True}

    @app.post("/api/thing")
    def post_thing():
        return {"ok": True}

    return TestClient(app, base_url="http://127.0.0.1:8808")


def test_host_names():
    assert host_name("LocalHost:8808") == "localhost"
    assert host_name("[::1]:8808") == "::1"
    assert host_name("::1") == "::1"
    assert host_name("example.com") == "example.com"


def test_this_machines_names_and_addresses_are_allowed_other_names_not():
    for host in ("localhost:8808", "127.0.0.1:8808", "[::1]:8808", "192.168.1.20:8808", "app.localhost:5173"):
        assert allowed_host(host), host
    for host in ("attacker.example:8808", "", "localhost.attacker.example"):
        assert not allowed_host(host), host
    assert allowed_host("studio-pc:8808", ("studio-pc",))


def test_same_origin():
    assert same_origin("http://127.0.0.1:8808", "127.0.0.1:8808")
    assert same_origin("http://localhost:5173", "localhost:5173")  # the dev server's proxy keeps Host
    assert not same_origin("https://attacker.example", "127.0.0.1:8808")
    assert not same_origin("null", "127.0.0.1:8808")


def test_a_rebound_host_name_is_refused():
    c = client()
    assert c.get("/api/thing").status_code == 200
    r = c.get("/api/thing", headers={"host": "attacker.example:8808"})
    assert r.status_code == 403
    assert "POLYVOX_ALLOWED_HOSTS" in r.json()["detail"]
    assert client(("attacker.example",)).get("/api/thing", headers={"host": "attacker.example:8808"}).status_code == 200


def test_changes_from_another_page_are_refused():
    c = client()
    assert c.post("/api/thing").status_code == 200  # no Origin: not a browser page
    assert c.post("/api/thing", headers={"origin": "http://127.0.0.1:8808"}).status_code == 200
    assert c.post("/api/thing", headers={"origin": "https://attacker.example"}).status_code == 403
    assert c.post("/api/thing", headers={"origin": "null"}).status_code == 403
    # Reading is left alone: the browser doesn't let the other page see the answer.
    assert c.get("/api/thing", headers={"origin": "https://attacker.example"}).status_code == 200
