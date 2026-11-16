"""The check for a newer release (no network: the fetch is made up)."""

import dataclasses

from fastapi.testclient import TestClient

from polyvox import updates
from polyvox.server import create_app


def test_versions_compare_as_versions():
    assert updates.newer("v0.10.0", "0.9.1") and updates.newer("0.6.1", "0.6.0")
    assert not updates.newer("v0.6.0", "0.6.0") and not updates.newer("v0.5.0", "0.6.0")
    assert not updates.newer("nightly", "0.6.0")


def test_a_newer_release_is_reported_and_an_older_or_unreachable_one_isnt(monkeypatch):
    monkeypatch.setattr(updates, "current_version", lambda: "0.6.0")
    url = "https://github.com/someone/project/releases/tag/v0.7.0"
    check = updates.UpdateCheck(fetch=lambda: {"tag_name": "v0.7.0", "html_url": url})
    assert check.check() == {"version": "0.7.0", "url": url}
    assert updates.UpdateCheck(fetch=lambda: {"tag_name": "v0.5.0"}).check() is None

    def offline():
        raise ConnectionError("no network")

    check.__init__(fetch=offline)
    check.info = {"version": "0.7.0", "url": url}  # a failed check keeps what an earlier one found
    assert check.check() == {"version": "0.7.0", "url": url}


def test_the_status_carries_the_version_and_a_newer_release(service, training, asr_service, monkeypatch):
    started = []
    monkeypatch.setattr(updates.UpdateCheck, "start", lambda self: started.append(self))
    client = TestClient(create_app(service.settings, service, training, asr_service))
    status = client.get("/api/status").json()
    assert status["version"] == updates.current_version() and status["update"] is None and not started  # off in tests
    settings = dataclasses.replace(service.settings, update_check=True)
    client = TestClient(create_app(settings, service, training, asr_service))
    started[0].info = {"version": "9.9.9", "url": "https://example.com/release"}
    assert client.get("/api/status").json()["update"] == {"version": "9.9.9", "url": "https://example.com/release"}
