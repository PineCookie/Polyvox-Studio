"""Is a newer release out? Checked on GitHub at startup and once a day (off with
POLYVOX_UPDATE_CHECK=0); the UI shows a notice, and update.bat / update.sh does the update.
Offline, or with GitHub unreachable, nothing is shown."""

from __future__ import annotations

import importlib.metadata
import logging
import threading

logger = logging.getLogger(__name__)

REPO = "PineCookie/Polyvox-Studio"  # also in tools/update.ps1 and tools/update.sh
LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
CHECK_EVERY_S = 24 * 3600
TIMEOUT_S = 5.0


def current_version() -> str:
    try:
        return importlib.metadata.version("polyvox")
    except importlib.metadata.PackageNotFoundError:
        return "0"


def newer(tag: str, current: str) -> bool:
    """Whether release ``tag`` ("v0.7.0") is newer than version ``current``."""
    from packaging.version import InvalidVersion, Version

    try:
        return Version(tag.removeprefix("v")) > Version(current)
    except InvalidVersion:
        return False


class UpdateCheck:
    """``info``: {"version", "url"} of a newer release, else None."""

    def __init__(self, fetch=None):
        self.info: dict | None = None
        self._fetch = fetch or self._latest
        self._stop = threading.Event()

    @staticmethod
    def _latest() -> dict:
        import httpx

        r = httpx.get(LATEST, timeout=TIMEOUT_S, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        return r.json()

    def check(self) -> dict | None:
        try:
            release = self._fetch()
        except Exception as e:  # offline, rate-limited, GitHub blocked: no notice
            logger.debug("Update check failed: %s", e)
            return self.info
        tag, current = str(release.get("tag_name") or ""), current_version()
        self.info = (
            {"version": tag.removeprefix("v"), "url": release.get("html_url") or ""} if newer(tag, current) else None
        )
        if self.info:
            logger.info("Polyvox Studio %s is available (this is %s)", self.info["version"], current)
        return self.info

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                self.check()
                self._stop.wait(CHECK_EVERY_S)

        threading.Thread(target=loop, daemon=True, name="update-check").start()

    def stop(self) -> None:
        self._stop.set()
