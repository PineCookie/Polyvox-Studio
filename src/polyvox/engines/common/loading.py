"""Building a model whose weights all come from a checkpoint, without first filling them with
random numbers: torch's random initializers (torch.nn.init.*, Tensor.uniform_ / normal_) do
nothing in the thread inside ``no_random_init()``. Random initialization of a 2B-parameter model
on the CPU takes ~12 s, all of it overwritten by the checkpoint right after.

Only for models loaded strictly afterwards (every parameter and persistent buffer from the file):
anything left out would hold whatever memory it got. Other threads keep the real initializers.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_INIT_FUNCTIONS = (
    "uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
    "xavier_uniform_", "xavier_normal_", "orthogonal_",
)  # fmt: skip
_TENSOR_METHODS = ("uniform_", "normal_")

_local = threading.local()
_lock = threading.Lock()
_users = 0
_originals: dict[tuple[object, str], object] = {}


def _skipping() -> bool:
    return getattr(_local, "depth", 0) > 0


def _wrap(original):
    def initializer(tensor, *args, **kwargs):
        return tensor if _skipping() else original(tensor, *args, **kwargs)

    initializer.__wrapped__ = original
    return initializer


def _install() -> None:
    import torch

    for owner, names in ((torch.nn.init, _INIT_FUNCTIONS), (torch.Tensor, _TENSOR_METHODS)):
        for name in names:
            original = getattr(owner, name)
            _originals[(owner, name)] = original
            setattr(owner, name, _wrap(original))


def _uninstall() -> None:
    for (owner, name), original in _originals.items():
        setattr(owner, name, original)
    _originals.clear()


@contextmanager
def no_random_init() -> Iterator[None]:
    """Inside: this thread's random weight initialization is skipped (see the module docstring)."""
    global _users
    with _lock:
        if _users == 0:
            _install()
        _users += 1
    _local.depth = getattr(_local, "depth", 0) + 1
    try:
        yield
    finally:
        _local.depth -= 1
        with _lock:
            _users -= 1
            if _users == 0:
                _uninstall()
