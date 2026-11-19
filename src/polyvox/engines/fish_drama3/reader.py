"""Stacks of same-shaped tensors read from safetensors shards in the background.

Drama 3's routed experts are 61,440 small tensors (16.6 GB) that end up as six stacks per layer
in pinned memory. Going through mapped files, a stacked copy and a pinned copy took most of the
load; here each tensor's bytes go from its shard straight to its place in the stack, on a few
threads, while the caller loads everything else.
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

THREADS = 8
DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32,
    "I64": torch.int64, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
}  # fmt: skip


def header(path: Path) -> tuple[int, dict]:
    """A safetensors file's (offset of its data, {tensor name: dtype, shape, data_offsets})."""
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        entries = json.loads(f.read(size))
    entries.pop("__metadata__", None)
    return 8 + size, entries


class Stacks:
    """``groups[i][name]`` = the keys of one stack; ``result()`` gives [{name: tensor [len(keys), ...]}]
    in that order once everything is read (and raises what a read raised). ``weight_map``: key ->
    shard file name in ``root``."""

    def __init__(self, root: Path, weight_map: dict[str, str], groups: list[dict[str, list[str]]], pin: bool = False):
        self.root, self.weight_map, self.groups, self.pin = root, weight_map, groups, pin
        self._stop = threading.Event()
        self._out: list[dict[str, torch.Tensor]] = []
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="drama3-experts", daemon=True)
        self._thread.start()

    def result(self) -> list[dict[str, torch.Tensor]]:
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._out

    def cancel(self) -> None:
        """Stop reading and drop what was read (the load failed elsewhere)."""
        self._stop.set()
        self._thread.join()
        self._out = []

    def _run(self) -> None:
        try:
            shards = {shard: header(self.root / shard) for group in self.groups for keys in group.values()
                      for shard in {self.weight_map[k] for k in keys}}  # fmt: skip
            with ThreadPoolExecutor(min(THREADS, os.cpu_count() or 1), thread_name_prefix="drama3-read") as pool:
                jobs = []
                for group in self.groups:
                    stacks = {}
                    for name, keys in group.items():
                        if self._stop.is_set():
                            return
                        stacks[name], raw, spans = self._plan(keys, shards)
                        jobs.append(pool.submit(self._read, raw, len(keys), spans))
                    self._out.append(stacks)
                for job in jobs:
                    job.result()
        except BaseException as error:  # handed to result()'s caller
            self._stop.set()
            self._error = error

    def _plan(self, keys: list[str], shards: dict[str, tuple[int, dict]]):
        """An empty stack for ``keys``, its bytes, and where each key's are: (shard, offset, row)."""
        first = shards[self.weight_map[keys[0]]][1][keys[0]]
        dtype, shape = DTYPES[first["dtype"]], (len(keys), *first["shape"])
        size = math.prod(first["shape"]) * dtype.itemsize
        raw = torch.empty(len(keys) * size, dtype=torch.uint8, pin_memory=self.pin)
        stack = raw.view(dtype).view(shape)
        spans = []
        for row, key in enumerate(keys):
            shard = self.weight_map[key]
            start, entries = shards[shard]
            entry = entries[key]
            lo, hi = entry["data_offsets"]
            if entry["shape"] != first["shape"] or entry["dtype"] != first["dtype"] or hi - lo != size:
                raise ValueError(f"{key} doesn't have the shape of {keys[0]}")
            spans.append((shard, start + lo, row))
        return stack, raw, sorted(spans)

    def _read(self, raw: torch.Tensor, rows: int, spans: list[tuple[str, int, int]]) -> None:
        view = memoryview(raw.numpy())
        size = len(view) // rows
        files: dict[str, object] = {}
        try:
            for shard, offset, row in spans:
                if self._stop.is_set():
                    return
                f = files.get(shard)
                if f is None:
                    f = files[shard] = open(self.root / shard, "rb", buffering=0)
                f.seek(offset)
                at, end = row * size, (row + 1) * size
                while at < end:  # (a raw read may return less than asked)
                    got = f.readinto(view[at:end])
                    if not got:
                        raise OSError(f"{shard} ends inside a tensor")
                    at += got
        finally:
            for f in files.values():
                f.close()
