"""What a dataset folder holds besides its clips: ``dataset.json``.

A dataset is a folder with its clips in ``train.jsonl`` (what training reads;
a clip's text stays empty until it is transcribed) and, here:

- ``recordings``: long recordings the clips are cut from, each with the
  slicing settings it was sliced with (None: not sliced yet);
- ``origins``: where each clip's audio came from: a recording and the
  seconds cut from it, or a whole source file copied in;
- ``steps``: the settings each processing step last ran with (slice,
  transcribe, check, ...), so the next run starts from them;
- ``cleaned``: clips the clean-up step processed (recordings are cleaned
  while they are sliced).

Folders without the file (prepared clips, datasets from before it) read as
all clips, no recordings. Check results, ASR-made texts and drops keep their
own files next to the manifest (manifests.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RECORD_FILE = "dataset.json"
VERSION = 1


@dataclass
class Recording:
    audio: str  # relative to the dataset folder when inside it, else absolute
    seconds: float
    rate: int | None = None
    text: str = ""  # the source's transcript of the whole recording, if it had one
    sliced: dict | None = None  # the slicing settings it was cut with; None: not sliced yet


@dataclass
class Origin:
    recording: str  # Recording.audio, or the source file copied in
    start: float  # seconds into it
    end: float


@dataclass
class Record:
    recordings: list[Recording] = field(default_factory=list)
    origins: dict[str, Origin] = field(default_factory=dict)  # clip key (manifests.row_key) -> origin
    steps: dict[str, dict] = field(default_factory=dict)  # step -> the settings it last ran with
    cleaned: dict[str, list[str]] = field(default_factory=dict)  # clip key -> clean-up models applied


def read_record(folder: Path) -> Record:
    """The folder's dataset.json; an empty record when there is none."""
    try:
        data = json.loads((folder / RECORD_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Record()
    return Record(
        recordings=[Recording(**r) for r in data.get("recordings", [])],
        origins={k: Origin(**o) for k, o in data.get("origins", {}).items()},
        steps=dict(data.get("steps", {})),
        cleaned={k: list(v) for k, v in data.get("cleaned", {}).items()},
    )


def write_record(folder: Path, record: Record) -> None:
    """Save atomically (a temporary file, then a rename)."""
    data = {"version": VERSION, **asdict(record)}
    tmp = folder / f".{RECORD_FILE}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(folder / RECORD_FILE)


def remember(folder: Path, step: str, settings: dict) -> None:
    """Note the settings a step ran with (over those it had, so a step made of
    parts, like tidy's trimming / fillers / loudness, keeps the others)."""
    record = read_record(folder)
    record.steps[step] = {**record.steps.get(step, {}), **settings}
    write_record(folder, record)


def trimmed(folder: Path, cuts: dict[str, tuple[float, float]]) -> None:
    """Move the origins of clips cut to [start, end) seconds of themselves."""
    record = read_record(folder)
    changed = False
    for key, (start, end) in cuts.items():
        if origin := record.origins.get(key):
            record.origins[key] = Origin(origin.recording, round(origin.start + start, 3), round(origin.start + end, 3))
            changed = True
    if changed:
        write_record(folder, record)
