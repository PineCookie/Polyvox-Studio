"""Reviewing a dataset clip by clip: pages of clips, one clip's words and audio, edits."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..data import manifests
from ..prep import checks
from ..prep.asr import SAMPLE_RATE as ASR_RATE
from ..prep.asr import language_of, load_audio
from .context import Studio, checked, found, has_checks
from .schemas import (
    ClipCheckIn,
    ClipDetailOut,
    ClipDropIn,
    ClipPageOut,
    ClipRangeIn,
    ClipRowOut,
    ClipTextIn,
    ClipTrimIn,
    DatasetReportOut,
    HeardOut,
    WordOut,
)

MIN_KEPT_S = 0.3  # a trimmed clip keeps at least this much


def trim_of(saved: dict) -> tuple[float, float] | None:
    return checks.ClipCheck.from_saved(saved).trim_range()


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings = s.settings

    def review_manifest(id: str | None, run: str | None) -> Path:
        """A dataset by manifest id (e.g. "speaker/train.jsonl"), or the one a run trains on."""
        if run:
            return found(s.training.train_manifest, run)
        if id:
            return found(manifests.resolve, settings.data_dir, id)
        raise HTTPException(422, "Give a dataset id or a run name.")

    @api.post("/api/dataset-clip/text")
    def set_clip_text(body: ClipTextIn) -> ClipDetailOut:
        """Change one clip's transcript. When the dataset has checks, the clip is
        checked again right away (with the dataset's aligner, compared with ASR
        if its last check was); otherwise its old check just goes stale."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        s.refuse_if_busy(gpu=False)
        checked(manifests.set_text, manifest, body.index, body.text)
        if has_checks(manifest) and s.gpu_free():
            s.check_clip_again(manifest, body.index)
        return dataset_clip(body.index, id=body.id)

    @api.post("/api/dataset-clip/check")
    def check_clip(body: ClipCheckIn) -> ClipDetailOut:
        """Check one clip again now (alignment; with ``compare_asr``, also what ASR hears)."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        s.refuse_if_busy(gpu=True)
        found(s.check_clip_again, manifest, body.index, body.compare_asr or None)
        return dataset_clip(body.index, id=body.id)

    @api.post("/api/dataset-clip/trim")
    def trim_clip(body: ClipTrimIn) -> ClipDetailOut:
        """Trim one clip in place (see trim-flagged): to [start, end) seconds when
        given, else at its checked edges."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        root = manifests.legacy_root(settings.data_dir)
        s.refuse_if_busy(gpu=False)
        keep: tuple[float, float] | None = None
        if body.start is not None or body.end is not None:
            seconds = found(manifests.clip_detail, manifest, root, body.index).row.seconds or 0.0
            keep = (max(0.0, body.start or 0.0), min(seconds, body.end if body.end is not None else seconds))
            if keep[1] - keep[0] < MIN_KEPT_S:
                raise HTTPException(422, f"Keep at least {MIN_KEPT_S} s of the clip.")
            if keep[0] < 0.01 and keep[1] > seconds - 0.01:
                raise HTTPException(409, "Nothing to trim: the range is the whole clip.")

        def cut_for(index: int, check: checks.ClipCheck | None) -> tuple[float, float] | None:
            if index != body.index:
                return None
            return keep or (check.trim_range() if check else None)

        result = checks.trim_clips(manifest, root, cut_for)
        if result["skipped"]:
            raise HTTPException(
                422, "This clip can't be trimmed in place (outside the dataset folder, or not WAV / FLAC / OGG)."
            )
        if not result["trimmed"]:
            raise HTTPException(409, "Nothing to trim: check the clip first.")
        # Align the trimmed clip afresh rather than keep the shifted old check.
        if has_checks(manifest) and s.gpu_free():
            s.check_clip_again(manifest, body.index)
        return dataset_clip(body.index, id=body.id)

    @api.post("/api/dataset-clip/hear")
    def hear_clip_part(body: ClipRangeIn) -> dict:
        """What ASR hears in [start, end) seconds of one clip, e.g. an edge sound
        the transcript lacks: {"text"}."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        audio = found(manifests.clip_audio, manifest, manifests.legacy_root(settings.data_dir), body.index)
        s.refuse_if_busy(gpu=True)
        part = load_audio(audio)[int(body.start * ASR_RATE) : int(body.end * ASR_RATE)]
        if len(part) < ASR_RATE // 10:
            raise HTTPException(422, "That part is too short to hear.")
        text = manifests.clip_detail(manifest, manifests.legacy_root(settings.data_dir), body.index).row.text
        return {"text": s.asr.transcribe([part], language=language_of(text))[0].text}

    @api.post("/api/dataset-clip/drop")
    def drop_clip(body: ClipDropIn) -> dict:
        """Leave one clip out of training (``dropped``), or bring it back."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        found(manifests.set_dropped, manifest, manifests.legacy_root(settings.data_dir), body.index, body.dropped)
        return {"index": body.index, "dropped": body.dropped}

    @api.get("/api/dataset-report")
    def dataset_report(id: str | None = None, run: str | None = None) -> DatasetReportOut:
        report = manifests.inspect(review_manifest(id, run), manifests.legacy_root(settings.data_dir))
        return DatasetReportOut(id=id or f"run:{run}", **dataclasses.asdict(report))

    @api.get("/api/dataset-rows")
    def dataset_rows(
        id: str | None = None,
        run: str | None = None,
        offset: int = 0,
        limit: int = Query(50, ge=1, le=500),
        q: str = "",
        flag: str | None = None,
        order: Literal["file", "worst"] = "file",
        keep: Annotated[list[int] | None, Query()] = None,
    ) -> ClipPageOut:
        """A page of a dataset's clips for review. ``flag``: a check flag, "any"
        flag, or "dropped" clips; ``order=worst`` puts the lowest scores first."""
        path = review_manifest(id, run)
        page = manifests.read_rows(
            path,
            manifests.legacy_root(settings.data_dir),
            max(0, offset),
            limit,
            q,
            flag or None,
            order == "worst",
            trim_of,
            frozenset(keep or ()),
        )
        return ClipPageOut(
            total=page.total,
            rows=[ClipRowOut(**dataclasses.asdict(r)) for r in page.rows],
            checks=page.checks,
            dropped=page.dropped,
        )

    @api.get("/api/dataset-clip")
    def dataset_clip(index: int, id: str | None = None, run: str | None = None) -> ClipDetailOut:
        """One clip with a finer waveform and, when checked, where each word is."""
        path = review_manifest(id, run)
        detail = found(manifests.clip_detail, path, manifests.legacy_root(settings.data_dir), index, trim_of)
        weak_below = checks.WEAK_WORD_BELOW.get(detail.aligner or "", 0.0)
        check = checks.ClipCheck.from_saved(detail.check) if detail.check else None
        return ClipDetailOut(
            sound_before=check.sound_before[0] if check and check.sound_before else None,
            sound_after=check.sound_after[0] if check and check.sound_after else None,
            speaker=check.speaker if check else None,
            clipped=check.clipped if check else None,
            **dataclasses.asdict(detail.row),
            aligner=detail.aligner,
            words=[
                WordOut(text=t, start=start, end=end, confidence=c, weak=c < weak_below)
                for t, start, end, c in detail.words
            ],
            heard=HeardOut(
                text=detail.asr["text"],
                extra=detail.asr["extra"],
                missing=detail.asr["missing"],
                changed=detail.asr["changed"],
                diff=checks.asr_diff(detail.row.text, detail.asr["text"]),
            )
            if detail.asr
            else None,
        )

    @api.get("/api/dataset-audio")
    def dataset_audio(index: int, id: str | None = None, run: str | None = None) -> FileResponse:
        """Audio of one clip, by its index in a dataset (never by path)."""
        path = review_manifest(id, run)
        return FileResponse(found(manifests.clip_audio, path, manifests.legacy_root(settings.data_dir), index))

    return api
