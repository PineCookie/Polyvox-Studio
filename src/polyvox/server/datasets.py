"""Datasets: list, import, whole-dataset tools (check, normalize, tidy, export, drop / trim flagged)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import timing
from ..data import manifests
from ..jobs import Job
from ..prep import checks, exporter, fillers, importer, loudness, slicer
from ..prep.asr import LanguageLock, load_audio
from ..prep.slicer import SliceOptions
from .context import Studio, checked, found, logged_timings
from .schemas import (
    CheckIn,
    DatasetIn,
    DatasetReportOut,
    DropFlaggedIn,
    ExportIn,
    ImportIn,
    ImportPreviewOut,
    ImportSourceIn,
    JobOut,
    NormalizeIn,
)


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings, jobs, asr = s.settings, s.jobs, s.asr

    @api.get("/api/datasets")
    def list_datasets() -> list[str]:
        return manifests.scan(settings.data_dir)

    @api.post("/api/datasets/import/preview")
    def preview_import(body: ImportSourceIn) -> ImportPreviewOut:
        preview = checked(importer.preview_import, body.source, body.audio_dir, settings.data_dir)
        return ImportPreviewOut(**dataclasses.asdict(preview))

    @api.post("/api/datasets/import", status_code=202)
    def import_dataset(body: ImportIn) -> JobOut:
        """Start an import job; poll /api/jobs/{id}. The job's result is
        {"id", "imported", "transcribed", "sliced"}."""
        available = s.cleanup_models()
        cleanup = [k for k in available if getattr(body, k)]
        # Fail fast on what can be checked now.
        try:
            if body.slice and not (body.copy_audio and body.transcribe):
                raise ValueError("Slicing makes new clips without text: it needs copying and transcription.")
            if body.slice and body.slice_min_seconds >= body.slice_max_seconds:
                raise ValueError("The minimum clip length must be below the maximum.")
            if body.trim and not (body.copy_audio and body.check):
                raise ValueError("Trimming rewrites the copied clips: it needs copying and checking.")
            for option in cleanup:
                if not available[option]:
                    raise ValueError(f"The model for '{option}' is not downloaded (see Models).")
            if cleanup and not body.copy_audio:
                raise ValueError("Clean-up writes new audio: it needs copying.")
            if body.normalize_lufs is not None and not body.copy_audio:
                raise ValueError("Normalizing loudness rewrites the clips: it needs copying.")
            importer.check_name(body.name, settings.data_dir)
            importer.detect(importer.clean_path(body.source))
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        s.refuse_if_busy(gpu=body.transcribe or body.check or bool(cleanup))

        lock = LanguageLock() if body.language == "auto" else None

        def heard(wavs: list) -> list[tuple[str, str | None]]:
            s.take_gpu(asr)
            forced = lock.language if lock else body.language
            results = asr.transcribe(wavs, language=forced, prompt=body.prompt)
            if lock:
                lock.add(wavs, results)
            # A forced language is not detected, so record what was forced.
            return [(r.text, r.language or forced) for r in results]

        def transcriber(paths: list[Path]) -> list[tuple[str, str | None]]:
            return heard([load_audio(p) for p in paths])

        def run(job: Job) -> dict:
            with timing.collect() as times:
                return {**import_(job), "timings": logged_timings(f"Import {body.name}", times)}

        def import_(job: Job) -> dict:
            result = importer.import_dataset(
                body.source,
                body.audio_dir,
                body.name,
                settings.data_dir,
                copy=body.copy_audio,
                transcriber=transcriber if body.transcribe else None,
                progress=job.progress,
                slicing=(
                    SliceOptions(
                        min_seconds=body.slice_min_seconds,
                        max_seconds=body.slice_max_seconds,
                        target_seconds=(body.slice_min_seconds + body.slice_max_seconds) / 2,
                    )
                    if body.slice
                    else None
                ),
                checker=s.checker(body.aligner, detect_singing=body.detect_singing) if body.check else None,
                aligner=body.aligner,
                trim=body.trim,
                guide=slicer.Guide(transcribe=heard, align=lambda wavs, texts: s.aligned(wavs, texts, body.aligner))
                if body.slice
                else None,
                separation=s.separated(s.separation.chain([available[k] for k in cleanup])) if cleanup else None,
                tidy_fillers=body.tidy_fillers,
                normalize=body.normalize_lufs,
            )
            return {
                "id": result.id,
                "imported": result.imported,
                "transcribed": result.transcribed,
                "sliced": result.sliced,
                "trimmed": result.trimmed,
                "checks": result.checks,
            }

        return JobOut(**jobs.start("import", f"Import {body.name}", run).public())

    @api.post("/api/datasets/tidy-fillers")
    def tidy_fillers(body: DatasetIn) -> dict:
        """Shorten runs of fillers in a dataset's transcripts (fillers.tidy), in
        place; the first time, the original manifest is kept as <name>.orig.
        Checks made for the old texts become stale."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        return {"changed": checked(fillers.tidy_manifest, manifest)}

    @api.post("/api/datasets/check", status_code=202)
    def check_dataset(body: CheckIn) -> JobOut:
        """Check every clip of a dataset against its text (forced alignment) and
        save the results as align.json. The job's result is the summary
        {"aligner", "checked", "flagged", "flags": {flag: count}}."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        s.refuse_if_busy(gpu=True)

        def run(job: Job) -> dict:
            with timing.collect() as times:
                summary = checks.check_dataset(
                    manifest,
                    manifests.legacy_root(settings.data_dir),
                    s.checker(body.aligner, body.compare_asr, body.detect_singing),
                    body.aligner,
                    load_audio,
                    job.progress,
                )
                return {**summary, "timings": logged_timings(f"Check {body.id}", times)}

        return JobOut(**jobs.start("check", f"Check {body.id}", run).public())

    @api.post("/api/datasets/normalize", status_code=202)
    def normalize_dataset(body: NormalizeIn) -> JobOut:
        """Bring every clip of a dataset to ``target_lufs`` in place (originals
        kept in originals/; checks stay valid). The job's result is
        loudness.normalize_dataset's summary."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        s.refuse_if_busy(gpu=False)

        def run(job: Job) -> dict:
            root = manifests.legacy_root(settings.data_dir)
            return loudness.normalize_dataset(manifest, root, body.target_lufs, job.progress)

        return JobOut(**jobs.start("normalize", f"Normalize {body.id}", run).public())

    @api.get("/api/export-formats")
    def export_formats() -> list[dict]:
        return [dict(f.__dict__) for f in exporter.FORMATS]

    @api.get("/api/datasets/export-dir")
    def export_dir(id: str, format: str) -> dict:
        """Where an export goes unless another folder is given."""
        found(manifests.resolve, settings.data_dir, id)
        return {"dir": str(exporter.default_dir(settings.data_dir, id, format))}

    @api.post("/api/datasets/export", status_code=202)
    def export_dataset(body: ExportIn) -> JobOut:
        """Write a dataset's training clips in another TTS tool's layout
        (exporter.py). The job's result is exporter.export_dataset's summary."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        if body.format not in exporter.BY_ID:
            raise HTTPException(422, f"Unknown export format: {body.format}")
        if body.language is not None and body.language not in exporter.GPT_SOVITS_LANGUAGES:
            raise HTTPException(422, f"Unknown language code: {body.language}")
        out_dir = (
            Path(body.out_dir)
            if body.out_dir.strip()
            else exporter.default_dir(settings.data_dir, body.id, body.format)
        )
        if not out_dir.is_absolute():
            raise HTTPException(422, "The export folder must be an absolute path.")
        if out_dir.exists() and any(out_dir.iterdir()):
            raise HTTPException(409, f"{out_dir} already exists and is not empty.")
        s.refuse_if_busy(gpu=False)
        speaker = manifest.parent.name

        def run(job: Job) -> dict:
            return exporter.export_dataset(
                manifest,
                manifests.legacy_root(settings.data_dir),
                out_dir,
                body.format,
                speaker,
                min_seconds=body.min_seconds,
                max_seconds=body.max_seconds,
                sample_rate=body.sample_rate,
                language=body.language,
                progress=job.progress,
            )

        return JobOut(**jobs.start("export", f"Export {body.id}", run).public())

    @api.post("/api/datasets/drop-flagged")
    def drop_flagged(body: DropFlaggedIn) -> dict:
        """Drop every clip whose check has ``flag`` (e.g. all sung clips)."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        if body.flag not in checks.FLAGS:
            raise HTTPException(422, f"Unknown flag '{body.flag}'.")
        return {"dropped": manifests.drop_flagged(manifest, manifests.legacy_root(settings.data_dir), body.flag)}

    @api.post("/api/datasets/trim-flagged")
    def trim_flagged(body: DropFlaggedIn) -> dict:
        """Trim every clip with ``flag`` (edge_sound or long_silence) at its
        checked edges, in place (originals kept in originals/). Clips also
        flagged mismatch / asr_more are left for review. Returns {"trimmed",
        "skipped" (format or location can't be rewritten), "review"}."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        if body.flag not in ("edge_sound", "long_silence"):
            raise HTTPException(422, "Only clips flagged for edge sound or long silence can be trimmed.")
        s.refuse_if_busy(gpu=False)
        return checks.trim_flagged(manifest, manifests.legacy_root(settings.data_dir), body.flag)

    # Last: the path parameter would also match the routes above.
    @api.get("/api/datasets/{dataset_id:path}")
    def inspect_dataset(dataset_id: str) -> DatasetReportOut:
        path = found(manifests.resolve, settings.data_dir, dataset_id)
        report = manifests.inspect(path, manifests.legacy_root(settings.data_dir))
        return DatasetReportOut(id=dataset_id, **dataclasses.asdict(report))

    return api
