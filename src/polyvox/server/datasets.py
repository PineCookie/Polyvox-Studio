"""Datasets: list, import, whole-dataset tools (check, normalize, tidy, export, drop / trim flagged)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import timing
from ..data import manifests, record
from ..jobs import Job
from ..prep import checks, exporter, fillers, importer, loudness, slicer, steps
from ..prep.asr import LanguageLock, load_audio
from ..prep.slicer import SliceOptions
from ..prep.steps import DEFAULTS as STEP_DEFAULTS
from .context import Studio, checked, found, logged_timings
from .schemas import (
    AddSourceIn,
    CheckIn,
    CreateDatasetIn,
    DatasetIn,
    DatasetReportOut,
    DropFlaggedIn,
    ExportIn,
    ImportIn,
    ImportPreviewOut,
    ImportSourceIn,
    JobOut,
    NormalizeIn,
    RunStepsIn,
    StepIn,
    StepStatusOut,
)


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings, jobs, asr = s.settings, s.jobs, s.asr

    @api.get("/api/datasets")
    def list_datasets() -> list[str]:
        return manifests.scan(settings.data_dir)

    def asr_tools(language: str | None, prompt: str | None):
        """(heard: 16 kHz clips -> [(text, language)], transcriber: files -> the
        same). ``language`` "auto" detects until the clips agree, then forces
        that; None / "" detects per clip; else forced."""
        lock = LanguageLock() if language == "auto" else None
        fixed = language or None

        def heard(wavs: list) -> list[tuple[str, str | None]]:
            s.take_gpu(asr)
            forced = lock.language if lock else fixed
            results = asr.transcribe(wavs, language=forced, prompt=prompt)
            if lock:
                lock.add(wavs, results)
            # A forced language is not detected, so record what was forced.
            return [(r.text, r.language or forced) for r in results]

        def transcriber(paths: list[Path]) -> list[tuple[str, str | None]]:
            return heard([load_audio(p) for p in paths])

        return heard, transcriber

    def step_settings(folder: Path, step: str, given: dict) -> dict:
        """A step's settings: given > its last run on this dataset > defaults."""
        defaults = STEP_DEFAULTS[step]
        saved = record.read_record(folder).steps.get(step, {})
        return {
            **defaults,
            **{k: v for k, v in saved.items() if k in defaults},
            **{k: v for k, v in given.items() if k in defaults and v is not None},
        }

    def run_step(folder: Path, step: str, given: dict, job: Job, others: dict[str, dict] | None = None) -> dict:
        """``others``: settings given for the other steps (slicing uses clean-up's
        models, transcription's language and prompt and the check's aligner)."""
        others = others or {}
        opts = step_settings(folder, step, given)
        if step == "clean_up":
            available = s.cleanup_models()
            if not opts["models"] or not all(available.get(m) for m in opts["models"]):
                raise ValueError("Choose clean-up models that are downloaded (see Models).")
            chain = s.separated(s.separation.chain([available[m] for m in opts["models"]]))
            return steps.clean_up(folder, chain, opts["models"], job.progress)
        if step == "slice":
            text = step_settings(folder, "transcribe", {**given, **others.get("transcribe", {})})
            aligner = step_settings(folder, "check", others.get("check", {}))["aligner"]
            heard, _ = asr_tools(text["language"], text["prompt"])
            guide = slicer.Guide(transcribe=heard, align=lambda wavs, texts: s.aligned(wavs, texts, aligner))
            models = step_settings(folder, "clean_up", others.get("clean_up", {}))["models"]
            available = s.cleanup_models()
            chain = (
                s.separated(s.separation.chain([available[m] for m in models]))
                if models and all(available.get(m) for m in models)
                else None
            )
            mid = (opts["min_seconds"] + opts["max_seconds"]) / 2
            cut = SliceOptions(min_seconds=opts["min_seconds"], max_seconds=opts["max_seconds"], target_seconds=mid)
            return steps.slice_recordings(folder, cut, guide=guide, chain=chain, progress=job.progress)
        if step == "transcribe":
            _, transcriber = asr_tools(opts["language"], opts["prompt"])
            keep = {"language": opts["language"], "prompt": opts["prompt"]}
            return steps.transcribe(folder, transcriber, opts["tidy_fillers"], job.progress, keep)
        if step == "check":
            checker = s.checker(opts["aligner"], opts["compare_asr"], opts["detect_singing"])
            more = {"compare_asr": opts["compare_asr"], "detect_singing": opts["detect_singing"]}
            everything = bool(given.get("everything"))
            return steps.check(folder, checker, opts["aligner"], load_audio, s.voices, job.progress, more, everything)
        return steps.tidy(folder, opts["trim"], opts["normalize_lufs"], job.progress)

    @api.get("/api/datasets/steps")
    def step_status(id: str) -> StepStatusOut:
        """What each processing step still has to do on a dataset, and its last settings."""
        manifest = found(manifests.resolve, settings.data_dir, id)
        return StepStatusOut(**steps.status(manifest.parent))

    @api.post("/api/datasets/add-source", status_code=202)
    def add_source(body: AddSourceIn) -> JobOut:
        """Add a source's clips and recordings to a dataset (a job: copying
        takes a while). The job's result is {"clips", "recordings"} added."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        checked(importer.detect, importer.clean_path(body.source))
        s.refuse_if_busy(gpu=False)

        def run(job: Job) -> dict:
            added = steps.add_source(
                manifest.parent, body.source, body.audio_dir, settings.data_dir, body.copy_audio, progress=job.progress
            )
            return {"clips": added.clips, "recordings": added.recordings}

        return JobOut(**jobs.start("import", f"Add to {body.id}", run).public())

    @api.post("/api/datasets/step", status_code=202)
    def dataset_step(body: StepIn) -> JobOut:
        """Run one processing step on what still needs it (a job)."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        s.refuse_if_busy(gpu=body.step != "tidy")
        given = body.model_dump(exclude={"id", "step"})

        def run(job: Job) -> dict:
            with timing.collect() as times:
                result = run_step(manifest.parent, body.step, given, job)
                return {**result, "timings": logged_timings(f"{body.step} {body.id}", times)}

        return JobOut(**jobs.start("step", f"{body.step.replace('_', ' ').capitalize()} {body.id}", run).public())

    def run_remaining_steps(folder: Path, given: dict[str, dict], job: Job) -> dict:
        """The steps with work left, in order: {step: its result}."""
        return {step: run_step(folder, step, given.get(step, {}), job, given) for step in steps.remaining(folder)}

    @api.post("/api/datasets/run-remaining", status_code=202)
    def run_remaining(body: RunStepsIn) -> JobOut:
        """Run the steps with work left (slice, transcribe, check), in order,
        each with the settings given, else its last ones. The job's result is
        {step: its result}."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        s.refuse_if_busy(gpu=True)

        def run(job: Job) -> dict:
            with timing.collect() as times:
                out = run_remaining_steps(manifest.parent, body.settings, job)
                return {**out, "timings": logged_timings(f"Steps {body.id}", times)}

        return JobOut(**jobs.start("step", f"Steps {body.id}", run).public())

    @api.post("/api/datasets/create", status_code=202)
    def create_dataset(body: CreateDatasetIn) -> JobOut:
        """A new dataset from a source (a job): its clips copied in, long
        recordings listed to slice; with ``run_steps``, the steps with work left
        run after. The job's result is {"id", "clips", "recordings", "steps"}."""
        try:
            importer.check_name(body.name, settings.data_dir)
            importer.detect(importer.clean_path(body.source))
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        s.refuse_if_busy(gpu=body.run_steps)

        def run(job: Job) -> dict:
            with timing.collect() as times:
                folder = steps.create(settings.data_dir, body.name)
                try:
                    added = steps.add_source(
                        folder, body.source, body.audio_dir, settings.data_dir, body.copy_audio, progress=job.progress
                    )
                    if not added.clips and not added.recordings:
                        raise ValueError("No audio found in the source; nothing to import.")
                except BaseException:
                    shutil.rmtree(folder, ignore_errors=True)
                    raise
                done = run_remaining_steps(folder, {}, job) if body.run_steps else {}
                return {
                    "id": f"{body.name}/{steps.MANIFEST}",
                    "clips": added.clips,
                    "recordings": added.recordings,
                    "steps": done,
                    "timings": logged_timings(f"Create {body.name}", times),
                }

        return JobOut(**jobs.start("import", f"Import {body.name}", run).public())

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
        s.refuse_if_busy(gpu=body.transcribe or body.check or body.slice or bool(cleanup))
        heard, transcriber = asr_tools(body.language, body.prompt)

        def run(job: Job) -> dict:
            with timing.collect() as times:
                return {**import_(job), "timings": logged_timings(f"Import {body.name}", times)}

        # The settings each step ran with, for the dataset's record (later runs start from them).
        noted: dict[str, dict] = {}
        if cleanup:
            noted["clean_up"] = {"models": cleanup}
        if body.slice:
            noted["slice"] = {"min_seconds": body.slice_min_seconds, "max_seconds": body.slice_max_seconds}
        if body.transcribe or body.slice:
            noted["transcribe"] = {"language": body.language, "prompt": body.prompt, "tidy_fillers": body.tidy_fillers}
        if body.check:
            noted["check"] = {"aligner": body.aligner, "compare_asr": False, "detect_singing": body.detect_singing}
        if body.trim or body.normalize_lufs is not None:
            noted["tidy"] = {"trim": body.trim, "normalize_lufs": body.normalize_lufs}

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
                embed=s.voices,
                steps=noted,
                cleanup_models=cleanup,
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
        changed = checked(fillers.tidy_manifest, manifest)
        record.remember(manifest.parent, "transcribe", {"tidy_fillers": True})
        return {"changed": changed}

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
                    s.voices,
                )
                record.remember(
                    manifest.parent,
                    "check",
                    {"aligner": body.aligner, "compare_asr": body.compare_asr, "detect_singing": body.detect_singing},
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
            result = loudness.normalize_dataset(manifest, root, body.target_lufs, job.progress)
            record.remember(manifest.parent, "tidy", {"normalize_lufs": body.target_lufs})
            return result

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
