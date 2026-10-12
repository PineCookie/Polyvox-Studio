"""Datasets: list, import, whole-dataset tools (check, normalize, tidy, export, drop / trim flagged)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import timing
from ..data import manifests, record
from ..jobs import Job
from ..prep import checks, exporter, importer, slicer, steps
from ..prep.asr import LanguageLock, load_audio
from ..prep.slicer import SliceOptions
from ..prep.steps import DEFAULTS as STEP_DEFAULTS
from .context import Studio, checked, found, logged_timings
from .schemas import (
    AddSourceIn,
    DatasetReportOut,
    DropFlaggedIn,
    ExportIn,
    ImportPreviewOut,
    ImportSourceIn,
    JobOut,
    NewDatasetIn,
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
        job = jobs.running_for(id)
        return StepStatusOut(**steps.status(manifest.parent), job=job.public() if job else None)

    @api.post("/api/datasets/new")
    def new_dataset(body: NewDatasetIn) -> dict:
        """An empty dataset to add sources to: {"id"}."""
        try:
            folder = steps.create(settings.data_dir, body.name)
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"id": f"{folder.name}/{steps.MANIFEST}"}

    @api.post("/api/datasets/add-source", status_code=202)
    def add_source(body: AddSourceIn) -> JobOut:
        """Add a source's clips and recordings to a dataset (a job: copying
        takes a while); with ``run_steps``, the steps with work left run after
        (with ``settings``, as run-remaining). The job's result is {"clips",
        "recordings"} added and {"steps": {step: its result}}."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        checked(importer.detect, importer.clean_path(body.source))
        s.refuse_if_busy(gpu=body.run_steps)

        def run(job: Job) -> dict:
            with timing.collect() as times:
                added = steps.add_source(
                    manifest.parent,
                    body.source,
                    body.audio_dir,
                    settings.data_dir,
                    body.copy_audio,
                    progress=job.progress,
                )
                if not added.clips and not added.recordings:
                    raise ValueError("No new audio found in the source; nothing was added.")
                done = run_remaining_steps(manifest.parent, body.settings, job) if body.run_steps else {}
                return {
                    "clips": added.clips,
                    "recordings": added.recordings,
                    "steps": done,
                    "timings": logged_timings(f"Add to {body.id}", times),
                }

        return JobOut(**jobs.start("import", f"Add to {body.id}", run, body.id, "add").public())

    @api.post("/api/datasets/step", status_code=202)
    def dataset_step(body: StepIn) -> JobOut:
        """Run one processing step on what still needs it (a job)."""
        manifest = found(manifests.resolve, settings.data_dir, body.id)
        given = body.model_dump(exclude={"id", "step"})
        if body.step == "slice":
            cut = step_settings(manifest.parent, "slice", given)
            if cut["min_seconds"] >= cut["max_seconds"]:
                raise HTTPException(422, "The minimum clip length must be below the maximum.")
        if body.step == "clean_up":
            available = s.cleanup_models()
            models = step_settings(manifest.parent, "clean_up", given)["models"]
            if not models:
                raise HTTPException(422, "Choose what to clean up: separation, de-reverb or denoise.")
            if missing := [m for m in models if not available.get(m)]:
                raise HTTPException(422, f"The model for '{missing[0]}' is not downloaded (see Models).")
        s.refuse_if_busy(gpu=body.step != "tidy")

        def run(job: Job) -> dict:
            with timing.collect() as times:
                result = run_step(manifest.parent, body.step, given, job)
                return {**result, "timings": logged_timings(f"{body.step} {body.id}", times)}

        return JobOut(
            **jobs.start(
                "step", f"{body.step.replace('_', ' ').capitalize()} {body.id}", run, body.id, body.step
            ).public()
        )

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

        return JobOut(**jobs.start("step", f"Steps {body.id}", run, body.id, "remaining").public())

    @api.post("/api/datasets/import/preview")
    def preview_import(body: ImportSourceIn) -> ImportPreviewOut:
        preview = checked(importer.preview_import, body.source, body.audio_dir, settings.data_dir)
        return ImportPreviewOut(**dataclasses.asdict(preview))

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
