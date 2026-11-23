"""Server status, engines, unloading, the models folder and background jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import models
from ..jobs import Job
from .context import Studio, checked, found
from .schemas import DownloadIn, JobOut, ModelOut, ModelsOut, SourceIn


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings, jobs = s.settings, s.jobs

    @api.get("/api/status")
    def status() -> dict:
        return s.status()

    @api.get("/api/engines")
    def list_engines() -> list[dict]:
        """Each engine: capabilities, synthesis settings (params) and whether it can run here."""
        return [e.public(s.profile) for e in s.registry.values()]

    @api.post("/api/model/unload")
    def unload() -> dict:
        s.unload_all()
        return s.status()

    @api.get("/api/models")
    def list_models() -> ModelsOut:
        out = []
        for spec in models.all_specs():
            path = settings.model_path(spec.setting)
            if settings.configured(spec.setting):
                state = "custom"
            else:
                state = "downloaded" if spec.downloaded(settings.models_dir) else "missing"
            out.append(
                ModelOut(
                    setting=spec.setting,
                    name=spec.name,
                    purpose=spec.purpose,
                    repo=spec.repo,
                    hub=spec.hub,
                    size_mb=spec.size_mb,
                    license=spec.license,
                    state=state,
                    path=path,
                    fallback=bool(spec.default),
                    group=spec.group,
                    ways=spec.ways(),
                )
            )
        running = jobs.running("download")
        return ModelsOut(
            models_dir=str(settings.models_dir.resolve()),
            endpoint=models.current_endpoint(),
            models=out,
            job=JobOut(**running.public()) if running else None,
        )

    @api.post("/api/models/source")
    def set_source(body: SourceIn) -> dict:
        """The download source (Hugging Face or a mirror): for the Models tab's downloads and for
        models that download themselves when first used; remembered in the models folder."""
        return {"endpoint": checked(models.save_endpoint, settings.models_dir, body.endpoint)}

    @api.post("/api/models/download", status_code=202)
    def download_models(body: DownloadIn) -> JobOut:
        """Download models into the models folder, one after another (job phase:
        the setting being downloaded; done / total in MB). A given ``endpoint`` becomes the
        download source (set_source)."""
        unknown = [m for m in body.settings if m not in models.by_setting()]
        if unknown:
            raise HTTPException(422, f"Unknown model: {', '.join(unknown)}")
        if body.endpoint is None:
            endpoint = models.current_endpoint()
        else:
            endpoint = checked(models.check_endpoint, body.endpoint)
        if running := jobs.running("download"):
            raise HTTPException(409, f"Wait for “{running.title}” to finish.")
        specs = [models.by_setting()[m] for m in body.settings]
        for setting, way in body.ways.items():
            spec = models.by_setting().get(setting)
            if spec is None or way not in {w["id"] for w in spec.ways()} | {"prebuilt"}:
                raise HTTPException(422, f"{setting} can't be downloaded as {way}.")
        if body.endpoint is not None:  # the source these come from is the source from now on
            models.save_endpoint(settings.models_dir, endpoint)

        def run(job: Job) -> dict:
            for spec in specs:
                models.download(
                    spec,
                    settings.models_dir,
                    endpoint or None,
                    lambda done, total, spec=spec: job.progress(spec.setting, done, total),
                    body.ways.get(spec.setting, "prebuilt"),
                )
            return {"downloaded": [spec.setting for spec in specs]}

        names = ", ".join(spec.name for spec in specs)
        return JobOut(**jobs.start("download", f"Download {names}", run).public())

    @api.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JobOut:
        return JobOut(**found(jobs.get, job_id).public())

    @api.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> JobOut:
        return JobOut(**found(jobs.cancel, job_id).public())

    return api
