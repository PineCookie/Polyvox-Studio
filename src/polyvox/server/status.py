"""Server status, engines, unloading, the models folder and background jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import models
from ..config import Settings
from ..jobs import Job
from .context import Studio, found
from .schemas import DownloadIn, JobOut, ModelOut, ModelsOut


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings, jobs = s.settings, s.jobs

    @api.get("/api/status")
    def status() -> dict:
        return s.status()

    @api.get("/api/engines")
    def list_engines() -> list[dict]:
        """Each engine: capabilities, synthesis settings (params) and whether it can run here."""
        return [e.public() for e in s.registry.values()]

    @api.post("/api/model/unload")
    def unload() -> dict:
        s.unload_all()
        return s.status()

    @api.get("/api/models")
    def list_models() -> ModelsOut:
        out = []
        for spec in models.MODELS:
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
                    required=spec.required,
                    state=state,
                    path=path,
                    fallback=bool(Settings.__dataclass_fields__[spec.setting].default),
                )
            )
        return ModelsOut(models_dir=str(settings.models_dir.resolve()), endpoint=settings.hf_endpoint, models=out)

    @api.post("/api/models/download", status_code=202)
    def download_models(body: DownloadIn) -> JobOut:
        """Download models into the models folder, one after another (job phase:
        the setting being downloaded; done / total in MB)."""
        unknown = [m for m in body.settings if m not in models.BY_SETTING]
        if unknown:
            raise HTTPException(422, f"Unknown model: {', '.join(unknown)}")
        endpoint = (settings.hf_endpoint if body.endpoint is None else body.endpoint).strip().rstrip("/")
        if endpoint and not endpoint.startswith(("http://", "https://")):
            raise HTTPException(422, "The download source must be an http(s) URL.")
        if running := jobs.running("download"):
            raise HTTPException(409, f"Wait for “{running.title}” to finish.")
        specs = [models.BY_SETTING[m] for m in body.settings]

        def run(job: Job) -> dict:
            for spec in specs:
                models.download(
                    spec,
                    settings.models_dir,
                    endpoint or None,
                    lambda done, total, spec=spec: job.progress(spec.setting, done, total),
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
