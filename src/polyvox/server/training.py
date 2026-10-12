"""Training runs: start, resume, stop, details, samples and their speaker similarity."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..training import RunDetail, RunSummary, TrainParams
from .context import Studio, checked, found
from .schemas import ResumeIn

logger = logging.getLogger(__name__)


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    training = s.training

    @api.get("/api/training/runs")
    def list_runs() -> list[RunSummary]:
        return training.list_runs()

    @api.post("/api/training/runs")
    def start_run(params: TrainParams) -> RunDetail:
        return checked(training.start, params)

    @api.get("/api/training/runs/{name}")
    def run_detail(name: str, since: int | None = None) -> RunDetail:
        """``since``: only the metric points after that step (polls while training)."""
        return found(training.detail, name, since)

    @api.post("/api/training/runs/{name}/resume")
    def resume_run(name: str, body: ResumeIn | None = None) -> RunDetail:
        return checked(training.resume, name, body.num_iters if body else None)

    @api.post("/api/training/stop")
    def stop_training() -> dict:
        training.stop()
        return {"stopping": training.active_run}

    @api.get("/api/training/runs/{name}/similarity")
    def run_similarity(name: str) -> dict:
        """Speaker similarity of the run's samples to its training voice (speaker.py)."""
        run_dir = found(training.run_dir, name)
        manifest = found(training.train_manifest, name)
        try:
            return s.speaker.score_run(run_dir, manifest)
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Similarity of %s failed: %s", name, e)
            raise HTTPException(503, f"Speaker similarity unavailable: {e}") from e

    @api.get("/api/training/runs/{name}/samples/{file}")
    def run_sample(name: str, file: str) -> FileResponse:
        return FileResponse(found(training.sample_file, name, file), media_type="audio/wav")

    return api
