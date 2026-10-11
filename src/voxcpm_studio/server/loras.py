"""The LoRA list, and export / import of shareable packages."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .. import lora_library, sharing
from .context import Studio, checked, found, save_upload
from .schemas import LoraConfigOut, LoraOut


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings = s.settings

    @api.get("/api/loras")
    def list_loras() -> list[LoraOut]:
        return [
            LoraOut(
                id=c.id,
                engine=c.engine,
                base_model=c.base_model,
                base_model_id=sharing.hub_id(c.base_model),
                step=sharing.checkpoint_step(c),
                rank=c.rank,
                alpha=c.alpha,
                config=LoraConfigOut(**c.config.model_dump()) if c.config else None,
            )
            for c in lora_library.scan(settings.lora_dir)
        ]

    @api.get("/api/loras/export", responses={200: {"content": {"application/zip": {}}}})
    def export_lora(id: str, name: str, base_model: str, half: bool = True) -> FileResponse:
        ckpt = found(lora_library.resolve, settings.lora_dir, id)
        path = checked(sharing.export_package, ckpt, name, base_model, half)
        return FileResponse(
            path, media_type="application/zip", filename=f"{name}.zip", background=BackgroundTask(path.unlink)
        )

    @api.post("/api/loras/import")
    def import_lora(package: Annotated[UploadFile, File()]) -> dict:
        path = save_upload(package)
        try:
            id, engine = sharing.import_package(settings.lora_dir, path, package.filename or "lora.zip")
            return {"id": id, "engine": engine}
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        finally:
            path.unlink(missing_ok=True)

    return api
