"""The LoRA / voice list, export / import of shareable packages, and GPT-SoVITS
voices (import of webui weights, default reference clips)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .. import engines, lora_library, sharing
from ..data import manifests, references
from ..engines.fish_s2 import lora as s2lora
from ..engines.gpt_sovits import voices
from ..prep import importer
from .context import Studio, checked, found, save_upload
from .schemas import GptSovitsImportIn, LoraConfigOut, LoraOut


def keeps_reference(engine: str) -> bool:
    """Whether the engine's voices keep a default reference clip."""
    cls = engines.classes().get(engine)
    return bool(cls and cls.capabilities.voice_reference)


def router(s: Studio) -> APIRouter:
    api = APIRouter()
    settings = s.settings

    def reference_text(ckpt: lora_library.LoraCheckpoint) -> str | None:
        ref = voices.read_reference(ckpt.path) if keeps_reference(ckpt.engine) else None
        return ref[1] if ref else None

    @api.get("/api/loras")
    def list_loras() -> list[LoraOut]:
        return [
            LoraOut(
                id=c.id,
                engine=c.engine,
                voice_path=c.voice_path,
                voice=c.voice,
                base_model=c.base_model,
                base_model_id=sharing.hub_id(c.base_model),
                step=sharing.checkpoint_step(c),
                rank=c.rank,
                alpha=c.alpha,
                config=LoraConfigOut(**c.config.model_dump()) if c.config else None,
                reference_text=reference_text(c),
                version=c.version,
            )
            for c in lora_library.scan(settings.lora_dir)
        ]

    @api.get("/api/loras/export", responses={200: {"content": {"application/zip": {}}}})
    def export_lora(id: str, name: str, base_model: str = "", half: bool = True) -> FileResponse:
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

    def voice_ckpt(id: str) -> lora_library.LoraCheckpoint:
        ckpt = found(lora_library.resolve, settings.lora_dir, id)
        if not keeps_reference(ckpt.engine):
            raise HTTPException(422, "This voice keeps no reference clip.")
        return ckpt

    def dataset_reference(dataset: str) -> tuple[Path, str]:
        manifest = found(manifests.resolve, settings.data_dir, dataset)
        return checked(voices.pick_reference, manifest, manifests.legacy_root(settings.data_dir))

    @api.post("/api/loras/import-gpt-sovits")
    def import_gpt_sovits(body: GptSovitsImportIn) -> dict:
        """A voice from the GPT-SoVITS webui's trained weights (v2Pro / v2ProPlus)."""
        if body.dataset:
            reference, origin = dataset_reference(body.dataset), f"dataset {body.dataset}"
        elif body.reference:
            reference, origin = (importer.clean_path(body.reference), body.reference_text), "file"
        else:
            reference, origin = None, ""  # added later in the Voices panel
        try:
            name = voices.import_webui(
                lora_library.engine_dir(settings.lora_dir, voices.ENGINE),
                body.name.strip(),
                importer.clean_path(body.sovits),
                importer.clean_path(body.gpt),
                reference,
                origin,
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        return {"id": f"{voices.ENGINE}/{name}", "engine": voices.ENGINE}

    @api.get("/api/loras/reference")
    def voice_reference(id: str) -> FileResponse:
        """A voice's default reference clip (GPT-SoVITS, S2-Pro)."""
        ref = voices.read_reference(voice_ckpt(id).path)
        if ref is None:
            raise HTTPException(404, "This voice has no reference clip.")
        return FileResponse(ref[0], media_type="audio/wav")

    @api.post("/api/loras/reference")
    def set_voice_reference(
        id: Annotated[str, Form()],
        text: Annotated[str, Form()] = "",
        # Pick from this dataset (manifest id) instead of an uploaded clip.
        dataset: Annotated[str | None, Form()] = None,
        audio: Annotated[UploadFile | None, File()] = None,
    ) -> LoraOut:
        """Change a voice's default reference clip: GPT-SoVITS takes the dataset's best
        clip (3-10 s), S2-Pro its best few joined (as training picked them)."""
        ckpt = voice_ckpt(id)
        folder, s2 = ckpt.path, ckpt.engine == s2lora.ENGINE
        if dataset:
            if s2:
                manifest = found(manifests.resolve, settings.data_dir, dataset)
                root = manifests.legacy_root(settings.data_dir)
                picked = checked(references.pick, manifest, s2lora.REFERENCE_CLIPS, root)
                checked(s2lora.write_reference_clips, folder, picked, f"dataset {dataset}")
            else:
                clip, transcript = dataset_reference(dataset)
                checked(voices.write_reference, folder, clip, transcript, f"dataset {dataset}")
        elif audio is not None:
            path = save_upload(audio)
            try:  # any format audio_io reads
                checked(s2lora.write_reference_file if s2 else voices.write_reference, folder, path, text, "upload")
            finally:
                path.unlink(missing_ok=True)
        else:
            raise HTTPException(422, "Give a clip, or a dataset to pick one from.")
        return next(lora for lora in list_loras() if lora.id == id)

    return api
