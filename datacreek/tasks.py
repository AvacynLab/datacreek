from __future__ import annotations

import os
from pathlib import Path

from celery import Celery

from datacreek.core.create import process_file as generate_data
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.ingest import process_file as ingest_file
from datacreek.core.save_as import convert_format
from datacreek.db import Dataset, SessionLocal, SourceData
from datacreek.services import create_dataset, create_source

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "memory://")
CELERY_BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "cache+memory://")

celery_app = Celery("datacreek", broker=CELERY_BROKER_URL, backend=CELERY_BACKEND_URL)
celery_app.conf.accept_content = ["json"]
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.task_always_eager = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false").lower() in {
    "1",
    "true",
}
celery_app.conf.task_store_eager_result = True


@celery_app.task
def ingest_task(user_id: int, path: str) -> dict:
    with SessionLocal() as db:
        content = ingest_file(path)
        src = create_source(db, user_id, path, content)
        return {"id": src.id}


@celery_app.task
def generate_task(
    user_id: int,
    src_id: int,
    content_type: str,
    num_pairs: int | None,
    *,
    provider: str | None = None,
    profile: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    config_path: str | None = None,
    generation: dict | None = None,
    prompts: dict | None = None,
) -> dict:
    with SessionLocal() as db:
        src = db.get(SourceData, src_id)
        if not src or src.owner_id != user_id:
            raise RuntimeError("Source not found")
        output_dir = Path("data/generated")
        output_dir.mkdir(parents=True, exist_ok=True)
        overrides = {}
        if generation is not None:
            overrides["generation"] = generation
        if prompts is not None:
            overrides["prompts"] = prompts
        out = generate_data(
            None,
            str(output_dir),
            Path(config_path) if config_path else None,
            api_base,
            model,
            content_type,
            num_pairs,
            False,
            provider=provider,
            profile=profile,
            document_text=src.content,
            config_overrides=overrides if overrides else None,
        )
        ds = create_dataset(db, user_id, src_id, out)
        return {"id": ds.id}


@celery_app.task
def curate_task(user_id: int, ds_id: int, threshold: float | None) -> dict:
    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        if not ds or ds.owner_id != user_id:
            raise RuntimeError("Dataset not found")
        output_path = Path(ds.path).with_name(Path(ds.path).stem + "_curated.json")
        curate_qa_pairs(ds.path, str(output_path), threshold, None, None, None, False)
        ds.path = str(output_path)
        db.commit()
        db.refresh(ds)
        return {"id": ds.id}


@celery_app.task
def save_task(user_id: int, ds_id: int, fmt: str) -> dict:
    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        if not ds or ds.owner_id != user_id:
            raise RuntimeError("Dataset not found")
        out = convert_format(ds.path, None, fmt, {}, "json")
        ds.path = out
        db.commit()
        db.refresh(ds)
        return {"id": ds.id}
