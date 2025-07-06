import json
from typing import Any, Literal

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Path, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from datacreek.backends import get_neo4j_driver, get_redis_client
from datacreek.core.dataset import MAX_NAME_LENGTH, NAME_PATTERN, DatasetBuilder
from datacreek.db import Dataset, SessionLocal, User, init_db
from datacreek.models.export_format import ExportFormat
from datacreek.schemas import (
    CurateParams,
    DatasetCreate,
    DatasetInit,
    DatasetName,
    DatasetOut,
    DatasetUpdate,
    GenerateParams,
    SaveParams,
    SourceCreate,
    SourceOut,
    UserCreate,
    UserOut,
    UserWithKey,
)
from datacreek.services import (
    _cache_dataset,
    create_dataset,
    create_user,
    create_user_with_generated_key,
    get_dataset_by_id,
    get_user_by_key,
)
from datacreek.tasks import (
    celery_app,
    curate_task,
    dataset_cleanup_task,
    dataset_delete_task,
    dataset_export_task,
    dataset_generate_task,
    dataset_ingest_task,
    generate_task,
    ingest_task,
    save_task,
)
from datacreek.utils import decode_hash

init_db()
app = FastAPI(title="Datacreek API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> User:
    user = get_user_by_key(db, api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


@app.post("/users", response_model=UserWithKey, summary="Create a user")
def create_user_route(payload: UserCreate, db: Session = Depends(get_db)):
    user, key = create_user_with_generated_key(db, payload.username)
    return {"id": user.id, "username": user.username, "api_key": key}


@app.get("/users/me/datasets", summary="List user's datasets")
def list_user_datasets(current_user: User = Depends(get_current_user)) -> list[str]:
    """Return dataset names owned by the authenticated user."""
    client = get_redis_client()
    if client is None:
        return []
    key = f"user:{current_user.id}:datasets"
    names = [n.decode() if isinstance(n, bytes) else n for n in client.smembers(key)]
    return sorted(names)


@app.get("/users/me/datasets/details", summary="List user's datasets with progress")
def list_user_datasets_details(current_user: User = Depends(get_current_user)) -> list[dict]:
    """Return datasets with their stage and progress information."""
    client = get_redis_client()
    if client is None:
        return []
    key = f"user:{current_user.id}:datasets"
    names = [n.decode() if isinstance(n, bytes) else n for n in client.smembers(key)]
    datasets: list[dict] = []
    for name in sorted(names):
        try:
            driver = get_neo4j_driver()
            ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
        except KeyError:
            continue
        finally:
            if driver:
                driver.close()

        progress_key = f"dataset:{name}:progress"
        progress = decode_hash(client.hgetall(progress_key))
        datasets.append({"name": name, "stage": ds.stage, "progress": progress})
    return datasets


def _load_dataset(name: str, current_user: User) -> DatasetBuilder:
    """Return ``DatasetBuilder`` for ``name`` if owned by ``current_user``."""

    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        driver = get_neo4j_driver()
        ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    finally:
        if driver:
            driver.close()
    if ds.owner_id not in {None, current_user.id}:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds.redis_client = client
    return ds


@app.post("/datasets/{name}", summary="Create persisted dataset", status_code=201)
def create_persisted_dataset(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    params: DatasetInit = Body(...),
    current_user: User = Depends(get_current_user),
):
    """Initialize an empty dataset in Redis and Neo4j."""

    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    key = f"dataset:{name}"
    if client.exists(key):
        raise HTTPException(status_code=409, detail="Dataset already exists")

    driver = get_neo4j_driver()
    ds = DatasetBuilder(params.dataset_type, name=name, redis_client=client, neo4j_driver=driver)
    ds.owner_id = current_user.id
    ds.history.append("Dataset created")
    ds._persist()

    client.sadd("datasets", name)
    client.sadd(f"user:{current_user.id}:datasets", name)
    if driver:
        driver.close()

    return {"name": name, "stage": ds.stage}


@app.post("/datasets", response_model=DatasetOut, summary="Add a dataset")
def add_dataset(
    payload: DatasetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = create_dataset(db, current_user.id, payload.source_id, payload.path)
    return ds


@app.delete("/datasets/{name}", summary="Delete persisted dataset", status_code=202)
def delete_persisted_dataset(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a persisted dataset from Redis and Neo4j."""

    _load_dataset(name, current_user)

    celery_task = dataset_delete_task.apply_async(args=[name, current_user.id])
    return {"task_id": celery_task.id}


@app.get("/datasets", response_model=list[DatasetOut], summary="List datasets")
def list_datasets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(Dataset).filter(Dataset.owner_id == current_user.id).all()


@app.get("/datasets/events", summary="Get global dataset events")
def global_dataset_events(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return recent dataset events across all datasets."""

    client = get_redis_client()
    if client is None:
        return []
    start = -(offset + limit)
    end = -offset - 1
    raw_events = client.lrange("dataset:events", start, end)
    events: list[dict] = []
    for raw in raw_events:
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            events.append(json.loads(raw))
        except Exception:
            continue
    events.reverse()
    return events


@app.get("/datasets/{ds_id}", response_model=DatasetOut, summary="Get dataset")
def get_dataset(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = get_dataset_by_id(db, ds_id)
    if not ds or ds.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@app.patch("/datasets/{ds_id}", response_model=DatasetOut, summary="Update dataset")
def update_dataset(
    ds_id: int,
    payload: DatasetUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = get_dataset_by_id(db, ds_id)
    if not record or record.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = db.get(Dataset, ds_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if payload.path:
        ds.path = payload.path
    db.commit()
    db.refresh(ds)
    _cache_dataset(ds)
    return ds


@app.delete("/datasets/{ds_id}", summary="Delete dataset")
def delete_dataset_route(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = get_dataset_by_id(db, ds_id)
    if not ds or ds.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(ds)
    db.commit()
    return {"status": "deleted"}


@app.get("/datasets/{name}/history", summary="Get dataset event history")
def dataset_history(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return stored dataset events from Redis."""
    ds = _load_dataset(name, current_user)
    client = ds.redis_client
    events: list[dict] = []
    key = f"dataset:{name}:events"
    for raw in client.lrange(key, 0, -1):
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            events.append(json.loads(raw))
        except Exception:
            continue
    return events


@app.get("/datasets/{name}/versions", summary="List dataset versions")
def dataset_versions(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return generation versions stored for ``name``."""
    ds = _load_dataset(name, current_user)
    return [{"index": i + 1, **v} for i, v in enumerate(ds.versions)]


@app.get("/datasets/{name}/versions/{index}", summary="Get dataset version")
def dataset_version_item(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    index: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return a single generation version for ``name``."""
    ds = _load_dataset(name, current_user)
    if index < 1 or index > len(ds.versions):
        raise HTTPException(status_code=404, detail="Version not found")
    return ds.versions[index - 1]


@app.delete("/datasets/{name}/versions/{index}", summary="Delete dataset version")
def delete_dataset_version_item(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    index: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a stored generation version for ``name``."""

    ds = _load_dataset(name, current_user)
    try:
        ds.delete_version(index)
    except IndexError:
        raise HTTPException(status_code=404, detail="Version not found") from None
    return {"status": "deleted"}


@app.post("/datasets/{name}/versions/{index}/restore", summary="Restore dataset version")
def restore_dataset_version_item(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    index: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Copy version ``index`` to the end of the version list."""

    ds = _load_dataset(name, current_user)
    try:
        ds.restore_version(index)
    except IndexError:
        raise HTTPException(status_code=404, detail="Version not found") from None
    return {"status": "restored"}


@app.get("/datasets/{name}/progress", summary="Get dataset progress")
def dataset_progress(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return progress information stored in Redis."""
    ds = _load_dataset(name, current_user)
    client = ds.redis_client
    key = f"dataset:{name}:progress"
    return decode_hash(client.hgetall(key))


@app.get("/graphs/{name}/progress", summary="Get graph progress")
def graph_progress(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return progress information stored for a knowledge graph."""
    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    try:
        driver = get_neo4j_driver()
        ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    except KeyError:
        raise HTTPException(status_code=404, detail="Graph not found")
    finally:
        if driver:
            driver.close()
    if ds.owner_id not in {None, current_user.id}:
        raise HTTPException(status_code=404, detail="Graph not found")
    key = f"graph:{name}:progress"
    return decode_hash(client.hgetall(key))


@app.get("/datasets/{name}/progress/history", summary="Get progress history")
def dataset_progress_history(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return progress status history stored in Redis."""
    ds = _load_dataset(name, current_user)
    client = ds.redis_client
    key = f"dataset:{name}:progress:history"
    raw = client.lrange(key, 0, -1)
    history: list[dict] = []
    for item in raw:
        if isinstance(item, bytes):
            item = item.decode()
        try:
            history.append(json.loads(item))
        except Exception:
            continue
    return history


@app.get("/graphs/{name}/progress/history", summary="Get graph progress history")
def graph_progress_history(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return progress status history stored for a knowledge graph."""
    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    try:
        driver = get_neo4j_driver()
        ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    except KeyError:
        raise HTTPException(status_code=404, detail="Graph not found")
    finally:
        if driver:
            driver.close()
    if ds.owner_id not in {None, current_user.id}:
        raise HTTPException(status_code=404, detail="Graph not found")
    key = f"graph:{name}:progress:history"
    raw = client.lrange(key, 0, -1)
    history: list[dict] = []
    for item in raw:
        if isinstance(item, bytes):
            item = item.decode()
        try:
            history.append(json.loads(item))
        except Exception:
            continue
    return history


@app.post("/datasets/{name}/ingest", summary="Ingest file into dataset")
def dataset_ingest_route(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    payload: SourceCreate = Body(...),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule ingestion of a file into a persisted dataset."""

    ds = _load_dataset(name, current_user)
    celery_task = dataset_ingest_task.apply_async(
        args=[name, payload.path, current_user.id],
        kwargs={
            "doc_id": payload.name,
            "high_res": payload.high_res or False,
            "ocr": payload.ocr or False,
            "use_unstructured": payload.use_unstructured,
            "extract_entities": payload.extract_entities or False,
            "extract_facts": payload.extract_facts or False,
        },
    )
    return {"task_id": celery_task.id}


@app.post("/datasets/{name}/generate", summary="Generate dataset asynchronously")
def dataset_generate_route(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    params: dict | None = Body(None),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule the generation pipeline for a persisted dataset."""

    _load_dataset(name, current_user)

    celery_task = dataset_generate_task.apply_async(args=[name, params or {}, current_user.id])
    return {"task_id": celery_task.id}


@app.post("/datasets/{name}/cleanup", summary="Cleanup dataset asynchronously")
def dataset_cleanup_route(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    params: dict | None = Body(None),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule cleanup operations on a persisted dataset."""

    _load_dataset(name, current_user)

    celery_task = dataset_cleanup_task.apply_async(args=[name, params or {}, current_user.id])
    return {"task_id": celery_task.id}


@app.post("/datasets/{name}/export", summary="Export dataset asynchronously")
def dataset_export_task_route(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    fmt: ExportFormat = ExportFormat.JSONL,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule dataset export to the requested format."""

    _load_dataset(name, current_user)

    celery_task = dataset_export_task.apply_async(args=[name, fmt, current_user.id])
    return {"task_id": celery_task.id}


@app.get("/datasets/{name}/export", summary="Get exported dataset")
def dataset_export_result(
    name: DatasetName = Path(..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
    fmt: ExportFormat = ExportFormat.JSONL,
    current_user: User = Depends(get_current_user),
) -> Response:
    """Return previously exported dataset from Redis."""
    ds = _load_dataset(name, current_user)
    client = ds.redis_client
    progress_key = f"dataset:{name}:progress"
    progress = client.hget(progress_key, "export")
    key: str | None = None
    if progress:
        try:
            entry = json.loads(progress)
            key = entry.get("key")
        except Exception:
            pass
    if not key:
        key = f"dataset:{name}:export:{fmt.value if isinstance(fmt, ExportFormat) else fmt}"
    data = client.get(key)
    if data is None:
        raise HTTPException(status_code=404, detail="Export not found")
    if isinstance(data, bytes):
        data = data.decode()
    filename = f"{name}.{fmt.value if isinstance(fmt, ExportFormat) else fmt}"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(data, media_type="application/json", headers=headers)


@app.get("/datasets/{ds_id}/download", summary="Download dataset")
def download_dataset(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = get_dataset_by_id(db, ds_id)
    if not ds or ds.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    filename = f"dataset_{ds_id}.json"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(ds.content or "{}", media_type="application/json", headers=headers)


# ----- Asynchronous task endpoints -----


@app.post("/tasks/ingest", summary="Ingest a file asynchronously")
async def ingest_async(
    payload: SourceCreate,
    current_user: User = Depends(get_current_user),
) -> dict:
    celery_task = ingest_task.apply_async(
        args=[current_user.id, payload.path],
        kwargs={
            "high_res": payload.high_res or False,
            "ocr": payload.ocr or False,
            "use_unstructured": payload.use_unstructured,
            "extract_entities": payload.extract_entities or False,
            "extract_facts": payload.extract_facts or False,
        },
    )
    return {"task_id": celery_task.id}


@app.post("/tasks/generate", summary="Generate dataset asynchronously")
async def generate_async(
    params: GenerateParams,
    current_user: User = Depends(get_current_user),
    x_config_path: str | None = Header(None, alias="X-Config-Path"),
) -> dict:
    celery_task = generate_task.apply_async(
        args=[current_user.id, params.src_id, params.content_type, params.num_pairs],
        kwargs={
            "provider": params.provider,
            "profile": params.profile,
            "model": params.model,
            "api_base": params.api_base,
            "config_path": x_config_path,
            "generation": (
                params.generation.model_dump(exclude_defaults=True, exclude_none=True)
                if params.generation
                else None
            ),
            "prompts": params.prompts,
        },
    )
    return {"task_id": celery_task.id}


@app.post("/tasks/curate", summary="Curate a dataset asynchronously")
async def curate_async(
    params: CurateParams,
    current_user: User = Depends(get_current_user),
) -> dict:
    celery_task = curate_task.apply_async(args=[current_user.id, params.ds_id, params.threshold])
    return {"task_id": celery_task.id}


@app.post("/tasks/save", summary="Convert dataset asynchronously")
async def save_async(
    params: SaveParams,
    current_user: User = Depends(get_current_user),
) -> dict:
    celery_task = save_task.apply_async(args=[current_user.id, params.ds_id, params.fmt.value])
    return {"task_id": celery_task.id}


@app.get("/tasks/{task_id}", summary="Get background task status")
async def get_task(task_id: str) -> dict:
    res = celery_app.AsyncResult(task_id)
    if res.state in {"PENDING", "STARTED"}:
        return {"status": "running"}
    if res.state == "SUCCESS":
        return {"status": "finished", "result": res.result}
    if res.state == "FAILURE":
        return {"status": "failed", "error": str(res.result)}
    return {"status": res.state.lower()}
