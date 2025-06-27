from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from datacreek.db import Dataset, SessionLocal, User, init_db
from datacreek.schemas import (
    CurateParams,
    DatasetCreate,
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
    create_dataset,
    create_user,
    create_user_with_generated_key,
    get_user_by_key,
)
from datacreek.tasks import celery_app, curate_task, generate_task, ingest_task, save_task

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


@app.post("/datasets", response_model=DatasetOut, summary="Add a dataset")
def add_dataset(
    payload: DatasetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = create_dataset(db, current_user.id, payload.source_id, payload.path)
    return ds


@app.get("/datasets", response_model=list[DatasetOut], summary="List datasets")
def list_datasets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(Dataset).filter(Dataset.owner_id == current_user.id).all()


@app.get("/datasets/{ds_id}", response_model=DatasetOut, summary="Get dataset")
def get_dataset(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, ds_id)
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
    ds = db.get(Dataset, ds_id)
    if not ds or ds.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if payload.path:
        ds.path = payload.path
    db.commit()
    db.refresh(ds)
    return ds


@app.delete("/datasets/{ds_id}", summary="Delete dataset")
def delete_dataset_route(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, ds_id)
    if not ds or ds.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(ds)
    db.commit()
    return {"status": "deleted"}


@app.get("/datasets/{ds_id}/download", summary="Download dataset")
def download_dataset(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, ds_id)
    if not ds or ds.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return FileResponse(ds.path)


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
    celery_task = save_task.apply_async(args=[current_user.id, params.ds_id, params.fmt])
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
