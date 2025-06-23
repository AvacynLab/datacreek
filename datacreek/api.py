from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from datacreek.core.ingest import process_file as ingest_file
from datacreek.core.create import process_file as generate_data
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.save_as import convert_format
from datacreek.db import SessionLocal, init_db, User, SourceData, Dataset
from datacreek.schemas import (
    UserCreate,
    UserOut,
    SourceCreate,
    SourceOut,
    GenerateParams,
    DatasetOut,
    CurateParams,
    SaveParams,
)
from datacreek.services import create_user, create_source, create_dataset

init_db()
app = FastAPI(title="Datacreek API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/users", response_model=UserOut, summary="Create a user")
def create_user_route(payload: UserCreate, db: Session = Depends(get_db)):
    user = create_user(db, payload.username, payload.api_key)
    return user


@app.post("/ingest", response_model=SourceOut, summary="Ingest a file")
def ingest(payload: SourceCreate, db: Session = Depends(get_db)):
    content = ingest_file(payload.path, None, payload.name, None)
    src = create_source(db, None, payload.path, content)
    return src


@app.post("/generate", response_model=DatasetOut, summary="Generate dataset from source")
def generate(params: GenerateParams, db: Session = Depends(get_db)):
    src = db.get(SourceData, params.src_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found")
    output_dir = Path("data/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = generate_data(
        None,
        str(output_dir),
        None,
        None,
        None,
        params.content_type,
        params.num_pairs,
        False,
        document_text=src.content,
    )
    ds = create_dataset(db, None, params.src_id, out)
    return ds


@app.post("/curate", response_model=DatasetOut, summary="Curate a dataset")
def curate(params: CurateParams, db: Session = Depends(get_db)):
    ds = db.get(Dataset, params.ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    output_path = Path(ds.path).with_name(Path(ds.path).stem + "_curated.json")
    curate_qa_pairs(ds.path, str(output_path), params.threshold, None, None, None, False)
    ds.path = str(output_path)
    db.commit()
    db.refresh(ds)
    return ds


@app.post("/save", response_model=DatasetOut, summary="Convert dataset format")
def save(params: SaveParams, db: Session = Depends(get_db)):
    ds = db.get(Dataset, params.ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    out = convert_format(ds.path, None, params.fmt, {}, "json")
    ds.path = out
    db.commit()
    db.refresh(ds)
    return ds

