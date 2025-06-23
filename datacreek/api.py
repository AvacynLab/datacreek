from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pathlib import Path

from datacreek.core.ingest import process_file as ingest_file
from datacreek.core.create import process_file as generate_data
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.save_as import convert_format
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./datacreek.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    api_key = Column(String, unique=True, index=True)

class SourceData(Base):
    __tablename__ = "sources"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    path = Column(String)

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    source_id = Column(Integer, ForeignKey("sources.id"))
    path = Column(String)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Datacreek API")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users", summary="Create a user")
def create_user(username: str, api_key: str, db: Session = Depends(get_db)):
    user = User(username=username, api_key=api_key)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id}

@app.post("/ingest", summary="Ingest a file")
def ingest(path: str, name: str | None = None, db: Session = Depends(get_db)):
    output_dir = Path("data/ingested")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = ingest_file(path, str(output_dir), name, None)
    src = SourceData(owner_id=None, path=out)
    db.add(src)
    db.commit()
    db.refresh(src)
    return {"id": src.id, "path": out}

@app.post("/generate", summary="Generate dataset from source")
def generate(src_id: int, content_type: str = "qa", num_pairs: int | None = None, db: Session = Depends(get_db)):
    src = db.get(SourceData, src_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found")
    output_dir = Path("data/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = generate_data(src.path, content_type, str(output_dir), None, None, None, num_pairs, False)
    ds = Dataset(owner_id=None, source_id=src_id, path=out)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"id": ds.id, "path": out}

@app.post("/curate", summary="Curate a dataset")
def curate(ds_id: int, threshold: float | None = None, db: Session = Depends(get_db)):
    ds = db.get(Dataset, ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    output_path = Path(ds.path).with_name(Path(ds.path).stem + "_curated.json")
    curate_qa_pairs(ds.path, str(output_path), threshold, None, None, None, False)
    ds.path = str(output_path)
    db.commit()
    return {"id": ds.id, "path": ds.path}

@app.post("/save", summary="Convert dataset format")
def save(ds_id: int, fmt: str = "jsonl", db: Session = Depends(get_db)):
    ds = db.get(Dataset, ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    out = convert_format(ds.path, None, fmt, {}, "json")
    ds.path = out
    db.commit()
    return {"id": ds.id, "path": out}

