from hashlib import sha256
from sqlalchemy.orm import Session

from datacreek.db import User, SourceData, Dataset


def hash_key(api_key: str) -> str:
    return sha256(api_key.encode()).hexdigest()


def create_user(db: Session, username: str, api_key: str) -> User:
    user = User(username=username, api_key=hash_key(api_key))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_source(db: Session, owner_id: int | None, path: str, content: str) -> SourceData:
    src = SourceData(owner_id=owner_id, path=path, content=content)
    db.add(src)
    db.commit()
    db.refresh(src)
    return src


def create_dataset(db: Session, owner_id: int | None, source_id: int, path: str) -> Dataset:
    ds = Dataset(owner_id=owner_id, source_id=source_id, path=path)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds
