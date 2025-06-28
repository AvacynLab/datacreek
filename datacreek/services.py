import json
import secrets
from hashlib import sha256

from sqlalchemy.orm import Session
from werkzeug.security import generate_password_hash

from datacreek.db import Dataset, SourceData, User


def hash_key(api_key: str) -> str:
    return sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Return a new random API key."""
    return secrets.token_hex(16)


def get_user_by_key(db: Session, api_key: str) -> User | None:
    hashed = hash_key(api_key)
    return db.query(User).filter_by(api_key=hashed).first()


def create_user(db: Session, username: str, api_key: str, password: str | None = None) -> User:
    user = User(
        username=username,
        api_key=hash_key(api_key),
        password_hash=generate_password_hash(password or ""),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_user_with_generated_key(
    db: Session, username: str, password: str | None = None
) -> tuple[User, str]:
    """Create a user and return the record along with the plain API key."""
    api_key = generate_api_key()
    user = create_user(db, username, api_key, password=password)
    return user, api_key


def create_source(
    db: Session,
    owner_id: int | None,
    path: str,
    content: str,
    *,
    entities: list[str] | None = None,
    facts: list[dict[str, str]] | None = None,
) -> SourceData:
    src = SourceData(
        owner_id=owner_id,
        path=path,
        content=content,
        entities=json.dumps(entities) if entities else None,
        facts=json.dumps(facts) if facts else None,
    )
    db.add(src)
    db.commit()
    db.refresh(src)
    return src


def create_dataset(
    db: Session,
    owner_id: int | None,
    source_id: int,
    *,
    path: str | None = None,
    content: str | None = None,
) -> Dataset:
    ds = Dataset(owner_id=owner_id, source_id=source_id, path=path or "", content=content)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds
