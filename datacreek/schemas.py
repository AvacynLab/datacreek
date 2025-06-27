from pydantic import BaseModel

from datacreek.config_models import GenerationSettingsModel


class UserCreate(BaseModel):
    username: str


class UserOut(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True


class SourceCreate(BaseModel):
    path: str
    name: str | None = None
    high_res: bool | None = False
    ocr: bool | None = False
    use_unstructured: bool | None = None
    extract_entities: bool | None = False
    extract_facts: bool | None = False


class SourceOut(BaseModel):
    id: int

    class Config:
        from_attributes = True


class GenerateParams(BaseModel):
    src_id: int
    content_type: str = "qa"
    num_pairs: int | None = None
    provider: str | None = None
    profile: str | None = None
    model: str | None = None
    api_base: str | None = None
    generation: GenerationSettingsModel | None = None
    prompts: dict | None = None


class DatasetOut(BaseModel):
    id: int
    path: str

    class Config:
        from_attributes = True


class CurateParams(BaseModel):
    ds_id: int
    threshold: float | None = None


class SaveParams(BaseModel):
    ds_id: int
    fmt: str = "jsonl"


class UserWithKey(UserOut):
    api_key: str


class DatasetCreate(BaseModel):
    source_id: int
    path: str


class DatasetUpdate(BaseModel):
    path: str | None = None
