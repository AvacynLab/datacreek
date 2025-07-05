from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import redis
from celery import Celery
from neo4j import GraphDatabase

from datacreek.core.create import process_file as generate_data
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.ingest import IngestOptions
from datacreek.core.ingest import process_file as ingest_file
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.core.save_as import convert_format
from datacreek.db import Dataset, SessionLocal, SourceData
from datacreek.models.export_format import ExportFormat
from datacreek.models.llm_client import LLMClient
from datacreek.models.task_status import TaskStatus
from datacreek.schemas import DatasetName
from datacreek.services import create_dataset, create_source
from datacreek.utils import extract_entities as extract_entities_func
from datacreek.utils import extract_facts as extract_facts_func
from datacreek.utils import load_config
from datacreek.utils.config import get_neo4j_config, get_redis_config, load_config_with_overrides

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


def get_redis_client() -> redis.Redis:
    """Return a Redis client based on configuration or environment variables."""
    cfg = get_redis_config(load_config_with_overrides(None))
    host = os.getenv("REDIS_HOST", cfg.get("host"))
    port = int(os.getenv("REDIS_PORT", cfg.get("port", 6379)))
    return redis.Redis(host=host, port=port, decode_responses=True)


def get_neo4j_driver():
    """Return a Neo4j driver using config or environment variables."""
    cfg = get_neo4j_config(load_config_with_overrides(None))
    uri = os.getenv("NEO4J_URI", cfg.get("uri"))
    user = os.getenv("NEO4J_USER", cfg.get("user"))
    password = os.getenv("NEO4J_PASSWORD", cfg.get("password"))
    if not uri or not user or not password:
        return None
    return GraphDatabase.driver(uri, auth=(user, password))


def _update_status(
    client: redis.Redis,
    key: str,
    status: TaskStatus | str,
    progress: float | None = None,
) -> None:
    """Set progress status information and record the timeline."""

    val = status.value if isinstance(status, TaskStatus) else status
    entry = {"status": val}
    if progress is not None:
        entry["progress"] = progress
    entry["time"] = datetime.now(timezone.utc).isoformat()
    try:
        client.hset(key, "status", val)
        if progress is not None:
            client.hset(key, "progress", progress)
        client.rpush(f"{key}:history", json.dumps(entry))
    except Exception:
        logger.exception("Failed to update task status for %s", key)


def _record_error(client: redis.Redis, key: str, exc: Exception) -> None:
    """Save task failure details in ``client`` under ``key``."""
    entry = {
        "status": TaskStatus.FAILED.value,
        "error": str(exc),
        "time": datetime.now(timezone.utc).isoformat(),
    }
    try:
        client.hset(key, mapping={"error": str(exc), "status": TaskStatus.FAILED.value})
        client.rpush(f"{key}:history", json.dumps(entry))
    except Exception:
        logger.exception("Failed to record error for %s", key)


@celery_app.task
def ingest_task(
    user_id: int,
    path: str,
    *,
    high_res: bool = False,
    ocr: bool = False,
    use_unstructured: bool | None = None,
    extract_entities: bool = False,
    extract_facts: bool = False,
) -> dict:
    with SessionLocal() as db:
        content = ingest_file(path, high_res=high_res, ocr=ocr, use_unstructured=use_unstructured)
        ents = extract_entities_func(content) if extract_entities else None
        facts = extract_facts_func(content) if extract_facts else None
        src = create_source(
            db,
            user_id,
            path,
            content,
            entities=ents,
            facts=facts,
        )
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
        from datacreek.utils.config import load_config_with_overrides

        overrides = {}
        if generation is not None:
            overrides["generation"] = generation
        if prompts is not None:
            overrides["prompts"] = prompts
        load_path = str(config_path) if config_path else None
        load_overrides = overrides if overrides else None
        load_config_with_overrides(load_path, load_overrides)
        out = generate_data(
            None,
            Path(config_path) if config_path else None,
            api_base,
            model,
            content_type,
            num_pairs,
            False,
            provider=provider,
            profile=profile,
            kg=KnowledgeGraph(),
            document_text=src.content,
            config_overrides=overrides if overrides else None,
        )
        ds = create_dataset(db, user_id, src_id, content=json.dumps(out))
        return {"id": ds.id}


@celery_app.task
def curate_task(user_id: int, ds_id: int, threshold: float | None) -> dict:
    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        if not ds or ds.owner_id != user_id:
            raise RuntimeError("Dataset not found")
        data = json.loads(ds.content or "{}")
        result = curate_qa_pairs(data, threshold, None, None, None, False, async_mode=False)
        ds.content = json.dumps(result)
        db.commit()
        db.refresh(ds)
        return {"id": ds.id}


@celery_app.task
def save_task(user_id: int, ds_id: int, fmt: ExportFormat) -> dict:
    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        if not ds or ds.owner_id != user_id:
            raise RuntimeError("Dataset not found")
        data = json.loads(ds.content or "{}")
        if isinstance(fmt, str):
            fmt = ExportFormat(fmt)
        out = convert_format(data, fmt.value, {}, "json")
        ds.content = out if isinstance(out, str) else json.dumps(out)
        db.commit()
        db.refresh(ds)
        return {"id": ds.id}


@celery_app.task
def dataset_ingest_task(name: DatasetName, path: str, user_id: int | None = None, **kwargs) -> dict:
    """Ingest a file into a persisted dataset."""
    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    opt_fields = IngestOptions.__dataclass_fields__.keys()
    opt_args = {k: kwargs.pop(k) for k in list(kwargs) if k in opt_fields}
    options = IngestOptions(**opt_args) if opt_args else None
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "ingest_start", start_ts)
    _update_status(client, key, TaskStatus.INGESTING, 0.0)
    if opt_args:
        client.hset(key, "ingestion_params", json.dumps(opt_args))
    try:
        doc_id = ds.ingest_file(path, options=options, **kwargs)
        _update_status(client, key, TaskStatus.INGESTING, 0.5)
        client.hincrby(key, "ingested", 1)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            f"ingested:{doc_id}",
            json.dumps({"path": path, "time": ts}),
        )
        client.hset(
            key,
            "last_ingested",
            json.dumps({"id": doc_id, "path": path, "time": ts}),
        )
        client.hset(key, "ingest_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage, "events": len(ds.events)}
    except Exception as exc:
        _record_error(client, key, exc)
        raise
    finally:
        if driver:
            driver.close()


@celery_app.task
def dataset_generate_task(
    name: DatasetName, params: dict | None = None, user_id: int | None = None
) -> dict:
    """Run the post-KG pipeline for a persisted dataset."""
    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    params = params or {}
    key = f"dataset:{name}:progress"
    ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "generate_start", ts)
    _update_status(client, key, TaskStatus.GENERATING, 0.0)
    if params:
        client.hset(key, "generation_params", json.dumps(params))
    try:
        ds.run_post_kg_pipeline(redis_client=client, **params)
        _update_status(client, key, TaskStatus.GENERATING, 0.5)
        end_ts = datetime.now(timezone.utc).isoformat()
        client.hset(key, "generated_version", json.dumps(len(ds.versions)))
        client.hset(key, "generate_finish", end_ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage, "versions": len(ds.versions)}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def dataset_cleanup_task(name: str, params: dict | None = None, user_id: int | None = None) -> dict:
    """Run cleanup operations on a persisted dataset."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    params = params or {}
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "cleanup_start", start_ts)
    _update_status(client, key, TaskStatus.CLEANUP, 0.0)
    if params:
        client.hset(key, "cleanup_params", json.dumps(params))

    try:
        removed, cleaned = ds.cleanup_graph(**params)
        _update_status(client, key, TaskStatus.CLEANUP, 0.5)

        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "cleanup",
            json.dumps({"removed": removed, "cleaned": cleaned, "time": ts}),
        )
        client.hset(key, "cleanup_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)

        return {"stage": ds.stage, "removed": removed, "cleaned": cleaned}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def dataset_export_task(
    name: DatasetName, fmt: ExportFormat = ExportFormat.JSONL, user_id: int | None = None
) -> dict:
    """Format the latest generation result and mark the dataset exported."""

    if isinstance(fmt, str):
        fmt = ExportFormat(fmt)
    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    data = None
    key: str
    progress_key = f"dataset:{name}:progress"
    _update_status(client, progress_key, TaskStatus.EXPORTING, 0.0)
    try:
        if ds.versions and ds.versions[-1].get("result") is not None:
            data = ds.versions[-1]["result"]
            formatted = convert_format(data, fmt.value, {}, "json")
            _update_status(client, progress_key, TaskStatus.EXPORTING, 0.5)
            key = f"dataset:{name}:export:{fmt.value}"
            client.set(key, formatted if isinstance(formatted, str) else json.dumps(formatted))
        else:
            key = f"dataset:{name}:export:json"
            client.set(key, json.dumps(ds.to_dict()))

        ds.mark_exported()
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            progress_key,
            "export",
            json.dumps({"fmt": fmt.value, "key": key, "time": ts}),
        )
        _update_status(client, progress_key, TaskStatus.COMPLETED, 1.0)

        return {"stage": ds.stage, "key": key}
    except Exception as exc:
        _record_error(client, progress_key, exc)
        raise


@celery_app.task
def dataset_save_neo4j_task(name: DatasetName, user_id: int | None = None) -> dict:
    """Persist the dataset graph to Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "save_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.0)
    try:
        ds.save_neo4j(driver)
        _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.5)
        driver.close()
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "save_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "save_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def dataset_load_neo4j_task(name: DatasetName, user_id: int | None = None) -> dict:
    """Load the dataset graph from Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "load_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.0)
    try:
        ds.load_neo4j(driver)
        _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.5)
        driver.close()
        ds.neo4j_driver = None
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "load_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "load_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def dataset_operation_task(
    name: DatasetName, operation: str, params: dict | None = None, user_id: int | None = None
) -> dict:
    """Run an arbitrary dataset method and persist the result."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    params = params or {}
    func = getattr(ds, operation)
    prog_key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(prog_key, f"{operation}_start", start_ts)
    client.hset(prog_key, "operation", operation)
    _update_status(client, prog_key, TaskStatus.OPERATION, 0.0)
    try:
        result = func(**params)
        _update_status(client, prog_key, TaskStatus.OPERATION, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        try:
            client.hset(
                prog_key,
                operation,
                json.dumps({"result": result, "time": ts}, default=str),
            )
        except Exception:
            client.hset(prog_key, operation, json.dumps({"time": ts}))
        _update_status(client, prog_key, TaskStatus.COMPLETED, 1.0)
        return {"result": result, "stage": ds.stage}
    except Exception as exc:
        _record_error(client, prog_key, exc)
        raise
    finally:
        if driver:
            driver.close()


@celery_app.task
def dataset_extract_facts_task(
    name: DatasetName,
    provider: str | None = None,
    profile: str | None = None,
    user_id: int | None = None,
) -> dict:
    """Run fact extraction asynchronously."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    llm_client = None
    if provider or profile:
        llm_client = LLMClient(provider=provider, profile=profile)
    key = f"dataset:{name}:progress"
    _update_status(client, key, TaskStatus.EXTRACTING_FACTS, 0.0)
    try:
        ds.extract_facts(llm_client)
        _update_status(client, key, TaskStatus.EXTRACTING_FACTS, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "extract_facts",
            json.dumps({"time": ts, "done": True}),
        )
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def dataset_extract_entities_task(
    name: DatasetName, model: str | None = None, user_id: int | None = None
) -> dict:
    """Run NER asynchronously."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    key = f"dataset:{name}:progress"
    _update_status(client, key, TaskStatus.EXTRACTING_ENTITIES, 0.0)
    try:
        ds.extract_entities(model=model)
        _update_status(client, key, TaskStatus.EXTRACTING_ENTITIES, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "extract_entities",
            json.dumps({"time": ts, "done": True}),
        )
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def dataset_delete_task(name: DatasetName, user_id: int | None = None) -> dict:
    """Remove a dataset from Redis and Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = None
    try:
        ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    except KeyError:
        pass
    if user_id is not None and ds and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    prog_key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(prog_key, "delete_start", start_ts)
    _update_status(client, prog_key, TaskStatus.DELETING, 0.0)
    try:
        for key in list(client.scan_iter(match=f"dataset:{name}*")):
            k = key.decode() if isinstance(key, bytes) else key
            if k != prog_key:
                client.delete(key)
        client.srem("datasets", name)
        if ds and ds.owner_id is not None:
            client.srem(f"user:{ds.owner_id}:datasets", name)
        _update_status(client, prog_key, TaskStatus.DELETING, 0.5)

        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    session.run(
                        "MATCH (n {dataset:$dataset}) DETACH DELETE n",
                        dataset=name,
                    )
            finally:
                driver.close()
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            prog_key,
            "delete",
            json.dumps({"deleted": True, "time": ts}),
        )
        client.hset(prog_key, "delete_finish", ts)
        client.hset(
            f"graph:{name}:progress",
            "delete",
            json.dumps({"deleted": True, "time": ts}),
        )
        _update_status(client, prog_key, TaskStatus.COMPLETED, 1.0)

        return {"deleted": name}
    except Exception as exc:
        _record_error(client, prog_key, exc)
        raise


@celery_app.task
def graph_save_neo4j_task(name: str, user_id: int | None = None) -> dict:
    """Persist a knowledge graph to Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"graph:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "save_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.0)
    try:
        ds.save_neo4j(driver)
        _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.5)
        driver.close()
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "save_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "save_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def graph_load_neo4j_task(name: str, user_id: int | None = None) -> dict:
    """Load a knowledge graph from Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"graph:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "load_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.0)
    try:
        ds.load_neo4j(driver)
        _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.5)
        driver.close()
        ds.neo4j_driver = None
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "load_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "load_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except Exception as exc:
        _record_error(client, key, exc)
        raise


@celery_app.task
def graph_delete_task(name: str, user_id: int | None = None) -> dict:
    """Remove a knowledge graph from Redis and Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    prog_key = f"graph:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(prog_key, "delete_start", start_ts)
    _update_status(client, prog_key, TaskStatus.DELETING, 0.0)
    try:
        keys = list(client.scan_iter(match=f"graph:{name}*"))
        for key in keys:
            k = key.decode() if isinstance(key, bytes) else key
            if k != prog_key:
                client.delete(key)
        client.srem("graphs", name)
        if ds.owner_id is not None:
            client.srem(f"user:{ds.owner_id}:graphs", name)
        _update_status(client, prog_key, TaskStatus.DELETING, 0.5)

        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    session.run(
                        "MATCH (n {dataset:$dataset}) DETACH DELETE n",
                        dataset=name,
                    )
            finally:
                driver.close()

        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            prog_key,
            "delete",
            json.dumps({"deleted": True, "time": ts}),
        )
        client.hset(prog_key, "delete_finish", ts)
        _update_status(client, prog_key, TaskStatus.COMPLETED, 1.0)

        return {"deleted": name}
    except Exception as exc:
        _record_error(client, prog_key, exc)
        raise
