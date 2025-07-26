import datacreek.models.stage as stage
import datacreek.models.task_status as task_status


def test_dataset_stage_values():
    expected = [stage.DatasetStage.CREATED,
                stage.DatasetStage.INGESTED,
                stage.DatasetStage.GENERATED,
                stage.DatasetStage.CURATED,
                stage.DatasetStage.EXPORTED]
    assert [s.value for s in expected] == list(range(5))
    assert list(stage.DatasetStage) == expected
    assert stage.DatasetStage(0) is stage.DatasetStage.CREATED
    assert stage.DatasetStage['EXPORTED'] is stage.DatasetStage.EXPORTED


def test_task_status_roundtrip():
    mapping = {
        'INGESTING': 'ingesting',
        'GENERATING': 'generating',
        'CLEANUP': 'cleanup',
        'EXPORTING': 'exporting',
        'SAVING_NEO4J': 'saving_neo4j',
        'LOADING_NEO4J': 'loading_neo4j',
        'SAVING_REDIS_GRAPH': 'saving_redis_graph',
        'LOADING_REDIS_GRAPH': 'loading_redis_graph',
        'DELETING': 'deleting',
        'EXTRACTING_FACTS': 'extracting_facts',
        'EXTRACTING_ENTITIES': 'extracting_entities',
        'OPERATION': 'operation',
        'COMPLETED': 'completed',
        'FAILED': 'failed',
    }
    # ensure enumeration contains exactly these pairs
    assert {s.name: s.value for s in task_status.TaskStatus} == mapping
    # round-trip from value back to enum
    for name, value in mapping.items():
        assert task_status.TaskStatus(value) is getattr(task_status.TaskStatus, name)
