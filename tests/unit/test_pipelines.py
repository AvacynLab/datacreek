import pytest
from pathlib import Path

from datacreek import pipelines
from datacreek.models.qa import QAPair
from datacreek.models.results import QAGenerationResult


class DummyClient:
    def ping(self):
        return True


class BadClient:
    def ping(self):
        raise RuntimeError


def test_get_redis_client(monkeypatch):
    monkeypatch.setattr(pipelines, 'backend_get_redis_client', lambda: DummyClient())
    assert isinstance(pipelines.get_redis_client(), DummyClient)


def test_get_redis_client_none(monkeypatch):
    monkeypatch.setattr(pipelines, 'backend_get_redis_client', lambda: BadClient())
    assert pipelines.get_redis_client() is None


def test_load_pipelines_from_file(tmp_path):
    yaml_text = (
        'qa:\n  steps: [ingest, generate_qa]\n  trainings: [sft]\n  description: foo\n'
    )
    path = tmp_path / 'p.yaml'
    path.write_text(yaml_text)
    data = pipelines.load_pipelines_from_file(path)
    pl = data[pipelines.DatasetType.QA]
    assert pl.steps == [pipelines.PipelineStep.INGEST, pipelines.PipelineStep.GENERATE_QA]
    assert pl.description == 'foo'


def test_generation_options_model_to_options(tmp_path):
    conf = tmp_path / 'c.yaml'
    model = pipelines.GenerationOptionsModel(start_step='ingest', config_path=str(conf))
    opts = model.to_options()
    assert opts.start_step is pipelines.PipelineStep.INGEST
    assert opts.config_path == conf


def test_validate_step_result_success():
    res = QAGenerationResult(summary='s', qa_pairs=[QAPair('q', 'a')])
    assert pipelines._validate_step_result(
        pipelines.DatasetType.QA,
        pipelines.PipelineStep.GENERATE_QA,
        res,
    ) is res


def test_validate_step_result_missing_field():
    with pytest.raises(ValueError):
        pipelines._validate_step_result(
            pipelines.DatasetType.QA,
            pipelines.PipelineStep.GENERATE_QA,
            {},
        )


def test_pipeline_execution_error():
    err = pipelines.PipelineExecutionError(pipelines.PipelineStep.INGEST, ValueError('bad'))
    info = err.info.to_dict()
    assert info['step'] == 'ingest'
    assert info['exc_type'] == 'ValueError'


def test_training_lookups():
    types_ = pipelines.get_dataset_types_for_training(pipelines.TrainingGoal.SFT)
    assert pipelines.DatasetType.QA in types_
    pipelines_list = pipelines.get_pipelines_for_training(pipelines.TrainingGoal.SFT)
    assert all(isinstance(p, pipelines.GenerationPipeline) for p in pipelines_list)


def test_serialize_nested_dataclass():
    qa = QAPair('q', 'a')
    res = QAGenerationResult(summary='s', qa_pairs=[qa])
    assert pipelines._serialize(res) == {
        'summary': 's',
        'qa_pairs': [{
            'question': 'q',
            'answer': 'a',
            'rating': None,
            'confidence': None,
            'chunk': None,
            'source': None,
            'facts': None,
        }],
    }


def test_generation_options_invalid_step():
    with pytest.raises(ValueError):
        pipelines.GenerationOptionsModel(start_step='unknown').to_options()


def test_serialize_collections():
    qa = QAPair('x', 'y')
    data = {'items': [qa]}
    result = pipelines._serialize(data)
    assert result['items'][0]['question'] == 'x'


def test_validate_step_result_pref_pair():
    payload = {'pairs': [{'question': 'q', 'chosen': 'c', 'rejected': 'r'}]}
    assert pipelines._validate_step_result(
        pipelines.DatasetType.PREF_PAIR,
        pipelines.PipelineStep.GENERATE_CANDIDATES,
        payload,
    ) == payload


def test_validate_step_result_pref_pair_invalid():
    with pytest.raises(ValueError):
        pipelines._validate_step_result(
            pipelines.DatasetType.PREF_PAIR,
            pipelines.PipelineStep.GENERATE_CANDIDATES,
            {'pairs': [{'question': 'q', 'chosen': 'c'}]},
        )
