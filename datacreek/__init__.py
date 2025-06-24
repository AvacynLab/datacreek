# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Datacreek: tools for preparing synthetic data for LLM fine-tuning."""

__version__ = "0.0.1"

from .pipelines import (
    GenerationPipeline,
    TrainingGoal,
    DatasetType,
    PIPELINES,
    get_pipeline,
    get_trainings_for_dataset,
    get_dataset_types_for_training,
    get_pipelines_for_training,
)
from .core.knowledge_graph import KnowledgeGraph
from .core.dataset import DatasetBuilder
from .core.ingest import process_file as ingest_file, to_kg, ingest_into_dataset

__all__ = [
    "GenerationPipeline",
    "TrainingGoal",
    "DatasetType",
    "PIPELINES",
    "get_pipeline",
    "get_trainings_for_dataset",
    "get_dataset_types_for_training",
    "get_pipelines_for_training",
    "KnowledgeGraph",
    "DatasetBuilder",
    "ingest_file",
    "to_kg",
    "ingest_into_dataset",
]

