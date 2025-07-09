"""Datacreek package."""

from typing import TYPE_CHECKING

__version__ = "0.0.2"

# Avoid heavy imports at module load time. Relevant classes and functions
# are available via submodules such as ``datacreek.pipelines`` or
# ``datacreek.core``.

__all__: list[str] = [
    "__version__",
    "DatasetBuilder",
    "Atom",
    "DatasetType",
    "GenerationPipeline",
    "TrainingGoal",
    "get_pipeline",
    "get_trainings_for_dataset",
    "get_dataset_types_for_training",
    "get_pipelines_for_training",
    "ingest_file",
    "to_kg",
    "ingest_into_dataset",
    "extract_facts",
    "KnowledgeGraph",
    "GenerationSettings",
    "run_generation_pipeline",
    "run_generation_pipeline_async",
    "curate_qa_pairs",
    "curate_qa_pairs_async",
    "filter_rated_pairs",
    "apply_curation_threshold",
    "box_cover",
    "box_counting_dimension",
    "persistence_entropy",
    "graphwave_embedding",
    "minimize_bottleneck_distance",
    "bottleneck_distance",
    "mdl_optimal_radius",
    "persistence_diagrams",
    "spectral_dimension",
    "laplacian_spectrum",
    "spectral_entropy",
    "spectral_gap",
    "laplacian_energy",
    "lacunarity",
    "poincare_embedding",
    "generate_graph_rnn_like",
    "spectral_density",
    "graph_fourier_transform",
    "compute_hyperbolic_hypergraph_embeddings",
    "inverse_graph_fourier_transform",
    "generate_graph_rnn_stateful",
    "generate_graph_rnn_sequential",
    "generate_netgan_like",
    "graph_information_bottleneck",
    "graph_entropy",
    "subgraph_entropy",
    "prototype_subgraph",
    "sheaf_laplacian",
    "sheaf_convolution",
    "sheaf_neural_network",
    "sheaf_first_cohomology",
    "resolve_sheaf_obstruction",
    "fractal_information_density",
    "diversification_score",
    "hyperbolic_neighbors",
    "compute_distmult_embeddings",
    "hyperbolic_reasoning",
    "hyperbolic_hypergraph_reasoning",
    "hyperbolic_multi_curvature_reasoning",
    "neighborhood_to_sentence",
    "subgraph_to_text",
    "graph_to_text",
    "gds_quality_check",
    "quality_check",
    "validate_topology",
    "optimize_topology_iterative",
    "md5_file",
    "caption_image",
    "detect_emotion",
    "detect_modality",
    "fractalize_graph",
    "fractalize_optimal",
    "build_fractal_hierarchy",
    "build_mdl_hierarchy",
    "PromptTemplate",
    "get_template",
    "validate_output",
    "betti_number",
    "coverage_stats",
    "invariants_dashboard",
    "mapper_nerve",
    "inverse_mapper",
    "fractal_net_prune",
    "graphwave_entropy",
    "embedding_entropy",
    "embedding_box_counting_dimension",
    "ensure_graphwave_entropy",
    "hyper_sagnn_embeddings",
    "select_mdl_motifs",
    "mdl_description_length",
    "detect_automorphisms",
    "automorphism_group_order",
    "quotient_by_symmetry",
    "prune_embeddings",
    "ingestion_layer",
    "quality_layer",
    "fractal_layer",
    "embedding_layer",
    "generation_layer",
    "compression_layer",
    "topological_perception_layer",
    "information_layer",
    "export_layer",
    "orchestrator",
]

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .analysis.fractal import (
        bottleneck_distance,
        box_counting_dimension,
        box_cover,
        build_fractal_hierarchy,
        build_mdl_hierarchy,
        fractal_information_density,
        fractalize_graph,
        fractalize_optimal,
        graph_fourier_transform,
        graph_lacunarity,
        graphwave_embedding,
        inverse_graph_fourier_transform,
        laplacian_energy,
        laplacian_spectrum,
        mdl_optimal_radius,
        minimize_bottleneck_distance,
        persistence_diagrams,
        persistence_entropy,
        poincare_embedding,
        spectral_density,
        spectral_dimension,
        spectral_entropy,
        spectral_gap,
    )
    from .analysis.generation import (
        generate_graph_rnn_like,
        generate_graph_rnn_sequential,
        generate_graph_rnn_stateful,
        generate_netgan_like,
    )
    from .analysis.hypergraph import hyper_sagnn_embeddings
    from .config_models import GenerationSettings
    from .core.dataset import DatasetBuilder
    from .core.ingest import ingest_into_dataset
    from .core.ingest import process_file as ingest_file
    from .core.ingest import to_kg
    from .core.knowledge_graph import KnowledgeGraph
    from .pipelines import (
        PIPELINES,
        DatasetType,
        GenerationPipeline,
        TrainingGoal,
        get_dataset_types_for_training,
        get_pipeline,
        get_pipelines_for_training,
        get_trainings_for_dataset,
        run_generation_pipeline,
        run_generation_pipeline_async,
    )
    from .utils.emotion import detect_emotion
    from .utils.fact_extraction import extract_facts
    from .utils.image_captioning import caption_image


def __getattr__(name: str):
    if name == "DatasetBuilder":
        from .core.dataset import DatasetBuilder as _DB

        return _DB
    if name == "Atom":
        from .core.dataset import Atom as _Atom

        return _Atom
    if name in {"ingest_file", "to_kg", "ingest_into_dataset"}:
        from .core.ingest import ingest_into_dataset as _ingest_into_dataset
        from .core.ingest import process_file as _ingest_file
        from .core.ingest import to_kg as _to_kg

        return {
            "ingest_file": _ingest_file,
            "to_kg": _to_kg,
            "ingest_into_dataset": _ingest_into_dataset,
        }[name]
    if name == "KnowledgeGraph":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG
    if name == "GenerationSettings":
        from .config_models import GenerationSettings as _GS

        return _GS
    if name in {
        "DatasetType",
        "GenerationPipeline",
        "TrainingGoal",
        "get_pipeline",
        "get_trainings_for_dataset",
        "get_dataset_types_for_training",
        "get_pipelines_for_training",
    }:
        from .pipelines import DatasetType as _DT
        from .pipelines import GenerationPipeline as _GP
        from .pipelines import TrainingGoal as _TG
        from .pipelines import get_dataset_types_for_training as _gdtft
        from .pipelines import get_pipeline as _gp
        from .pipelines import get_pipelines_for_training as _gpft
        from .pipelines import get_trainings_for_dataset as _gtfd

        mapping = {
            "DatasetType": _DT,
            "GenerationPipeline": _GP,
            "TrainingGoal": _TG,
            "get_pipeline": _gp,
            "get_trainings_for_dataset": _gtfd,
            "get_dataset_types_for_training": _gdtft,
            "get_pipelines_for_training": _gpft,
        }

        return mapping[name]
    if name == "extract_facts":
        from .utils.fact_extraction import extract_facts as _ef

        return _ef
    if name in {"run_generation_pipeline", "run_generation_pipeline_async"}:
        from .pipelines import run_generation_pipeline as _rgp
        from .pipelines import run_generation_pipeline_async as _rgp_async

        return {
            "run_generation_pipeline": _rgp,
            "run_generation_pipeline_async": _rgp_async,
        }[name]
    if name in {
        "curate_qa_pairs",
        "curate_qa_pairs_async",
        "filter_rated_pairs",
        "apply_curation_threshold",
    }:
        from .core.curate import apply_curation_threshold as _act
        from .core.curate import curate_qa_pairs as _cqp
        from .core.curate import curate_qa_pairs_async as _cqpa
        from .core.curate import filter_rated_pairs as _frp

        return {
            "curate_qa_pairs": _cqp,
            "curate_qa_pairs_async": _cqpa,
            "filter_rated_pairs": _frp,
            "apply_curation_threshold": _act,
        }[name]
    if name == "box_cover":
        from .analysis.fractal import box_cover as _bc

        return _bc
    if name == "box_counting_dimension":
        from .analysis.fractal import box_counting_dimension as _bcd

        return _bcd
    if name == "persistence_entropy":
        from .analysis.fractal import persistence_entropy as _pe

        return _pe
    if name == "graphwave_embedding":
        from .analysis.fractal import graphwave_embedding as _ge

        return _ge
    if name == "minimize_bottleneck_distance":
        from .analysis.fractal import minimize_bottleneck_distance as _mbd

        return _mbd
    if name == "bottleneck_distance":
        from .analysis.fractal import bottleneck_distance as _bd

        return _bd
    if name == "mdl_optimal_radius":
        from .analysis.fractal import mdl_optimal_radius as _mr

        return _mr
    if name == "persistence_diagrams":
        from .analysis.fractal import persistence_diagrams as _pd

        return _pd
    if name == "spectral_dimension":
        from .analysis.fractal import spectral_dimension as _sd

        return _sd
    if name == "laplacian_spectrum":
        from .analysis.fractal import laplacian_spectrum as _ls

        return _ls
    if name == "spectral_entropy":
        from .analysis.fractal import spectral_entropy as _se

        return _se
    if name == "spectral_gap":
        from .analysis.fractal import spectral_gap as _sg

        return _sg
    if name == "laplacian_energy":
        from .analysis.fractal import laplacian_energy as _le

        return _le
    if name == "lacunarity":
        from .analysis.fractal import graph_lacunarity as _gl

        return _gl
    if name == "spectral_density":
        from .analysis.fractal import spectral_density as _sdn

        return _sdn
    if name == "graph_fourier_transform":
        from .analysis.fractal import graph_fourier_transform as _gft

        return _gft
    if name == "inverse_graph_fourier_transform":
        from .analysis.fractal import inverse_graph_fourier_transform as _igft

        return _igft
    if name == "graph_information_bottleneck":
        from .analysis.information import graph_information_bottleneck as _gib

        return _gib
    if name == "graph_entropy":
        from .analysis.information import graph_entropy as _ge

        return _ge
    if name == "subgraph_entropy":
        from .analysis.information import subgraph_entropy as _se

        return _se
    if name == "prototype_subgraph":
        from .analysis.information import prototype_subgraph as _ps

        return _ps
    if name == "sheaf_laplacian":
        from .analysis.sheaf import sheaf_laplacian as _sl

        return _sl
    if name == "sheaf_convolution":
        from .analysis.sheaf import sheaf_convolution as _sc

        return _sc
    if name == "sheaf_neural_network":
        from .analysis.sheaf import sheaf_neural_network as _snn

        return _snn
    if name == "sheaf_first_cohomology":
        from .analysis.sheaf import sheaf_first_cohomology as _sfc

        return _sfc
    if name == "resolve_sheaf_obstruction":
        from .analysis.sheaf import resolve_sheaf_obstruction as _rso

        return _rso
    if name == "quality_check":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.quality_check
    if name == "gds_quality_check":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.gds_quality_check
    if name == "validate_topology":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.validate_topology
    if name == "optimize_topology_iterative":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.optimize_topology_iterative
    if name == "quality_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_quality_layer
    if name == "fractal_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_fractal_layer
    if name == "embedding_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_embedding_layer
    if name == "generation_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_generation_layer
    if name == "topological_perception_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_topological_perception_layer
    if name == "compression_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_compression_layer
    if name == "information_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_information_layer
    if name == "export_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_export_layer
    if name == "orchestrator":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_orchestrator
    if name == "poincare_embedding":
        from .analysis.fractal import poincare_embedding as _peb

        return _peb
    if name == "compute_hyperbolic_hypergraph_embeddings":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.compute_hyperbolic_hypergraph_embeddings
    if name == "fractalize_graph":
        from .analysis.fractal import fractalize_graph as _fg

        return _fg
    if name == "fractalize_optimal":
        from .analysis.fractal import fractalize_optimal as _fo

        return _fo
    if name == "build_fractal_hierarchy":
        from .analysis.fractal import build_fractal_hierarchy as _bfh

        return _bfh
    if name == "build_mdl_hierarchy":
        from .analysis.fractal import build_mdl_hierarchy as _bmh

        return _bmh
    if name == "caption_image":
        from .utils.image_captioning import caption_image as _ci

        return _ci
    if name == "detect_emotion":
        from .utils.emotion import detect_emotion as _de

        return _de
    if name == "detect_modality":
        from .utils.modality import detect_modality as _dm

        return _dm
    if name == "md5_file":
        from .utils.checksum import md5_file as _md5

        return _md5
    if name == "generate_graph_rnn_like":
        from .analysis.generation import generate_graph_rnn_like as _gg

        return _gg
    if name == "generate_graph_rnn_stateful":
        from .analysis.generation import generate_graph_rnn_stateful as _grs

        return _grs
    if name == "generate_graph_rnn_sequential":
        from .analysis.generation import generate_graph_rnn_sequential as _grs2

        return _grs2
    if name == "generate_netgan_like":
        from .analysis.generation import generate_netgan_like as _gn

        return _gn
    if name == "fractal_information_density":
        from .analysis.fractal import fractal_information_density as _fid

        return _fid
    if name == "diversification_score":
        from .analysis.fractal import diversification_score as _ds

        return _ds
    if name == "hyperbolic_neighbors":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_neighbors
    if name == "hyperbolic_reasoning":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_reasoning
    if name == "hyperbolic_hypergraph_reasoning":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_hypergraph_reasoning
    if name == "hyperbolic_multi_curvature_reasoning":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_multi_curvature_reasoning
    if name == "compute_distmult_embeddings":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.compute_distmult_embeddings
    if name == "neighborhood_to_sentence":
        from .utils.graph_text import neighborhood_to_sentence as _nts

        return _nts
    if name == "subgraph_to_text":
        from .utils.graph_text import subgraph_to_text as _st

        return _st
    if name == "graph_to_text":
        from .utils.graph_text import graph_to_text as _gt

        return _gt
    if name == "betti_number":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.betti_number
    if name == "coverage_stats":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.coverage_stats
    if name == "invariants_dashboard":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.invariants_dashboard
    if name == "mapper_nerve":
        from .analysis.mapper import mapper_nerve as _mn

        return _mn
    if name == "inverse_mapper":
        from .analysis.mapper import inverse_mapper as _im

        return _im
    if name == "fractal_net_prune":
        from .analysis.fractal import fractal_net_prune as _fp

        return _fp
    if name == "graphwave_entropy":
        from .analysis.fractal import graphwave_entropy as _ge

        return _ge
    if name == "embedding_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.embedding_entropy
    if name == "graph_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.graph_entropy
    if name == "subgraph_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.subgraph_entropy
    if name == "embedding_box_counting_dimension":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.embedding_box_counting_dimension
    if name == "ensure_graphwave_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.ensure_graphwave_entropy
    if name == "hyper_sagnn_embeddings":
        from .analysis.hypergraph import hyper_sagnn_embeddings as _hs

        return _hs
    if name == "select_mdl_motifs":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.select_mdl_motifs
    if name == "mdl_description_length":
        from .analysis.information import mdl_description_length as _mdl

        return _mdl
    if name == "prune_embeddings":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.prune_embeddings
    if name == "ingestion_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_ingestion_layer
    if name in {"detect_automorphisms", "quotient_by_symmetry", "automorphism_group_order"}:
        from .core.dataset import DatasetBuilder as _DB

        mapping = {
            "detect_automorphisms": _DB.detect_automorphisms,
            "automorphism_group_order": _DB.automorphism_group_order,
            "quotient_by_symmetry": _DB.quotient_by_symmetry,
        }

        return mapping[name]
    if name == "get_template" or name == "PromptTemplate" or name == "validate_output":
        from .templates.library import PromptTemplate as _PT
        from .templates.library import get_template as _gtmpl
        from .templates.library import validate_output as _vo

        return {"get_template": _gtmpl, "PromptTemplate": _PT, "validate_output": _vo}[name]
    raise AttributeError(name)
