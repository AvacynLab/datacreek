"""Utilities for graph analysis and fractal measures."""

__all__ = [
    "bottleneck_distance",
    "persistence_wasserstein_distance",
    "box_counting_dimension",
    "box_cover",
    "graphwave_embedding",
    "embedding_box_counting_dimension",
    "colour_box_dimension",
    "mdl_optimal_radius",
    "persistence_diagrams",
    "persistence_entropy",
    "fractalize_graph",
    "fractalize_optimal",
    "minimize_bottleneck_distance",
    "build_fractal_hierarchy",
    "build_mdl_hierarchy",
    "poincare_embedding",
    "spectral_dimension",
    "laplacian_spectrum",
    "spectral_entropy",
    "spectral_density",
    "graph_lacunarity",
    "graph_fourier_transform",
    "inverse_graph_fourier_transform",
    "graph_information_bottleneck",
    "graph_entropy",
    "subgraph_entropy",
    "structural_entropy",
    "prototype_subgraph",
    "sheaf_laplacian",
    "sheaf_convolution",
    "sheaf_neural_network",
    "sheaf_first_cohomology",
    "sheaf_first_cohomology_blocksmith",
    "sheaf_consistency_score_batched",
    "resolve_sheaf_obstruction",
    "sheaf_consistency_score",
    "fractal_information_density",
    "fractal_level_coverage",
    "diversification_score",
    "generate_graph_rnn_stateful",
    "generate_graph_rnn_sequential",
    "hyperbolic_reasoning",
    "hyperbolic_hypergraph_reasoning",
    "automorphisms",
    "automorphism_orbits",
    "automorphism_group_order",
    "quotient_graph",
    "mapper_nerve",
    "inverse_mapper",
    "fractal_net_prune",
    "fractalnet_compress",
    "prune_fractalnet",
    "graphwave_entropy",
    "embedding_entropy",
    "hyper_sagnn_embeddings",
    "hyper_sagnn_head_drop_embeddings",
    "hyperedge_attention_scores",
    "mdl_description_length",
    "select_mdl_motifs",
    "product_embedding",
    "train_product_manifold",
    "aligned_cca",
    "hybrid_score",
    "multiview_contrastive_loss",
    "meta_autoencoder",
    "tpl_correct_graph",
    "alignment_correlation",
    "average_hyperbolic_radius",
    "scale_bias_wasserstein",
    "governance_metrics",
    "AutoTuneState",
    "autotune_step",
    "svgp_ei_propose",
    "kw_gradient",
    "spectral_bound_exceeded",
    "filter_semantic_cycles",
    "entropy_triangle_threshold",
    "rollback_gremlin_diff",
    "SheafSLA",
]


def __getattr__(name: str):
    if name in {
        "bottleneck_distance",
        "box_counting_dimension",
        "box_cover",
        "fractalize_graph",
        "fractalize_optimal",
        "graph_fourier_transform",
        "graphwave_embedding",
        "inverse_graph_fourier_transform",
        "laplacian_spectrum",
        "mdl_optimal_radius",
        "minimize_bottleneck_distance",
        "build_fractal_hierarchy",
        "build_mdl_hierarchy",
        "persistence_diagrams",
        "persistence_entropy",
        "persistence_wasserstein_distance",
        "poincare_embedding",
        "spectral_density",
        "spectral_dimension",
        "spectral_entropy",
        "graph_lacunarity",
        "fractal_information_density",
        "fractal_level_coverage",
        "diversification_score",
        "hyperbolic_reasoning",
        "hyperbolic_hypergraph_reasoning",
        "fractal_net_prune",
        "fractalnet_compress",
        "prune_fractalnet",
        "graphwave_entropy",
        "embedding_entropy",
        "embedding_box_counting_dimension",
        "colour_box_dimension",
    }:
        from . import fractal as _f

        return getattr(_f, name)
    if name in {
        "hyper_sagnn_embeddings",
        "hyper_sagnn_head_drop_embeddings",
        "hyperedge_attention_scores",
    }:
        from .hypergraph import hyper_sagnn_embeddings as _hs
        from .hypergraph import hyper_sagnn_head_drop_embeddings as _hd
        from .hypergraph import hyperedge_attention_scores as _att

        return {
            "hyper_sagnn_embeddings": _hs,
            "hyper_sagnn_head_drop_embeddings": _hd,
            "hyperedge_attention_scores": _att,
        }[name]
    if name in {"mdl_description_length", "select_mdl_motifs"}:
        from .information import mdl_description_length as _mdl_desc
        from .information import select_mdl_motifs as _mdl_sel

        return _mdl_desc if name == "mdl_description_length" else _mdl_sel
    if name in {
        "product_embedding",
        "train_product_manifold",
        "aligned_cca",
        "hybrid_score",
        "multiview_contrastive_loss",
        "meta_autoencoder",
        "tpl_correct_graph",
        "alignment_correlation",
        "average_hyperbolic_radius",
        "scale_bias_wasserstein",
        "governance_metrics",
    }:
        from . import governance as _g
        from . import multiview as _mv

        if hasattr(_mv, name):
            return getattr(_mv, name)
        return getattr(_g, name)
    if name == "tpl_correct_graph":
        from .tpl import tpl_correct_graph as _tcg

        return _tcg
    if name in {
        "automorphisms",
        "automorphism_orbits",
        "quotient_graph",
        "automorphism_group_order",
    }:
        from . import symmetry as _s

        return getattr(_s, name)
    if name in {"mapper_nerve", "inverse_mapper"}:
        from . import mapper as _m

        return getattr(_m, name)
    if name == "graph_information_bottleneck":
        from .information import graph_information_bottleneck as _gib

        return _gib
    if name == "graph_entropy":
        from .information import graph_entropy as _ge

        return _ge
    if name == "subgraph_entropy":
        from .information import subgraph_entropy as _se

        return _se
    if name == "structural_entropy":
        from .information import structural_entropy as _str_e

        return _str_e
    if name == "prototype_subgraph":
        from .information import prototype_subgraph as _ps

        return _ps
    if name in {"AutoTuneState", "autotune_step", "svgp_ei_propose", "kw_gradient"}:
        from .autotune import AutoTuneState as _AS
        from .autotune import autotune_step as _at
        from .autotune import kw_gradient as _kw
        from .autotune import svgp_ei_propose as _sv

        return {
            "AutoTuneState": _AS,
            "autotune_step": _at,
            "svgp_ei_propose": _sv,
            "kw_gradient": _kw,
        }[name]
    if name == "sheaf_laplacian":
        from .sheaf import sheaf_laplacian as _sl

        return _sl
    if name == "sheaf_convolution":
        from .sheaf import sheaf_convolution as _sc

        return _sc
    if name == "sheaf_neural_network":
        from .sheaf import sheaf_neural_network as _snn

        return _snn
    if name == "sheaf_first_cohomology":
        from .sheaf import sheaf_first_cohomology as _sfc

        return _sfc
    if name == "sheaf_first_cohomology_blocksmith":
        from .sheaf import sheaf_first_cohomology_blocksmith as _sfcbs

        return _sfcbs
    if name == "resolve_sheaf_obstruction":
        from .sheaf import resolve_sheaf_obstruction as _rso

        return _rso
    if name == "sheaf_consistency_score":
        from .sheaf import sheaf_consistency_score as _scs

        return _scs
    if name == "sheaf_consistency_score_batched":
        from .sheaf import sheaf_consistency_score_batched as _scsb

        return _scsb
    if name == "spectral_bound_exceeded":
        from .sheaf import spectral_bound_exceeded as _sbe

        return _sbe
    if name == "generate_graph_rnn_stateful":
        from .generation import generate_graph_rnn_stateful as _grs

        return _grs
    if name == "generate_graph_rnn_sequential":
        from .generation import generate_graph_rnn_sequential as _grs2

        return _grs2
    if name == "filter_semantic_cycles":
        from .filtering import filter_semantic_cycles as _fsc

        return _fsc
    if name == "entropy_triangle_threshold":
        from .filtering import entropy_triangle_threshold as _ett

        return _ett
    if name == "rollback_gremlin_diff":
        from .rollback import rollback_gremlin_diff as _rgd

        return _rgd
    if name == "SheafSLA":
        from .rollback import SheafSLA as _sla

        return _sla
    raise AttributeError(name)
