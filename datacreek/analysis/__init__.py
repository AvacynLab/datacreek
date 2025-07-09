"""Utilities for graph analysis and fractal measures."""

__all__ = [
    "bottleneck_distance",
    "box_counting_dimension",
    "box_cover",
    "graphwave_embedding",
    "embedding_box_counting_dimension",
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
    "prototype_subgraph",
    "sheaf_laplacian",
    "sheaf_convolution",
    "sheaf_neural_network",
    "sheaf_first_cohomology",
    "resolve_sheaf_obstruction",
    "fractal_information_density",
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
    "graphwave_entropy",
    "embedding_entropy",
    "hyper_sagnn_embeddings",
    "mdl_description_length",
    "select_mdl_motifs",
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
        "poincare_embedding",
        "spectral_density",
        "spectral_dimension",
        "spectral_entropy",
        "graph_lacunarity",
        "fractal_information_density",
        "diversification_score",
        "hyperbolic_reasoning",
        "hyperbolic_hypergraph_reasoning",
        "fractal_net_prune",
        "graphwave_entropy",
        "embedding_entropy",
        "embedding_box_counting_dimension",
    }:
        from . import fractal as _f

        return getattr(_f, name)
    if name == "hyper_sagnn_embeddings":
        from .hypergraph import hyper_sagnn_embeddings as _hs

        return _hs
    if name in {"mdl_description_length", "select_mdl_motifs"}:
        from .information import mdl_description_length as _mdl_desc
        from .information import select_mdl_motifs as _mdl_sel

        return _mdl_desc if name == "mdl_description_length" else _mdl_sel
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
    if name == "prototype_subgraph":
        from .information import prototype_subgraph as _ps

        return _ps
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
    if name == "resolve_sheaf_obstruction":
        from .sheaf import resolve_sheaf_obstruction as _rso

        return _rso
    if name == "generate_graph_rnn_stateful":
        from .generation import generate_graph_rnn_stateful as _grs

        return _grs
    if name == "generate_graph_rnn_sequential":
        from .generation import generate_graph_rnn_sequential as _grs2

        return _grs2
    raise AttributeError(name)
