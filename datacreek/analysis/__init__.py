"""Utilities for graph analysis and fractal measures."""

__all__ = [
    "bottleneck_distance",
    "box_counting_dimension",
    "box_cover",
    "graphwave_embedding",
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
    "prototype_subgraph",
    "sheaf_laplacian",
    "sheaf_convolution",
    "sheaf_neural_network",
    "fractal_information_density",
    "diversification_score",
    "generate_graph_rnn_stateful",
    "generate_graph_rnn_sequential",
    "hyperbolic_reasoning",
    "hyperbolic_hypergraph_reasoning",
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
    }:
        from . import fractal as _f

        return getattr(_f, name)
    if name == "graph_information_bottleneck":
        from .information import graph_information_bottleneck as _gib

        return _gib
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
    if name == "generate_graph_rnn_stateful":
        from .generation import generate_graph_rnn_stateful as _grs

        return _grs
    if name == "generate_graph_rnn_sequential":
        from .generation import generate_graph_rnn_sequential as _grs2

        return _grs2
    raise AttributeError(name)
