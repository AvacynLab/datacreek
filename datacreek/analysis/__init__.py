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
    "poincare_embedding",
    "spectral_dimension",
    "laplacian_spectrum",
    "spectral_entropy",
    "spectral_density",
    "graph_fourier_transform",
    "inverse_graph_fourier_transform",
    "graph_information_bottleneck",
    "sheaf_laplacian",
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
        "persistence_diagrams",
        "persistence_entropy",
        "poincare_embedding",
        "spectral_density",
        "spectral_dimension",
        "spectral_entropy",
    }:
        from . import fractal as _f

        return getattr(_f, name)
    if name == "graph_information_bottleneck":
        from .information import graph_information_bottleneck as _gib

        return _gib
    if name == "sheaf_laplacian":
        from .sheaf import sheaf_laplacian as _sl

        return _sl
    raise AttributeError(name)
