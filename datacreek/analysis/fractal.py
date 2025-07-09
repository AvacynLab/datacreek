import math
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # optional dependency
    import gudhi as gd
    import gudhi.representations as gr
except Exception:  # pragma: no cover - optional dependency missing
    gd = None  # type: ignore
    gr = None  # type: ignore
import networkx as nx
import numpy as np

try:
    from scipy.linalg import eigh  # type: ignore
    from scipy.sparse import csgraph  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    from numpy.linalg import eigh  # type: ignore

    csgraph = None  # type: ignore


def _laplacian(graph: nx.Graph, *, normed: bool = False) -> np.ndarray:
    """Return Laplacian matrix of ``graph``."""

    if csgraph is not None:
        a = nx.to_scipy_sparse_array(graph)
        return csgraph.laplacian(a, normed=normed).toarray()
    a = nx.to_numpy_array(graph)
    deg = a.sum(axis=1)
    lap = np.diag(deg) - a
    if normed:
        with np.errstate(divide="ignore"):
            inv_sqrt = 1.0 / np.sqrt(deg)
        inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
        lap = (inv_sqrt[:, None] * lap) * inv_sqrt
    return lap


def box_cover(graph: nx.Graph, radius: int) -> List[set[int]]:
    """Return a box covering of ``graph`` with radius ``radius``.

    Parameters
    ----------
    graph:
        Input graph.
    radius:
        Radius ``l_B`` used for the covering.

    Returns
    -------
    list[set[int]]
        List of node sets, each representing a box of diameter ``2 * radius``.
    """

    remaining = set(graph.nodes())
    boxes: List[set[int]] = []
    while remaining:
        start = remaining.pop()
        seen = nx.single_source_shortest_path_length(graph, start, cutoff=radius)
        box = {n for n in seen.keys() if n in remaining or n == start}
        remaining.difference_update(box)
        boxes.append(box)
    return boxes


def box_counting_dimension(
    graph: nx.Graph, radii: Iterable[int]
) -> Tuple[float, List[Tuple[int, int]]]:
    """Estimate fractal dimension of ``graph`` using box covering.

    Parameters
    ----------
    graph:
        Input graph.
    radii:
        Iterable of box radii ``l_B``. Larger radii correspond to coarser covers.

    Returns
    -------
    float
        Estimated fractal dimension based on the slope of ``log(N_B)`` vs ``log(1/l_B)``.
    List[Tuple[int, int]]
        List of ``(radius, box_count)`` pairs used for the estimation.
    """

    counts: List[Tuple[int, int]] = []
    nodes = list(graph.nodes())
    for r in radii:
        remaining = set(nodes)
        boxes = 0
        while remaining:
            start = remaining.pop()
            seen = nx.single_source_shortest_path_length(graph, start, cutoff=r)
            remaining.difference_update(seen.keys())
            boxes += 1
        counts.append((r, boxes))

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0, counts
    slope, _ = np.polyfit(xs, ys, 1)
    return slope, counts


def persistence_entropy(graph: nx.Graph, dimension: int = 0) -> float:
    """Return the persistence entropy for ``graph`` in a given dimension.

    Parameters
    ----------
    graph:
        Input graph.
    dimension:
        Homology dimension for which to compute the persistence entropy.

    Returns
    -------
    float
        Persistence entropy of the diagram in ``dimension``. Returns ``0.0`` if
        no finite intervals are present.
    """

    if gd is None or gr is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    st = gd.SimplexTree()
    for node in graph.nodes():
        st.insert([node], filtration=0.0)
    for u, v in graph.edges():
        st.insert([u, v], filtration=1.0)

    st.compute_persistence(persistence_dim_max=True)
    diag = np.array(st.persistence_intervals_in_dimension(dimension))
    if diag.size == 0:
        return 0.0
    diag = diag[np.isfinite(diag[:, 1])]
    if diag.size == 0:
        return 0.0
    entropy = gr.Entropy()
    return float(entropy.fit_transform([diag])[0])


def persistence_diagrams(graph: nx.Graph, max_dim: int = 2) -> Dict[int, np.ndarray]:
    """Return persistence diagrams of ``graph`` up to ``max_dim`` using the clique complex.

    Parameters
    ----------
    graph:
        Input graph.
    max_dim:
        Highest homology dimension for which to compute the diagram.

    Returns
    -------
    dict
        Mapping ``dimension -> array`` with birth-death pairs for each diagram.
    """

    st = gd.SimplexTree()
    for node in graph.nodes():
        st.insert([node], filtration=0.0)

    # Insert all cliques up to size ``max_dim + 1`` to build the clique complex
    for clique in nx.enumerate_all_cliques(graph):
        dim = len(clique) - 1
        if dim == 0 or dim > max_dim:
            continue
        st.insert(clique, filtration=float(dim))

    st.compute_persistence(persistence_dim_max=True)
    diags: Dict[int, np.ndarray] = {}
    for dim in range(max_dim + 1):
        diag = np.array(st.persistence_intervals_in_dimension(dim))
        diag = diag[np.isfinite(diag[:, 1])]
        diags[dim] = diag
    return diags


def graphwave_embedding(
    graph: nx.Graph, scales: Iterable[float], num_points: int = 10
) -> Dict[int, np.ndarray]:
    """Return GraphWave embeddings for ``graph``.

    Parameters
    ----------
    graph:
        Input graph.
    scales:
        Iterable of diffusion scales used for the wavelets.
    num_points:
        Number of sample points for the characteristic function.

    Returns
    -------
    dict
        Mapping of node to embedding vector of length ``len(scales) * num_points * 2``.
    """

    nodes = list(graph.nodes())
    a = nx.to_numpy_array(graph, nodelist=nodes)
    lap = _laplacian(graph, normed=False)
    evals, evecs = eigh(lap)
    ts = np.linspace(0, 2 * np.pi, num_points)
    emb: Dict[int, List[float]] = {n: [] for n in nodes}

    for s in scales:
        heat = evecs @ np.diag(np.exp(-s * evals)) @ evecs.T
        for idx, node in enumerate(nodes):
            coeffs = np.exp(1j * np.outer(ts, heat[idx, :])).sum(axis=1)
            emb[node].extend(coeffs.real)
            emb[node].extend(coeffs.imag)

    return {n: np.asarray(v, dtype=float) for n, v in emb.items()}


def graphwave_entropy(embeddings: Dict[object, Iterable[float]]) -> float:
    """Return differential entropy of GraphWave ``embeddings``.

    The vectors are assumed to follow a continuous distribution approximated
    by a multivariate Gaussian.  The entropy then reads

    .. math:: H = \tfrac{1}{2}\log\bigl((2\pi e)^d \det \Sigma\bigr)

    where ``d`` is the embedding dimension and ``\Sigma`` the covariance
    matrix.

    Parameters
    ----------
    embeddings:
        Mapping of nodes to their embedding vectors.

    Returns
    -------
    float
        Estimated differential entropy.
    """

    arr = np.vstack([np.asarray(v, dtype=float) for v in embeddings.values()])
    if arr.size == 0:
        return 0.0
    cov = np.cov(arr, rowvar=False)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov + 1e-8 * np.eye(d))
    if sign <= 0:
        return float("nan")
    return 0.5 * (logdet + d * math.log(2 * math.pi * math.e))


def embedding_entropy(embeddings: Dict[object, Iterable[float]]) -> float:
    """Return differential entropy for an embedding dictionary.

    This is a generic variant of :func:`graphwave_entropy` that applies to any
    set of embedding vectors. The computation assumes a multivariate Gaussian
    distribution and uses the covariance determinant as

    .. math:: H = \tfrac{1}{2}\log\bigl((2\pi e)^d \det \Sigma\bigr).

    Parameters
    ----------
    embeddings:
        Mapping of arbitrary keys to embedding vectors.

    Returns
    -------
    float
        Estimated differential entropy of the embeddings.
    """

    arr = np.vstack([np.asarray(v, dtype=float) for v in embeddings.values()])
    if arr.size == 0:
        return 0.0
    cov = np.cov(arr, rowvar=False)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov + 1e-8 * np.eye(d))
    if sign <= 0:
        return float("nan")
    return 0.5 * (logdet + d * math.log(2 * math.pi * math.e))


def bottleneck_distance(
    g1: nx.Graph, g2: nx.Graph, dimension: int = 0, approx_epsilon: float | None = None
) -> float:
    """Return bottleneck distance between ``g1`` and ``g2`` diagrams.

    Parameters
    ----------
    g1, g2:
        Graphs to compare.
    dimension:
        Homology dimension used for the persistence diagrams.
    approx_epsilon:
        Additive error tolerated by :func:`gudhi.bottleneck_distance`. ``None``
        (the default) means use the smallest positive ``float`` for exact
        computation.

    Returns
    -------
    float
        Bottleneck distance between the diagrams of ``g1`` and ``g2`` in the
        chosen dimension.
    """

    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    def _diagram(graph: nx.Graph) -> np.ndarray:
        st = gd.SimplexTree()
        mapping = {n: i for i, n in enumerate(graph.nodes())}
        for node, idx in mapping.items():
            st.insert([idx], filtration=0.0)
        for u, v in graph.edges():
            st.insert([mapping[u], mapping[v]], filtration=1.0)

        st.compute_persistence(persistence_dim_max=True)
        diag = np.array(st.persistence_intervals_in_dimension(dimension))
        if diag.size == 0:
            return np.empty((0, 2))
        diag = diag[np.isfinite(diag[:, 1])]
        return diag

    d1 = _diagram(g1)
    d2 = _diagram(g2)
    return float(gd.bottleneck_distance(d1, d2, e=approx_epsilon))


def mdl_optimal_radius(counts: List[Tuple[int, int]]) -> int:
    """Return radius index minimizing a simple MDL criterion.

    Parameters
    ----------
    counts:
        Sequence of ``(radius, box_count)`` pairs returned by
        :func:`box_counting_dimension`. Must be ordered by increasing radius.

    Returns
    -------
    int
        Index of ``counts`` giving the minimal description length. ``counts``
        must contain at least two elements; otherwise 0 is returned.
    """

    if len(counts) < 2:
        return 0

    mdls = []
    for end in range(2, len(counts) + 1):
        sub = counts[:end]
        xs = [-math.log(float(r)) for r, _ in sub]
        ys = [math.log(float(n)) for _, n in sub]
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = [slope * x + intercept for x in xs]
        rss = sum((y - p) ** 2 for y, p in zip(ys, pred))
        n = len(xs)
        k = 2  # slope and intercept
        mdl = n * math.log(rss / n + 1e-12) + k * math.log(n)
        mdls.append((mdl, end - 1))

    best = min(mdls, key=lambda x: x[0])[1]
    return best


def mdl_value(counts: List[Tuple[int, int]]) -> float:
    """Return MDL value for a sequence of ``(radius, box_count)`` pairs."""

    if len(counts) < 2:
        return 0.0

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = [slope * x + intercept for x in xs]
    rss = sum((y - p) ** 2 for y, p in zip(ys, pred))
    n = len(xs)
    k = 2  # slope and intercept
    return float(n * math.log(rss / n + 1e-12) + k * math.log(n))


def _slope(counts: List[Tuple[int, int]]) -> float:
    """Return slope of ``log(N_B)`` vs ``log(1/l_B)`` for ``counts``."""

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def dichotomic_radius(counts: List[Tuple[int, int]], target: float) -> int:
    """Return index whose slope is closest to ``target`` using dichotomy."""

    left = 1
    right = len(counts) - 1
    best = 1
    best_diff = abs(_slope(counts[: best + 1]) - target)
    while left <= right:
        mid = (left + right) // 2
        s = _slope(counts[: mid + 1])
        diff = abs(s - target)
        if diff < best_diff:
            best = mid
            best_diff = diff
        if s > target:
            left = mid + 1
        else:
            right = mid - 1
    return best


def spectral_dimension(
    graph: nx.Graph, times: Iterable[float]
) -> Tuple[float, List[Tuple[float, float]]]:
    """Estimate the spectral dimension of ``graph`` using heat trace scaling.

    The heat trace :math:`\mathrm{Tr}(e^{-tL})` of the graph Laplacian
    asymptotically scales like ``t**(-d/2)`` where ``d`` is the spectral
    dimension.  By computing the heat trace for a range of ``times`` and
    fitting a line to ``log(trace)`` versus ``log(time)`` we obtain an estimate
    of ``d``.

    Parameters
    ----------
    graph:
        Input graph.
    times:
        Iterable of diffusion times ``t``. Larger values probe coarser
        structures of the graph.

    Returns
    -------
    float
        Estimated spectral dimension.
    list[tuple[float, float]]
        Pairs ``(t, trace)`` used for the fit.
    """

    nodes = list(graph.nodes())
    a = nx.to_numpy_array(graph, nodelist=nodes)
    lap = _laplacian(graph, normed=False)
    evals, _ = eigh(lap)

    traces: List[Tuple[float, float]] = []
    for t in times:
        trace = float(np.sum(np.exp(-t * evals)))
        traces.append((float(t), trace))

    xs = [math.log(t) for t, _ in traces]
    ys = [math.log(tr) for _, tr in traces]
    if len(xs) < 2:
        return 0.0, traces
    slope, _ = np.polyfit(xs, ys, 1)
    dim = -2 * slope
    return float(dim), traces


def embedding_box_counting_dimension(
    coords: Dict[object, Iterable[float]], radii: Iterable[float]
) -> Tuple[float, List[Tuple[float, int]]]:
    """Estimate fractal dimension from point coordinates.

    Parameters
    ----------
    coords:
        Mapping of node IDs to embedding vectors.
    radii:
        Iterable of ball radii. Larger radii correspond to coarser covers.

    Returns
    -------
    float
        Estimated fractal dimension based on the slope of ``log(N_B)`` vs
        ``log(1/r)``.
    list[tuple[float, int]]
        ``(radius, count)`` pairs used for the estimation.
    """

    pts = np.asarray(list(coords.values()), dtype=float)
    counts: List[Tuple[float, int]] = []
    for r in radii:
        remaining = set(range(len(pts)))
        boxes = 0
        while remaining:
            i = remaining.pop()
            dists = np.linalg.norm(pts[list(remaining | {i})] - pts[i], axis=1)
            cover = {j for j, d in zip(list(remaining | {i}), dists) if d <= r}
            remaining.difference_update(cover)
            boxes += 1
        counts.append((float(r), boxes))

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0, counts
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope), counts


def laplacian_spectrum(graph: nx.Graph, *, normed: bool = True) -> np.ndarray:
    """Return the Laplacian eigenvalues of ``graph``.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to compute the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    numpy.ndarray
        Array of eigenvalues sorted in ascending order.
    """

    a = nx.to_numpy_array(graph, nodelist=list(graph.nodes()))
    lap = _laplacian(graph, normed=normed)
    evals = eigh(lap, eigvals_only=True)
    return np.sort(evals)


def spectral_entropy(graph: nx.Graph, *, normed: bool = True) -> float:
    """Return the Shannon entropy of the Laplacian spectrum.

    The eigenvalues of the Laplacian are normalized to form a probability
    distribution before computing the entropy.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to use the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    float
        Shannon entropy of the normalized spectrum.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    total = float(np.sum(evals))
    if total == 0:
        return 0.0
    probs = evals / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def spectral_gap(graph: nx.Graph, *, normed: bool = True) -> float:
    """Return the spectral gap of ``graph``.

    The spectral gap is the difference between the two smallest eigenvalues
    of the (normalized) Laplacian. For a connected graph with a normalized
    Laplacian this is simply ``lambda_2``.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to use the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    float
        The spectral gap value. Returns ``0.0`` if there are fewer than two
        eigenvalues.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    if len(evals) < 2:
        return 0.0
    return float(evals[1] - evals[0])


def laplacian_energy(graph: nx.Graph, *, normed: bool = True) -> float:
    """Return the Laplacian energy of ``graph``.

    The Laplacian energy is defined as

    .. math:: \sum_{i} |\lambda_i - 2m/n|

    where :math:`\lambda_i` are the Laplacian eigenvalues, ``m`` is the number
    of edges and ``n`` is the number of nodes.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to use the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    float
        Laplacian energy value.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    avg = 2 * m / n
    return float(np.sum(np.abs(evals - avg)))


def spectral_density(
    graph: nx.Graph, bins: int = 50, *, normed: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the density of Laplacian eigenvalues.

    Parameters
    ----------
    graph:
        Input graph.
    bins:
        Number of histogram bins.
    normed:
        Whether to use the normalized Laplacian.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Histogram counts and the corresponding bin edges.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    hist, edges = np.histogram(evals, bins=bins, density=True)
    return hist.astype(float), edges.astype(float)


def graph_lacunarity(graph: nx.Graph, radius: int = 1) -> float:
    """Return lacunarity of ``graph`` for neighborhood radius ``radius``.

    The lacunarity measures the heterogeneity of mass distribution
    at a given scale. For each node we count the number of nodes
    within ``radius`` hops and compute

    .. math:: \Lambda = \frac{\mathrm{Var}(M)}{\mathrm{E}[M]^2} + 1,

    where :math:`M` denotes the local mass.
    """

    masses = [nx.ego_graph(graph, n, radius=radius).number_of_nodes() for n in graph.nodes()]
    if not masses:
        return 0.0
    arr = np.asarray(masses, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.var() / (mean * mean) + 1.0)


def graph_fourier_transform(
    graph: nx.Graph, signal: Dict[int, float] | np.ndarray, *, normed: bool = True
) -> np.ndarray:
    """Return the graph Fourier transform of ``signal``.

    Parameters
    ----------
    graph:
        Input graph.
    signal:
        Mapping from node to value or array of values ordered by ``graph.nodes``.
    normed:
        Whether to use the normalized Laplacian.

    Returns
    -------
    numpy.ndarray
        Fourier coefficients ordered by increasing Laplacian eigenvalue.
    """

    nodes = list(graph.nodes())
    lap = _laplacian(graph, normed=normed)
    evals, evecs = eigh(lap)
    if isinstance(signal, dict):
        vec = np.array([signal[n] for n in nodes], dtype=float)
    else:
        vec = np.asarray(signal, dtype=float)
    return evecs.T @ vec


def inverse_graph_fourier_transform(
    graph: nx.Graph, coeffs: np.ndarray, *, normed: bool = True
) -> np.ndarray:
    """Return the inverse graph Fourier transform of ``coeffs``."""

    nodes = list(graph.nodes())
    lap = _laplacian(graph, normed=normed)
    _, evecs = eigh(lap)
    return evecs @ np.asarray(coeffs, dtype=float)


def fractal_information_metrics(
    graph: nx.Graph, radii: Iterable[int], *, max_dim: int = 1
) -> Dict[str, Any]:
    """Return fractal dimension and persistence entropies.

    Parameters
    ----------
    graph:
        Input graph.
    radii:
        Iterable of box radii used for the covering.
    max_dim:
        Highest homology dimension for which to compute the entropy.

    Returns
    -------
    dict
        Mapping with keys ``dimension`` and ``entropy`` (per homology dimension).
    """

    dim, _ = box_counting_dimension(graph, radii)
    entropies: Dict[int, float] = {}
    if gd is not None and gr is not None:
        for d in range(max_dim + 1):
            try:
                entropies[d] = persistence_entropy(graph, dimension=d)
            except Exception:  # pragma: no cover - optional dep failure
                entropies[d] = float("nan")
    else:  # pragma: no cover - optional dep missing
        entropies = {d: float("nan") for d in range(max_dim + 1)}

    return {"dimension": dim, "entropy": entropies}


def fractal_information_density(
    graph: nx.Graph, radii: Iterable[int], *, max_dim: int = 1
) -> float:
    """Return a simple information density from fractal dimension and entropy.

    The density is defined as ``dimension / (1 + sum(entropies))`` so that
    higher entropies lower the returned value. It is meant as a lightweight
    indicator of how much structural information is carried per fractal degree.
    """

    metrics = fractal_information_metrics(graph, radii, max_dim=max_dim)
    dim = metrics["dimension"]
    ent_sum = float(sum(metrics["entropy"].values()))
    return dim / (1.0 + ent_sum)


def diversification_score(
    global_graph: nx.Graph,
    batch_graph: nx.Graph,
    radii: Iterable[int],
    *,
    max_dim: int = 1,
    dimension: int = 0,
) -> float:
    """Return a diversification score mixing MDL and bottleneck distance.

    The score is the sum of the MDL difference between ``global_graph`` and
    ``batch_graph`` and the bottleneck distance of their persistence diagrams in
    ``dimension``. Lower scores indicate higher redundancy of ``batch_graph``
    with respect to ``global_graph``.
    """

    _, global_counts = box_counting_dimension(global_graph, radii)
    _, batch_counts = box_counting_dimension(batch_graph, radii)
    mdl_global = mdl_value(global_counts)
    mdl_batch = mdl_value(batch_counts)

    try:
        dist = bottleneck_distance(global_graph, batch_graph, dimension=dimension)
    except Exception:
        dist = float("nan")

    return float((mdl_global - mdl_batch) + dist)


def poincare_embedding(
    graph: nx.Graph,
    dim: int = 2,
    negative: int = 5,
    epochs: int = 50,
    learning_rate: float = 0.1,
    burn_in: int = 10,
) -> Dict[int, np.ndarray]:
    """Return Poincar\u00e9 embeddings for ``graph``.

    Parameters
    ----------
    graph:
        Input graph whose edges define the hierarchy.
    dim:
        Dimension of the hyperbolic embedding space.
    negative:
        Number of negative samples used during training.
    epochs:
        Number of training epochs.
    learning_rate:
        Learning rate for the optimizer.
    burn_in:
        Number of burn-in epochs before using negative sampling.

    Returns
    -------
    dict
        Mapping of node to embedding vector of length ``dim``.
    """

    try:  # pragma: no cover - dependency not always present
        from gensim.models.poincare import PoincareModel
    except Exception as e:  # pragma: no cover - dependency missing
        raise RuntimeError("gensim is required for Poincare embeddings") from e

    edges = [(str(u), str(v)) for u, v in graph.edges()]
    model = PoincareModel(
        edges,
        size=dim,
        negative=negative,
        burn_in=burn_in,
        alpha=learning_rate,
    )
    model.train(epochs)

    embeddings: Dict[int, np.ndarray] = {}
    for node in graph.nodes():
        key = str(node)
        if key in model.kv:
            embeddings[node] = np.asarray(model.kv[key], dtype=float)
    return embeddings


def fractalize_graph(graph: nx.Graph, radius: int) -> Tuple[nx.Graph, Dict[object, int]]:
    """Return a coarse-grained graph obtained via box covering.

    Parameters
    ----------
    graph:
        Input graph to coarse-grain.
    radius:
        Radius ``l_B`` used for the covering.

    Returns
    -------
    tuple
        A tuple ``(G, mapping)`` where ``G`` is the coarse-grained graph and
        ``mapping`` maps each original node to its box index.
    """

    boxes = box_cover(graph, radius)
    mapping: Dict[object, int] = {}
    coarse = nx.Graph()
    for idx, box in enumerate(boxes):
        coarse.add_node(idx)
        for n in box:
            mapping[n] = idx

    for u, v in graph.edges():
        b1 = mapping[u]
        b2 = mapping[v]
        if b1 != b2:
            coarse.add_edge(b1, b2)

    return coarse, mapping


def fractalize_optimal(
    graph: nx.Graph, radii: Iterable[int]
) -> Tuple[nx.Graph, Dict[object, int], int]:
    """Coarse-grain ``graph`` using the MDL-optimal radius.

    Parameters
    ----------
    graph:
        Input graph to coarse-grain.
    radii:
        Candidate box radii ``l_B`` used to search for the optimal cover.

    Returns
    -------
    tuple
        ``(G, mapping, radius)`` where ``G`` is the coarse-grained graph,
        ``mapping`` maps each original node to its box index, and ``radius``
        is the chosen radius.
    """

    _, counts = box_counting_dimension(graph, radii)
    if not counts:
        raise ValueError("radii must not be empty")
    idx = mdl_optimal_radius(counts)
    radius = counts[idx][0]
    coarse, mapping = fractalize_graph(graph, radius)
    return coarse, mapping, radius


def build_fractal_hierarchy(
    graph: nx.Graph, radii: Iterable[int], *, max_levels: int = 5
) -> List[Tuple[nx.Graph, Dict[object, int], int]]:
    """Return a hierarchy of coarse graphs using MDL-optimal radii.

    Parameters
    ----------
    graph:
        Input graph to coarse-grain recursively.
    radii:
        Candidate box radii ``l_B`` used to search for the optimal cover at each
        level.
    max_levels:
        Maximum number of coarse-graining iterations. The process stops early if
        the graph can no longer be reduced.

    Returns
    -------
    list
        Sequence ``[(G1, mapping1, r1), (G2, mapping2, r2), ...]`` describing
        the hierarchy from fine to coarse. ``Gi`` is the graph at level ``i`` and
        ``mappingi`` maps nodes of ``G_{i-1}`` to boxes of ``Gi``.
    """

    levels: List[Tuple[nx.Graph, Dict[object, int], int]] = []
    current = graph
    for _ in range(max_levels):
        coarse, mapping, radius = fractalize_optimal(current, radii)
        levels.append((coarse, mapping, radius))
        if coarse.number_of_nodes() >= current.number_of_nodes() or coarse.number_of_nodes() <= 1:
            break
        current = coarse
    return levels


def build_mdl_hierarchy(
    graph: nx.Graph, radii: Iterable[int], *, max_levels: int = 5, slope_tol: float = 0.1
) -> List[Tuple[nx.Graph, Dict[object, int], int]]:
    """Return a hierarchy using MDL to stop when description length grows."""

    levels: List[Tuple[nx.Graph, Dict[object, int], int]] = []
    current = graph
    prev_mdl = float("inf")
    target_dim = None
    for _ in range(max_levels):
        dim, counts = box_counting_dimension(current, radii)
        if not counts:
            break
        idx = mdl_optimal_radius(counts)
        if target_dim is not None:
            cand = dichotomic_radius(counts, target_dim)
            if abs(_slope(counts[: cand + 1]) - target_dim) <= slope_tol:
                idx = cand
        mdl_curr = mdl_value(counts[: idx + 1])
        if mdl_curr >= prev_mdl:
            break
        prev_mdl = mdl_curr
        radius = counts[idx][0]
        coarse, mapping = fractalize_graph(current, radius)
        levels.append((coarse, mapping, radius))
        if coarse.number_of_nodes() >= current.number_of_nodes() or coarse.number_of_nodes() <= 1:
            break
        target_dim = _slope(counts[: idx + 1])
        current = coarse
    return levels


def minimize_bottleneck_distance(
    graph: nx.Graph,
    target: nx.Graph,
    *,
    dimension: int = 1,
    epsilon: float = 0.0,
    max_iter: int = 100,
    seed: int | None = None,
) -> Tuple[nx.Graph, float]:
    """Return a graph whose topology approximates ``target``.

    The function greedily edits edges to minimize the bottleneck distance
    between ``graph`` and ``target`` in ``dimension``. Edges are added or
    removed at random and a change is kept if it improves the distance by at
    least ``epsilon``.

    Parameters
    ----------
    graph:
        Input graph to modify. A copy is used internally.
    target:
        Reference graph defining the desired topology.
    dimension:
        Homology dimension for the bottleneck distance.
    epsilon:
        Minimum improvement required to accept a modification.
    max_iter:
        Maximum number of edge edits to attempt.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    tuple
        ``(G, dist)`` where ``G`` is the optimized graph and ``dist`` is the
        final bottleneck distance to ``target``.
    """

    rng = random.Random(seed)

    best = graph.copy()
    best_dist = bottleneck_distance(best, target, dimension=dimension)

    for _ in range(max_iter):
        candidate = best.copy()
        if rng.random() < 0.5 and list(nx.non_edges(candidate)):
            u, v = rng.choice(list(nx.non_edges(candidate)))
            candidate.add_edge(u, v)
        elif candidate.number_of_edges() > 0:
            u, v = rng.choice(list(candidate.edges()))
            candidate.remove_edge(u, v)
        else:
            continue

        dist = bottleneck_distance(candidate, target, dimension=dimension)
        if dist + epsilon < best_dist:
            best = candidate
            best_dist = dist
        if best_dist <= epsilon:
            break

    return best, float(best_dist)


def hyperbolic_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Poincar\u00e9 distance between ``x`` and ``y``."""

    x2 = np.dot(x, x)
    y2 = np.dot(y, y)
    diff = np.linalg.norm(x - y) ** 2
    denom = (1 - x2) * (1 - y2)
    if denom <= 0:
        return float("inf")
    cosh = 1 + 2 * diff / denom
    return float(np.arccosh(max(1.0, cosh)))


def hyperbolic_nearest_neighbors(
    embeddings: Dict[object, Iterable[float]], k: int = 5
) -> Dict[object, List[tuple[object, float]]]:
    """Return ``k`` nearest neighbors for each node in hyperbolic space."""

    vecs = {n: np.asarray(v, dtype=float) for n, v in embeddings.items()}
    neighbors: Dict[object, List[tuple[object, float]]] = {}
    for u, vec_u in vecs.items():
        dists = []
        for v, vec_v in vecs.items():
            if u == v:
                continue
            dist = hyperbolic_distance(vec_u, vec_v)
            dists.append((v, dist))
        dists.sort(key=lambda t: t[1])
        neighbors[u] = dists[:k]
    return neighbors


def hyperbolic_reasoning(
    embeddings: Dict[object, Iterable[float]],
    start: object,
    goal: object,
    *,
    max_steps: int = 5,
) -> List[object]:
    """Return a greedy path from ``start`` to ``goal`` in hyperbolic space."""

    vecs = {n: np.asarray(v, dtype=float) for n, v in embeddings.items()}
    if start not in vecs or goal not in vecs:
        raise ValueError("start or goal missing from embeddings")

    current = start
    path = [current]
    for _ in range(max_steps):
        if current == goal:
            break

        candidates = [n for n in vecs if n != current]
        if not candidates:
            break
        next_node = min(
            candidates,
            key=lambda n: hyperbolic_distance(vecs[current], vecs[n])
            + hyperbolic_distance(vecs[n], vecs[goal]),
        )
        if next_node == current or next_node in path:
            break
        path.append(next_node)
        current = next_node
    return path


def hyperbolic_hypergraph_reasoning(
    embeddings: Dict[object, Iterable[float]],
    hyperedges: Iterable[object],
    start: object,
    goal: object,
    *,
    penalty: float = 1.0,
    max_steps: int = 5,
) -> List[object]:
    """Return a greedy path in hyperbolic space considering hyperedges."""

    vecs = {n: np.asarray(v, dtype=float) for n, v in embeddings.items()}
    if start not in vecs or goal not in vecs:
        raise ValueError("start or goal missing from embeddings")
    hyper_set = set(hyperedges)

    current = start
    path = [current]
    for _ in range(max_steps):
        if current == goal:
            break

        candidates = [n for n in vecs if n != current]
        if not candidates:
            break
        next_node = min(
            candidates,
            key=lambda n: hyperbolic_distance(vecs[current], vecs[n])
            + hyperbolic_distance(vecs[n], vecs[goal])
            + (penalty if n in hyper_set else 0.0),
        )
        if next_node == current or next_node in path:
            break
        path.append(next_node)
        current = next_node
    return path


def hyperbolic_multi_curvature_reasoning(
    embeddings: Dict[float, Dict[object, Iterable[float]]],
    start: object,
    goal: object,
    *,
    weights: Optional[Dict[float, float]] = None,
    max_steps: int = 5,
) -> List[object]:
    """Return a greedy path combining several hyperbolic curvatures."""

    if not embeddings:
        return []
    base_key = next(iter(embeddings))
    nodes = set(embeddings[base_key])
    if start not in nodes or goal not in nodes:
        raise ValueError("start or goal missing from embeddings")

    weights = weights or {c: 1.0 for c in embeddings}
    vecs = {
        c: {n: np.asarray(v, dtype=float) for n, v in embs.items()}
        for c, embs in embeddings.items()
    }

    def _dist(u: object, v: object) -> float:
        return sum(
            weights.get(c, 1.0) * hyperbolic_distance(vecs[c][u], vecs[c][v])
            for c in vecs
            if u in vecs[c] and v in vecs[c]
        )

    current = start
    path = [current]
    for _ in range(max_steps):
        if current == goal:
            break
        candidates = [n for n in nodes if n != current]
        if not candidates:
            break
        next_node = min(candidates, key=lambda n: _dist(current, n) + _dist(n, goal))
        if next_node == current or next_node in path:
            break
        path.append(next_node)
        current = next_node

    return path


def fractal_net_prune(
    embeddings: Dict[object, Iterable[float]],
    *,
    tol: float = 1e-3,
) -> Tuple[Dict[int, np.ndarray], Dict[object, int]]:
    """Return pruned embedding centers and node mapping.

    The function greedily merges embedding vectors whose Euclidean distance is
    below ``tol``. Each node is assigned to a cluster represented by the mean
    of its members. The returned dictionary maps cluster index to centroid
    vector, while ``mapping`` tells which cluster each node belongs to.
    """

    centers: List[np.ndarray] = []
    mapping: Dict[object, int] = {}

    for node, vec in embeddings.items():
        arr = np.asarray(vec, dtype=float)
        assigned = False
        for idx, c in enumerate(centers):
            if np.linalg.norm(arr - c) <= tol:
                centers[idx] = (c + arr) / 2.0
                mapping[node] = idx
                assigned = True
                break
        if not assigned:
            centers.append(arr)
            mapping[node] = len(centers) - 1

    pruned = {i: centers[i] for i in range(len(centers))}
    return pruned, mapping
