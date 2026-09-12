"""Graph-based community detection (clustering).

Mirrors Seurat's FindClusters().
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from functools import cache

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _res_label(r: float) -> str:
    """Format a resolution as R's ``as.character()`` would, for the column name.

    Seurat names the column ``paste0(graph.name, "_res.", resolution)``, so the
    label is R's numeric-to-string conversion and not Python's. The two differ on
    the values users actually type: ``str(1.0)`` is ``"1.0"`` where R gives
    ``"1"``, which would put the partition in ``RNA_snn_res.1.0`` and leave a
    ported script reading ``RNA_snn_res.1`` with a ``KeyError``.

    R's rule (at the default ``scipen = 0``) is to render both fixed and
    scientific and keep whichever is shorter, ties going to fixed — so 0.001
    stays ``0.001`` while 0.0001 becomes ``1e-04``. Verified against R 4.x on 21
    values spanning 1e-7 to 1234567.
    """
    r = float(r)
    fixed = np.format_float_positional(r, trim="-")
    sci = np.format_float_scientific(r, trim="-", exp_digits=2)
    return sci if len(sci) < len(fixed) else fixed


def find_clusters(
    seurat,
    resolution: float | Sequence[float] = 0.8,
    algorithm: int = 1,
    graph_name: str | None = None,
    random_seed: int = 0,
    n_iter: int = 10,
    group_singletons: bool = True,
    cluster_name: str | Sequence[str] | None = None,
    modularity_fxn: int = 1,
    n_start: int = 10,
    optimizer: str = "seurat",
    n_iterations: int | None = None,
) -> None:
    """Cluster the cells on a neighbour graph, as Seurat's ``FindClusters`` does.

    Mirrors R's ``FindClusters(pbmc)``, including its vector form
    ``FindClusters(pbmc, resolution = c(0.4, 0.8, 1.2))``.

    Algorithms 1–3 run Seurat's own modularity optimiser, translated from the C++
    behind ``RunModularityClusteringCpp``. It makes ``n_start`` random starts from
    one random stream, iterates each up to ``n_iter`` times, and keeps the
    partition with the highest modularity. Given the same graph and settings,
    truecell returns Seurat's partition label for label. Algorithm 4 is Leiden,
    run by leidenalg.

    Each resolution is written to its own metadata column, named
    ``{graph_name}_res.{resolution}`` as Seurat names it. ``seurat_clusters`` and
    the active identities are set from the **last** resolution in the sequence —
    last as given, not largest, which is what Seurat does.

    Parameters
    ----------
    resolution   : higher values give more / finer clusters. A sequence runs each
                   in turn, as R's ``resolution = c(...)`` does.
    algorithm    : 1 = Louvain, 2 = Louvain with multilevel refinement,
                   3 = smart local moving (SLM), 4 = Leiden.
    graph_name   : SNN graph to use (defaults to '{assay}_snn')
    random_seed  : seeds the optimiser. Leiden needs a seed above 0, so there a
                   seed of 0 or below becomes 1, with a warning, as in Seurat.
    n_iter       : iterations per random start. Louvain stops early once an
                   iteration changes nothing; SLM runs all of them. For Leiden it
                   is leidenalg's iteration count, and a negative value runs until
                   the partition is stable.
    group_singletons : absorb size-1 clusters into their best-connected
                   neighbour, as Seurat's ``GroupSingletons`` does. With
                   ``False`` they are all pooled into one ``"singleton"``
                   cluster instead — again matching Seurat.
    cluster_name : override the generated column name(s); one name, or one per
                   resolution. Seurat's ``cluster.name``. ``seurat_clusters`` is
                   still written either way.
    modularity_fxn : 1 = standard modularity; 2 = Seurat's alternative, in which
                   every cell weighs 1 and ``resolution`` must be at most 1.
    n_start      : random starts; the partition with the highest modularity wins.
    optimizer    : ``"seurat"`` runs Seurat's optimiser for algorithms 1–3.
                   ``"igraph"`` runs one pass of igraph's ``community_multilevel``
                   for algorithm 1 or 2, which is how truecell clustered up to
                   1.2; it reads neither ``n_start`` nor ``n_iter``.
    n_iterations : deprecated name for ``n_iter``, accepted for one more release.

    Notes
    -----
    Seurat's partition can depend on the processor. Built for arm64, its C++
    fuses a multiply-add in the rule that decides whether a cell moves, and that
    flips moves whose gain is within one rounding of zero. Such near-ties need
    edge weights that tie exactly, such as small fractions; they did not occur on
    PBMC 3k's shared nearest-neighbour graph. truecell uses plain IEEE arithmetic,
    which is Seurat as built for x86_64.

    Each resolution is clustered from the same seed rather than from a running
    RNG stream, so a partition does not depend on which resolutions preceded it
    or on the order they were given in. Verified against Seurat 5.5.1, where
    resolution 0.8 gives the same partition alone, in ``c(0.4, 0.8, 1.2)``, and
    in ``c(1.2, 0.8, 0.4)``.
    """
    if n_iterations is not None:
        warnings.warn(
            "`n_iterations` is deprecated and will be removed in the next release; "
            "use `n_iter`, which is Seurat's name for it.",
            FutureWarning,
            stacklevel=2,
        )
        n_iter = n_iterations

    assay_name = seurat.active_assay
    if graph_name is None:
        graph_name = f"{assay_name}_snn"
        if graph_name not in seurat.graphs:
            # Try knn graph
            graph_name = f"{assay_name}_nn"

    if graph_name not in seurat.graphs:
        raise KeyError(
            f"Graph '{graph_name}' not found. Run find_neighbors() first."
        )

    graph = seurat.graphs[graph_name]
    mat = graph._matrix  # scipy sparse (cells × cells)

    # Every argument is checked before the first resolution is clustered, so a
    # bad one raises before any clustering runs; a failure inside a later
    # resolution cannot leave earlier columns behind either, because nothing is
    # written until every resolution has succeeded (below).
    if algorithm not in (1, 2, 3, 4):
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Use 1 (Louvain), 2 (Louvain with "
            "multilevel refinement), 3 (SLM) or 4 (Leiden)."
        )
    if optimizer not in ("seurat", "igraph"):
        raise ValueError(f"Unknown optimizer {optimizer!r}. Use 'seurat' or 'igraph'.")
    seurat_optimizer = algorithm != 4 and optimizer == "seurat"
    if algorithm == 3 and optimizer == "igraph":
        raise ValueError("igraph has no smart local moving algorithm; algorithm=3 "
                         "needs optimizer='seurat'.")
    if algorithm in (1, 2) and optimizer == "igraph" and modularity_fxn != 1:
        raise ValueError("igraph optimises the standard modularity only; "
                         "modularity_fxn=2 needs optimizer='seurat'.")
    if seurat_optimizer:
        if modularity_fxn not in (1, 2):
            raise ValueError(f"`modularity_fxn` must be 1 (standard) or 2 "
                             f"(alternative); got {modularity_fxn!r}.")
        if n_start < 1:
            raise ValueError(f"`n_start` must be at least 1; got {n_start!r}.")
        if n_iter < 1:
            raise ValueError(f"`n_iter` must be at least 1; got {n_iter!r}.")

    # `np.number` is here for the numpy scalars a caller gets out of an array;
    # np.float64 subclasses float but np.float32 does not, and iterating one
    # raises rather than falling through to the sequence branch.
    if isinstance(resolution, (int, float, np.number)):
        resolutions = [float(resolution)]
    else:
        resolutions = [float(r) for r in resolution]
    if not resolutions:
        raise ValueError("`resolution` is empty; give at least one value.")
    if seurat_optimizer and modularity_fxn == 2 and max(resolutions) > 1.0:
        raise ValueError("The alternative modularity (modularity_fxn=2) needs "
                         f"resolution <= 1; got {max(resolutions)}.")

    if cluster_name is None:
        names = [f"{graph_name}_res.{_res_label(r)}" for r in resolutions]
    else:
        names = [cluster_name] if isinstance(cluster_name, str) else list(cluster_name)
        if len(names) != len(resolutions):
            raise ValueError(
                f"`cluster_name` has {len(names)} name(s) for "
                f"{len(resolutions)} resolution(s); give one per resolution."
            )

    leiden_seed = random_seed
    if algorithm == 4 and random_seed <= 0:
        warnings.warn(
            "`random_seed` must be greater than 0 for Leiden clustering; using 1, "
            "as Seurat does.",
            UserWarning,
            stacklevel=2,
        )
        leiden_seed = 1

    # The graph is converted once, not once per resolution.
    if seurat_optimizer:
        try:
            from ._modularity import build_network, run_modularity_clustering
        except ImportError as exc:  # numba is in the [analysis] extra
            raise ImportError(
                "find_clusters needs numba to run Seurat's modularity optimiser; "
                "install it with `pip install truecell[analysis]`."
            ) from exc
        network = build_network(mat, modularity_fxn)
    else:
        igraph_graph = _sparse_to_igraph(mat)

    columns = []
    for res, name in zip(resolutions, names):
        if seurat_optimizer:
            labels = run_modularity_clustering(network, res, algorithm, n_start, n_iter,
                                               random_seed, modularity_fxn)
        elif algorithm == 4:
            labels = _leiden_clustering(igraph_graph, res, leiden_seed, n_iter)
        else:
            labels = _louvain_clustering(igraph_graph, res, random_seed)

        str_labels = _group_singletons(
            np.asarray([str(c) for c in labels]), mat, group_singletons
        )
        present = sorted(set(str_labels),
                         key=lambda s: (not s.isdigit(), s.isdigit() and int(s), s))
        columns.append((name, pd.Categorical(str_labels, categories=present)))

    # Seurat builds every column before it assigns any, so a resolution that fails
    # part way leaves the object as it was.
    for name, series in columns:
        seurat.meta_data[name] = series
    # The last resolution given, not the largest — Seurat takes the last column
    # of its results frame, so `resolution = c(1.2, 0.8, 0.4)` leaves the object
    # sitting on 0.4.
    cluster_series = columns[-1][1]
    seurat.meta_data["seurat_clusters"] = cluster_series
    seurat.idents = cluster_series


# ------------------------------------------------------------------
# Singleton absorption — Seurat's GroupSingletons
# ------------------------------------------------------------------

def _group_singletons(
    ids: np.ndarray,
    snn: sp.spmatrix,
    group_singletons: bool,
) -> np.ndarray:
    """Absorb size-1 clusters, mirroring Seurat's ``GroupSingletons``.

    Every cluster holding exactly one cell is reassigned to whichever
    *non*-singleton cluster it is most connected to, scored as the mean SNN
    weight from that cell to every cell of the candidate cluster (Seurat:
    ``sum(subSNN) / (nrow * ncol)``). With ``group_singletons = False`` they are
    instead pooled into a single ``"singleton"`` cluster.

    Singletons are taken in order of first appearance, and each one scores the
    clusters as they stand by then: a singleton absorbed earlier counts towards
    its new cluster for the singletons after it, because Seurat looks the
    cluster's cells up again each time. The candidate clusters themselves are
    fixed before the loop, so one singleton never absorbs another.

    A tie goes where Seurat's ``set.seed(1); sample()`` sends it, over the tied
    clusters in order of first appearance. A cell with no edges ties across every
    cluster.
    """
    ids = ids.copy()
    values, first, counts = np.unique(ids, return_index=True, return_counts=True)
    singletons = set(values[counts == 1].tolist())
    if not singletons:
        return ids
    if not group_singletons:
        # `ids` is a fixed-width unicode array sized to the labels it already
        # holds, so a plain assignment would truncate "singleton" to "si".
        ids = ids.astype(object)
        ids[np.isin(ids, list(singletons))] = "singleton"
        return ids.astype(str)

    by_appearance = np.argsort(first, kind="stable")
    order = [str(values[k]) for k in by_appearance]
    position = {str(values[k]): int(first[k]) for k in by_appearance}
    # Candidate targets are fixed before the loop, as in Seurat: a cluster that
    # has just absorbed a singleton does not itself become a new target.
    targets = [v for v in order if v not in singletons]
    if not targets:
        return ids
    snn = snn.tocsr()
    members = {t: np.flatnonzero(ids == t).tolist() for t in targets}

    for s in (v for v in order if v in singletons):
        cell = position[s]
        row = snn[cell].toarray().ravel()
        connectivity = np.array([row[members[t]].mean() for t in targets])
        tied = np.flatnonzero(connectivity == connectivity.max())
        best = targets[tied[_r_sample_index(len(tied))]]
        ids[cell] = best
        members[best].append(cell)
    return ids


@cache
def _r_sample_index(m: int) -> int:
    """The 0-based index R draws for ``set.seed(1); sample.int(m, 1)``.

    ``GroupSingletons`` calls ``set.seed(1)`` right before its tie-breaking
    ``sample()``, so the draw depends on ``m`` alone. R (3.6 onwards) seeds its
    Mersenne Twister by scrambling the seed through the 69069 congruential
    generator and filling the state from it, then samples by rejection below the
    next power of two (``R_unif_index``). Checked against R 4.6.1 for m = 1..300
    and nine larger values.
    """
    if m <= 1:
        return 0
    s = 1
    for _ in range(50):
        s = (69069 * s + 1) & 0xFFFFFFFF
    s = (69069 * s + 1) & 0xFFFFFFFF  # the stream position, which R then resets
    key = np.empty(624, dtype=np.uint32)
    for k in range(624):
        s = (69069 * s + 1) & 0xFFFFFFFF
        key[k] = s
    generator = np.random.MT19937()
    generator.state = {"bit_generator": "MT19937", "state": {"key": key, "pos": 624}}
    bits = math.ceil(math.log2(m))
    while True:
        v, n = 0, 0
        while n <= bits:
            # unif_rand() never returns 0 or 1; a 32-bit draw can only hit 0.
            u = float(generator.random_raw()) * 2.3283064365386963e-10
            if u <= 0.0:
                u = 0.5 * 2.328306437080797e-10
            v = 65536 * v + math.floor(u * 65536)
            n += 16
        draw = v & ((1 << bits) - 1)
        if draw < m:
            return draw


def _key(label: str) -> float:
    """Sort key that keeps numeric cluster labels in numeric order."""
    try:
        return float(label)
    except ValueError:
        return float("inf")


# ------------------------------------------------------------------
# Louvain via igraph
# ------------------------------------------------------------------

def _seed_igraph(seed: int) -> None:
    """Seed igraph's own RNG (igraph does not use numpy's global RNG)."""
    import random as _random
    import igraph as ig

    _random.seed(seed)
    try:
        ig.set_random_number_generator(_random)
    except Exception:
        # Older igraph: fall back to seeding the stdlib RNG igraph reads from.
        pass


def _louvain_clustering(
    g,
    resolution: float,
    seed: int,
) -> np.ndarray:
    """One pass of igraph's multilevel Louvain (``optimizer="igraph"``)."""
    _seed_igraph(seed)

    try:
        result = g.community_multilevel(
            weights="weight",
            resolution=resolution,
            return_levels=False,
        )
    except TypeError:
        # Older igraph builds lack the `resolution` parameter. Keep the edge
        # weights — never silently downgrade to an unweighted clustering.
        warnings.warn(
            "This python-igraph version does not support the 'resolution' "
            "parameter for community_multilevel; clustering at the default "
            "resolution (1.0).",
            RuntimeWarning,
        )
        result = g.community_multilevel(weights="weight", return_levels=False)

    labels = np.array(result.membership)

    # Re-number clusters by size (largest = 0) — mirrors Seurat behavior
    labels = _renumber_by_size(labels)
    return labels


# ------------------------------------------------------------------
# Leiden via leidenalg
# ------------------------------------------------------------------

def _leiden_clustering(
    g,
    resolution: float,
    seed: int,
    n_iterations: int,
) -> np.ndarray:
    """Leiden community detection."""
    import leidenalg

    _seed_igraph(seed)

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight" if g.is_weighted() else None,
        n_iterations=n_iterations,
        seed=seed,
    )

    labels = np.array(partition.membership)
    labels = _renumber_by_size(labels)
    return labels


# ------------------------------------------------------------------
# Helper: sparse matrix → igraph Graph
# ------------------------------------------------------------------

def _sparse_to_igraph(mat: sp.spmatrix):
    """Convert scipy sparse adjacency matrix to an igraph Graph."""
    import igraph as ig

    mat = mat.tocoo()
    n = mat.shape[0]

    # Only upper triangle (undirected)
    mask = mat.row < mat.col
    edges = np.column_stack((mat.row[mask], mat.col[mask]))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = mat.data[mask]
    return g


def _renumber_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels so 0 is the largest cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    order = unique[np.argsort(-counts)]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[lab] for lab in labels])
