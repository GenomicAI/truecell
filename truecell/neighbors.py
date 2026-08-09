"""KNN and SNN graph construction.

Mirrors Seurat's FindNeighbors().
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from .command import log_truecell_command
from .graph import Graph
from .neighbor import Neighbor


def find_neighbors(
    seurat,
    dims: Optional[Union[list[int], range]] = None,
    k_param: int = 20,
    assay: Optional[str] = None,
    reduction: str = "pca",
    graph_name: Optional[str] = None,
    return_neighbor: bool = False,
    compute_snn: Optional[bool] = None,
    prune_snn: float = 1 / 15,
    seed: int = 42,
) -> None:
    """Build KNN and SNN graphs from a low-dimensional embedding.

    Mirrors R's ``FindNeighbors(pbmc, dims = 1:10)``. Stores ``Graph`` objects in
    ``seurat.graphs[f"{graph_name}_nn"]`` and ``seurat.graphs[f"{graph_name}_snn"]``.

    Parameters
    ----------
    dims        : which PCs to use (0-indexed; default all available)
    k_param     : number of nearest neighbors, **including the cell itself**
    reduction   : which reduction to use ('pca' by default)
    graph_name  : prefix for graph names (defaults to active assay name). With
                  ``return_neighbor=True`` it names the ``Neighbor`` outright,
                  rather than acting as a prefix.
    return_neighbor : store the raw KNN result — indices and distances — as a
                  ``Neighbor`` in ``seurat.neighbors`` instead of building
                  graphs. Seurat's ``return.neighbor``.
    compute_snn : build the SNN graph. Defaults to ``not return_neighbor``,
                  as in Seurat, which cannot do both.
    prune_snn   : edges with Jaccard index below this are pruned (Seurat default 1/15)

    Notes
    -----
    ``return_neighbor=True`` stores a ``Neighbor`` under ``f"{assay}.nn"`` — a
    **dot**, where the graphs use an underscore (``RNA.nn`` against ``RNA_nn`` /
    ``RNA_snn``). That is Seurat's naming, and the separator is the only thing
    distinguishing the two in a printout.

    The stored indices are **0-based**, where R's ``Indices()`` are 1-based.
    Everything else — the k columns, self first at distance 0, the ordering — is
    the same.
    """
    assay_name = assay or seurat.active_assay
    if compute_snn is None:
        compute_snn = not return_neighbor
    elif compute_snn and return_neighbor:
        # R warns and computes no SNN rather than refusing the call.
        warnings.warn(
            "The SNN graph is not computed if return_neighbor is True.",
            stacklevel=2,
        )
        compute_snn = False

    # Get embeddings
    if reduction not in seurat.reductions:
        raise KeyError(f"Reduction '{reduction}' not found. Run run_pca() first.")
    dr = seurat.reductions[reduction]
    embeddings = dr.cell_embeddings  # (cells × dims)

    if dims is None:
        emb = embeddings
    else:
        dims_list = list(dims)
        emb = embeddings[:, dims_list]

    cells = seurat.cell_names()
    n_cells = len(cells)

    # Build KNN
    nn_idx, nn_dist = _build_knn(emb, k_param, seed)

    if return_neighbor:
        # The distances have always been computed here and thrown away; this is
        # the branch that keeps them. Seurat stores no graph at all in this mode.
        seurat.neighbors[graph_name or f"{assay_name}.nn"] = Neighbor(
            nn_idx=nn_idx,
            nn_dist=nn_dist,
            cell_names=cells,
            alg_info={"k_param": k_param, "reduction": reduction,
                      "assay": assay_name},
        )
    else:
        prefix = graph_name or assay_name
        seurat.graphs[f"{prefix}_nn"] = Graph(
            matrix=_knn_to_sparse(nn_idx, n_cells),
            cell_names=cells, assay_used=assay_name,
        )
        if compute_snn:
            seurat.graphs[f"{prefix}_snn"] = Graph(
                matrix=_build_snn(nn_idx, n_cells, k_param, prune_snn),
                cell_names=cells, assay_used=assay_name,
            )

    log_truecell_command(
        seurat, "FindNeighbors", assay=assay_name, reduction=reduction,
        params={"k_param": k_param, "prune_snn": prune_snn,
                "return_neighbor": return_neighbor, "compute_snn": compute_snn,
                "dims": list(dims) if dims is not None else None},
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_knn(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nn_idx, nn_dist) of shape (n_cells, k), *including self*.

    Matches Seurat, whose ``k.param`` neighbourhood includes the cell itself
    (column 0, at distance 0), i.e. k total entries / k-1 other cells.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return indices, distances


def _knn_to_sparse(nn_idx: np.ndarray, n_cells: int) -> sp.csc_matrix:
    """Build the binary KNN adjacency matrix (nn_idx includes self).

    Seurat's ``nn`` graph is the raw *directed* KNN —
    ``sparseMatrix(i = i, j = j, x = 1)`` over the ranked neighbour table, with
    no symmetrisation. So every row has exactly ``k`` entries and the column
    sums vary with how often a cell is chosen as someone else's neighbour;
    ``nnz`` is exactly ``n * k``. Symmetrising here would both change those
    counts and destroy the in-degree signal (hub cells appear in many
    neighbourhoods). Consumers that need symmetry — ``run_umap``, the graph
    branch of ``run_spca`` — symmetrise at the point of use, as Seurat does.
    """
    n, k = nn_idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.flatten()
    data = np.ones(len(rows), dtype=np.float64)
    return sp.coo_matrix(
        (data, (rows, cols)), shape=(n_cells, n_cells)
    ).tocsc()


def _build_snn(
    nn_idx: np.ndarray,
    n_cells: int,
    k: int,
    prune_snn: float,
) -> sp.csc_matrix:
    """Build sparse SNN graph with Jaccard similarity weights.

    SNN[i,j] = |NN(i) ∩ NN(j)| / |NN(i) ∪ NN(j)|, where each neighbourhood has
    ``k`` members (self included, matching Seurat). Edges with Jaccard below
    ``prune_snn`` are dropped.

    The diagonal is *kept*: a cell shares its whole neighbourhood with itself,
    so ``SNN[i,i] = k / (2k − k) = 1``. Seurat's ``ComputeSNN`` stores it (all
    ``n`` diagonal entries are 1) and downstream code relies on that — notably
    ``RunUMAP.Graph``, which zeroes the diagonal itself rather than assuming it
    is absent. Dropping it here made the stored graph disagree with Seurat's in
    both ``nnz`` and ``sum``.

    The intersection is computed via a sparse matrix product and kept sparse
    throughout — never materialised as a dense n×n array — so this scales to
    large cell counts.
    """
    n = nn_idx.shape[0]
    # Membership matrix M[i, j] = 1 if j ∈ NN(i). nn_idx already includes self,
    # so each row has exactly k ones.
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.flatten()
    vals = np.ones(len(rows), dtype=np.float64)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float64)

    # |NN(i) ∩ NN(j)| as a *sparse* product (nonzero only for overlapping
    # neighbourhoods). Seurat computes this in double; float32 here put ~3e-08
    # on every stored weight. That did not change *which* edges survive — the
    # weak Python ``prune_snn`` is cast down to float32 for the comparison, so
    # both sides round the same way — but the weights are exact ratios of small
    # integers, so agreeing with Seurat to machine epsilon is free.
    inter = (M @ M.T).tocoo()
    r, c, inter_vals = inter.row, inter.col, inter.data

    # |NN(i) ∪ NN(j)| = k + k − |intersection|.
    union = 2 * k - inter_vals
    jaccard = inter_vals / np.maximum(union, 1.0)

    keep = jaccard >= prune_snn
    snn = sp.csc_matrix(
        (jaccard[keep], (r[keep], c[keep])), shape=(n_cells, n_cells)
    )
    return snn
