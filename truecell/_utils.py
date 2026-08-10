from __future__ import annotations


import numpy as np
import scipy.sparse as sp



def calc_n(matrix, margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Compute nCount (sums) and nFeature (non-zero counts) along one margin.

    ``margin=2`` (the default) works down **columns** — one pair of numbers per
    cell, which is what Seurat's ``.CalcN`` does and what every caller here
    wants. ``margin=1`` works across **rows**, giving the same two numbers per
    feature.

    R's ``apply`` convention, not NumPy's: margin 1 is rows and 2 is columns, so
    ``margin`` names the axis being *indexed*, not the axis being reduced over.

    Returns (nCount, nFeature) as 1-D float arrays.
    """
    from .lazy import is_lazy

    if margin not in (1, 2):
        raise ValueError(f"margin must be 1 (rows) or 2 (columns), got {margin!r}")

    if is_lazy(matrix):
        # Seurat has a dedicated `.CalcN.IterableMatrix` for exactly this, and
        # for the same reason: these two numbers are wanted for every cell the
        # moment an object is built, so computing them densely would end an
        # on-disk layer's life in the constructor. Both are streaming
        # reductions over the store's own index arrays.
        if margin == 2:
            return (matrix.sum(axis=0).astype(float),
                    matrix.nnz_per_col().astype(float))
        # No streaming per-row nnz on the store, and densifying is exactly what
        # a lazy layer exists to avoid — so walk it in cell blocks instead.
        total = np.zeros(matrix.shape[0])
        nz = np.zeros(matrix.shape[0])
        for _start, _stop, block in matrix.col_blocks():
            total += np.asarray(block.sum(axis=1)).ravel()
            nz += np.diff(block.tocsr().indptr)
        return total.astype(float), nz.astype(float)

    axis = 0 if margin == 2 else 1
    if sp.issparse(matrix):
        ncount = np.asarray(matrix.sum(axis=axis)).flatten()
        indptr = (matrix.tocsc() if margin == 2 else matrix.tocsr()).indptr
        nfeature = np.diff(indptr).astype(float)
    else:
        arr = np.asarray(matrix)
        ncount = arr.sum(axis=axis).flatten()
        nfeature = (arr != 0).sum(axis=axis).flatten()
    return ncount.astype(float), nfeature.astype(float)


def match_cells(
    query: list[str],
    target: list[str],
    error_missing: bool = False,
) -> np.ndarray:
    """Return integer indices of query items in target (like R match()).

    Returns -1 for items not found. Raises if error_missing=True and any are absent.
    """
    target_index = {v: i for i, v in enumerate(target)}
    idx = np.array([target_index.get(q, -1) for q in query], dtype=int)
    if error_missing and (idx == -1).any():
        missing = [q for q, i in zip(query, idx) if i == -1]
        raise KeyError(f"Cells not found in target: {missing}")
    return idx


def intersect_names(a: list[str], b: list[str]) -> list[str]:
    b_set = set(b)
    return [x for x in a if x in b_set]


def unique_names(names: list[str]) -> bool:
    return len(names) == len(set(names))


def validate_cell_names(names) -> list[str]:
    names = list(names)
    if not unique_names(names):
        raise ValueError("Cell names must be unique.")
    return names


def validate_feature_names(names) -> list[str]:
    names = list(names)
    if not unique_names(names):
        raise ValueError("Feature names must be unique.")
    return names
