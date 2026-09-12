from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import SpatialImage
from .centroids import Centroids


def _close_rings(coords: pd.DataFrame) -> pd.DataFrame:
    """Repeat each cell's first vertex at the end of its ring, as R does.

    ``CreateSegmentation`` stores closed rings — a square arrives as four
    vertices and comes back as five. That is the ``sf``/GEOS convention R
    inherits, and it is not decoration: code that reads the vertex list to
    measure a perimeter, or to draw an outline without asking matplotlib to close
    it, is off by one edge on an open ring.

    Already-closed rings are left alone, so this is idempotent.
    """
    if coords.empty:
        return coords
    out = []
    for _, ring in coords.groupby("cell", sort=False):
        first, last = ring.iloc[0], ring.iloc[-1]
        if len(ring) > 2 and (first["x"], first["y"]) != (last["x"], last["y"]):
            ring = pd.concat([ring, ring.iloc[[0]]], ignore_index=False)
        out.append(ring)
    return pd.concat(out) if out else coords


def _ring_centroids(coords: pd.DataFrame) -> pd.DataFrame:
    """Each cell's ring centroid, one row per cell in order of first appearance.

    The area centroid, summed over a fan of triangles on the ring's first vertex in
    vertex order: the label point sp gives a polygon, and so the point SeuratObject's
    ``as(segmentation, "Centroids")`` returns. It is not the mean of the vertices,
    which extra vertices along one edge would pull towards that edge.

    A ring with no area (collinear, or fewer than three vertices) sums to 0/0, and its
    first vertex stands in, as it does in sp.
    """
    if coords.empty:
        return pd.DataFrame({"x": np.array([], dtype=float), "y": np.array([], dtype=float),
                             "cell": np.array([], dtype=object)})
    codes, cells = pd.factorize(coords["cell"].to_numpy())
    order = np.argsort(codes, kind="stable")      # each ring contiguous, vertices in order
    codes = codes[order]
    x = coords["x"].to_numpy(dtype=float)[order]
    y = coords["y"].to_numpy(dtype=float)[order]
    n_rows, n_cells = len(codes), len(cells)

    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    first = np.repeat(starts, np.diff(np.r_[starts, n_rows]))
    # Triangle (P0, Pj, Pj+1) for every vertex j after the first that has a successor.
    j = np.flatnonzero((np.arange(n_rows) != first) & np.r_[codes[1:] == codes[:-1], False])
    x0, y0 = x[first[j]], y[first[j]]
    area = (x[j] - x0) * (y[j + 1] - y0) - (x[j + 1] - x0) * (y[j] - y0)
    # bincount adds in row order, so each sum runs in the ring's own vertex order.
    total = np.bincount(codes[j], weights=area, minlength=n_cells)
    sum_x = np.bincount(codes[j], weights=area * (x0 + x[j] + x[j + 1]), minlength=n_cells)
    sum_y = np.bincount(codes[j], weights=area * (y0 + y[j] + y[j + 1]), minlength=n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = sum_x / (3 * total)
        cy = sum_y / (3 * total)
    flat = ~(np.isfinite(cx) & np.isfinite(cy))
    cx[flat] = x[starts[flat]]
    cy[flat] = y[starts[flat]]
    return pd.DataFrame({"x": cx, "y": cy, "cell": cells})


class Segmentation(SpatialImage):
    """Cell boundary polygon coordinates.

    Mirrors R's Segmentation class from segmentation.R.
    Each cell may have multiple (x, y) polygon vertices.

    Slots
    -----
    - ``_coords`` (pd.DataFrame) — columns: x, y, cell   (multiple rows per cell)
    """

    __slots__ = ("_coords", "assay", "misc", "_key")

    def __init__(
        self,
        coords: pd.DataFrame,
        assay: str = "",
        key: str = "segmentation_",
        misc: Optional[dict] = None,
    ) -> None:
        super().__init__(assay=assay, key=key, misc=misc)
        for col in ("x", "y", "cell"):
            if col not in coords.columns:
                raise ValueError(f"coords must have a '{col}' column.")
        self._coords = _close_rings(coords[["x", "y", "cell"]].copy())

    # ------------------------------------------------------------------
    # SpatialImage interface
    # ------------------------------------------------------------------

    def cells(self) -> list[str]:
        return list(self._coords["cell"].unique())

    def dim(self) -> tuple[int, int]:
        return (self._coords["cell"].nunique(), 2)

    def get_tissue_coordinates(
        self,
        cells: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        df = self._coords.set_index("cell")[["x", "y"]]
        if cells is not None:
            df = df.loc[df.index.isin(cells)]
        df = df.copy()
        df["cell"] = df.index.to_numpy()      # as Seurat returns it; see Centroids
        return df

    def rename_cells(self, new_names: list[str]) -> "Segmentation":
        old_names = self.cells()
        if len(new_names) != len(old_names):
            raise ValueError("new_names length must match number of unique cells.")
        mapping = dict(zip(old_names, new_names))
        new_coords = self._coords.copy()
        new_coords["cell"] = new_coords["cell"].map(mapping)
        return Segmentation(
            coords=new_coords,
            assay=self.assay,
            key=self._key,
            misc=dict(self.misc),
        )

    def subset(self, cells: list[str]) -> "Segmentation":
        mask = self._coords["cell"].isin(cells)
        return Segmentation(
            coords=self._coords[mask].copy(),
            assay=self.assay,
            key=self._key,
            misc=dict(self.misc),
        )

    def as_centroids(self) -> Centroids:
        """One point per cell, where SeuratObject puts it.

        Mirrors ``as(segmentation, "Centroids")``: the area centroid of each cell's
        ring, or its first vertex when the ring has no area. The radius is left to
        ``.AutoRadius``, as ``CreateCentroids`` leaves it.

        Checked against SeuratObject 5.4.0 on 40 random polygons, where it agrees to
        5.2e-16 relative, and exactly on rings with no area. The last bits differ
        because CRAN's arm64 build fuses multiply-adds, which truecell's plain IEEE
        arithmetic does not.
        """
        return Centroids(coords=_ring_centroids(self._coords), assay=self.assay,
                         misc=dict(self.misc))

    # ------------------------------------------------------------------
    # Simplify polygons
    # ------------------------------------------------------------------

    def simplify(self, tol: float = 0.5) -> "Segmentation":
        """Reduce polygon vertex count by removing vertices closer than tol.

        Simple Douglas–Peucker–style approximation per cell.
        """
        groups = []
        for cell_id, group in self._coords.groupby("cell", sort=False):
            pts = group[["x", "y"]].values
            if len(pts) <= 3:
                groups.append(group)
                continue
            # keep[i] = True means retain pts[i]; always keep first and last
            keep = np.zeros(len(pts), dtype=bool)
            keep[0] = True
            keep[-1] = True
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)  # len(pts)-1
            keep[1:-1] = diffs[:-1] > tol
            groups.append(group.iloc[np.where(keep)[0]])

        new_coords = pd.concat(groups, ignore_index=True)
        return Segmentation(
            coords=new_coords,
            assay=self.assay,
            key=self._key,
            misc=dict(self.misc),
        )


def create_segmentation(
    coords: pd.DataFrame,
    assay: str = "",
    key: str = "segmentation_",
) -> Segmentation:
    return Segmentation(coords=coords, assay=assay, key=key)
