"""Spatial technology loaders — Xenium / Visium / CosMx / MERSCOPE.

Mirror Seurat's ``LoadXenium`` / ``Load10X_Spatial`` / ``LoadNanostring`` /
``LoadVizgen``: read a platform's on-disk output into a Truecell object with the
expression assay AND a populated ``seurat.images`` (per-FOV centroids), so the
spatial accessors and ``truecell.spatial.analysis`` functions work immediately.

What each has been checked against:

* ``load_visium``: ``Load10X_Spatial`` on 10x's Visium mouse-brain slide.
* ``load_xenium``: ``LoadXenium`` on 10x's Xenium mouse brain (Xenium Onboard
  Analysis 1.0.1), and on synthetic bundles in the XOA 1.x and 2.0+ layouts,
  ``cells.parquet`` included.
* ``load_cosmx`` and ``load_merscope``: ``LoadNanostring`` and ``LoadVizgen`` on
  synthetic bundles in each platform's layout only.

The synthetic comparisons are ``tests/test_loaders_vs_seurat.py``.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..io import read_10x
from .centroids import _auto_radius
from .fov import create_fov, create_fovs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    """Read a cell/metadata table (.parquet, .csv, or .csv.gz)."""
    s = str(path)
    if s.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _feature_types(mtx_dir: Path) -> Optional[list[str]]:
    """Read the 3rd column (feature type) of a 10x features.tsv[.gz], if present.

    Xenium/Visium feature tables tag every row with a type — ``Gene Expression``
    for real genes and ``Negative Control Probe`` / ``Negative Control
    Codeword`` / ``Blank Codeword`` / ``Deprecated Codeword`` for QC controls.
    Returns one type per matrix row (file order), or ``None`` if unavailable.
    """
    feat = _first_existing(mtx_dir, ["features.tsv.gz", "features.tsv"])
    if feat is None:
        return None
    opn = gzip.open if str(feat).endswith(".gz") else open
    types: list[str] = []
    with opn(feat, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            types.append(parts[2] if len(parts) > 2 else "Gene Expression")
    return types


def _first_existing(base: Path, names: list[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


def _file_like_r(directory: Path, pattern: str) -> Optional[Path]:
    """The file an R reader's ``list.files(pattern = )`` lookup picks.

    ``ReadNanostring`` and ``ReadVizgen`` search every file name in the directory
    for a regular expression and take the first match in decreasing order, which
    is how a run's ``Lung5_Rep1_exprMat_file.csv`` is found without being named.
    """
    if not directory.is_dir():
        return None
    matches = sorted((p for p in directory.iterdir()
                      if p.is_file() and re.search(pattern, p.name)), reverse=True)
    return matches[0] if matches else None


def _build_spatial_object(
    counts: sp.spmatrix,
    feature_names: list[str],
    cell_names: list[str],
    coords: pd.DataFrame,
    assay: str,
    project: str,
    fov: Optional[Union[str, np.ndarray]] = None,
    meta_data: Optional[pd.DataFrame] = None,
    image_name: Optional[str] = None,
    radius: Optional[float] = None,
):
    """Assemble a Truecell object + images from expression + coordinate parts.

    ``image_name`` names the single FOV when the run is not split into several.
    It defaults to the lowercased assay name; the loaders pass the name Seurat's
    reader gives it. ``radius`` is that FOV's centroid radius. Seurat's readers
    build the centroids from every row of the coordinate table and only then
    keep the object's cells, so its automatic radius spans the whole table; the
    imaging loaders pass that. ``None`` takes the automatic radius of the cells
    placed, as Visium's plain FOV always has.
    """
    from ..truecell import create_truecell_object

    obj = create_truecell_object(
        counts, assay=assay, project=project,
        feature_names=feature_names, cell_names=cell_names,
    )
    kept = obj.cell_names()
    # The object's cells, in its order, selected rather than reindexed: a reindex
    # fills the cells a table lacks with NaN, which turned an integer FOV column
    # into floats and named the per-FOV images "1.0", "2.0", ...
    present = set(coords["cell"])
    coords = coords.set_index("cell").loc[[c for c in kept if c in present]]
    coords["cell"] = coords.index.to_numpy()
    coords = coords.dropna(subset=["x", "y"])
    fov_labels = coords[fov].to_numpy() if isinstance(fov, str) and fov in coords else None
    if fov_labels is None:
        name = image_name or assay.lower()
        obj.images = {name: create_fov(coords[["x", "y", "cell"]], type_="centroids",
                                       radius=radius, assay=assay, key=f"{name}_")}
    else:
        obj.images = create_fovs(coords[["x", "y", "cell"]], fov=fov_labels, assay=assay)
    if meta_data is not None:
        md = meta_data.reindex(kept)
        for c in md.columns:
            if c not in obj.meta_data.columns:
                obj.meta_data[c] = md[c].values
    return obj


# ---------------------------------------------------------------------------
# Xenium
# ---------------------------------------------------------------------------

def load_xenium(
    path: Union[str, Path],
    assay: str = "Xenium",
    fov: str = "fov",
    fov_column: Optional[str] = None,
    project: str = "Xenium",
    keep_controls: bool = False,
):
    """Load a 10x Xenium output bundle into a Truecell object with images.

    Expects (from the Xenium output folder):
      * ``cell_feature_matrix/`` — 10x MTX triplet (barcodes/features/matrix)
      * ``cells.parquet`` or ``cells.csv[.gz]`` — with ``cell_id``,
        ``x_centroid``, ``y_centroid`` (and optionally ``fov`` / transcript QC)

    Builds what Seurat's ``LoadXenium(molecule.coordinates = FALSE)`` builds from
    the same files:

    * the assay holds the ``Gene Expression`` features. Seurat routes the control
      and codeword features to separate assays; ``load_xenium`` drops them, and
      ``keep_controls=True`` keeps every row in the one assay instead;
    * ``cells.parquet`` is read when pandas has a parquet engine, as
      ``ReadXenium`` reads it when ``arrow`` is installed, and ``cells.csv.gz``
      otherwise. A ``cell_id`` stored as bytes is decoded, as ``ReadXenium``
      decodes it;
    * one image, named ``fov`` (``LoadXenium``'s argument, with its default),
      holds the centroids;
    * ``segmentation_method`` comes from the cells table, or is ``"cell"`` when
      the table has none, as ``LoadXenium`` fills it.

    truecell also carries the cells table's other columns into ``meta_data``.
    ``fov_column``, if given and present in the cells table, splits the cells into
    one image per value instead, named by the value.

    Checked against ``LoadXenium`` on 10x's Xenium Onboard Analysis 1.0.1 mouse
    brain, and on synthetic bundles in the XOA 1.x and 2.0+ layouts.
    ``LoadXenium`` can also read ``cell_feature_matrix.h5``; ``load_xenium``
    needs the MTX directory.
    """
    path = Path(path)
    mtx_dir = path / "cell_feature_matrix"
    if not mtx_dir.exists():
        raise FileNotFoundError(
            f"{mtx_dir} not found. load_xenium expects the unpacked "
            "cell_feature_matrix/ MTX directory."
        )
    counts, feats, cells = read_10x(mtx_dir)

    if not keep_controls:
        types = _feature_types(mtx_dir)
        if types is not None and len(types) == len(feats) and "Gene Expression" in types:
            mask = np.array([t == "Gene Expression" for t in types])
            counts = counts[mask, :]
            feats = [f for f, m in zip(feats, mask) if m]

    cell_candidates = [path / n for n in ("cells.parquet", "cells.csv.gz", "cells.csv")
                       if (path / n).exists()]
    if not cell_candidates:
        raise FileNotFoundError(f"No cells.parquet/csv found in {path}.")
    cdf = None
    for cf in cell_candidates:                      # prefer parquet, fall back to csv
        try:
            cdf = _read_table(cf)
            break
        except ImportError:                         # no parquet engine → try next
            continue
    if cdf is None:
        raise ImportError(
            "cells.parquet found but no parquet engine is installed. Install "
            "pyarrow, or provide cells.csv[.gz] alongside it."
        )
    rename = {"cell_id": "cell", "x_centroid": "x", "y_centroid": "y"}
    cdf = cdf.rename(columns={k: v for k, v in rename.items() if k in cdf.columns})
    if not {"cell", "x", "y"} <= set(cdf.columns):
        raise ValueError("cells table must contain cell_id, x_centroid, y_centroid.")
    # A cell_id column stored as bytes comes back from pandas as `bytes`, which
    # str() would turn into "b'aaabinlp-1'" and match no barcode.
    cdf["cell"] = [c.decode() if isinstance(c, bytes) else str(c) for c in cdf["cell"]]
    if "segmentation_method" not in cdf.columns:
        cdf["segmentation_method"] = "cell"

    split = fov_column if (fov_column and fov_column in cdf.columns) else None
    coords = cdf[["cell", "x", "y"] + ([split] if split else [])]
    meta = cdf.set_index("cell").drop(columns=["x", "y"], errors="ignore")
    return _build_spatial_object(counts, feats, [str(c) for c in cells], coords,
                                 assay, project, fov=split, meta_data=meta,
                                 image_name=fov, radius=_auto_radius(coords))


# ---------------------------------------------------------------------------
# Visium
# ---------------------------------------------------------------------------

def load_visium(
    path: Union[str, Path],
    assay: str = "Spatial",
    project: str = "Visium",
    image: bool = True,
    image_resolution: str = "lowres",
    filter_by_tissue: bool = True,
    slice_name: str = "slice1",
):
    """Load a 10x Visium output into a Truecell object with spot coordinates.

    Expects:
      * ``filtered_feature_bc_matrix/`` — 10x MTX triplet
      * ``spatial/tissue_positions.csv`` (or ``tissue_positions_list.csv``) with
        barcode, in_tissue, array row/col and pixel row/col columns

    Optionally (``image=True``, the default) also reads
    ``spatial/tissue_{hires,lowres}_image.png`` and ``spatial/scalefactors_json.json``,
    producing a :class:`~truecell.spatial.visium.VisiumV2` image that carries the H&E
    tissue photo — what ``spatial_dim_plot`` / ``spatial_feature_plot`` draw on. A
    bundle with no PNG still loads; you just get a plain FOV, as before.

    Parameters
    ----------
    image            : read the tissue image + scale factors (default True).
    image_resolution : 'lowres' (default) or 'hires'; falls back to whichever is
                       present. Matches ``Read10X_Image``'s
                       ``image.name = "tissue_lowres_image.png"``.
    filter_by_tissue : keep only spots with ``in_tissue == 1`` (default True),
                       matching ``Read10X_Image``'s ``filter.matrix = TRUE``.
    slice_name       : key for the FOV in ``obj.images`` (default ``"slice1"``,
                       the name ``Load10X_Spatial`` uses).

    Notes
    -----
    Spot coordinates stay in **full-resolution pixels**, matching
    ``tissue_positions.csv``. The scale factors convert them to image pixels —
    see :meth:`VisiumV2.scale_coordinates`.

    The three defaults above changed to match Seurat. Reading a bundle the way
    earlier versions did is ``load_visium(path, image_resolution="hires",
    filter_by_tissue=False, slice_name="spatial")``.
    """
    from .visium import VisiumV2, read_scale_factors, read_tissue_image

    path = Path(path)
    mtx_dir = _first_existing(path, ["filtered_feature_bc_matrix", "raw_feature_bc_matrix"])
    if mtx_dir is None:
        raise FileNotFoundError(f"No filtered_feature_bc_matrix/ in {path}.")
    counts, feats, cells = read_10x(mtx_dir)

    pos_file = _first_existing(path / "spatial",
                               ["tissue_positions.csv", "tissue_positions_list.csv"])
    if pos_file is None:
        raise FileNotFoundError(f"No spatial/tissue_positions.csv in {path}.")
    header = 0 if pos_file.name == "tissue_positions.csv" else None
    pos = pd.read_csv(pos_file, header=header)
    if header is None:
        pos.columns = ["barcode", "in_tissue", "array_row", "array_col",
                       "pxl_row_in_fullres", "pxl_col_in_fullres"]
    pos = pos.rename(columns={"barcode": "cell", "pxl_col_in_fullres": "x",
                              "pxl_row_in_fullres": "y"})
    pos["cell"] = pos["cell"].astype(str)
    cells = [str(c) for c in cells]

    if filter_by_tissue and "in_tissue" in pos.columns:
        pos = pos[pos["in_tissue"].astype(int) == 1]
        keep = set(pos["cell"])
        idx = [i for i, c in enumerate(cells) if c in keep]
        counts = counts[:, idx]              # drop off-tissue spots from the matrix too
        cells = [cells[i] for i in idx]

    coords = pos[["cell", "x", "y"]]
    obj = _build_spatial_object(counts, feats, cells, coords, assay, project,
                                image_name=slice_name)

    if not image:
        return obj

    sf_file = path / "spatial" / "scalefactors_json.json"
    sf = read_scale_factors(sf_file) if sf_file.exists() else None
    read = read_tissue_image(path / "spatial", resolution=image_resolution)
    if read is None and sf is None:
        return obj                      # nothing image-ish on disk; plain FOV is right
    img, res = read if read is not None else (None, image_resolution)
    obj.images = {
        name: VisiumV2.from_fov(fov, image=img, scale_factors=sf, image_resolution=res)
        for name, fov in obj.images.items()
    }
    return obj


# ---------------------------------------------------------------------------
# CosMx / Nanostring
# ---------------------------------------------------------------------------

def _cosmx_cell_names(df: pd.DataFrame) -> np.ndarray:
    """``ReadNanostring``'s cell names, ``paste0(cell_ID, "_", fov)``."""
    return (df["cell_ID"].astype(str) + "_" + df["fov"].astype(str)).to_numpy()


def load_cosmx(
    path: Union[str, Path],
    expr_file: Optional[str] = None,
    meta_file: Optional[str] = None,
    assay: str = "Nanostring",
    fov: str = "fov",
    fov_column: Optional[str] = None,
    project: str = "CosMx",
):
    """Load NanoString CosMx output (exprMat + metadata CSVs) into a Truecell object.

    Builds what Seurat's ``LoadNanostring`` builds from the same files, for the
    expression matrix and the cell centroids:

    * cells are named ``<cell_ID>_<fov>``, as ``ReadNanostring`` names them;
    * the ``cell_ID`` 0 row each FOV carries, for transcripts outside any cell, is
      dropped, and so is every cell with no counts;
    * every column but ``fov`` and ``cell_ID`` is a feature, the negative probes
      included;
    * one image, named ``fov`` (``LoadNanostring``'s argument), holds each cell's
      ``CenterX_global_px`` / ``CenterY_global_px``, for the cells in the object.

    ``LoadNanostring`` also loads the cell polygons (``*-polygons.csv``) and the
    transcript coordinates (``*_tx_file.csv``) into that image; ``load_cosmx``
    reads neither. truecell carries the metadata file's columns into
    ``meta_data``, which Seurat does not. ``fov_column`` splits the cells into one
    image per value of that metadata column instead, named by the value.

    ``expr_file`` / ``meta_file`` default to the files ``ReadNanostring`` finds in
    ``path``: the last name, in sorted order, matching ``*_exprMat_file.csv`` and
    ``*_metadata_file.csv``.

    Checked against ``LoadNanostring`` (Seurat 5.5.1) on a synthetic bundle in
    CosMx's file layout; not compared with R on a real run.
    """
    path = Path(path)
    expr = (Path(expr_file) if expr_file
            else _file_like_r(path, r"[_a-zA-Z0-9]*_exprMat_file.csv"))
    meta = (Path(meta_file) if meta_file
            else _file_like_r(path, r"[_a-zA-Z0-9]*_metadata_file.csv"))
    if expr is None or meta is None:
        raise FileNotFoundError(
            f"Could not locate *_exprMat_file.csv / *_metadata_file.csv in {path}.")

    edf = pd.read_csv(expr)
    mdf = pd.read_csv(meta)
    for label, df in (("exprMat", edf), ("metadata", mdf)):
        missing = [c for c in ("fov", "cell_ID") if c not in df.columns]
        if missing:
            raise ValueError(f"The CosMx {label} file has no {' or '.join(missing)} column.")

    edf = edf[edf["cell_ID"] != 0]
    gene_cols = [c for c in edf.columns if c not in ("fov", "cell_ID")]
    counts = sp.csc_matrix(edf[gene_cols].to_numpy(dtype=float).T)   # genes × cells
    cell_ids = _cosmx_cell_names(edf)
    has_counts = np.asarray(counts.sum(axis=0)).ravel() != 0
    counts, cell_ids = counts[:, has_counts], cell_ids[has_counts]

    mdf = mdf.copy()
    mdf["cell"] = _cosmx_cell_names(mdf)
    mcoord = mdf.rename(columns={"CenterX_global_px": "x", "CenterY_global_px": "y"})
    if not {"x", "y"} <= set(mcoord.columns):
        raise ValueError("The CosMx metadata file has no CenterX_global_px / "
                         "CenterY_global_px columns.")
    split = fov_column if (fov_column and fov_column in mcoord.columns) else None
    coords = mcoord[["cell", "x", "y"] + ([split] if split else [])]
    return _build_spatial_object(counts, gene_cols, list(cell_ids), coords,
                                 assay, project, fov=split,
                                 meta_data=mdf.set_index("cell"),
                                 image_name=fov, radius=_auto_radius(coords))


# ---------------------------------------------------------------------------
# MERSCOPE / Vizgen
# ---------------------------------------------------------------------------

def load_merscope(
    path: Union[str, Path],
    expr_file: Optional[str] = None,
    meta_file: Optional[str] = None,
    assay: str = "Vizgen",
    fov: str = "fov",
    fov_column: Optional[str] = None,
    project: str = "MERSCOPE",
    keep_controls: bool = False,
):
    """Load a Vizgen MERSCOPE output into a Truecell object with images.

    Expects, in ``path``:
      * ``cell_by_gene.csv`` — cell × gene counts (leading column = cell id)
      * ``cell_metadata.csv`` — with ``center_x`` / ``center_y`` (and usually
        ``fov``, ``volume``)

    Builds what Seurat's ``LoadVizgen`` builds from the same files, for the
    expression matrix and the cell centroids:

    * cell ids are the leading column of both files, as ``ReadVizgen`` reads them;
    * features matching ``^Blank-``, the blank barcodes, are dropped as
      ``LoadVizgen`` drops them. The match is case-sensitive, so ``blank-3``
      stays; ``keep_controls=True`` keeps them all;
    * cells with no counts stay, as they do in ``LoadVizgen``;
    * one image, named ``fov`` (``LoadVizgen``'s argument), holds the
      ``center_x`` / ``center_y`` centroids of the cells in the object.

    ``LoadVizgen`` also loads the cell boundaries (``cell_boundaries/*.hdf5``) and
    the transcripts of one z-plane into that image, and drops from the image any
    cell without a boundary; ``load_merscope`` reads neither and keeps those
    cells. truecell carries the metadata file's columns into ``meta_data``, which
    Seurat does not. ``fov_column`` splits the cells into one image per value of
    that metadata column instead, named by the value.

    ``expr_file`` / ``meta_file`` default to the files ``ReadVizgen`` finds in
    ``path``: the last name, in sorted order, matching ``cell_by_gene*.csv`` and
    ``cell_metadata*.csv``.

    Checked against ``LoadVizgen`` (Seurat 5.5.1) on a synthetic bundle in
    MERSCOPE's file layout; not compared with R on a real run.
    """
    path = Path(path)
    expr = (Path(expr_file) if expr_file
            else _file_like_r(path, r"cell_by_gene[_a-zA-Z0-9]*.csv"))
    meta = (Path(meta_file) if meta_file
            else _file_like_r(path, r"cell_metadata[_a-zA-Z0-9]*.csv"))
    if expr is None or meta is None:
        raise FileNotFoundError(
            f"Could not locate cell_by_gene.csv / cell_metadata.csv in {path}."
        )

    edf = pd.read_csv(expr)
    mdf = pd.read_csv(meta)

    gene_cols = [str(c) for c in edf.columns[1:]]
    if not keep_controls:
        gene_cols = [g for g in gene_cols if not re.match(r"Blank-", g)]
    if not gene_cols:
        raise ValueError(f"No gene columns found in {expr}.")
    cell_ids = edf.iloc[:, 0].astype(str).to_numpy()
    counts = sp.csc_matrix(edf[gene_cols].to_numpy(dtype=float).T)   # genes × cells

    mdf = mdf.copy()
    id_column = mdf.columns[0]
    mdf["cell"] = mdf[id_column].astype(str)
    mcoord = mdf.rename(columns={"center_x": "x", "center_y": "y"})
    if not {"x", "y"} <= set(mcoord.columns):
        raise ValueError("cell_metadata must contain center_x / center_y columns.")
    split = fov_column if (fov_column and fov_column in mcoord.columns) else None
    coords = mcoord[["cell", "x", "y"] + ([split] if split else [])]
    meta_data = mdf.set_index("cell").drop(columns=[id_column], errors="ignore")
    return _build_spatial_object(counts, gene_cols, list(cell_ids), coords,
                                 assay, project, fov=split, meta_data=meta_data,
                                 image_name=fov, radius=_auto_radius(coords))
