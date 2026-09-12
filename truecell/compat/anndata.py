"""AnnData ↔ Truecell conversion helpers.

Requires: pip install "truecell[anndata]"
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:
    # `truecell.truecell` pulls in the assay and reduction stack, which this module
    # is itself imported from — hence the deferred import inside `from_anndata`.
    # Annotation-only, so the return type resolves without reinstating the cycle.
    from ..truecell import Truecell

_INSTALL = 'anndata is required. Install with: pip install "truecell[anndata]"'


def as_anndata(
    seurat,
    assay: Optional[str] = None,
    spatial_key: str = "spatial",
    fov_key: str = "fov",
):
    """Convert a Truecell object to anndata.AnnData.

    Mapping
    -------
    active_assay counts/data layer → adata.X  (+ adata.layers for extras)
    meta_data                       → adata.obs
    assay.meta_features / meta_data → adata.var
    reductions["pca"].embeddings    → adata.obsm["X_pca"]
    reductions["pca"].loadings      → adata.varm["PCs"]
    graphs                          → adata.obsp
    misc                            → adata.uns
    images                          → adata.obsm[spatial_key], adata.obs[fov_key],
                                      adata.uns[spatial_key]

    Space goes where Scanpy and Squidpy look for it, and where ``from_anndata``
    reads it back:

    - ``obsm[spatial_key]`` holds each cell's (x, y), row for row with ``obs_names``,
      from the first image that places the cell: its ``Centroids`` point, or, in an
      image with no centroids, the centroid of its ``Segmentation`` ring
      (:meth:`~truecell.spatial.segmentation.Segmentation.as_centroids`). A cell no
      image places is NaN. Visium spots stay in full-resolution pixels, column
      first, the order ``scanpy.read_visium`` writes.
    - ``obs[fov_key]`` names the image each cell came from, as a categorical in the
      order of ``seurat.images``. A ``meta_data`` column already under that name is
      written the same way when it names the images, as a platform's own FOV column
      does, and kept as it is, with a warning, when it does not.
    - ``uns[spatial_key][name]`` holds each ``VisiumV2``'s tissue image, under its
      resolution, and its scale factors, under ``scalefactors_json.json``'s keys.

    AnnData has nowhere to put cell polygons or molecules, so those stay behind, as
    does a second image of cells another image already placed, such as a crop.
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(_INSTALL) from None

    from ..assay5 import StdAssay
    from ..truecell import Truecell

    if not isinstance(seurat, Truecell):
        raise TypeError(f"Expected Truecell, got {type(seurat).__name__}.")

    assay_name = assay or seurat.active_assay
    assay_obj = seurat.assays.get(assay_name)
    if assay_obj is None:
        raise KeyError(f"Assay '{assay_name}' not found.")

    cells = seurat.cell_names()

    # ---- X ----
    if isinstance(assay_obj, StdAssay):
        default_layer = assay_obj.default_layer
        X = assay_obj.layers.get(default_layer) if default_layer else None
        extra_layers = {
            k: v for k, v in assay_obj.layers.items() if k != default_layer
        }
        var_df = assay_obj.meta_data.copy() if assay_obj.meta_data is not None else pd.DataFrame()
        feature_names = assay_obj._all_feature_names
    else:
        from .._sparse import is_matrix_empty
        X = assay_obj.data if not is_matrix_empty(assay_obj.data) else assay_obj.counts
        extra_layers = {}
        if not is_matrix_empty(assay_obj.counts):
            extra_layers["counts"] = assay_obj.counts
        if not is_matrix_empty(assay_obj.scale_data):
            extra_layers["scale_data"] = assay_obj.scale_data
        var_df = assay_obj.meta_features.copy()
        feature_names = assay_obj._feature_names

    if X is None:
        X = sp.csc_matrix((len(feature_names), len(cells)))

    # anndata wants obs × var (cells × features), so transpose
    X_t = X.T if sp.issparse(X) else X.T

    # ---- obs ----
    obs = seurat.meta_data.copy()
    obs["ident"] = list(seurat.idents)

    # ---- var ----
    var = var_df.copy()
    var.index = feature_names

    # ---- layers ----
    layers_out = {}
    for layer_name, mat in extra_layers.items():
        layers_out[layer_name] = mat.T if sp.issparse(mat) else mat.T

    # ---- obsm ----
    obsm = {}
    for red_name, dr in seurat.reductions.items():
        key = f"X_{red_name.lower()}"
        obsm[key] = dr.cell_embeddings

    # ---- varm ----
    varm = {}
    for red_name, dr in seurat.reductions.items():
        if dr.feature_loadings.shape[0] == len(feature_names):
            varm[red_name.upper() + "s"] = dr.feature_loadings

    # ---- obsp ----
    obsp = {}
    for g_name, g in seurat.graphs.items():
        obsp[g_name] = g._matrix

    # ---- uns ----
    uns = dict(seurat.misc)
    uns["project_name"] = seurat.project_name
    uns["active_assay"] = assay_name

    # ---- space ----
    xy, placed_in = _cell_positions(seurat.images, cells)
    if xy is not None:
        obsm[spatial_key] = xy
        if fov_key not in obs.columns:
            obs[fov_key] = placed_in
        elif _names_the_images(obs[fov_key], placed_in):
            obs[fov_key] = _in_image_order(obs[fov_key], placed_in.categories)
        else:
            warnings.warn(
                f"meta_data already has a {fov_key!r} column that does not name the "
                f"images, so it is kept and the image names are not written. Pass "
                f"fov_key= to write them under another column.",
                stacklevel=2,
            )
    libraries = _visium_libraries(seurat.images, uns.get(spatial_key))
    if libraries:
        uns[spatial_key] = libraries

    return anndata.AnnData(
        X=X_t,
        obs=obs,
        var=var,
        layers=layers_out,
        obsm=obsm,
        varm=varm,
        obsp=obsp,
        uns=uns,
    )


def from_anndata(
    adata,
    assay: str = "RNA",
    spatial_key: str = "spatial",
    fov_key: str = "fov",
    image_resolution: str = "lowres",
) -> "Truecell":
    """Convert an anndata.AnnData to a Truecell object.

    Mapping
    -------
    adata.X                   → Assay5 'counts' layer  (transposed → features × cells)
    adata.layers              → additional Assay5 layers
    adata.obs                 → seurat.meta_data
    adata.var                 → assay.meta_data
    adata.obsm["X_pca"]       → seurat.reductions["pca"].cell_embeddings
    adata.obsm[spatial_key]   → seurat.images  (Centroids/FOV, split by obs[fov_key])
    adata.uns[spatial_key]    → the tissue image and scale factors of a VisiumV2
    adata.varm["PCs"]         → seurat.reductions["pca"].feature_loadings
    adata.obsp["connectivities"] → seurat.graphs
    adata.uns                 → seurat.misc

    ``spatial_key`` (default ``"spatial"``) is treated as physical coordinates
    and reconstructed into ``seurat.images`` — NOT as a dimensional reduction —
    so ``get_tissue_coordinates`` and the spatial-analysis functions work. If
    ``obs[fov_key]`` exists it splits the cells into one image per FOV, in
    category order when the column is categorical. A cell with NaN coordinates
    stays in the object and out of the images.

    An image named after a library in ``uns[spatial_key]``, the layout
    ``scanpy.read_visium`` and ``as_anndata`` write, becomes a ``VisiumV2`` with
    that library's tissue image and scale factors. With no ``obs[fov_key]``
    column, a single library names the single image. ``image_resolution`` picks
    which of the library's images to keep, falling back to the other, as
    ``load_visium`` does. The image and scale factors leave ``misc``; anything
    else the library holds, such as Scanpy's ``metadata``, stays there.
    """
    try:
        import anndata  # noqa: F401  — probed for availability, not used here
    except ImportError:
        raise ImportError(_INSTALL) from None
    if image_resolution not in ("hires", "lowres"):
        raise ValueError(
            f"image_resolution must be 'hires' or 'lowres', got {image_resolution!r}."
        )

    from ..assay5 import Assay5
    from ..dimreduc import DimReduc
    from ..graph import Graph
    from ..truecell import Truecell, _VERSION

    cells = list(adata.obs_names)
    features = list(adata.var_names)

    # ---- Build Assay5 layers ----
    X = adata.X
    if sp.issparse(X):
        X_t = X.T.tocsc()
    else:
        X_t = np.asarray(X).T

    layers: dict = {"counts": X_t}
    for layer_name, mat in adata.layers.items():
        if layer_name is None:
            continue                     # anndata 0.13 lists X itself as layers[None]
        if sp.issparse(mat):
            layers[layer_name] = mat.T.tocsc()
        else:
            layers[layer_name] = np.asarray(mat).T

    meta_data_var = adata.var.copy() if adata.var is not None else pd.DataFrame(index=features)
    assay_obj = Assay5(
        layers=layers,
        feature_names=features,
        cell_names=cells,
        meta_data=meta_data_var,
        key=f"{assay.lower()}_",
    )

    # ---- Metadata ----
    meta_data = adata.obs.copy() if adata.obs is not None else pd.DataFrame(index=cells)

    # ---- misc ----
    misc = dict(adata.uns) if adata.uns is not None else {}
    project_name = misc.pop("project_name", "SeuratProject")

    # ---- Spatial images (obsm[spatial_key] → Centroids/FOV, uns → VisiumV2) ----
    images: dict = {}
    if spatial_key in adata.obsm:
        images, used = _spatial_images(adata, assay, spatial_key, fov_key, image_resolution)
        if used:
            left = {}
            for name, library in misc[spatial_key].items():
                if name in used:
                    library = {k: v for k, v in library.items()
                               if k not in ("images", "scalefactors")}
                    if not library:
                        continue
                left[name] = library
            if left:
                misc[spatial_key] = left
            else:
                del misc[spatial_key]

    # ---- Reductions ----
    reductions: dict = {}
    for obsm_key, emb in adata.obsm.items():
        if obsm_key == spatial_key:
            continue                         # handled as images, not a reduction
        if obsm_key.startswith("X_"):
            red_name = obsm_key[2:]
        else:
            red_name = obsm_key

        varm_key = red_name.upper() + "s"
        loadings = adata.varm.get(varm_key) if adata.varm is not None else None

        dr = DimReduc(
            cell_embeddings=np.asarray(emb),
            cell_names=cells,
            feature_loadings=np.asarray(loadings) if loadings is not None else None,
            feature_names=features if loadings is not None else None,
            assay_used=assay,
            key=f"{red_name.upper()}_",
        )
        reductions[red_name] = dr

    # ---- Graphs ----
    graphs: dict = {}
    if adata.obsp is not None:
        for obsp_key, mat in adata.obsp.items():
            g = Graph(matrix=mat if sp.issparse(mat) else sp.csc_matrix(mat), cell_names=cells)
            graphs[obsp_key] = g

    return Truecell(
        assays={assay: assay_obj},
        meta_data=meta_data,
        active_assay=assay,
        graphs=graphs,
        reductions=reductions,
        images=images,
        project_name=project_name,
        misc=misc,
        version=_VERSION,
    )


# ---------------------------------------------------------------------------
# Space, both ways
# ---------------------------------------------------------------------------

def _centroids_of(image):
    """The Centroids that place an image's cells, or None when nothing does.

    An FOV's own centroids win, the default boundary before the others. Failing
    those, the centroids of its segmentation, as SeuratObject derives them.
    """
    from ..spatial.centroids import Centroids
    from ..spatial.fov import FOV
    from ..spatial.segmentation import Segmentation

    if isinstance(image, FOV):
        default = image.default_boundary()
        ordered = sorted(image.boundaries.items(), key=lambda item: item[0] != default)
        boundaries = [boundary for _, boundary in ordered]
    else:
        boundaries = [image]
    for boundary in boundaries:
        if isinstance(boundary, Centroids):
            return boundary
    for boundary in boundaries:
        if isinstance(boundary, Segmentation):
            return boundary.as_centroids()
    return None


def _cell_positions(images: Mapping, cells: list[str]):
    """Each cell's (x, y) from the first image that places it, and that image's name.

    Returns ``(None, None)`` when no image places a cell. Integer coordinates stay
    integers when every cell is placed, as Scanpy keeps a Space Ranger 1 slide's
    pixel positions; otherwise a cell no image places is NaN.
    """
    index = pd.Index(cells)
    source = np.full(len(cells), -1, dtype=np.intp)
    names: list[str] = []
    parts = []
    for name, image in images.items():
        points = _centroids_of(image)
        if points is None:
            continue
        frame = points.get_tissue_coordinates()
        rows = index.get_indexer(frame["cell"])
        take = np.flatnonzero(rows >= 0)
        take = take[source[rows[take]] < 0]
        if len(take) == 0:
            continue
        source[rows[take]] = len(names)
        names.append(name)
        parts.append((rows[take], frame["x"].to_numpy()[take], frame["y"].to_numpy()[take]))
    if not names:
        return None, None

    dtype = np.result_type(*[values for _, x, y in parts for values in (x, y)])
    placed = source >= 0
    if not np.issubdtype(dtype, np.number) or (
            not placed.all() and not np.issubdtype(dtype, np.floating)):
        dtype = np.dtype(np.float64)
    xy = np.full((len(cells), 2), np.nan if np.issubdtype(dtype, np.floating) else 0,
                 dtype=dtype)
    for rows, x, y in parts:
        xy[rows, 0] = x
        xy[rows, 1] = y
    return xy, pd.Categorical.from_codes(source, categories=names)


def _names_the_images(column: pd.Series, placed_in: pd.Categorical) -> bool:
    """Whether an existing column names, for every placed cell, its image."""
    placed = ~pd.isna(placed_in)
    return bool(np.array_equal(column.astype(str).to_numpy()[placed],
                               np.asarray(placed_in.astype(str))[placed]))


def _in_image_order(column: pd.Series, names) -> pd.Categorical:
    """A column that names the images, as a categorical of strings in image order.

    Left alone, a column of strings reaches an h5ad as a categorical with sorted
    categories, and ``from_anndata`` would rebuild the images in that order.
    """
    labels = column.astype(str).where(column.notna())
    names = [str(name) for name in names]
    known = set(names)
    extra = [label for label in pd.unique(labels.dropna()) if label not in known]
    return pd.Categorical(labels, categories=names + extra)


def _visium_libraries(images: Mapping, existing) -> dict:
    """``uns[spatial_key]`` in Scanpy's layout, one library per ``VisiumV2``.

    Entries already under the key are kept, so anything else a library holds, such
    as Scanpy's ``metadata``, survives a round trip.
    """
    from ..spatial.visium import VisiumV2

    visium = {name: image for name, image in images.items() if isinstance(image, VisiumV2)}
    if not visium:
        return {}
    libraries = dict(existing) if isinstance(existing, Mapping) else {}
    for name, image in visium.items():
        entry = dict(libraries[name]) if isinstance(libraries.get(name), Mapping) else {}
        entry["images"] = {} if image.image is None else {image.image_resolution: image.image}
        entry["scalefactors"] = ({} if image.scale_factors is None
                                 else image.scale_factors.to_dict())
        libraries[name] = entry
    return libraries


def _spatial_images(adata, assay: str, spatial_key: str, fov_key: str,
                    image_resolution: str) -> tuple[dict, set]:
    """``seurat.images`` from ``obsm[spatial_key]``, and the Visium libraries used."""
    from ..spatial.fov import create_fovs
    from ..spatial.visium import ScaleFactors, VisiumV2

    xy = np.asarray(adata.obsm[spatial_key])[:, :2]
    placed = ~pd.isna(xy).any(axis=1)
    if not placed.any():
        return {}, set()
    libraries = adata.uns.get(spatial_key) if adata.uns is not None else None
    libraries = libraries if isinstance(libraries, Mapping) else {}

    coords = pd.DataFrame({"x": xy[placed, 0], "y": xy[placed, 1],
                           "cell": adata.obs_names.to_numpy()[placed]})
    labels = adata.obs[fov_key].iloc[placed] if fov_key in adata.obs.columns else None
    default_name = assay.lower()
    if labels is None and len(libraries) == 1:
        default_name = str(next(iter(libraries)))
    images = create_fovs(coords, fov=labels, assay=assay, default_name=default_name)

    used = set()
    other = "hires" if image_resolution == "lowres" else "lowres"
    for name, fov in list(images.items()):
        library = libraries.get(name)
        if not isinstance(library, Mapping):
            continue
        stored = library.get("images")
        stored = stored if isinstance(stored, Mapping) else {}
        factors = library.get("scalefactors")
        if not stored and not isinstance(factors, Mapping):
            continue
        res = image_resolution if image_resolution in stored or other not in stored else other
        images[name] = VisiumV2.from_fov(
            fov,
            image=np.asarray(stored[res]) if res in stored else None,
            scale_factors=(ScaleFactors.from_dict(factors)
                           if isinstance(factors, Mapping) else None),
            image_resolution=res,
        )
        used.add(name)
    return images, used
