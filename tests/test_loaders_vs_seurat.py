"""The spatial loaders read each platform's files the way Seurat's readers do.

`tests/data/loaders/` holds one synthetic bundle per file layout, and
`loader_r_reference.json` records what Seurat 5.5.1's `LoadNanostring`,
`LoadVizgen` and `LoadXenium` built from the same files
(`make_loader_r_reference.R`). The bundles plant the rows R's readers treat
specially: the background row each CosMx FOV carries, cells with no counts, cells
missing from one of two files, `Blank-` controls beside names that only look like
them, and cell ids stored as bytes. A loader that reads a layout differently fails
here rather than on someone's slide.

What this cannot show is whether a platform's current software still writes these
layouts: the bundles are synthetic.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import truecell

DATA = Path(__file__).resolve().parent / "data"
BUNDLES = DATA / "loaders"
REFERENCE = json.loads((DATA / "loader_r_reference.json").read_text())

LOADERS = {
    "cosmx": truecell.load_cosmx,
    "merscope": truecell.load_merscope,
    "xenium_csv": truecell.load_xenium,
    "xenium_parquet": truecell.load_xenium,
    "xenium_parquet_binary": truecell.load_xenium,
}
XENIUM = [name for name in LOADERS if name.startswith("xenium")]


def _load(name, **kwargs):
    if name.startswith("xenium_parquet"):
        # These bundles carry no cells.csv.gz to fall back to, so without a
        # parquet engine load_xenium cannot read them. The dev extra installs one.
        pytest.importorskip("pyarrow")
    return LOADERS[name](BUNDLES / name, **kwargs)


@pytest.fixture(scope="module", params=list(LOADERS))
def loaded(request):
    return _load(request.param), REFERENCE[request.param]


def test_the_reference_is_seurat_s():
    assert (REFERENCE["seurat"], REFERENCE["seurat_object"]) == ("5.5.1", "5.4.0")
    assert set(LOADERS) <= set(REFERENCE)


def test_the_cells_features_and_counts_are_seurat_s(loaded):
    obj, r = loaded
    assert obj.active_assay == r["assay"]
    assert obj.cell_names() == r["cells"]
    assert obj.feature_names() == r["features"]
    counts = obj.assays[r["assay"]].layers["counts"]
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    np.testing.assert_array_equal(counts, np.array(r["counts"], dtype=float))
    np.testing.assert_array_equal(obj.meta_data[f"nCount_{r['assay']}"].to_numpy(), r["n_count"])
    np.testing.assert_array_equal(obj.meta_data[f"nFeature_{r['assay']}"].to_numpy(),
                                  r["n_feature"])


def test_one_image_under_seurat_s_name_with_seurat_s_centroids(loaded):
    obj, r = loaded
    assert obj.image_names() == r["images"]
    centroids = obj.images[r["images"][0]].boundaries["centroids"]
    coords = centroids.get_tissue_coordinates()
    assert list(coords["cell"]) == r["centroid_cells"] == r["image_cells"]
    np.testing.assert_array_equal(coords["x"].to_numpy(), r["centroid_x"])
    np.testing.assert_array_equal(coords["y"].to_numpy(), r["centroid_y"])
    # R builds the centroids from every row of the coordinate table and then keeps
    # the object's cells, so the automatic radius spans the whole table.
    assert centroids.radius() == pytest.approx(r["centroid_radius"], rel=1e-12, abs=0)


@pytest.mark.parametrize("name", list(LOADERS))
def test_the_image_takes_its_name_from_fov(name):
    assert _load(name, fov="slide").image_names() == ["slide"]


@pytest.mark.parametrize("name", XENIUM)
def test_xenium_segmentation_method_is_seurat_s(name):
    obj = _load(name)
    assert list(obj.meta_data["segmentation_method"]) == REFERENCE[name]["segmentation_method"]


@pytest.mark.parametrize("name", XENIUM)
def test_xenium_keep_controls_keeps_what_seurat_routes_to_other_assays(name):
    r = REFERENCE[name]
    obj = _load(name, keep_controls=True)
    routed = [f for assay in r["assays"] for f in r["assay_features"][assay]]
    assert set(routed) <= set(obj.feature_names())
    assert obj.feature_names()[:len(r["features"])] == r["features"]


def test_cosmx_fov_column_gives_one_image_per_fov():
    obj = _load("cosmx", fov_column="fov")
    assert obj.image_names() == ["1", "2", "3"]
    placed = [cell for name in obj.image_names()
              for cell in obj.images[name].boundaries["centroids"].cells()]
    assert placed == REFERENCE["cosmx"]["image_cells"]


def test_merscope_keep_controls_keeps_the_blank_barcodes():
    obj = _load("merscope", keep_controls=True)
    assert obj.feature_names() == ["Gad1", "Sst", "Slc17a7", "Pvalb-2", "Blank-1", "Blank-2",
                                   "blank-3", "BlankLike"]
