"""AnnData carries space: `as_anndata` writes it and `from_anndata` reads it back.

The layout is the one Scanpy and Squidpy read: each cell's (x, y) in
``obsm["spatial"]``, the image it came from in ``obs["fov"]``, and a Visium library's
tissue image and scale factors in ``uns["spatial"][library]``. The Frontiers revision
found 1.2.0's `as_anndata` wrote none of it: on a 36,602-cell Xenium slide ``obsm``
came back empty, so a truecell object had no way into Squidpy or SpatialData.
"""
import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp

anndata = pytest.importorskip("anndata", reason="anndata not installed")

import truecell
from truecell.compat.anndata import as_anndata, from_anndata
from truecell.spatial.centroids import Centroids
from truecell.spatial.fov import FOV, create_fovs
from truecell.spatial.segmentation import Segmentation
from truecell.spatial.visium import ScaleFactors, VisiumV2

DATA = Path(__file__).resolve().parent / "data"
SCALEFACTORS = {"spot_diameter_fullres": 89.47199235723474, "tissue_hires_scalef": 0.17211704,
                "fiducial_diameter_fullres": 144.5316799616869,
                "tissue_lowres_scalef": 0.051635113}


def _counts(n_genes, n_cells, seed):
    return np.random.default_rng(seed).poisson(2.0, size=(n_genes, n_cells)).astype(float)


def _object(cells, seed=0):
    return truecell.create_truecell_object(
        counts=sp.csc_matrix(_counts(4, len(cells), seed)),
        feature_names=[f"g{i}" for i in range(4)], cell_names=list(cells))


def _centroids_fov(cells, x, y):
    return FOV(boundaries={"centroids": Centroids(pd.DataFrame({"x": x, "y": y, "cell": cells}))})


def _h5ad(adata, tmp_path):
    path = tmp_path / "space.h5ad"
    adata.write_h5ad(path)
    return anndata.read_h5ad(path)


# ---------------------------------------------------------------------------
# On-disk bundles for each platform's loader
# ---------------------------------------------------------------------------

def _write_xenium(root, fovs):
    genes = ["Gad1", "Slc17a7", "Sox9", "Pdgfra"]
    cells = [f"cell-{i}" for i in range(12)]
    mtx = root / "cell_feature_matrix"
    mtx.mkdir(parents=True)
    with gzip.open(mtx / "matrix.mtx.gz", "wb") as fh:
        sio.mmwrite(fh, sp.csc_matrix(_counts(len(genes), len(cells), 1)))
    with gzip.open(mtx / "features.tsv.gz", "wt") as fh:
        fh.writelines(f"{g}\t{g}\tGene Expression\n" for g in genes)
    with gzip.open(mtx / "barcodes.tsv.gz", "wt") as fh:
        fh.writelines(f"{c}\n" for c in cells)
    rng = np.random.default_rng(2)
    table = pd.DataFrame({"cell_id": cells, "x_centroid": rng.uniform(0, 3000, len(cells)),
                          "y_centroid": rng.uniform(0, 3000, len(cells))})
    if fovs:
        table["fov"] = ["B", "A", "B", "A", "C", "A", "B", "C", "A", "B", "C", "A"]
    table.to_csv(root / "cells.csv", index=False)
    return truecell.load_xenium(root, fov_column="fov" if fovs else None)


def _write_visium(root, integer_pixels=False):
    genes = ["Gad1", "Slc17a7", "Sox9"]
    barcodes = [f"AAAC{i}-1" for i in range(8)]
    mtx = root / "filtered_feature_bc_matrix"
    mtx.mkdir(parents=True)
    sio.mmwrite(mtx / "matrix.mtx", sp.csr_matrix(_counts(len(genes), len(barcodes), 3)))
    pd.DataFrame({"bc": barcodes}).to_csv(mtx / "barcodes.tsv", sep="\t", header=False,
                                          index=False)
    pd.DataFrame({"id": [f"ENSG{i}" for i in range(len(genes))], "sym": genes}).to_csv(
        mtx / "genes.tsv", sep="\t", header=False, index=False)

    spatial = root / "spatial"
    spatial.mkdir()
    positions = pd.DataFrame({
        "barcode": barcodes, "in_tissue": [1] * 8,
        "array_row": np.arange(8), "array_col": np.arange(8),
        "pxl_row_in_fullres": np.arange(8) * 97.0 + 13.25,
        "pxl_col_in_fullres": np.arange(8)[::-1] * 211.0 + 7.5,
    })
    if integer_pixels:                    # Space Ranger 1: no header, whole pixels
        for col in ("pxl_row_in_fullres", "pxl_col_in_fullres"):
            positions[col] = positions[col].round().astype(int)
        positions.to_csv(spatial / "tissue_positions_list.csv", header=False, index=False)
    else:
        positions.to_csv(spatial / "tissue_positions.csv", index=False)
    (spatial / "scalefactors_json.json").write_text(json.dumps(SCALEFACTORS))
    import matplotlib.image as mpimg
    rng = np.random.default_rng(4)
    for res, size in (("hires", 40), ("lowres", 12)):
        mpimg.imsave(spatial / f"tissue_{res}_image.png", rng.random((size, size, 3)))
    return truecell.load_visium(root), positions


def _write_cosmx(root):
    genes = ["CD3D", "MS4A1", "LYZ"]
    fov = [1, 1, 2, 2, 2, 1, 3, 3]
    expr = pd.DataFrame({"fov": fov, "cell_ID": [1, 2, 1, 2, 3, 3, 1, 2]})
    for gene, row in zip(genes, _counts(len(genes), len(fov), 5)):
        expr[gene] = row.astype(int)
    root.mkdir(parents=True)
    expr.to_csv(root / "run_exprMat_file.csv", index=False)
    rng = np.random.default_rng(6)
    meta = expr[["fov", "cell_ID"]].assign(CenterX_global_px=rng.uniform(0, 5e4, len(fov)),
                                           CenterY_global_px=rng.uniform(0, 5e4, len(fov)))
    meta.to_csv(root / "run_metadata_file.csv", index=False)
    return truecell.load_cosmx(root)


def _write_merscope(root):
    genes = ["Gad1", "Sst", "Blank-1"]
    cells = [str(1000 + i) for i in range(9)]
    by_gene = pd.DataFrame({"cell": cells})
    for gene, row in zip(genes, _counts(len(genes), len(cells), 7)):
        by_gene[gene] = row.astype(int)
    root.mkdir(parents=True)
    by_gene.to_csv(root / "cell_by_gene.csv", index=False)
    rng = np.random.default_rng(8)
    pd.DataFrame({"cell": cells, "fov": [0, 0, 1, 1, 1, 0, 2, 2, 2],
                  "center_x": rng.uniform(0, 9000, len(cells)),
                  "center_y": rng.uniform(0, 9000, len(cells))}).to_csv(
        root / "cell_metadata.csv", index=False)
    return truecell.load_merscope(root)


PLATFORMS = {
    "xenium": lambda root: _write_xenium(root, fovs=False),
    "xenium_fovs": lambda root: _write_xenium(root, fovs=True),
    "visium": lambda root: _write_visium(root)[0],
    "visium_integer_pixels": lambda root: _write_visium(root, integer_pixels=True)[0],
    "cosmx": _write_cosmx,
    "merscope": _write_merscope,
}


def _assert_same_space(back, obj):
    assert back.cell_names() == obj.cell_names()
    assert list(back.images) == list(obj.images)
    for name, image in obj.images.items():
        got = back.images[name]
        assert type(got) is type(image), name
        assert (got._key, got.assay) == (image._key, image.assay)
        pd.testing.assert_frame_equal(got.get_tissue_coordinates(),
                                      image.get_tissue_coordinates(), check_exact=True)
        assert list(got.boundaries) == list(image.boundaries)
        for boundary, points in image.boundaries.items():
            assert got.boundaries[boundary].radius() == points.radius()
        if isinstance(image, VisiumV2):
            assert got.radius() == image.radius()
            assert got.image_resolution == image.image_resolution
            assert got.scale_factors == image.scale_factors
            assert got.image.dtype == image.image.dtype
            np.testing.assert_array_equal(got.image, image.image)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("on_disk", [False, True], ids=["in_memory", "h5ad"])
@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_a_loaded_slide_comes_back_from_anndata_unchanged(tmp_path, platform, on_disk):
    """Coordinates to the bit and in dtype, cell and image order, keys, radii, and
    Visium's image and scale factors, through a file as well as in memory."""
    obj = PLATFORMS[platform](tmp_path / platform)
    adata = as_anndata(obj)
    if on_disk:
        adata = _h5ad(adata, tmp_path)
    back = from_anndata(adata, assay=obj.active_assay)
    _assert_same_space(back, obj)


def test_a_non_spatial_object_gains_no_spatial_fields(small_seurat):
    adata = as_anndata(small_seurat)
    assert "spatial" not in adata.obsm
    assert "fov" not in adata.obs.columns
    assert "spatial" not in adata.uns


def test_obsm_spatial_follows_obs_names_not_the_images_order():
    cells = [f"c{i}" for i in range(10)]
    obj = _object(cells)
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, size=(10, 2))
    order = rng.permutation(10)
    obj.images = {"slide": _centroids_fov([cells[i] for i in order], xy[order, 0], xy[order, 1])}

    adata = as_anndata(obj)

    np.testing.assert_array_equal(adata.obsm["spatial"], xy)
    assert list(adata.obs["fov"]) == ["slide"] * 10


def test_visium_spots_are_written_column_first_as_scanpy_reads_them(tmp_path):
    """scanpy 1.12.3's `read_visium` names 10x's fifth and sixth position columns
    `pxl_col_in_fullres` and `pxl_row_in_fullres` (the reverse of 10x's own header)
    and stacks them as [its row, its col], so `obsm["spatial"]` is 10x's (column,
    row): x across the image, then y down it. `uns["spatial"][library]` holds
    `images` by resolution and `scalefactors` under the JSON's own keys."""
    obj, positions = _write_visium(tmp_path)
    adata = as_anndata(obj)

    np.testing.assert_array_equal(adata.obsm["spatial"][:, 0], positions["pxl_col_in_fullres"])
    np.testing.assert_array_equal(adata.obsm["spatial"][:, 1], positions["pxl_row_in_fullres"])
    library = adata.uns["spatial"]["slice1"]
    assert list(library["images"]) == ["lowres"]
    np.testing.assert_array_equal(library["images"]["lowres"], obj.images["slice1"].get_image())
    assert library["scalefactors"] == SCALEFACTORS
    assert list(adata.obs["fov"].cat.categories) == ["slice1"]


def test_images_keep_their_order_when_the_cells_would_reverse_it(tmp_path):
    """`create_fovs` orders images by first appearance, which here is B's cells, so
    only the categorical's order can bring A back first."""
    cells = [f"c{i}" for i in range(12)]
    obj = _object(cells)
    x = np.arange(12, dtype=float)
    obj.images = {"A": _centroids_fov(cells[6:], x[6:], x[6:]),
                  "B": _centroids_fov(cells[:6], x[:6], x[:6])}

    back = from_anndata(_h5ad(as_anndata(obj), tmp_path))

    assert list(back.images) == ["A", "B"]
    assert back.images["A"].cells() == cells[6:]


def test_create_fovs_takes_categories_in_order_and_skips_unused_ones():
    coords = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "cell": ["a", "b", "c"]})
    labels = pd.Categorical(["a", "z", "a"], categories=["m", "z", "a"])
    assert list(create_fovs(coords, fov=labels)) == ["z", "a"]
    assert list(create_fovs(coords, fov=np.asarray(labels))) == ["a", "z"]


# ---------------------------------------------------------------------------
# Which point stands for a cell
# ---------------------------------------------------------------------------

def _r_segmentation_reference():
    ref = json.loads((DATA / "r_segmentation_centroids.json").read_text())
    hx = float.fromhex
    frame = pd.DataFrame([(hx(x), hx(y), p["cell"]) for p in ref["polygons"]
                          for x, y in zip(p["x"], p["y"])], columns=["x", "y", "cell"])
    expected = pd.DataFrame({"x": [hx(v) for v in ref["centroid_x"]],
                             "y": [hx(v) for v in ref["centroid_y"]],
                             "cell": [p["cell"] for p in ref["polygons"]]})
    return frame, expected


def test_a_segmentation_centroid_is_seuratobjects():
    """R: SeuratObject 5.4.0 `as(segmentation, "Centroids")`, written by
    `tests/data/make_segmentation_centroid_reference.R`.

    The 40 star-shaped rings agree to 5.1e-16 relative, not to the bit: CRAN's arm64
    build fuses multiply-adds, and truecell's arithmetic is plain IEEE. The mean of
    the vertices, the easy mistake, is about 1e-4 away on them. The other six are
    exact: a rectangle given closed, a triangle with extra vertices along one edge
    (centroid (1, 1), vertex mean (1.2, 0.6)), and four rings with no area, which
    fall back to their first vertex.
    """
    frame, expected = _r_segmentation_reference()
    got = Segmentation(frame).as_centroids().get_tissue_coordinates()

    assert list(got["cell"]) == list(expected["cell"])
    star = expected["cell"].str.startswith("star").to_numpy()
    assert star.sum() == 40 and (~star).sum() == 6
    np.testing.assert_allclose(got[["x", "y"]].to_numpy()[star],
                               expected[["x", "y"]].to_numpy()[star], rtol=1e-15, atol=0)
    np.testing.assert_array_equal(got[["x", "y"]].to_numpy()[~star],
                                  expected[["x", "y"]].to_numpy()[~star])


def test_an_image_with_only_segmentation_writes_its_centroids():
    frame, _ = _r_segmentation_reference()
    segmentation = Segmentation(frame)
    obj = _object(segmentation.cells())
    obj.images = {"cells": FOV(boundaries={"segmentation": segmentation})}

    adata = as_anndata(obj)

    expected = segmentation.as_centroids().get_tissue_coordinates()
    np.testing.assert_array_equal(adata.obsm["spatial"], expected[["x", "y"]].to_numpy())


def test_an_fovs_own_centroids_win_over_its_segmentation():
    """Xenium, CosMx and MERSCOPE report their own centroids, which are not the
    polygons' centroids; the default boundary being the segmentation changes nothing."""
    frame, _ = _r_segmentation_reference()
    segmentation = Segmentation(frame)
    derived = segmentation.as_centroids().get_tissue_coordinates()
    reported = Centroids(derived.assign(x=derived["x"] + 0.25, y=derived["y"] - 0.5)
                         .reset_index(drop=True))
    obj = _object(segmentation.cells())
    obj.images = {"cells": FOV(boundaries={"segmentation": segmentation, "centroids": reported})}
    assert obj.images["cells"].default_boundary() == "segmentation"

    adata = as_anndata(obj)

    np.testing.assert_array_equal(adata.obsm["spatial"],
                                  reported.get_tissue_coordinates()[["x", "y"]].to_numpy())


def test_a_cell_in_two_images_is_written_from_the_first():
    cells = [f"c{i}" for i in range(12)]
    x = np.arange(12, dtype=float)
    slide = _centroids_fov(cells, x, x)
    zoom = _centroids_fov(cells[:4], x[:4] + 1, x[:4] + 1)      # offset so the point tells

    obj = _object(cells)
    obj.images = {"slide": slide, "zoom": zoom}
    adata = as_anndata(obj)
    assert list(adata.obs["fov"].cat.categories) == ["slide"]
    np.testing.assert_array_equal(adata.obsm["spatial"][:, 0], x)

    obj.images = {"zoom": zoom, "slide": slide}
    adata = as_anndata(obj)
    assert list(adata.obs["fov"]) == ["zoom"] * 4 + ["slide"] * 8
    np.testing.assert_array_equal(adata.obsm["spatial"][:4, 0], x[:4] + 1)


def test_a_cell_no_image_places_is_nan_and_stays_out_of_the_images(tmp_path):
    cells = [f"c{i}" for i in range(12)]
    obj = _object(cells)
    whole = np.arange(8)                                        # integer coordinates
    obj.images = {"slide": _centroids_fov(cells[:8], whole, whole * 2)}

    adata = as_anndata(obj)
    assert adata.obsm["spatial"].dtype == np.float64            # NaN needs a float
    assert np.isnan(adata.obsm["spatial"][8:]).all()
    assert adata.obs["fov"].isna().tolist() == [False] * 8 + [True] * 4

    back = from_anndata(_h5ad(adata, tmp_path))
    assert back.cell_names() == cells
    assert back.image_names() == ["slide"]
    assert back.images["slide"].cells() == cells[:8]
    np.testing.assert_array_equal(back.get_tissue_coordinates()[["x", "y"]].to_numpy(),
                                  np.column_stack([whole, whole * 2]))

    # Without an FOV column nothing else keeps the NaN rows out of the one image.
    del adata.obs["fov"]
    alone = from_anndata(adata)
    assert alone.image_names() == ["rna"]
    assert alone.images["rna"].cells() == cells[:8]


# ---------------------------------------------------------------------------
# obs["fov"] when meta_data already has one
# ---------------------------------------------------------------------------

def test_an_existing_fov_column_that_names_the_images_becomes_their_categorical(tmp_path):
    """MERSCOPE's metadata carries its own `fov` (0, 1, 2), which the images are named
    after. It is written, without a warning, as the categorical in image order that
    Squidpy's `library_key` wants. Left as it was, a column of strings would reach an
    h5ad as a categorical with its categories sorted, and the images would come back
    in that order: the `xenium_fovs` round trip above caught exactly that."""
    obj = _write_merscope(tmp_path / "merscope")
    assert obj.meta_data["fov"].dtype.kind == "i"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adata = as_anndata(obj)
    assert list(adata.obs["fov"].cat.categories) == obj.image_names()
    assert adata.obs["fov"].astype(str).tolist() == obj.meta_data["fov"].astype(str).tolist()


def test_an_existing_fov_column_that_names_something_else_is_kept_with_a_warning():
    cells = [f"c{i}" for i in range(6)]
    obj = _object(cells)
    obj.images = {"slide": _centroids_fov(cells, np.arange(6.0), np.arange(6.0))}
    obj.meta_data["fov"] = ["run1"] * 6

    with pytest.warns(UserWarning, match="does not name the images"):
        adata = as_anndata(obj)
    assert adata.obs["fov"].tolist() == ["run1"] * 6

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adata = as_anndata(obj, fov_key="image")
    assert adata.obs["image"].tolist() == ["slide"] * 6
    assert from_anndata(adata, fov_key="image").image_names() == ["slide"]


# ---------------------------------------------------------------------------
# A file Scanpy wrote
# ---------------------------------------------------------------------------

def _scanpy_visium(libraries_images=("hires", "lowres")):
    """An AnnData laid out as `scanpy.read_visium` lays one out."""
    rng = np.random.default_rng(9)
    barcodes = [f"AAAC{i}-1" for i in range(6)]
    images = {"hires": rng.random((40, 40, 3), dtype=np.float32),
              "lowres": rng.random((12, 12, 3), dtype=np.float32)}
    return anndata.AnnData(
        X=sp.csr_matrix(_counts(6, 3, 10)),
        obs=pd.DataFrame({"in_tissue": 1, "array_row": np.arange(6), "array_col": np.arange(6)},
                         index=barcodes),
        var=pd.DataFrame(index=["Gad1", "Slc17a7", "Sox9"]),
        obsm={"spatial": np.column_stack([np.arange(6) * 211, np.arange(6) * 97]).astype(np.int64)},
        uns={"spatial": {"V1_Mouse_Brain": {
            "images": {r: images[r] for r in libraries_images},
            "scalefactors": dict(SCALEFACTORS),
            "metadata": {"chemistry_description": "Visium V1", "software_version": "1.1.0"},
        }}},
    ), images


@pytest.mark.parametrize("on_disk", [False, True], ids=["in_memory", "h5ad"])
def test_a_scanpy_visium_file_becomes_a_visiumv2(tmp_path, on_disk):
    adata, images = _scanpy_visium()
    if on_disk:
        adata = _h5ad(adata, tmp_path)

    obj = from_anndata(adata, assay="Spatial")

    assert obj.image_names() == ["V1_Mouse_Brain"]              # the library names the image
    (image,) = obj.images.values()
    assert isinstance(image, VisiumV2)
    assert image.image_resolution == "lowres"                   # load_visium's default
    np.testing.assert_array_equal(image.get_image(), images["lowres"])
    assert image.scale_factors == ScaleFactors.from_dict(SCALEFACTORS)
    assert image.radius() == SCALEFACTORS["spot_diameter_fullres"] / 2
    np.testing.assert_array_equal(image.get_tissue_coordinates()[["x", "y"]].to_numpy(),
                                  adata.obsm["spatial"])
    # The image and scale factors moved into the VisiumV2; the metadata did not.
    assert set(obj.misc["spatial"]) == {"V1_Mouse_Brain"}
    assert set(obj.misc["spatial"]["V1_Mouse_Brain"]) == {"metadata"}

    again = as_anndata(obj)
    library = again.uns["spatial"]["V1_Mouse_Brain"]
    assert set(library) == {"images", "scalefactors", "metadata"}
    assert library["metadata"]["software_version"] == "1.1.0"
    np.testing.assert_array_equal(again.obsm["spatial"], adata.obsm["spatial"])


def test_a_library_with_no_image_or_scale_factors_leaves_a_plain_fov():
    """As `load_visium` gives a plain FOV for a bundle with neither."""
    adata, _ = _scanpy_visium()
    adata.uns["spatial"]["V1_Mouse_Brain"] = {"metadata": {"software_version": "1.1.0"}}

    obj = from_anndata(adata)

    (image,) = obj.images.values()
    assert type(image) is FOV
    assert obj.misc["spatial"] == {"V1_Mouse_Brain": {"metadata": {"software_version": "1.1.0"}}}


def test_image_resolution_picks_the_image_and_falls_back_as_load_visium_does():
    adata, images = _scanpy_visium()
    hires = from_anndata(adata, image_resolution="hires").images["V1_Mouse_Brain"]
    assert hires.image_resolution == "hires"
    np.testing.assert_array_equal(hires.get_image(), images["hires"])

    lowres_only, _ = _scanpy_visium(libraries_images=("lowres",))
    fallback = from_anndata(lowres_only, image_resolution="hires").images["V1_Mouse_Brain"]
    assert fallback.image_resolution == "lowres"

    with pytest.raises(ValueError, match="hires"):
        from_anndata(adata, image_resolution="fullres")


def test_scale_factors_missing_from_a_file_are_nan():
    sf = ScaleFactors.from_dict({"tissue_lowres_scalef": 0.05})
    assert sf.lowres == 0.05
    assert np.isnan([sf.spot, sf.fiducial, sf.hires]).all()
    assert ScaleFactors.from_dict(sf.to_dict()).lowres == 0.05


def test_the_import_errors_name_truecells_extra(monkeypatch, small_seurat):
    monkeypatch.setitem(sys.modules, "anndata", None)
    with pytest.raises(ImportError, match=r'"truecell\[anndata\]"'):
        as_anndata(small_seurat)
    with pytest.raises(ImportError, match=r'"truecell\[anndata\]"'):
        from_anndata(None)
