"""A subset keeps every slot aligned, whatever order its cells are given in.

`Truecell.subset(cells=...)` used to take the caller's order for the metadata, the
assay's cell axis, the reductions and the graphs, while every layer, the
identities and the image coordinates stayed in the object's own order. Each slot
was right under its own labels, and anything that pairs two slots by position —
`nCount` against the counts, expression against coordinates — paired one cell
with another. The Frontiers revision found it as a Moran's I gap against R on a
Xenium subset whose cells were listed in sorted barcode order.

R-derived expectations are marked "R:" and were read off SeuratObject 5.4.0:
`subset.Seurat` resolves cells with `intersect(colnames(x), cells)`,
`subset.StdAssay` resolves both axes with `MatchCells(ordered = TRUE)`, and the
identities are rebuilt with `Idents(x, drop = TRUE) <- Idents(x)[cells]`.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from truecell import (
    DimReduc,
    Graph,
    create_truecell_object,
    find_spatially_variable_features,
    get_tissue_coordinates,
    local_neighborhood,
)
from truecell.preprocessing import normalize_data, scale_data
from truecell.spatial.fov import create_fovs

N_CELLS, N_GENES = 60, 30
LEVELS = ["10", "2", "1"]     # deliberately not in lexicographic order


def _dense(m):
    return np.asarray(m.toarray() if sp.issparse(m) else m)


@pytest.fixture
def obj():
    """Every slot a subset has to carry: layers with spans of their own, idents
    with non-lexicographic levels, a reduction, a graph and centroid coordinates."""
    rng = np.random.default_rng(7)
    cells = [f"c{i:02d}" for i in range(N_CELLS)]
    genes = [f"G{i:02d}" for i in range(N_GENES)]
    xy = rng.uniform(0, 100, size=(N_CELLS, 2))
    counts = rng.poisson(2.0, size=(N_GENES, N_CELLS)).astype(float)
    # A gradient along x, so reading expression against the wrong coordinates
    # moves Moran's I by a great deal rather than by noise.
    counts[0] = np.round(xy[:, 0] / 5)
    o = create_truecell_object(counts=sp.csc_matrix(counts), feature_names=genes,
                               cell_names=cells)
    normalize_data(o)
    scale_data(o, features=genes[:12])            # scale.data spans only 12 features
    o.idents = pd.Categorical([LEVELS[i % 3] for i in range(N_CELLS)], categories=LEVELS)
    o.meta_data["label"] = [f"L{i % 4}" for i in range(N_CELLS)]
    o.reductions["pca"] = DimReduc(
        cell_embeddings=rng.normal(size=(N_CELLS, 4)), cell_names=cells,
        key="PC_", assay_used="RNA",
    )
    adjacency = rng.random((N_CELLS, N_CELLS)) * (rng.random((N_CELLS, N_CELLS)) < 0.2)
    o.graphs["RNA_snn"] = Graph(sp.csr_matrix(adjacency), cells, "RNA")
    o.images = create_fovs(
        pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cell": cells}),
        assay="RNA", default_name="fov",
    )
    return o


def _assert_same_state(got, ref):
    """Every slot that anything reads by position, compared one by one."""
    assert got.cell_names() == ref.cell_names()
    pd.testing.assert_frame_equal(got.meta_data, ref.meta_data)

    assert list(got.idents) == list(ref.idents)
    assert list(got.idents.categories) == list(ref.idents.categories)

    a, b = got.assays["RNA"], ref.assays["RNA"]
    assert a.cells() == b.cells()
    assert a.features() == b.features()
    assert a.layers_list() == b.layers_list()
    for layer in a.layers_list():
        assert a.cells(layer) == b.cells(layer)
        assert a.features(layer) == b.features(layer)
        np.testing.assert_array_equal(_dense(a.layer_data(layer)), _dense(b.layer_data(layer)))

    assert got.reductions["pca"].cells() == ref.reductions["pca"].cells()
    np.testing.assert_array_equal(got.reductions["pca"].cell_embeddings,
                                  ref.reductions["pca"].cell_embeddings)

    assert got.graphs["RNA_snn"].cells() == ref.graphs["RNA_snn"].cells()
    np.testing.assert_array_equal(_dense(got.graphs["RNA_snn"]._matrix),
                                  _dense(ref.graphs["RNA_snn"]._matrix))

    pd.testing.assert_frame_equal(get_tissue_coordinates(got), get_tissue_coordinates(ref))


def test_a_subset_in_object_order_matches_the_parent_by_name(obj):
    """The reference the order tests compare against must itself be right."""
    cells = obj.cell_names()[::3]
    sub = obj.subset(cells=cells)
    parent, child = obj.assays["RNA"], sub.assays["RNA"]

    pd.testing.assert_frame_equal(sub.meta_data, obj.meta_data.loc[cells])
    for layer in parent.layers_list():
        keep = [c for c in parent.cells(layer) if c in set(cells)]
        want = parent.layer_data(layer, cells=keep, features=parent.features(layer))
        np.testing.assert_array_equal(_dense(child.layer_data(layer)), _dense(want))

    by_name = pd.Series(list(obj.idents), index=obj.cell_names())
    assert list(sub.idents) == list(by_name.loc[cells])

    rows = [obj.cell_names().index(c) for c in cells]
    np.testing.assert_array_equal(sub.reductions["pca"].cell_embeddings,
                                  obj.reductions["pca"].cell_embeddings[rows])

    full = get_tissue_coordinates(obj).set_index("cell")
    part = get_tissue_coordinates(sub).set_index("cell")
    pd.testing.assert_frame_equal(part[["x", "y"]], full.loc[part.index, ["x", "y"]])


@pytest.mark.parametrize("seed", range(4))
def test_the_order_cells_arrive_in_does_not_change_the_subset(obj, seed):
    """R: `subset.Seurat` takes `intersect(colnames(x), cells)` — object order."""
    rng = np.random.default_rng(seed)
    names = obj.cell_names()
    picked = [names[i] for i in sorted(rng.choice(N_CELLS, size=40, replace=False))]
    shuffled = [picked[i] for i in rng.permutation(len(picked))]
    assert shuffled != picked                      # the request really is out of order

    ref = obj.subset(cells=picked)
    got = obj.subset(cells=shuffled)

    assert got.cell_names() == picked
    _assert_same_state(got, ref)


def test_ncount_is_the_column_sum_of_counts_by_position(obj):
    sub = obj.subset(cells=obj.cell_names()[::-2])     # reversed, every other cell
    counts = sub.assays["RNA"].layer_data("counts")
    np.testing.assert_allclose(sub.meta_data["nCount_RNA"].to_numpy(),
                               np.asarray(counts.sum(axis=0)).ravel())


def test_spatial_statistics_do_not_depend_on_the_request_order(obj):
    picked = obj.cell_names()[::2]
    in_order = find_spatially_variable_features(obj.subset(cells=picked), layer="data")
    reversed_ = find_spatially_variable_features(obj.subset(cells=picked[::-1]), layer="data")
    pd.testing.assert_frame_equal(reversed_, in_order)

    # And the fixture can see a misalignment at all: the same cells with their
    # expression columns reversed under unchanged labels — what the defect did —
    # move Moran's I by a lot (0.54 at most, measured), not by noise.
    misaligned = obj.subset(cells=picked)
    assay = misaligned.assays["RNA"]
    assay.set_layer_data("data", assay.layer_data("data")[:, ::-1],
                         cell_names=assay.cells("data"))
    moved = find_spatially_variable_features(misaligned, layer="data")["moransi"].reindex(in_order.index)
    assert (moved - in_order["moransi"]).abs().max() > 0.3

    pd.testing.assert_frame_equal(
        local_neighborhood(obj.subset(cells=picked[::-1]), group_by="label", k=5),
        local_neighborhood(obj.subset(cells=picked), group_by="label", k=5),
    )


def test_morans_i_finds_columns_by_the_layers_own_cell_names(obj):
    """A data layer stored in a different column order from `cell_names()` is
    still read cell by cell — the second line of defence behind `subset`."""
    want = find_spatially_variable_features(obj.subset(cells=obj.cell_names()), layer="data")

    shuffled = obj.subset(cells=obj.cell_names())
    assay = shuffled.assays["RNA"]
    names = assay.cells("data")
    perm = np.random.default_rng(0).permutation(len(names))
    assay.set_layer_data("data", assay.layer_data("data")[:, perm],
                         cell_names=[names[i] for i in perm])
    assert assay.cells("data") != shuffled.cell_names()

    pd.testing.assert_frame_equal(find_spatially_variable_features(shuffled, layer="data"), want)


def test_identity_levels_keep_their_order(obj):
    """R: levels `10 2 1` survive a subset. Rebuilding the factor from its
    values sorted them to `1 10 2`, which reorders plots and marker searches."""
    sub = obj.subset(cells=obj.cell_names()[:30])
    assert list(sub.idents.categories) == LEVELS


def test_a_level_no_retained_cell_carries_is_dropped(obj):
    """R: `Idents(x, drop = TRUE)` — levels `2 0 1` with every `0` removed leave
    `2 1`, still in that order."""
    keep = [c for c, i in zip(obj.cell_names(), obj.idents) if i != "2"]
    sub = obj.subset(cells=keep[::-1])
    assert list(sub.idents.categories) == ["10", "1"]
    by_name = pd.Series(list(obj.idents), index=obj.cell_names())
    assert list(sub.idents) == list(by_name.loc[sub.cell_names()])


def test_assay5_subset_keeps_its_own_order_on_both_axes(obj):
    """R: `subset.StdAssay` matches cells and features with
    `MatchCells(ordered = TRUE)`, so the assay's order wins over the request's.
    Called directly, not through the object, so nothing upstream can reorder."""
    assay = obj.assays["RNA"]
    cells, features = assay.cells()[::-3], assay.features()[::-2]
    sub = assay.subset(cells=cells, features=features)

    assert sub.cells() == [c for c in assay.cells() if c in set(cells)]
    assert sub.features() == [f for f in assay.features() if f in set(features)]
    assert list(sub.meta_data.index) == sub.features()
    for layer in assay.layers_list():
        assert sub.cells(layer) == [c for c in assay.cells(layer) if c in set(cells)]
        want = assay.layer_data(layer, cells=sub.cells(layer), features=sub.features(layer))
        np.testing.assert_array_equal(_dense(sub.layer_data(layer)), _dense(want))


def test_a_v3_assay_is_aligned_through_the_object():
    rng = np.random.default_rng(3)
    counts = sp.csc_matrix(rng.poisson(2.0, size=(10, 12)).astype(float))
    o = create_truecell_object(counts=counts, feature_names=[f"G{i}" for i in range(10)],
                               cell_names=[f"c{i:02d}" for i in range(12)], use_v5=False)
    sub = o.subset(cells=o.cell_names()[::-1])
    assert sub.cell_names() == sub.assays["RNA"].cells() == o.cell_names()
    np.testing.assert_allclose(sub.meta_data["nCount_RNA"].to_numpy(),
                               np.asarray(sub.assays["RNA"].counts.sum(axis=0)).ravel())


def test_a_cell_the_object_does_not_have_is_reported(obj):
    """Seurat drops an unknown name silently; a misspelt barcode is better raised."""
    with pytest.raises(KeyError, match="not_a_cell"):
        obj.subset(cells=["c00", "not_a_cell"])
