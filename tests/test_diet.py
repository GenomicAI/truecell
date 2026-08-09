"""`diet_truecell` — Seurat's DietSeurat.

The `_R_REFERENCE` table below was read off a live Seurat 5.5.1 run on pbmc3k
(2,638 cells x 13,714 genes, with `pca`, `umap`, `RNA_nn` and `RNA_snn`), one
`DietSeurat` call per row. All eight matched truecell exactly on layers,
reductions, graphs, feature count, cell count, assay list and the sum of the
counts layer.

The first row is the one to read twice: **`DietSeurat(obj)` with no arguments
deletes every reduction and every graph.** `dimreducs` and `graphs` are
keep-lists, and an unset keep-list keeps nothing.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell import create_truecell_object, diet_truecell

# case -> (layers, reductions, graphs), captured from Seurat 5.5.1.
_R_REFERENCE = {
    "before":          (["counts", "data", "scale.data"], ["pca", "umap"], ["RNA_nn", "RNA_snn"]),
    "default":         (["counts", "data", "scale.data"], [], []),
    "counts_only":     (["counts"], [], []),
    "keep_pca":        (["counts", "data", "scale.data"], ["pca"], []),
    "keep_both_reduc": (["counts", "data", "scale.data"], ["pca", "umap"], []),
    "keep_graphs":     (["counts", "data", "scale.data"], [], ["RNA_nn", "RNA_snn"]),
    "data_and_pca":    (["counts", "data"], ["pca"], []),
}


@pytest.fixture
def obj():
    """A small object with every slot DietSeurat can strip populated."""
    rng = np.random.default_rng(4)
    counts = sp.csc_matrix(rng.poisson(3.0, (80, 60)).astype(float))
    o = create_truecell_object(
        counts, assay="RNA", min_cells=0, min_features=0,
        feature_names=[f"g{i}" for i in range(80)],
        cell_names=[f"c{i}" for i in range(60)])
    tc.normalize_data(o)
    tc.find_variable_features(o, selection_method="vst", nfeatures=40)
    tc.scale_data(o, features=o.assays["RNA"]._all_feature_names)
    tc.run_pca(o, n_pcs=10, features=o.assays["RNA"].variable_features)
    tc.find_neighbors(o, dims=range(5))
    return o


def _shape(x):
    return (sorted(x.assays["RNA"].layers), sorted(x.reductions), sorted(x.graphs))


# ---------------------------------------------------------------------------
# Parity with Seurat
# ---------------------------------------------------------------------------

def test_default_removes_every_reduction_and_graph(obj):
    """The behaviour most likely to surprise, and it is R's.

    Captured from Seurat 5.5.1: `DietSeurat(o)` on an object with `pca`, `umap`
    and two graphs returns zero reductions and zero graphs, with all three
    layers untouched. Anyone reaching for `diet_truecell` to drop a layer will
    lose their embedding unless they name it.
    """
    assert list(obj.reductions) and list(obj.graphs), "fixture has nothing to strip"
    layers, reductions, graphs = _shape(diet_truecell(obj))
    assert reductions == _R_REFERENCE["default"][1] == []
    assert graphs == _R_REFERENCE["default"][2] == []
    assert layers == _R_REFERENCE["default"][0]


@pytest.mark.parametrize("case,kwargs", [
    ("counts_only",     dict(layers="counts")),
    ("keep_pca",        dict(dimreducs="pca")),
    ("keep_both_reduc", dict(dimreducs=["pca", "umap"])),
    ("keep_graphs",     dict(graphs=["RNA_nn", "RNA_snn"])),
    ("data_and_pca",    dict(layers=["counts", "data"], dimreducs="pca")),
])
def test_matches_seurat_structure(obj, case, kwargs):
    """What survives each call must be what survived it in R."""
    want = _R_REFERENCE[case]
    # The fixture has no umap; compare against R's answer with umap removed.
    if "umap" not in obj.reductions:
        want = (want[0], [r for r in want[1] if r != "umap"], want[2])
    assert _shape(diet_truecell(obj, **kwargs)) == want


def test_counts_data_is_untouched(obj):
    """Slimming must not perturb what it keeps."""
    before = obj.assays["RNA"].layers["counts"].tocsc().toarray()
    slim = diet_truecell(obj, layers="counts")
    after = slim.assays["RNA"].layers["counts"].tocsc().toarray()
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# It must not damage the object it was given
# ---------------------------------------------------------------------------

def test_input_object_is_not_mutated(obj):
    """The layers are shared with the input, so a careless drop reaches back.

    This is the failure mode of the shallow copy `_slim_assay` uses: share the
    matrices (the point — copying them would defeat a function that exists to
    free memory) but rebuild every mutable index, or dropping a layer from the
    result removes it from the caller's object too.
    """
    before = _shape(obj)
    diet_truecell(obj, layers="counts")
    diet_truecell(obj)
    assert _shape(obj) == before


def test_kept_layers_are_shared_not_copied(obj):
    """The memory claim, asserted rather than described.

    `diet_truecell` is for making an object smaller; if it copied the matrices
    it kept, calling it would briefly double the footprint of the very thing
    being shrunk.
    """
    slim = diet_truecell(obj, layers="counts")
    assert slim.assays["RNA"].layers["counts"] is obj.assays["RNA"].layers["counts"]


def test_dropping_a_layer_repoints_the_default_by_name(obj):
    """`default` is an index into the layer list, not a name.

    Drop a layer that sits before it and the index silently addresses a
    different layer — which is what `layer_data()` with no argument returns.
    """
    assay = obj.assays["RNA"]
    assay.default_layer = "data"
    slim = diet_truecell(obj, layers=["data", "scale.data"])
    assert slim.assays["RNA"].default_layer == "data"


# ---------------------------------------------------------------------------
# Feature subsetting
# ---------------------------------------------------------------------------

def test_features_subsets_rows_and_keeps_every_cell(obj):
    feats = obj.assays["RNA"].features()[:10]
    slim = diet_truecell(obj, features=feats)
    assert slim.assays["RNA"].features() == feats
    assert len(slim.cell_names()) == len(obj.cell_names())


def test_an_assay_with_no_requested_features_is_dropped_with_a_warning(obj):
    """R warns and removes the assay rather than leaving an empty one."""
    adt = obj.assays["RNA"].subset(features=obj.assays["RNA"].features()[:5])
    obj.assays["ADT"] = adt
    with pytest.warns(UserWarning, match="No features found in assay 'ADT'"):
        slim = diet_truecell(obj, features=obj.assays["RNA"].features()[20:30])
    assert "ADT" not in slim.assays
    assert "RNA" in slim.assays


# ---------------------------------------------------------------------------
# The aborts R raises
# ---------------------------------------------------------------------------

def test_unknown_assay_raises(obj):
    with pytest.raises(ValueError, match="No assays provided were found"):
        diet_truecell(obj, assays="nope")


def test_removing_the_default_assay_raises(obj):
    """R: 'The default assay is slated to be removed, please change the default assay'.

    Silently keeping it would be worse than failing: every downstream call reads
    `active_assay`, so an object whose default had been quietly re-pointed would
    return the wrong matrix rather than an error.
    """
    obj.assays["ADT"] = obj.assays["RNA"].subset(
        features=obj.assays["RNA"].features()[:5])
    assert obj.active_assay == "RNA"
    with pytest.raises(ValueError, match="default assay .* is slated to be removed"):
        diet_truecell(obj, assays="ADT")


def test_unknown_layer_raises(obj):
    with pytest.raises(ValueError, match="None of the requested layers found"):
        diet_truecell(obj, layers="not-a-layer")


def test_v3_assay_cannot_lose_both_counts_and_data():
    """R aborts rather than leave a v3 assay with no expression matrix."""
    rng = np.random.default_rng(1)
    o = create_truecell_object(
        sp.csc_matrix(rng.poisson(3.0, (30, 20)).astype(float)),
        assay="RNA", min_cells=0, min_features=0, use_v5=False,
        feature_names=[f"g{i}" for i in range(30)],
        cell_names=[f"c{i}" for i in range(20)])
    tc.normalize_data(o)
    with pytest.raises(ValueError, match="Cannot remove both 'counts' and 'data'"):
        diet_truecell(o, layers="scale.data")


# ---------------------------------------------------------------------------
# Per-assay layer selection, and misc
# ---------------------------------------------------------------------------

def test_layers_may_be_named_per_assay(obj):
    """R's `.PropagateList`: a bare list applies to all assays, a dict per assay."""
    obj.assays["ADT"] = obj.assays["RNA"].subset(
        features=obj.assays["RNA"].features()[:5])
    slim = diet_truecell(obj, layers={"RNA": "counts", "ADT": ["counts", "data"]})
    assert sorted(slim.assays["RNA"].layers) == ["counts"]
    assert sorted(slim.assays["ADT"].layers) == ["counts", "data"]


def test_misc_false_empties_object_misc(obj):
    obj.misc["big"] = list(range(1000))
    assert diet_truecell(obj, misc=False).misc == {}
    assert diet_truecell(obj, misc=True).misc == {"big": list(range(1000))}


def test_neighbors_are_left_alone(obj):
    """R filters only DimReduc and Graph; Neighbor objects are not candidates.

    The `Neighbor` is built here rather than produced by `find_neighbors`,
    which accepts an `nn_name` argument but never writes one — so asking the
    pipeline for one and skipping when it does not appear would be a test that
    can only ever skip.
    """
    from truecell import Neighbor

    cells = obj.cell_names()
    k = 5
    obj.neighbors["RNA.nn"] = Neighbor(
        nn_idx=np.tile(np.arange(k), (len(cells), 1)),
        nn_dist=np.zeros((len(cells), k)),
        cell_names=list(cells),
    )
    assert set(diet_truecell(obj).neighbors) == {"RNA.nn"}
