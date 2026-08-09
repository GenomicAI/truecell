"""Guards for parameters that used to be accepted and never read.

An AST sweep of `truecell/` for parameters never loaded in their own function
body turned up nine real cases after triage; these cover the user-facing ones.
The pattern they share with the `nn_name` defect that prompted the sweep is that
the docstring promised behaviour the code did not deliver, so each test drives
the parameter to a value where ignoring it gives a *different, wrong* answer.

R-derived expectations are marked "R:" and were read off a live Seurat 5.5.1
session, not from truecell's own output.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell import DimReduc, create_truecell_object


# ---------------------------------------------------------------------------
# calc_n(margin)
# ---------------------------------------------------------------------------

_A = np.array([[1.0, 0.0, 2.0],
               [0.0, 0.0, 3.0],
               [4.0, 5.0, 0.0]])


@pytest.mark.parametrize("mat", [_A, sp.csc_matrix(_A), sp.csr_matrix(_A)],
                         ids=["dense", "csc", "csr"])
def test_calc_n_margin_selects_the_axis(mat):
    """`margin` was documented and ignored — margin=1 returned column results.

    R's `apply` convention: 1 is rows, 2 is columns. The fixture is deliberately
    not square in its answers (column sums are all 5, row sums are 3/3/9), so a
    function ignoring `margin` cannot accidentally agree on both.
    """
    from truecell._utils import calc_n

    ncount_c, nfeat_c = calc_n(mat, margin=2)
    ncount_r, nfeat_r = calc_n(mat, margin=1)

    assert np.array_equal(ncount_c, _A.sum(axis=0))
    assert np.array_equal(nfeat_c, (_A != 0).sum(axis=0))
    assert np.array_equal(ncount_r, _A.sum(axis=1))
    assert np.array_equal(nfeat_r, (_A != 0).sum(axis=1))
    # Anti-vacuity: the two margins must actually differ on this fixture.
    assert not np.array_equal(ncount_c, ncount_r)


def test_calc_n_margin_works_on_a_lazy_matrix(tmp_path):
    """The out-of-core path has its own branch, and must not densify to serve it."""
    from truecell import open_lazy_matrix, write_lazy_matrix
    from truecell._utils import calc_n

    path = tmp_path / "m"
    write_lazy_matrix(sp.csc_matrix(_A), str(path))
    lazy = open_lazy_matrix(str(path))

    assert np.array_equal(calc_n(lazy, margin=2)[0], _A.sum(axis=0))
    assert np.array_equal(calc_n(lazy, margin=1)[0], _A.sum(axis=1))
    assert np.array_equal(calc_n(lazy, margin=1)[1], (_A != 0).sum(axis=1))


def test_calc_n_rejects_a_margin_that_is_neither():
    from truecell._utils import calc_n

    with pytest.raises(ValueError, match="margin must be 1"):
        calc_n(_A, margin=0)


# ---------------------------------------------------------------------------
# DimReduc.features(projected)
# ---------------------------------------------------------------------------

def _dr(n_feat=6, n_proj=0):
    rng = np.random.default_rng(0)
    kw = {}
    if n_proj:
        kw = {"feature_loadings_projected": rng.normal(size=(n_proj, 3)),
              "feature_names_projected": [f"p{i}" for i in range(n_proj)]}
    return DimReduc(
        cell_embeddings=rng.normal(size=(10, 3)),
        cell_names=[f"c{i}" for i in range(10)],
        feature_loadings=rng.normal(size=(n_feat, 3)),
        feature_names=[f"g{i}" for i in range(n_feat)],
        **kw,
    )


def test_features_projected_is_empty_when_there_are_no_projected_loadings():
    """It used to claim N features for a 0-row matrix.

    R: `Features.DimReduc` returns `rownames(Loadings(projected = projected))`,
    and NULL when that is empty. truecell never populates projected loadings
    (there is no `ProjectDim`), so every reduction hit this.
    """
    dr = _dr()
    assert dr.loadings(projected=True).shape[0] == 0
    assert dr.features(projected=True) == []
    assert dr.features() == [f"g{i}" for i in range(6)]


def test_features_projected_reports_its_own_axis():
    """The projected axis is a different feature set, not the same one.

    `ProjectDim` scores every gene in the assay where the reduction itself was
    computed on the variable features, so the two lists differ in length and
    content — which is why one shared list could not have been right.
    """
    dr = _dr(n_feat=6, n_proj=9)
    assert dr.features(projected=True) == [f"p{i}" for i in range(9)]
    assert dr.features(projected=False) == [f"g{i}" for i in range(6)]
    assert len(dr.features(True)) != len(dr.features(False))


@pytest.mark.parametrize("projected", [False, True])
def test_features_follows_the_loadings_not_the_stored_names(projected):
    """Names without a matrix must report nothing — R reads `rownames(Loadings())`.

    The realistic case is a reduction that carries feature names but no loadings
    at all (UMAP, t-SNE). Without this the accessor would announce N features
    for a matrix with zero rows, which is the same inconsistency the projected
    side had. Both branches are covered because each has its own guard, and a
    fixture that only ever supplies loadings cannot tell either of them apart.
    """
    rng = np.random.default_rng(2)
    kw = {"feature_names_projected": ["p0", "p1"]} if projected else \
         {"feature_names": ["g0", "g1"]}
    dr = DimReduc(
        cell_embeddings=rng.normal(size=(6, 2)),
        cell_names=[f"c{i}" for i in range(6)],
        **kw,
    )
    assert dr.loadings(projected=projected).shape[0] == 0
    assert dr.features(projected=projected) == []


def test_set_loadings_can_name_the_projected_features():
    dr = _dr()
    rng = np.random.default_rng(1)
    dr.set_loadings(rng.normal(size=(4, 3)), projected=True,
                    feature_names=["a", "b", "c", "d"])
    assert dr.features(projected=True) == ["a", "b", "c", "d"]


def test_the_projected_axis_survives_subset_and_rename():
    dr = _dr(n_feat=6, n_proj=4)
    kept = dr.subset(cells=[f"c{i}" for i in range(5)])
    assert kept.features(projected=True) == [f"p{i}" for i in range(4)]
    renamed = dr.rename_cells([f"x{i}" for i in range(10)])
    assert renamed.features(projected=True) == [f"p{i}" for i in range(4)]


def test_the_generic_routes_projected_through(_generic_check=None):
    """`Features(dr, projected=True)` is the public path this reaches users by."""
    from truecell import generics

    dr = _dr(n_feat=6, n_proj=4)
    assert generics.features(dr, True) == [f"p{i}" for i in range(4)]
    assert generics.features(dr, False) == [f"g{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# Truecell.reorder_ident(var)
# ---------------------------------------------------------------------------

@pytest.fixture
def four_idents():
    rng = np.random.default_rng(1)
    obj = create_truecell_object(
        sp.csc_matrix(rng.poisson(5.0, (60, 40)).astype(float)),
        assay="RNA", min_cells=0, min_features=0,
        feature_names=[f"g{i}" for i in range(60)],
        cell_names=[f"c{i}" for i in range(40)])
    obj.idents = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
    # per-ident means A=3 B=2 C=4 D=1, so ascending order is D,B,A,C
    obj.meta_data["score"] = [3.0] * 10 + [2.0] * 10 + [4.0] * 10 + [1.0] * 10
    return obj


def test_reorder_ident_orders_by_the_variable(four_idents):
    """R: `ReorderIdent(o, var="score")` gives levels D,B,A,C on this fixture.

    The old signature was `reorder_ident(ident, order)` and never read `ident` —
    it just applied `order`. This is Seurat's function instead.
    """
    got = list(four_idents.reorder_ident("score").idents.categories)
    assert got == ["D", "B", "A", "C"]


def test_reorder_ident_reverse_actually_reverses(four_idents):
    """A deliberate divergence: R's `reverse` is a no-op.

    R applies `max(x)+1-x` to the values of an already-sorted named vector and
    reads `names()` off the result, which leaves the order untouched — verified
    on Seurat 5.5.1, which returns D,B,A,C either way. Replicating that would
    ship yet another argument that does nothing.
    """
    assert list(four_idents.reorder_ident("score", reverse=True).idents.categories) \
        == ["C", "A", "B", "D"]


def test_reorder_ident_honours_afxn(four_idents):
    """`afxn` picks the summary; a spread-out ident reorders under max but not mean."""
    obj = four_idents
    # One outlier in B, sized so its *mean* stays below A's (2.8 < 3) while its
    # *max* clears C's (10 > 4). Too large a value moves both and the test would
    # pass for the wrong reason.
    scores = list(obj.meta_data["score"])
    scores[10] = 10.0
    obj.meta_data["score"] = scores
    by_mean = list(obj.reorder_ident("score", afxn=np.mean).idents.categories)
    obj.idents = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
    by_max = list(obj.reorder_ident("score", afxn=np.max).idents.categories)
    assert by_mean != by_max
    assert by_max[-1] == "B"


def test_reorder_ident_accepts_a_gene(four_idents):
    """R fetches `var` with FetchData, so a gene name works as well as metadata."""
    got = four_idents.reorder_ident("g0")
    assert set(got.idents.categories) == {"A", "B", "C", "D"}


def test_reorder_ident_keeps_every_cell_assigned(four_idents):
    """Levels missing from the summary would silently turn their cells into NaN."""
    out = four_idents.reorder_ident("score")
    assert not pd_isna(out.idents).any()
    assert set(map(str, out.idents)) == {"A", "B", "C", "D"}


def pd_isna(cat):
    import pandas as pd

    return pd.isna(pd.Categorical(cat))


# ---------------------------------------------------------------------------
# stitch_matrix is gone
# ---------------------------------------------------------------------------

def test_stitch_matrix_is_removed():
    """It had no callers, no tests, and ignored both of its name arguments —
    it `hstack`ed the blocks, where a real `StitchMatrix` aligns *by* the names."""
    import truecell._sparse as _sparse
    from truecell import generics

    assert not hasattr(_sparse, "stitch_matrix")
    assert not hasattr(generics, "stitch_matrix")
