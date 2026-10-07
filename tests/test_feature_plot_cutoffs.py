"""``feature_plot`` cutoffs: Seurat's ``SetQuantile``, and a scale that cannot collapse silently.

The skills recommend ``min_cutoff="q05", max_cutoff="q95"`` as the fix for a washed-out
colour scale. ``feature_plot`` took those percentiles over every cell, so for a gene
detected in under 5% of cells both were 0 and the colour bar spanned (0, 1e-9): every
expressing cell the same colour. Seurat's ``SetQuantile`` takes the percentile over the
cells that express the feature, so the same call works there, and it accepts any
``q`` and one or two digits where ``feature_plot`` recognised only the strings
``"q05"`` and ``"q95"``, and failed on the rest.

The reference values below are from ``Seurat:::SetQuantile`` on Seurat 5.5.1.
"""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell.plotting import _set_quantile

# x = c(0, 0, 0, 0, 0.5, 1, 2, 3, 10, 20), run through Seurat:::SetQuantile
X = np.array([0, 0, 0, 0, 0.5, 1, 2, 3, 10, 20], dtype=float)
R_QUANTILES = {"q05": 0.625, "q5": 0.625, "q10": 0.75, "q50": 2.5, "q95": 17.5, "q99": 19.5, "q00": 0.5}


@pytest.mark.parametrize("cutoff,expected", sorted(R_QUANTILES.items()))
def test_a_quantile_is_seurats_over_the_expressing_cells(cutoff, expected):
    assert _set_quantile(cutoff, X) == pytest.approx(expected, rel=1e-12)


def test_the_zeros_do_not_count_toward_a_quantile():
    # Over every cell the 5th percentile of X is 0; over the expressing cells it is 0.625.
    assert np.percentile(X, 5) == 0.0
    assert _set_quantile("q05", X) == pytest.approx(0.625)
    # R's sparse example: 97 zeros and three expressers
    sparse = np.r_[np.zeros(97), [0.5, 1.5, 3.0]]
    assert _set_quantile("q05", sparse) == pytest.approx(0.6)
    assert _set_quantile("q95", sparse) == pytest.approx(2.85)


def test_negatives_are_not_expression_either():
    """`data[data > 0]`, so a scaled value below zero is left out, as in R (0.575)."""
    assert _set_quantile("q05", np.array([-2, -1, 0, 0.5, 1, 2, 3.0])) == pytest.approx(0.575)


def test_a_number_is_itself_and_a_feature_nothing_expresses_is_zero():
    assert _set_quantile(2.5, X) == 2.5 and _set_quantile(0, X) == 0.0
    assert _set_quantile("q05", np.zeros(5)) == 0.0                 # R gives NA; a scale needs a number


@pytest.mark.parametrize("bad", ["q100", "q", "q1.5", "95", "Q05", "q05 ", "", "median", "q-5"])
def test_anything_else_is_an_error_that_says_what_is_allowed(bad):
    with pytest.raises(ValueError, match="a number or a quantile such as 'q05'"):
        _set_quantile(bad, X)


@pytest.fixture(scope="module")
def obj():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    o = tc.create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(2.0, size=(30, 400)).astype(float)),
        feature_names=[f"G{i}" for i in range(30)], cell_names=[f"c{i}" for i in range(400)])
    tc.normalize_data(o)
    tc.find_variable_features(o, nfeatures=20)
    tc.scale_data(o)
    tc.run_pca(o, n_pcs=4)
    # exact expression vectors, as metadata columns
    sparse = np.zeros(400)
    sparse[:12] = np.linspace(0.5, 3.0, 12)                          # 3% of cells express it
    o.meta_data["sparse"] = sparse
    o.meta_data["dense"] = np.linspace(0.0, 10.0, 400)
    o.meta_data["flat"] = np.r_[np.zeros(380), np.ones(20)]          # every expresser is 1
    o.meta_data["silent"] = np.zeros(400)
    return o


def _clim(fig):
    """The colour scale of the first panel's expressing cells."""
    return tuple(round(float(v), 9) for v in fig.axes[0].collections[1].get_clim())


def _draw(obj, feature, **kwargs):
    plt = pytest.importorskip("matplotlib.pyplot")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fig = tc.feature_plot(obj, feature, reduction="pca", **kwargs)
    clim = _clim(fig)
    plt.close(fig)
    return clim, [str(w.message) for w in seen if "colour scale no range" in str(w.message)]


def test_the_documented_quantiles_work_for_a_gene_few_cells_express(obj):
    """The report: the colour bar was (0, 1e-9) and the expression gradient was gone."""
    expressed = obj.meta_data["sparse"].to_numpy()
    expressed = expressed[expressed > 0]
    clim, warned = _draw(obj, "sparse", min_cutoff="q05", max_cutoff="q95")
    assert clim == (round(np.percentile(expressed, 5), 9), round(np.percentile(expressed, 95), 9))
    assert clim[1] - clim[0] > 1.0                                    # a range, not 1e-9
    assert not warned


def test_every_quantile_is_accepted_not_only_q05_and_q95(obj):
    expressed = obj.meta_data["dense"].to_numpy()
    expressed = expressed[expressed > 0]
    clim, _ = _draw(obj, "dense", min_cutoff="q10", max_cutoff="q99")
    assert clim == (round(np.percentile(expressed, 10), 9), round(np.percentile(expressed, 99), 9))


def test_numbers_and_the_defaults_are_as_they_were(obj):
    assert _draw(obj, "dense", min_cutoff=2.0, max_cutoff=7.0)[0] == (2.0, 7.0)
    assert _draw(obj, "dense")[0] == (0.0, 10.0)
    assert _draw(obj, "dense", min_cutoff="q05")[0][1] == 10.0       # the max default is the feature's max


@pytest.mark.parametrize("kwargs", [
    {"min_cutoff": 5.0, "max_cutoff": 5.0},                          # equal
    {"min_cutoff": 7.0, "max_cutoff": 2.0},                          # inverted
    {"max_cutoff": 0.0},                                             # a maximum of zero
])
def test_cutoffs_that_leave_no_range_warn_and_fall_back_to_the_features_range(obj, kwargs):
    clim, warned = _draw(obj, "dense", **kwargs)
    assert clim == (0.0, 10.0)
    assert len(warned) == 1 and "'dense'" in warned[0] and "0 to 10" in warned[0]


def test_quantiles_that_collapse_warn_too(obj):
    """Every expresser is 1, so q05 and q95 are both 1: one colour for every expresser."""
    clim, warned = _draw(obj, "flat", min_cutoff="q05", max_cutoff="q95")
    assert clim == (0.0, 1.0) and len(warned) == 1


def test_a_feature_nothing_expresses_is_not_a_cutoff_problem(obj):
    clim, warned = _draw(obj, "silent", min_cutoff="q05", max_cutoff="q95")
    assert clim == (0.0, 1e-9) and not warned


def test_split_by_still_shares_one_scale_computed_over_every_cell(obj):
    plt = pytest.importorskip("matplotlib.pyplot")
    obj.meta_data["half"] = ["a"] * 200 + ["b"] * 200
    fig = tc.feature_plot(obj, "dense", reduction="pca", min_cutoff="q05", max_cutoff="q95",
                          split_by="half")
    try:
        clims = {tuple(round(float(v), 9) for v in ax.collections[1].get_clim())
                 for ax in fig.axes if len(ax.collections) > 1}
    finally:
        plt.close(fig)
    expressed = obj.meta_data["dense"].to_numpy()
    expressed = expressed[expressed > 0]
    assert clims == {(round(np.percentile(expressed, 5), 9), round(np.percentile(expressed, 95), 9))}
