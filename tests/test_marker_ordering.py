"""Seurat's marker tables beyond the statistics: which fractions, which genes, what order.

Three details `find_markers` and `find_all_markers` share with Seurat 5.5.1, each
read from Seurat's source and checked in a live R session (marked "R:"):

* ``FoldChange`` rounds ``pct.1`` / ``pct.2`` to three decimals, and ``min.pct``
  filters on the rounded values;
* ``FindMarkers`` and ``FindAllMarkers`` order rows by
  ``order(p_val, -abs(pct.1 - pct.2))``;
* ``FindAllMarkers`` with ``test.use = "roc"`` swaps the default
  ``return.thresh`` of 0.01 for 0.7 before filtering on ``myAUC``.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from truecell import create_truecell_object, find_all_markers, find_markers
from truecell.markers import _round_like_r
from truecell.preprocessing import normalize_data

R_ROUNDED = [0.0, 0.002, 0.005, 0.01, 0.012]


def test_fractions_round_the_way_r_rounds_them():
    """R: `round(c(1, 5, 11, 19, 23) / 2000, digits = 3)` is 0.000 0.002 0.005 0.010 0.012.

    Over every k/n for 16 group sizes up to 2,638 the port matched R on all
    11,624 values, where numpy's `round` missed 260 and Python's 234.
    """
    x = np.array([1, 5, 11, 19, 23]) / 2000
    assert _round_like_r(x).tolist() == R_ROUNDED
    # Anti-vacuity: these five really do separate R's rule from the two obvious ones.
    assert np.round(x, 3).tolist() != R_ROUNDED
    assert [round(v, 3) for v in x] != R_ROUNDED


def test_min_pct_filters_on_the_rounded_fraction():
    """A gene in 19 of 2,000 cells sits at 0.0095, which R rounds to 0.010 — so it
    passes `min.pct = 0.01` in Seurat, and has to here too."""
    n = 2000
    rng = np.random.default_rng(0)
    counts = rng.poisson(3, size=(20, 2 * n)).astype(float)
    rare = np.zeros(2 * n)
    rare[:19] = 5.0                                   # 19 of group 1's 2,000 cells
    obj = create_truecell_object(
        counts=sp.csc_matrix(np.vstack([rare, counts])),
        feature_names=["rare"] + [f"bg{i}" for i in range(20)],
        cell_names=[f"c{i}" for i in range(2 * n)],
    )
    normalize_data(obj)
    obj.idents = ["g1"] * n + ["g2"] * n

    res = find_markers(obj, "g1", "g2", min_pct=0.01, logfc_threshold=0)
    assert "rare" in res.index
    assert res.loc["rare", "pct.1"] == 0.01           # R: 0.010, not 0.0095


def _separated_markers():
    """Three genes all of group 1 expresses and group 2 partly does, each with a
    complete rank separation, so all three Wilcoxon p-values underflow to 0 — the
    tie the ordering exists for. Group 2 expresses A in 0 %, C in 10 % and B in
    30 %; they are listed B, C, A, so falling back on feature order is visibly
    wrong.
    """
    n = 1000
    rng = np.random.default_rng(1)
    background = rng.poisson(5, size=(50, 2 * n)).astype(float)
    a = np.r_[np.full(n, 20.0), np.zeros(n)]
    b = np.r_[np.full(n, 20.0), np.ones(300), np.zeros(n - 300)]
    c = np.r_[np.full(n, 20.0), np.ones(100), np.zeros(n - 100)]
    obj = create_truecell_object(
        counts=sp.csc_matrix(np.vstack([b, c, a, background])),
        feature_names=["B", "C", "A"] + [f"bg{i}" for i in range(50)],
        cell_names=[f"c{i}" for i in range(2 * n)],
    )
    normalize_data(obj)
    obj.idents = ["g1"] * n + ["g2"] * n
    return obj


def test_p_value_ties_go_to_the_larger_pct_difference():
    """R, on the same design: `FindMarkers` returns A, C, B, all at p_val = 0 —
    |pct.1 - pct.2| of 1.0, 0.9 and 0.7."""
    res = find_markers(_separated_markers(), "g1", "g2", test_use="wilcox",
                       logfc_threshold=0, min_pct=0)
    top = res.iloc[:3]
    assert (top["p_val"] == 0).all()                  # the tie is real
    assert list(top.index) == ["A", "C", "B"]


@pytest.fixture
def roc_object():
    """Two groups, eight genes whose separation runs from none to complete."""
    rng = np.random.default_rng(3)
    n = 150
    rows = [np.r_[rng.poisson(4 + 2 * step, n), rng.poisson(4, n)] for step in range(8)]
    obj = create_truecell_object(
        counts=sp.csc_matrix(np.vstack(rows).astype(float)),
        feature_names=[f"g{i}" for i in range(8)],
        cell_names=[f"c{i}" for i in range(2 * n)],
    )
    normalize_data(obj)
    obj.idents = ["A"] * n + ["B"] * n
    return obj


def test_roc_swaps_the_default_threshold_for_seurats_auc_cutoff(roc_object):
    """R: `if ((test.use == "roc") && (return.thresh == 0.01)) return.thresh <- 0.7`."""
    everything = find_all_markers(roc_object, test_use="roc", return_thresh=None)
    auc = everything["myAUC"]
    # Anti-vacuity: some genes must sit between 0.3 and 0.7, or the cutoff never bites.
    assert ((auc >= 0.3) & (auc <= 0.7)).any()

    default = find_all_markers(roc_object, test_use="roc")
    assert ((default["myAUC"] > 0.7) | (default["myAUC"] < 0.3)).all()
    assert len(default) < len(everything)

    # Seurat tests the value rather than whether it was given, so 0.01 is swapped
    # when passed explicitly too; any other value is used as it is.
    assert len(find_all_markers(roc_object, test_use="roc", return_thresh=0.01)) == len(default)
    assert len(find_all_markers(roc_object, test_use="roc", return_thresh=0.5)) > len(default)
