"""Group order in the plots when some groups are numbered and others named.

Renaming some clusters and not others is ordinary: name the ones you recognise,
leave the rest as numbers. Every group-colouring plot sorted its groups with a key
that returned an ``int`` for a numbered group and a ``str`` for a named one, so a
mixed set could not be compared and all six plots raised ``TypeError``. Seurat's
``DimPlot`` and friends draw such identities without complaint.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell._utils import ident_sort_key

# Twelve clusters so that string order ("10" before "2") differs from numeric order.
MIXED_ORDER = ["1", "2", "3", "4", "6", "7", "8", "9", "10", "11", "B", "T cell"]


@pytest.mark.parametrize("labels,expected", [
    (["10", "2", "0"], ["0", "2", "10"]),
    (["B", "2", "10", "CD4 T"], ["2", "10", "B", "CD4 T"]),
    # Noise labels from density clustering are negative numbers.
    (["3", "-1", "0"], ["-1", "0", "3"]),
    # Not numbers, although `lstrip("-").isdigit()` or `isdigit()` says they are:
    # int() raises on both.
    (["--1", "2"], ["2", "--1"]),
    (["²", "1"], ["1", "²"]),
    (["1.5", "1"], ["1", "1.5"]),
])
def test_ident_sort_key_puts_numbers_first_then_names(labels, expected):
    assert sorted(labels, key=ident_sort_key) == expected
    assert sorted(np.array(labels), key=ident_sort_key) == expected


@pytest.fixture(scope="module")
def mixed():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    n_genes, n_cells = 20, 120
    counts = rng.poisson(2.0, size=(n_genes, n_cells)).astype(float)
    o = tc.create_truecell_object(
        counts=sp.csc_matrix(counts), feature_names=[f"G{i:02d}" for i in range(n_genes)],
        cell_names=[f"C{i:03d}" for i in range(n_cells)], project="mixed",
    )
    tc.normalize_data(o)
    tc.find_variable_features(o, nfeatures=15)
    tc.scale_data(o, features=o.assays["RNA"]._all_feature_names)
    tc.run_pca(o, n_pcs=6)
    tc.run_umap(o, dims=range(5), seed=42)
    o.meta_data["cl"] = [str(i % 12) for i in range(n_cells)]
    o.idents = o.meta_data["cl"]
    o.rename_idents({"0": "T cell", "5": "B"})
    o.meta_data["mixed_split"] = ["B" if i % 3 == 0 else str(i % 3 + 8) for i in range(n_cells)]
    assert sorted({str(i) for i in o.idents}, key=ident_sort_key) == MIXED_ORDER
    return o


def _legend(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def _xticks(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def _yticks(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


@pytest.mark.parametrize("name,call,read", [
    ("dim_plot", lambda o: tc.dim_plot(o, label=False), lambda f: _legend(f.axes[0])),
    ("vln_plot", lambda o: tc.vln_plot(o, features=["G00"]), lambda f: _xticks(f.axes[0])),
    ("feature_scatter", lambda o: tc.feature_scatter(o, "G00", "G01"),
     lambda f: _legend(f.axes[0])),
    ("do_heatmap", lambda o: tc.do_heatmap(o, features=["G00", "G01"]),
     lambda f: [t.get_text() for t in f.axes[0].texts]),
    ("ridge_plot", lambda o: tc.ridge_plot(o, features=["G00"]),
     lambda f: _yticks(f.axes[0])[::-1]),
    ("dot_plot", lambda o: tc.dot_plot(o, features=["G00", "G01"]),
     lambda f: _yticks(f.axes[0])),
])
def test_partly_renamed_identities_plot_numbers_first(mixed, name, call, read):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = call(mixed)
    try:
        assert read(fig) == MIXED_ORDER, name
    finally:
        plt.close(fig)


def test_split_by_a_column_of_numbers_and_names(mixed):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = tc.dim_plot(mixed, group_by="cl", split_by="mixed_split", label=False)
    try:
        titles = [a.get_title() for a in fig.axes if a.get_visible() and a.get_title()]
        assert titles == ["9", "10", "B"]
    finally:
        plt.close(fig)
