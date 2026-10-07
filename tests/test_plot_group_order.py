"""Group order in the plots when some groups are numbered and others named.

Renaming some clusters and not others is ordinary: name the ones you recognise,
leave the rest as numbers. Every group-colouring plot sorted its groups with a key
that returned an ``int`` for a numbered group and a ``str`` for a named one, so a
mixed set could not be compared and all six plots raised ``TypeError``. Seurat's
``DimPlot`` and friends draw such identities without complaint.
"""
import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# A categorical's own order
# ---------------------------------------------------------------------------
# R draws a factor's levels in the order they are declared, so a lineage-ordered
# DotPlot is a matter of setting the levels, and ReorderIdent changes the plots by
# changing them. dot_plot and vln_plot sorted the groups instead, whatever the
# categories said, and so did the four other plots that colour or lay out by group.

DECLARED = ["Zeta", "Mu", "Alpha"]               # deliberately not alphabetical


@pytest.fixture(scope="module")
def declared():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(1)
    n_genes, n_cells = 20, 90
    o = tc.create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(2.0, size=(n_genes, n_cells)).astype(float)),
        feature_names=[f"G{i:02d}" for i in range(n_genes)],
        cell_names=[f"C{i:03d}" for i in range(n_cells)], project="declared",
    )
    tc.normalize_data(o)
    tc.find_variable_features(o, nfeatures=15)
    tc.scale_data(o, features=o.assays["RNA"]._all_feature_names)
    tc.run_pca(o, n_pcs=6)
    tc.run_umap(o, dims=range(5), seed=42)
    labels = rng.choice(DECLARED, n_cells)
    # An unordered categorical with its categories set, which is how Scanpy users do it.
    o.meta_data["ct"] = pd.Categorical(labels, categories=DECLARED)
    o.idents = pd.Categorical(labels, categories=DECLARED)
    return o


PLOTS = [
    ("dim_plot", lambda o, g: tc.dim_plot(o, group_by=g, label=False), lambda f: _legend(f.axes[0])),
    ("vln_plot", lambda o, g: tc.vln_plot(o, features=["G00"], group_by=g), lambda f: _xticks(f.axes[0])),
    ("feature_scatter", lambda o, g: tc.feature_scatter(o, "G00", "G01", group_by=g),
     lambda f: _legend(f.axes[0])),
    ("do_heatmap", lambda o, g: tc.do_heatmap(o, features=["G00", "G01"], group_by=g),
     lambda f: [t.get_text() for t in f.axes[0].texts]),
    ("ridge_plot", lambda o, g: tc.ridge_plot(o, features=["G00"], group_by=g),
     lambda f: _yticks(f.axes[0])[::-1]),
    ("dot_plot", lambda o, g: tc.dot_plot(o, features=["G00", "G01"], group_by=g),
     lambda f: _yticks(f.axes[0])),
]


@pytest.mark.parametrize("group_by", [None, "ct"], ids=["active identity", "metadata column"])
@pytest.mark.parametrize("name,call,read", PLOTS, ids=[p[0] for p in PLOTS])
def test_groups_are_drawn_in_their_categories_order(declared, name, call, read, group_by):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = call(declared, group_by)
    try:
        assert read(fig) == DECLARED, name
    finally:
        plt.close(fig)


@pytest.mark.parametrize("name,call,read", [PLOTS[1], PLOTS[5]], ids=["vln_plot", "dot_plot"])
def test_reorder_ident_changes_the_plots(name, call, read):
    """ReorderIdent sorts the levels by a per-identity summary, so the plots move with it."""
    plt = pytest.importorskip("matplotlib.pyplot")
    rng = np.random.default_rng(2)
    o = tc.create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(2.0, size=(20, 60)).astype(float)),
        feature_names=[f"G{i:02d}" for i in range(20)], cell_names=[f"C{i:03d}" for i in range(60)])
    tc.normalize_data(o)
    labels = np.array(["Alpha", "Mu", "Zeta"] * 20)
    o.idents = pd.Categorical(labels)
    o.meta_data["score"] = pd.Series(labels).map({"Zeta": 0.0, "Mu": 1.0, "Alpha": 2.0}).to_numpy() \
        + rng.normal(scale=0.1, size=60)
    fig = call(o, None)
    try:
        assert read(fig) == ["Alpha", "Mu", "Zeta"]                   # before: the sorted default
    finally:
        plt.close(fig)
    o.reorder_ident("score")
    assert list(o.idents.categories) == DECLARED
    fig = call(o, None)
    try:
        assert read(fig) == DECLARED
    finally:
        plt.close(fig)


def _levels(values, categories=None, ordered=False, column=True):
    """`_group_levels` on a one-column object, with a categorical meta_data column or idents."""
    from truecell.plotting import _group_levels

    class Obj:
        pass

    obj = Obj()
    col = pd.Categorical(values, categories=categories, ordered=ordered)
    obj.meta_data = pd.DataFrame({"g": col})
    obj.idents = col
    groups = np.array([str(v) for v in values])
    return (_group_levels(obj, "g", groups), _group_levels(obj, None, groups))


@pytest.mark.parametrize("categories,expected", [
    (["B", "A", "C"], ["B", "A", "C"]),
    # numeric-looking labels in a deliberate numeric order, as find_clusters leaves them
    (["0", "1", "2", "10", "11"], ["0", "1", "2", "10", "11"]),
    (["10", "2", "1"], ["10", "2", "1"]),                  # a deliberate reverse, not the default
])
def test_a_declared_order_is_followed(categories, expected):
    values = list(categories) * 2
    assert _levels(values, categories) == (expected, expected)


def test_the_default_sorted_categories_say_nothing_about_the_order_wanted():
    """`pd.Categorical(["10", "2", "1"])` and `rename_idents` leave string-sorted categories:
    "1", "10", "2". Those are not an order anyone chose, so numbers sort as numbers."""
    for values in (["10", "2", "1", "B", "T cell"], ["2", "10", "1"]):
        cat = pd.Categorical(values)
        assert list(cat.categories) == sorted(cat.categories)           # the premise
        by_column, by_ident = _levels(values)
        assert by_column == by_ident == sorted(values, key=ident_sort_key)
    assert _levels(["10", "2", "1"])[0] == ["1", "2", "10"]


def test_a_category_no_cell_has_is_not_drawn():
    assert _levels(["B", "B", "A"], categories=["C", "B", "A"]) == (["B", "A"], ["B", "A"])


def test_a_missing_value_is_still_a_group_and_comes_last():
    from truecell.plotting import _group_levels

    class Obj:
        pass

    obj = Obj()
    obj.meta_data = pd.DataFrame({"g": pd.Categorical(["B", None, "A"], categories=["B", "A"])})
    groups = obj.meta_data["g"].astype(str).to_numpy()
    levels = _group_levels(obj, "g", groups)
    # `astype(str)` leaves the missing value "nan" on pandas 2 and NaN on pandas 3.
    assert levels[:2] == ["B", "A"] and len(levels) == 3 and str(levels[2]) == "nan"


def test_a_plain_column_is_still_sorted_numbers_first():
    from truecell.plotting import _group_levels

    class Obj:
        pass

    obj = Obj()
    obj.meta_data = pd.DataFrame({"g": ["10", "B", "2", "1"]})
    assert _group_levels(obj, "g", obj.meta_data["g"].to_numpy()) == ["1", "2", "10", "B"]
