"""Tests for the multimodal CITE-seq tutorial's cell annotation and WNN section.

Builds a tiny synthetic two-assay (RNA + ADT) object with cleanly-separated
signal and checks the combined protein-priority / RNA-fallback gating in
annotate_cells(), the run_wnn() joint-clustering flow, and the figures the
walkthrough's Step 8 embeds. Network-free.
"""
import ast
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truecell.truecell import create_truecell_object  # noqa: E402
from truecell.assay5 import create_assay5_object  # noqa: E402
from truecell.preprocessing import (  # noqa: E402
    normalize_data, find_variable_features, scale_data,
)
from truecell.reduction import run_pca  # noqa: E402
from truecell.plotting import dim_plot, hue_pal, vln_plot  # noqa: E402
from truecell._utils import ident_sort_key
from tutorials.cbmc_citeseq_tutorial import CELL_TYPES, annotate_cells, run_wnn  # noqa: E402
from tutorials.generate_multimodal_plots import _group_panel, adt_weight_violins  # noqa: E402

TUTORIALS = Path(__file__).resolve().parent.parent / "tutorials"


def _two_assay_object(rna_levels, adt_levels, n=6):
    clusters = list(rna_levels)
    rna_genes = sorted({g for lv in rna_levels.values() for g in lv} | {"MALAT1"})
    adt_prots = sorted({p for lv in adt_levels.values() for p in lv})
    rg = {g: i for i, g in enumerate(rna_genes)}
    ag = {p: i for i, p in enumerate(adt_prots)}

    cells, idents = [], []
    rmat = np.zeros((len(rna_genes), len(clusters) * n))
    amat = np.zeros((len(adt_prots), len(clusters) * n))
    col = 0
    for cl in clusters:
        for _ in range(n):
            rmat[rg["MALAT1"], col] = 50
            for g, v in rna_levels[cl].items():
                rmat[rg[g], col] = v
            for p, v in adt_levels[cl].items():
                amat[ag[p], col] = v
            cells.append(f"cell_{col}")
            idents.append(cl)
            col += 1

    obj = create_truecell_object(
        counts=sp.csc_matrix(rmat), assay="RNA",
        feature_names=rna_genes, cell_names=cells,
    )
    normalize_data(obj)
    obj.assays["ADT"] = create_assay5_object(
        counts=sp.csc_matrix(amat), feature_names=adt_prots,
        cell_names=cells, key="adt_",
    )
    normalize_data(obj, assay="ADT", normalization_method="CLR", margin=2)
    obj.idents = idents
    return obj


def test_annotate_cells_protein_then_rna_fallback():
    obj = _two_assay_object(
        rna_levels={
            "t": {}, "b": {}, "nk": {}, "mono": {},
            "plt": {"PPBP": 200, "PF4": 200},     # protein-panel-less → RNA
        },
        adt_levels={
            "t":   {"CD3": 200, "CD4": 200},
            "b":   {"CD19": 200},
            "nk":  {"CD16": 200, "CD56": 200},
            "mono": {"CD14": 200},
            "plt": {},
        },
    )
    anno = annotate_cells(obj)
    assert anno["t"] == "CD4 T"
    assert anno["b"] == "B"
    assert anno["nk"] == "NK"
    assert anno["mono"] == "CD14+ Mono"
    assert anno["plt"] == "Platelet"      # resolved by RNA, not protein


def test_annotate_cells_cd8_split():
    obj = _two_assay_object(
        rna_levels={"cd4": {}, "cd8": {}, "b": {}},
        adt_levels={
            "cd4": {"CD3": 200, "CD4": 200, "CD8": 1},
            "cd8": {"CD3": 200, "CD8": 200, "CD4": 1},
            "b":   {"CD19": 200},
        },
    )
    anno = annotate_cells(obj)
    assert anno["cd4"] == "CD4 T"
    assert anno["cd8"] == "CD8 T"
    assert anno["b"] == "B"


# ---------------------------------------------------------------------------
# WNN section (run_wnn)
# ---------------------------------------------------------------------------

def _wnn_ready_object(seed=0, per=30):
    """RNA (workflow run through PCA) + CLR-normalised ADT, ready for run_wnn."""
    rng = np.random.default_rng(seed)
    groups = ("A", "B", "C")
    n = len(groups) * per
    Grna, Padt = 120, 30
    rna = rng.gamma(0.3, size=(Grna, n)) + 0.05
    adt = rng.gamma(0.3, size=(Padt, n)) + 0.05
    cells = []
    for ci in range(n):
        g = groups[ci // per]
        if g == "A":
            rna[0:30, ci] += 6.0      # RNA separates A from {B,C}
        if g == "C":
            adt[0:10, ci] += 6.0      # ADT separates C from {A,B}
        cells.append(f"cell{ci}")

    obj = create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(rna).astype(float)), assay="RNA",
        feature_names=[f"g{i}" for i in range(Grna)], cell_names=cells,
    )
    normalize_data(obj)
    find_variable_features(obj, selection_method="vst", nfeatures=100)
    scale_data(obj)
    run_pca(obj, n_pcs=15)

    obj.assays["ADT"] = create_assay5_object(
        counts=sp.csc_matrix(rng.poisson(adt).astype(float)),
        feature_names=[f"prot{i}" for i in range(Padt)], cell_names=cells, key="adt_",
    )
    normalize_data(obj, assay="ADT", normalization_method="CLR", margin=2)
    return obj, n


def test_run_wnn_builds_joint_graphs_umap_and_weights():
    obj, n = _wnn_ready_object()
    run_wnn(obj, rna_dims=range(10), resolution=1.0)

    # ADT reduction + joint graphs + joint UMAP produced.
    assert "apca" in obj.reductions
    assert "wknn" in obj.graphs and "wsnn" in obj.graphs
    assert obj.reductions["wnn_umap"].cell_embeddings.shape == (n, 2)
    assert np.isfinite(obj.reductions["wnn_umap"].cell_embeddings).all()

    # Joint clustering column + per-cell modality weights summing to 1.
    assert "wnn_clusters" in obj.meta_data.columns
    w = obj.meta_data[["RNA.weight", "ADT.weight"]].to_numpy()
    assert np.all(w >= 0) and np.all(w <= 1)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Step 8 figures (generate_multimodal_plots)
# ---------------------------------------------------------------------------

def test_wnn_umap_plots_on_the_joint_embedding():
    """Figure 08 — dim_plot has to accept the joint reduction, not just "umap"."""
    obj, _ = _wnn_ready_object()
    run_wnn(obj, rna_dims=range(10), resolution=1.0)
    fig = dim_plot(obj, reduction="wnn_umap", group_by="wnn_clusters", label=True)
    assert fig.axes and fig.axes[0].collections


def test_modality_weight_plots_as_a_metadata_feature():
    """Figure 10 — ADT.weight is a metadata column, not a gene.

    Seurat's VlnPlot resolves features against metadata; truecell's vln_plot does
    the same via _get_expression, and that fallback is what the figure rides on.
    """
    obj, n = _wnn_ready_object()
    run_wnn(obj, rna_dims=range(10), resolution=1.0)
    obj.meta_data["ct"] = ["x" if i < n // 2 else "y" for i in range(n)]
    fig = vln_plot(obj, "ADT.weight", group_by="ct")
    assert fig.axes
    # The violin must span the real weight spread, not collapse to a point.
    lo, hi = fig.axes[0].get_ylim()
    assert hi > lo


def test_group_panel_colours_labels_consistently():
    """Figure 09 draws two embeddings; a cell type must keep its colour."""
    import matplotlib.pyplot as plt

    emb = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    labels = np.array(["B", "A", "A", "B"])
    fig, axes = plt.subplots(1, 2)
    _group_panel(axes[0], emb, labels, "left")
    _group_panel(axes[1], emb[::-1], labels[::-1], "right", legend=True)

    # One collection per group, in sorted-label order, matching colours across
    # panels even though the second panel sees the cells in reverse.
    assert len(axes[0].collections) == len(axes[1].collections) == 2
    for left, right in zip(axes[0].collections, axes[1].collections):
        assert np.allclose(left.get_facecolor(), right.get_facecolor())
    assert axes[0].get_title() == "left"

    # The legend is the reliable key when a centroid label lands off-cluster.
    assert axes[0].get_legend() is None
    assert [t.get_text() for t in axes[1].get_legend().get_texts()] == ["A", "B"]
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10 is an R | Truecell pair
# ---------------------------------------------------------------------------
# Reviewer 2 of the Frontiers paper found R's and Truecell's ADT-weight violins
# in different orders and colours. Both scripts now take both from CELL_TYPES.

def _strings(node):
    return {c.value for c in ast.walk(node)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)}


def _python_labels():
    """Every label the Python annotate_cells can return, read from its source."""
    import tutorials.cbmc_citeseq_tutorial as tutorial

    tree = ast.parse(Path(tutorial.__file__).read_text())
    func = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "annotate_cells")
    labels = set(tutorial._RNA_FALLBACK)
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        # assignment[c] = "B", or = "CD8 T" if cd8 > 1.0 else "CD4 T"
        if isinstance(target, ast.Subscript) and getattr(target.value, "id", None) == "assignment":
            labels |= _strings(node.value)
        # rna_fallback's default: best, best_score = "Other", 0.30
        if isinstance(target, ast.Tuple) and any(getattr(e, "id", None) == "best"
                                                 for e in target.elts):
            labels |= _strings(node.value)
    return labels


def _r_source():
    return (TUTORIALS / "cbmc_citeseq_verify.R").read_text()


def _r_labels():
    """Every label the R port of annotate_cells can return, read from its source."""
    text = _r_source()
    fallback = text[text.index("RNA_FALLBACK <- list("):text.index("annotate_cells <- function")]
    labels = set(re.findall(r"(\w+) = c\(", fallback))
    for line in text.splitlines():
        if "assignment[c] <-" in line:        # the gene names sit left of the arrow
            labels |= set(re.findall(r'"([^"]+)"', line.split("assignment[c] <-", 1)[1]))
    return labels | set(re.findall(r'best <- "([^"]+)"', text))


def test_both_scripts_draw_the_violins_from_one_list():
    """R keeps its own copy of CELL_TYPES, since it cannot import Python's. A copy
    edited on one side only would put the two panels out of step again."""
    text = _r_source()
    block = text[text.index("CELL_TYPES <- c("):]
    assert re.findall(r'"([^"]+)"', block[:block.index(")")]) == list(CELL_TYPES)


def test_the_list_is_in_the_order_vln_plot_draws_groups():
    """vln_plot sorts its groups and matches the palette to them by position, and
    R draws in the list's order, so the list has to be vln_plot's order already."""
    assert list(CELL_TYPES) == sorted(CELL_TYPES, key=ident_sort_key)
    assert len(set(CELL_TYPES)) == len(CELL_TYPES)


def test_the_list_is_every_label_annotate_cells_can_return():
    """The colours are spread over the whole list, so a type one side lacks keeps
    its slot. A label missing from it would stop the R script, and one that
    annotate_cells can no longer return would shift every colour after it."""
    assert _python_labels() == set(CELL_TYPES)
    assert _r_labels() == set(CELL_TYPES)


def _weights_by_type(labels, seed=0):
    rng = np.random.default_rng(seed)
    obj = create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(2.0, size=(5, len(labels))).astype(float)),
        feature_names=[f"g{i}" for i in range(5)],
        cell_names=[f"c{i}" for i in range(len(labels))])
    obj.meta_data["protein_celltype"] = labels
    obj.meta_data["ADT.weight"] = rng.uniform(0.0, 1.0, size=len(labels))
    return obj


def test_adt_weight_violins_keep_each_type_in_its_colour_slot():
    """Three of the twelve types: drawn in list order, each in the colour its place
    in the whole list gives it, with no points and the paired titles."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    fig = adt_weight_violins(_weights_by_type(["pDC", "B", "NK"] * 10))
    ax = fig.axes[0]
    present = ["B", "NK", "pDC"]
    assert [t.get_text() for t in ax.get_xticklabels()] == present
    # One filled violin per type and nothing else: points would add collections.
    palette = hue_pal(len(CELL_TYPES))
    assert [to_hex(c.get_facecolor()[0]) for c in ax.collections] == \
        [palette[CELL_TYPES.index(g)].lower() for g in present]
    assert ax.get_title() == "ADT weight by cell type"
    assert ax.get_ylabel() == "ADT weight"
    plt.close(fig)


def test_adt_weight_violins_refuse_a_type_outside_the_list():
    with pytest.raises(ValueError, match="Mystery"):
        adt_weight_violins(_weights_by_type(["B", "Mystery"] * 10))


# ----------------------------------------------------------------------
# The numeric handoff against R
# ----------------------------------------------------------------------


def _handoff_object():
    """A WNN-run object carrying the three metadata columns the dump reads."""
    obj, n = _wnn_ready_object()
    run_wnn(obj, rna_dims=range(10), resolution=0.6)
    obj.meta_data["rna_clusters"] = ["0"] * (n // 2) + ["1"] * (n - n // 2)
    obj.meta_data["protein_celltype"] = ["B"] * (n // 2) + ["NK"] * (n - n // 2)
    return obj, n


def test_adt_clr_summary_is_per_protein_across_cells(tmp_path, monkeypatch):
    """One row per protein, summarising over cells — not the transpose.

    The axis matters and has been wrong in this codebase before: CLR's `margin`
    was inverted against Seurat's for a whole release. A transposed summary here
    would still write a well-formed CSV, and the report would compare it against
    R's happily, so this checks the numbers against an independent row-wise
    computation rather than checking the file merely exists.
    """
    from tutorials import cbmc_citeseq_tutorial as T

    obj, _ = _handoff_object()
    monkeypatch.setattr(T, "FIGURES", tmp_path)
    T.write_anchors(obj)

    got = pd.read_csv(tmp_path / "py_adt_clr.csv")
    adt = obj.assays["ADT"]
    data = adt.layers["data"]
    dense = data.toarray() if sp.issparse(data) else np.asarray(data)

    assert list(got["protein"]) == list(adt._all_feature_names)
    assert len(got) == dense.shape[0], "one row per protein, not per cell"
    assert got["mean"].to_numpy() == pytest.approx(dense.mean(axis=1), abs=1e-12)
    assert got["max"].to_numpy() == pytest.approx(dense.max(axis=1), abs=1e-12)
    # A transpose would only be caught by the values if the matrix were square.
    assert dense.shape[0] != dense.shape[1]


def test_cell_weights_are_written_in_object_order(tmp_path, monkeypatch):
    """The whole R comparison joins on barcode, so the key must align.

    If the `cell` column were ever written independently of the weights — a
    sorted copy, say — every downstream number would be a silent mismatch of
    cell to weight while still looking like a clean join.
    """
    from tutorials import cbmc_citeseq_tutorial as T

    obj, n = _handoff_object()
    monkeypatch.setattr(T, "FIGURES", tmp_path)
    T.write_anchors(obj)

    got = pd.read_csv(tmp_path / "py_cell_weights.csv")
    assert list(got["cell"]) == list(obj.cell_names())
    assert len(got) == n
    assert got["ADT.weight"].to_numpy() == pytest.approx(
        obj.meta_data["ADT.weight"].to_numpy(), abs=1e-12)
    # The two modality weights are a softmax pair, so they must sum to one
    # per cell — a guard that a re-ordering of one column alone would break.
    assert (got["RNA.weight"] + got["ADT.weight"]).to_numpy() == pytest.approx(
        np.ones(n), abs=1e-9)


def test_anchor_scalars_describe_the_object(tmp_path, monkeypatch):
    from tutorials import cbmc_citeseq_tutorial as T

    obj, n = _handoff_object()
    monkeypatch.setattr(T, "FIGURES", tmp_path)
    T.write_anchors(obj)

    got = json.loads((tmp_path / "py_anchors.json").read_text())
    assert got["n_cells"] == n
    assert got["n_proteins"] == len(obj.assays["ADT"]._all_feature_names)
    assert got["n_rna_clusters"] == 2
    assert got["mean_adt_weight"] == pytest.approx(
        float(obj.meta_data["ADT.weight"].mean()))
    assert got["adt_weight_sum"] == pytest.approx(
        got["mean_adt_weight"] * n, rel=1e-9)
