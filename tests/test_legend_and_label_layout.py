"""Legends and labels that sat on the data, found by sweeping the layout check.

``tests/_layout.py`` was run over every figure the tutorials draw. These are the
package's own plotting functions it caught at the size the tutorials show them:

* ``variable_feature_plot`` printed the top gene names over one another where
  their points crowd, which is where the most variable genes are;
* ``viz_dim_loadings`` put its legend over the positive loadings' bars;
* ``image_dim_plot`` and ``spatial_dim_plot`` kept a fixed 12% strip for the
  legend, and a legend of cell-type names is wider, so it lay over the tissue.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell.spatial.fov import create_fovs

plt = pytest.importorskip("matplotlib.pyplot")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _layout


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# variable_feature_plot(repel=True)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def crowded_hvgs():
    """Twelve genes with the same counts in every cell, so the ten of them that
    lead the variable features all sit on one point of the plot.

    vst measures a gene against genes of similar mean, so the background spans
    a range of means: alone at theirs, the twelve would set that fit themselves.
    """
    rng = np.random.default_rng(0)
    n_cells = 300
    means = np.exp(rng.uniform(np.log(0.5), np.log(20), size=300))
    background = rng.poisson(means[:, None], size=(300, n_cells))
    bursty = rng.negative_binomial(0.5, 0.5 / (0.5 + 5.0), size=n_cells)
    counts = np.vstack([background, np.tile(bursty, (12, 1))]).astype(float)
    genes = [f"BG{i:03d}" for i in range(300)] + [f"HOTGENE{i:02d}" for i in range(12)]
    obj = tc.create_truecell_object(
        counts=sp.csc_matrix(counts), assay="RNA", feature_names=genes,
        cell_names=[f"C{i:03d}" for i in range(n_cells)], project="hvg")
    tc.normalize_data(obj)
    tc.find_variable_features(obj, selection_method="vst", nfeatures=50)
    top = obj.assays["RNA"].variable_features[:10]
    assert all(g.startswith("HOTGENE") for g in top), "premise: the duplicates lead"
    return obj


def _not_on_points(problems):
    # The names may sit on other genes' points; only these kinds must be absent.
    return [p for p in problems if not p.startswith("annotation overlaps")]


def test_variable_feature_names_collide_without_repel(crowded_hvgs):
    fig = tc.variable_feature_plot(crowded_hvgs, label=True, n_label=10)
    assert any(p.startswith("text overlaps text") for p in _layout.check_layout(fig))


@pytest.mark.parametrize("n_label", [1, 10])
@pytest.mark.parametrize("shrink", [False, True])
def test_repel_keeps_variable_feature_names_apart_and_off_their_points(crowded_hvgs, shrink, n_label):
    """One name as well as ten: alone, nothing but its own point keeps it off."""
    fig = tc.variable_feature_plot(crowded_hvgs, label=True, n_label=n_label, repel=True)
    if shrink:
        _layout.shrink(fig)
    assert _not_on_points(_layout.check_layout(fig)) == []

    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    top = crowded_hvgs.assays["RNA"].variable_features[:n_label]
    hvgs = next(c for c in ax.collections if c.get_label() == "Variable features")
    names = [t for t in ax.texts if t.get_text() in top]
    assert sorted(t.get_text() for t in names) == sorted(top)
    # Every named gene's point stays clear of every name.
    points = hvgs.get_offset_transform().transform(np.asarray(hvgs.get_offsets()))
    for text in names:
        box = text.get_window_extent(renderer)
        inside = ((points[:, 0] > box.x0) & (points[:, 0] < box.x1)
                  & (points[:, 1] > box.y0) & (points[:, 1] < box.y1))
        assert not inside.any(), text.get_text()


def test_without_repel_variable_feature_names_are_still_annotations(crowded_hvgs):
    """The default call draws what it always drew."""
    from matplotlib.text import Annotation
    fig = tc.variable_feature_plot(crowded_hvgs, label=True, n_label=10)
    assert len(fig.axes[0].texts) == 10
    assert all(isinstance(t, Annotation) for t in fig.axes[0].texts)


# ---------------------------------------------------------------------------
# viz_dim_loadings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pca_obj():
    rng = np.random.default_rng(1)
    n_genes, n_cells = 60, 200
    counts = rng.poisson(2.0, size=(n_genes, n_cells)).astype(float)
    counts[:15, :100] += rng.poisson(12, size=(15, 100))
    counts[15:30, 100:] += rng.poisson(12, size=(15, 100))
    obj = tc.create_truecell_object(
        counts=sp.csc_matrix(counts), assay="RNA",
        feature_names=[f"GENE{i:02d}" for i in range(n_genes)],
        cell_names=[f"C{i:03d}" for i in range(n_cells)], project="pca")
    tc.normalize_data(obj)
    tc.find_variable_features(obj, nfeatures=40)
    tc.scale_data(obj, features=obj.assays["RNA"]._all_feature_names)
    tc.run_pca(obj, n_pcs=5, features=obj.assays["RNA"]._all_feature_names)
    return obj


@pytest.mark.parametrize("shrink", [False, True])
def test_the_loadings_legend_sits_off_the_bars(pca_obj, shrink):
    fig = tc.viz_dim_loadings(pca_obj, dims=[1, 2], n_features=15)
    if shrink:
        _layout.shrink(fig)
    assert not [p for p in _layout.check_layout(fig) if p.startswith("legend covers")]


# ---------------------------------------------------------------------------
# image_dim_plot / spatial_dim_plot
# ---------------------------------------------------------------------------

LONG_TYPES = ["Excitatory neuron, layer 2/3", "Oligodendrocyte precursor",
              "Vascular leptomeningeal cell", "Astrocyte"]


@pytest.fixture
def imaging_obj():
    rng = np.random.default_rng(2)
    n = 400
    cells = [f"cell_{i}" for i in range(n)]
    obj = tc.create_truecell_object(
        sp.csc_matrix(rng.poisson(1.0, size=(5, n)).astype(float)), assay="Xenium",
        feature_names=[f"gene-{i}" for i in range(5)], cell_names=cells)
    coords = pd.DataFrame({"x": rng.uniform(0, 100, n), "y": rng.uniform(0, 100, n), "cell": cells})
    obj.images = create_fovs(coords, fov=np.array(["fov"] * n))
    obj.meta_data["cell_type"] = [LONG_TYPES[i % 4] for i in range(n)]
    return obj


@pytest.mark.parametrize("shrink", [False, True])
def test_the_image_legend_sits_beside_the_tissue(imaging_obj, shrink):
    fig = tc.image_dim_plot(imaging_obj, group_by="cell_type")
    if shrink:
        _layout.shrink(fig)
    assert _layout.check_layout(fig) == []


def test_the_visium_legend_sits_beside_the_tissue(tmp_path):
    from test_spatial_plots import N_SPOTS, _spots, _write_visium

    barcodes = _write_visium(tmp_path)
    obj = tc.load_visium(tmp_path)
    obj.add_meta_data(pd.Series([LONG_TYPES[i % 4] for i in range(N_SPOTS)], index=barcodes),
                      col_name="region")
    fig = tc.spatial_dim_plot(obj, group_by="region", figsize=(3, 3))
    assert len(_spots(fig).get_offsets()) == N_SPOTS, "premise: the spots are drawn"
    assert _layout.check_layout(fig) == []
