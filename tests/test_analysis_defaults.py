"""Analysis defaults brought to Seurat 5.5.1's, each checked against R's own code.

`tests/data/r_analysis_defaults.json` is written by
`tests/data/make_analysis_defaults_reference.R`, which runs Seurat's code rather
than a restatement of it: the phase function inside `CellCycleScoring`,
`DimHeatmap`'s cell selection through `SeuratObject:::Top`, and `ScaleData` on a
matrix, the call `RunMixscape(slot = "scale.data")` makes.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

matplotlib.use("Agg")

from truecell import create_truecell_object, find_spatially_variable_features, module_score
from truecell.mixscape import _mixscape_em, _scale_rows
from truecell.module_score import cell_cycle_scoring
from truecell.plotting import _dim_heatmap_cells, dim_heatmap
from truecell.preprocessing import normalize_data, scale_data
from truecell.reduction import run_pca
from truecell.spatial.fov import create_fovs

REF = json.loads((Path(__file__).parent / "data" / "r_analysis_defaults.json").read_text())


def _object(n_cells, n_genes=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3.0, size=(n_genes, n_cells)).astype(float) + 1
    return create_truecell_object(
        counts=sp.csc_matrix(counts), feature_names=[f"g{i}" for i in range(n_genes)],
        cell_names=[f"c{i}" for i in range(n_cells)])


# ---------------------------------------------------------------------------
# cell_cycle_scoring
# ---------------------------------------------------------------------------

def _stub_module_score(monkeypatch, s, g2m):
    """Stand in for add_module_score with fixed scores, recording each call."""
    calls = []

    def stub(seurat, features, **kwargs):
        calls.append(dict(kwargs, features=features))
        seurat.meta_data["S.Score"] = list(s)
        seurat.meta_data["G2M.Score"] = list(g2m)
        return seurat

    monkeypatch.setattr(module_score, "add_module_score", stub)
    return calls


def test_ctrl_defaults_to_the_size_of_the_smaller_gene_set(monkeypatch):
    """CellCycleScoring: `ctrl <- min(vapply(X = features, FUN = length, ...))`."""
    obj = _object(4)
    calls = _stub_module_score(monkeypatch, [0.1] * 4, [0.2] * 4)
    cell_cycle_scoring(obj, s_features=["g0", "g1", "g2"], g2m_features=["g3", "g4"])
    cell_cycle_scoring(obj, s_features=["g0", "g1", "g2"], g2m_features=["g3", "g4"], ctrl=7)
    cell_cycle_scoring(obj)
    built_in = min(len(module_score.CC_GENES["s_genes"]), len(module_score.CC_GENES["g2m_genes"]))
    assert [c["ctrl"] for c in calls] == [2, 7, built_in]


def test_phase_follows_seurats_rule(monkeypatch):
    """All scores below zero is G1; a tie for the highest is Undecided."""
    cases = REF["phase"]
    # The pairs that separate R's rule from `<= 0` and from sending ties to G2M.
    assert {"S", "Undecided"} <= {c["phase"] for c in cases}
    obj = _object(len(cases))
    _stub_module_score(monkeypatch, [c["s"] for c in cases], [c["g2m"] for c in cases])
    cell_cycle_scoring(obj, s_features=["g0"], g2m_features=["g1"])
    assert list(obj.meta_data["Phase"]) == [c["phase"] for c in cases]


# ---------------------------------------------------------------------------
# dim_heatmap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", REF["dim_heatmap_cells"],
    ids=lambda c: f"n{len(c['scores'])}-cells{c['cells']}-{'balanced' if c['balanced'] else 'abs'}")
def test_dim_heatmap_picks_and_orders_cells_as_seurat_does(case):
    got = _dim_heatmap_cells(np.asarray(case["scores"], dtype=float), case["cells"],
                             case["balanced"])
    assert list(got) == case["order"]


def test_dim_heatmap_draws_every_cell_by_default_highest_score_first():
    obj = _object(40, n_genes=30, seed=1)
    normalize_data(obj)
    scale_data(obj)
    run_pca(obj, n_pcs=5)
    fig = dim_heatmap(obj, dims=1)
    try:
        ax = fig.axes[0]
        image = np.asarray(ax.images[0].get_array())
        genes = [t.get_text() for t in ax.get_yticklabels()]
        assay = obj.get_assay()
        rows = [assay.features(layer="scale.data").index(g) for g in genes]
        scaled = assay.layer_data("scale.data")
        scaled = np.asarray(scaled.toarray() if sp.issparse(scaled) else scaled)
        order = _dim_heatmap_cells(obj.reductions["pca"].cell_embeddings[:, 0], 40, True)
        assert image.shape == (len(genes), 40)
        np.testing.assert_array_equal(image, np.clip(scaled[rows][:, order], -2.5, 2.5))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# run_mixscape
# ---------------------------------------------------------------------------

def test_scale_rows_is_seurats_scale_data():
    """A z-score by the n-1 standard deviation, clipped at 10 above and not below."""
    x = np.asarray(REF["scale_data"]["input"], dtype=float)
    want = np.asarray(REF["scale_data"]["scaled"], dtype=float)
    assert want.max() == 10 and want.min() < -10          # the fixture reaches both sides
    np.testing.assert_allclose(_scale_rows(x), want, rtol=0, atol=1e-12)


def test_first_round_score_projects_the_scaled_signature():
    """R scales the DE genes over the guide's and the NT cells, then ProjectVec."""
    rng = np.random.default_rng(4)
    sig = rng.normal(size=(6, 30))
    de_rows, nt_idx, gene_local = [0, 2, 5], np.arange(15), np.arange(15, 30)
    *_, score = _mixscape_em(sig, de_rows, nt_idx, gene_local, 1, 0, scale=True)
    dat = _scale_rows(sig[np.ix_(de_rows, np.concatenate([nt_idx, gene_local]))])
    vec = dat[:, 15:].mean(axis=1) - dat[:, :15].mean(axis=1)
    np.testing.assert_allclose(score, (vec @ dat) / (vec @ vec), rtol=1e-12)


# ---------------------------------------------------------------------------
# find_spatially_variable_features
# ---------------------------------------------------------------------------

def _spatial(n=60, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n, 2))
    X = rng.poisson(5, size=(8, n)).astype(float)
    X[0] += xy[:, 0] * 3
    obj = create_truecell_object(
        counts=sp.csc_matrix(X), feature_names=[f"g{i}" for i in range(8)],
        cell_names=[f"c{i}" for i in range(n)])
    coords = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cell": obj.cell_names()})
    obj.images = create_fovs(coords, assay="RNA", default_name="rna")
    normalize_data(obj)
    return obj


def test_svf_reads_scale_data_by_default():
    """FindSpatiallyVariableFeatures.Seurat: layer = "scale.data", features in that layer."""
    obj = _spatial()
    scale_data(obj, features=["g0", "g1", "g2"])
    res = find_spatially_variable_features(obj)
    assert set(res.index) == {"g0", "g1", "g2"}
    explicit = find_spatially_variable_features(obj, layer="scale.data")
    pd.testing.assert_series_equal(res["moransi"], explicit["moransi"])


def test_svf_without_scale_data_says_what_to_do():
    obj = _spatial()
    with pytest.raises(ValueError, match="scale_data"):
        find_spatially_variable_features(obj)
    assert len(find_spatially_variable_features(obj, layer="data")) == 8


# ---------------------------------------------------------------------------
# run_mixscape, end to end
# ---------------------------------------------------------------------------

def _screen():
    """test_mixscape's synthetic screen, with its perturbation signature."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.test_mixscape import _screen_object
    from truecell.mixscape import calc_perturb_sig

    obj, _, _ = _screen_object()
    calc_perturb_sig(obj)
    return obj


def _first_round_scores(obj):
    genes = obj.misc["mixscape"]["PRTB"]["genes"]
    return {g: info["scores"]["pvec"].to_numpy() for g, info in genes.items()
            if info.get("scores") is not None}


def test_run_mixscape_scales_the_signature_by_default():
    """RunMixscape's slot = "scale.data" is the default, and it moves the scores."""
    from truecell.mixscape import run_mixscape

    runs = {}
    for name, kwargs in {"default": {}, "scale.data": {"layer": "scale.data"},
                         "data": {"layer": "data"}}.items():
        obj = _screen()
        run_mixscape(obj, **kwargs)
        runs[name] = _first_round_scores(obj)
    assert runs["default"], "no gene reached the mixture"
    for gene, scores in runs["default"].items():
        np.testing.assert_array_equal(scores, runs["scale.data"][gene])
        assert not np.allclose(scores, runs["data"][gene])


def test_run_mixscape_rejects_a_layer_seurat_does_not_read():
    from truecell.mixscape import run_mixscape

    with pytest.raises(ValueError, match="scale.data"):
        run_mixscape(_screen(), layer="counts")
