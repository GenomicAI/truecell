"""Functions that take ``layer=`` read the layer's own rows, not the assay's.

``scale_data()`` scales only the variable features unless it is given others, so
``scale.data`` is narrower than the assay and row *i* of it is not feature *i* of
the assay. ``dim_heatmap`` and the helper behind ``dot_plot``, ``feature_plot``,
``feature_scatter``, ``ridge_plot``, ``vln_plot`` and the spatial plots looked a
gene up in the assay's full list and used that position to index ``scale.data``:
an ``IndexError`` when the position fell past the last scaled row, and another
gene's row, silently, when it did not. ``add_module_score(layer="scale.data")``
did the same.

Every other plotting test scales *all* features, which makes a gene's position in
the assay equal its row in the layer, so none of them could see it. The fixture
here is the standard workflow, and checks its own premise.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell.plotting import _get_expression

N_VARIABLE = 60


@pytest.fixture(scope="module")
def narrow():
    """The standard workflow, with ``scale.data`` holding only the variable features."""
    rng = np.random.default_rng(0)
    n_genes, n_cells = 600, 240
    group = rng.integers(0, 3, n_cells)
    base = rng.gamma(0.5, 1.0, n_genes) * 0.5
    mult = np.ones((n_genes, 3))
    for g in range(3):
        mult[g * 60:(g + 1) * 60, g] = 6
    counts = rng.poisson(base[:, None] * mult[:, group] * 2)
    # Shuffled, so the variable genes sit anywhere in the feature list, as in real data.
    perm = rng.permutation(n_genes)
    obj = tc.create_truecell_object(
        counts=sp.csc_matrix(counts[perm]), feature_names=[f"G{i}" for i in perm],
        cell_names=[f"c{i}" for i in range(n_cells)], min_cells=3, min_features=10)
    tc.normalize_data(obj)
    tc.find_variable_features(obj, nfeatures=N_VARIABLE)
    tc.scale_data(obj)
    tc.run_pca(obj, n_pcs=6)
    a = obj.get_assay()
    scaled, full = list(a._scaled_features), list(a._all_feature_names)
    assert len(scaled) == N_VARIABLE < len(full)
    # Were the scaled genes the assay's first rows, looking a gene up by its place in
    # the assay would be right by accident, and nothing below could fail.
    assert scaled != full[:len(scaled)]
    return obj


def _scaled(obj):
    """``(scaled gene names, scale.data as a dense array)`` read straight from the layer."""
    a = obj.get_assay()
    sd = a.layer_data("scale.data")
    return list(a._scaled_features), (sd.toarray() if sp.issparse(sd) else np.asarray(sd))


def test_get_expression_reads_the_genes_own_scale_data_row(narrow):
    scaled, sd = _scaled(narrow)
    for gene in (scaled[0], scaled[len(scaled) // 2], scaled[-1]):
        got = _get_expression(narrow, gene, layer="scale.data")
        assert np.array_equal(got, sd[scaled.index(gene)]), gene


def test_get_expression_without_a_layer_still_reads_every_gene(narrow):
    # The default layer is `data`, which does hold every gene: only a gene that
    # scale.data lacks is out of reach there.
    a = narrow.get_assay()
    full, data = list(a._all_feature_names), a.layer_data("data")
    unscaled = next(g for g in full if g not in set(a._scaled_features))
    assert np.array_equal(_get_expression(narrow, unscaled), data[full.index(unscaled)].toarray().ravel())


def test_a_gene_the_layer_lacks_is_named_as_such_not_read_from_another_row(narrow):
    a = narrow.get_assay()
    unscaled = next(g for g in a._all_feature_names if g not in set(a._scaled_features))
    with pytest.raises(KeyError, match=rf"not in layer 'scale.data'.*{N_VARIABLE} of the assay's"):
        _get_expression(narrow, unscaled, layer="scale.data")
    with pytest.raises(KeyError, match="not found in assay or metadata"):
        _get_expression(narrow, "NoSuchGene", layer="scale.data")


def test_feature_scatter_plots_the_two_genes_it_was_asked_for(narrow):
    plt = pytest.importorskip("matplotlib.pyplot")
    scaled, sd = _scaled(narrow)
    g1, g2 = scaled[10], scaled[40]
    fig = tc.feature_scatter(narrow, g1, g2, layer="scale.data")
    try:
        pts = np.vstack([np.asarray(c.get_offsets()) for c in fig.axes[0].collections])
    finally:
        plt.close(fig)
    assert np.array_equal(np.sort(pts[:, 0]), np.sort(sd[scaled.index(g1)]))
    assert np.array_equal(np.sort(pts[:, 1]), np.sort(sd[scaled.index(g2)]))


def test_dim_heatmap_draws_each_labelled_gene_s_own_row(narrow):
    plt = pytest.importorskip("matplotlib.pyplot")
    scaled, sd = _scaled(narrow)
    fig = tc.dim_heatmap(narrow, dims=[1, 2], cells=None)       # every cell, so rows compare whole
    try:
        for ax in fig.axes:
            if not ax.images:
                continue
            labels = [t.get_text() for t in ax.get_yticklabels()]
            shown = np.asarray(ax.images[0].get_array())
            assert len(labels) == shown.shape[0] > 0
            for gene, row in zip(labels, shown):
                # DimHeatmap clips at +-2.5 and orders the cells by score, so compare
                # the clipped values as a multiset.
                want = np.clip(sd[scaled.index(gene)], -2.5, 2.5)
                assert np.array_equal(np.sort(row), np.sort(want)), gene
    finally:
        plt.close(fig)


def test_add_module_score_reads_the_scale_data_layer_by_its_own_features(narrow):
    # One bin and a control pool that is the whole pool fixes the control set, so the
    # score is exactly the program's mean minus the pool's mean, whatever the seed.
    scaled, sd = _scaled(narrow)
    program, pool = scaled[5:10], scaled[:40]
    tc.add_module_score(narrow, [program], pool=pool, nbin=1, ctrl=len(pool),
                        layer="scale.data", name="M")
    want = sd[[scaled.index(g) for g in program]].mean(axis=0) - sd[[scaled.index(g) for g in pool]].mean(axis=0)
    np.testing.assert_allclose(narrow.meta_data["M1"].to_numpy(), want, rtol=0, atol=1e-12)
