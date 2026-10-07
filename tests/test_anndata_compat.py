import warnings

import numpy as np
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata", reason="anndata not installed")

from truecell import create_truecell_object, find_variable_features, normalize_data, scale_data
from truecell.compat import as_anndata, from_anndata


def test_as_anndata(small_seurat):
    adata = as_anndata(small_seurat)
    assert adata.n_obs == 20
    assert adata.n_vars == 50


def test_as_anndata_obs(small_seurat):
    adata = as_anndata(small_seurat)
    assert "ident" in adata.obs.columns


def test_as_anndata_uns(small_seurat):
    adata = as_anndata(small_seurat)
    assert "project_name" in adata.uns


def test_from_anndata(small_seurat):
    adata = as_anndata(small_seurat)
    seurat2 = from_anndata(adata, assay="RNA")
    assert len(seurat2) == 20
    assert len(seurat2.feature_names()) == 50


def test_roundtrip_cells(small_seurat):
    adata = as_anndata(small_seurat)
    seurat2 = from_anndata(adata, assay="RNA")
    assert set(seurat2.cell_names()) == set(small_seurat.cell_names())


def test_roundtrip_features(small_seurat):
    adata = as_anndata(small_seurat)
    seurat2 = from_anndata(adata, assay="RNA")
    assert set(seurat2.feature_names()) == set(small_seurat.feature_names())


def test_from_anndata_makes_no_layer_out_of_x(small_seurat):
    """anndata 0.13 lists X among the layers, under the key None. `from_anndata`
    copied it into a layer named None, and `as_anndata` then refused the object."""
    seurat2 = from_anndata(as_anndata(small_seurat), assay="RNA")
    assert list(seurat2.assays["RNA"].layers) == ["counts"]
    assert as_anndata(seurat2).n_obs == 20


# ---------------------------------------------------------------------------
# The standard workflow: scale_data() scales only the variable features
# ---------------------------------------------------------------------------

N_GENES, N_CELLS, N_VARIABLE = 300, 120, 40

# (use_v5, the scale layer's name, the one non-default layer that is full width).
# Assay5 makes `counts` the default layer, so `data` goes to `layers`; the legacy
# Assay puts `data` in X and `counts` in `layers`. The two name scale.data differently.
ASSAYS = [(True, "scale.data", "data"), (False, "scale_data", "counts")]


def _standard(use_v5, every_feature=False):
    """normalize, find variable features, scale: what the docs and the tutorials run."""
    rng = np.random.default_rng(0)
    obj = create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(2.0, size=(N_GENES, N_CELLS)).astype(float)),
        feature_names=[f"g{i}" for i in range(N_GENES)],
        cell_names=[f"c{j}" for j in range(N_CELLS)],
        use_v5=use_v5,
    )
    normalize_data(obj)
    find_variable_features(obj, nfeatures=N_VARIABLE)
    scale_data(obj, features=obj.feature_names() if every_feature else None)
    return obj


def _dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


@pytest.mark.parametrize("use_v5,scale_layer,other", ASSAYS)
def test_a_scale_layer_narrower_than_the_assay_is_left_out_with_a_warning(use_v5, scale_layer, other):
    """AnnData layers are as wide as var. scale_data() scales only the variable
    features, so the layer has 40 rows against 300, and as_anndata raised a
    ValueError from inside AnnData on every object the standard workflow makes."""
    obj = _standard(use_v5)
    with pytest.warns(UserWarning, match=rf"left out '{scale_layer}' \({N_VARIABLE} features\).*{N_GENES} features"):
        adata = as_anndata(obj)
    assert adata.shape == (N_CELLS, N_GENES)
    assert scale_layer not in adata.layers
    # What does fit is all still there.
    assert other in adata.layers and adata.layers[other].shape == (N_CELLS, N_GENES)


@pytest.mark.parametrize("use_v5,scale_layer,other", ASSAYS)
def test_a_scale_layer_that_covers_every_feature_is_kept(use_v5, scale_layer, other):
    obj = _standard(use_v5, every_feature=True)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        adata = as_anndata(obj)
    assert not [w for w in seen if "left out" in str(w.message)]
    source = obj.get_assay().layer_data("scale.data") if use_v5 else obj.get_assay().scale_data
    assert adata.layers[scale_layer].shape == (N_CELLS, N_GENES)
    np.testing.assert_array_equal(_dense(adata.layers[scale_layer]), _dense(source).T)


def test_a_converted_standard_workflow_object_comes_back_without_the_narrow_layer():
    obj = _standard(use_v5=True)
    with pytest.warns(UserWarning, match="left out"):
        adata = as_anndata(obj)
    back = from_anndata(adata, assay="RNA")
    assert back.feature_names() == obj.feature_names()
    assert "scale.data" not in back.assays["RNA"].layers
