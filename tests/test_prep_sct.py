"""`prep_sct_find_markers` — Seurat's PrepSCTFindMarkers.

The R reference in ``data/prep_sct_r_reference.json`` was captured from a live
Seurat 5.5.1 run: pbmc3k split in two, batch B thinned to ~35% of its counts so
the medians genuinely differ (2196 vs 770), each half SCTransformed, merged, and
``PrepSCTFindMarkers`` run. It carries R's *fitted models* as well as its output,
because truecell's SCTransform is deliberately not bit-identical to R's — R
samples its step-1 genes at random — so feeding R's parameters in is what makes
the comparison a test of the re-correction rather than of the model fit. The
same method pinned ``FindWeightsC`` in T-int.

On the full 9,967 x 1,400 matrix truecell reproduced R exactly: 0 of 13,953,800
entries differing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from truecell import create_truecell_object, prep_sct_find_markers, sctransform
from truecell.assay5 import Assay5

REFERENCE = Path(__file__).parent / "data" / "prep_sct_r_reference.json"


def _object_from_r_reference():
    """A minimal object carrying R's two models and the raw counts they read."""
    ref = json.loads(REFERENCE.read_text())
    genes, cells = ref["genes"], ref["cells"]
    raw = sp.csc_matrix(np.array(ref["raw_counts"], dtype=float))
    obj = create_truecell_object(raw, assay="RNA", min_cells=0, min_features=0,
                                 feature_names=genes, cell_names=cells)
    sct = Assay5(
        layers={"counts": sp.csc_matrix((len(genes), len(cells)))},
        feature_names=genes, cell_names=cells, key="sct_",
    )
    models = {}
    for key in ("model1", "model2"):
        m = ref[key]
        # The cell attributes are kept whole even though the fixture's matrices
        # cover only 24 cells: R takes the observed median over every cell the
        # model was fitted on, so a truncated table would pick a different scale
        # factor (796.5 rather than 770) and silently compare against the wrong
        # reference. Only the cells that exist in the assay get corrected.
        ca = pd.DataFrame({"umi": m["umi"], "log_umi": m["log_umi"]},
                          index=m["cell_index"])
        models[key] = {
            "feature_attributes": pd.DataFrame(m["features"], index=genes),
            "cell_attributes": ca,
            "median_umi": float(np.median(ca["umi"])),
            "umi_assay": "RNA",
        }
    sct.misc["SCTModel.list"] = models
    obj.assays["SCT"] = sct
    return obj, ref


# ---------------------------------------------------------------------------
# Parity with Seurat
# ---------------------------------------------------------------------------

def test_recorrection_matches_seurat_exactly():
    """Given R's own models, the corrected counts must be R's, entry for entry.

    Not `approx`: the correction ends in `round()`, so every entry is an
    integer-valued double and there is no tolerance to spend. A near-miss here
    means the formula is wrong, not that the arithmetic drifted.
    """
    obj, ref = _object_from_r_reference()
    prep_sct_find_markers(obj, verbose=False)

    got = np.asarray(obj.assays["SCT"].layers["counts"].todense())
    want = np.array(ref["expected_counts"], dtype=float)
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        f"{int((got != want).sum())} of {got.size} entries differ from Seurat; "
        f"max abs diff {np.abs(got - want).max()}"
    )
    # Anti-vacuity: an all-zero expectation would pass the assertion above
    # while proving nothing about the correction.
    assert (want > 0).sum() > 500


def test_data_layer_is_log1p_of_the_corrected_counts():
    obj, _ = _object_from_r_reference()
    prep_sct_find_markers(obj, verbose=False)
    sct = obj.assays["SCT"]
    counts = np.asarray(sct.layers["counts"].todense())
    data = np.asarray(sct.layers["data"].todense())
    assert np.allclose(data, np.log1p(counts), rtol=0, atol=0)


def test_models_are_restamped_with_the_shared_depth():
    """R sets every model's median_umi to the minimum; the guard below reads it."""
    obj, ref = _object_from_r_reference()
    expected = ref["min_median_umi"]
    prep_sct_find_markers(obj, verbose=False)
    stamped = {n: m["median_umi"]
               for n, m in obj.assays["SCT"].misc["SCTModel.list"].items()}
    assert set(stamped.values()) == {expected}


# ---------------------------------------------------------------------------
# The two early returns, and idempotence
# ---------------------------------------------------------------------------

def _one_batch(depth, seed, prefix, n_genes=120, n_cells=90):
    """One SCTransformed batch at the given sequencing depth."""
    r = np.random.default_rng(seed)
    base = np.exp(r.normal(0.0, 1.0, (n_genes, 1))) * depth
    counts = sp.csc_matrix(
        r.poisson(base * np.ones((n_genes, n_cells))).astype(float))
    obj = create_truecell_object(
        counts, assay="RNA", min_cells=0, min_features=0,
        feature_names=[f"g{i}" for i in range(n_genes)],
        cell_names=[f"{prefix}{i}" for i in range(n_cells)])
    sctransform(obj, verbose=False, set_default=False)
    return obj


def _two_batch_object(seed=11, depth_b=4.0):
    """Two batches SCTransformed separately and merged, at different depths."""
    return _one_batch(1.0, seed, "a").merge(_one_batch(depth_b, seed + 1, "b"))


def test_merge_keeps_both_models():
    """Before this, `Assay5.merge` kept only the first assay's misc.

    That is the failure the whole function exists to prevent: the merged object
    looked complete, carried two batches corrected to two different depths, and
    had no record that there had ever been more than one model.
    """
    merged = _two_batch_object()
    models = merged.assays["SCT"].misc["SCTModel.list"]
    assert list(models) == ["model1", "model2"]
    # Each model's cells must address the merged assay, or the correction has
    # nothing to write into.
    cells = set(merged.assays["SCT"].cells())
    for m in models.values():
        assert set(m["cell_attributes"].index) <= cells
    assert models["model1"]["median_umi"] != models["model2"]["median_umi"]


def test_single_model_is_left_alone():
    """R: 'Only one SCT model is stored - skipping recalculating corrected counts'."""
    r = np.random.default_rng(3)
    counts = sp.csc_matrix(r.poisson(3.0, (100, 80)).astype(float))
    obj = create_truecell_object(counts, assay="RNA", min_cells=0, min_features=0,
                                 feature_names=[f"g{i}" for i in range(100)],
                                 cell_names=[f"c{i}" for i in range(80)])
    sctransform(obj, verbose=False)
    before = obj.assays["SCT"].layers["counts"].tocsc().toarray()
    prep_sct_find_markers(obj, verbose=False)
    after = obj.assays["SCT"].layers["counts"].tocsc().toarray()
    assert np.array_equal(before, after)


def test_skips_when_every_stored_median_is_above_the_minimum():
    """R's second guard: `all(model_median_umis > min_median_umi)` returns early."""
    merged = _two_batch_object()
    models = merged.assays["SCT"].misc["SCTModel.list"]
    observed_min = min(float(np.median(m["cell_attributes"]["umi"]))
                       for m in models.values())
    for m in models.values():
        m["median_umi"] = observed_min + 1_000_000.0
    before = merged.assays["SCT"].layers["counts"].tocsc().toarray()
    prep_sct_find_markers(merged, verbose=False)
    after = merged.assays["SCT"].layers["counts"].tocsc().toarray()
    assert np.array_equal(before, after)


def test_is_idempotent():
    """Re-correction always reads the raw UMI counts, never the corrected ones.

    If it read its own output the second call would correct already-corrected
    values and the matrix would drift on every run — a bug that only shows up
    when someone runs the function twice.
    """
    merged = _two_batch_object()
    prep_sct_find_markers(merged, verbose=False)
    once = merged.assays["SCT"].layers["counts"].tocsc().toarray()
    prep_sct_find_markers(merged, verbose=False)
    twice = merged.assays["SCT"].layers["counts"].tocsc().toarray()
    assert np.array_equal(once, twice)


def test_it_actually_closes_the_depth_gap():
    """The point of the function, stated as a measurement.

    Two batches sequenced 4x apart leave the SCT counts 4x apart, and a fold
    change across the merge then partly measures sequencing depth. After the
    re-correction they sit on one scale.
    """
    merged = _two_batch_object(depth_b=4.0)
    n = 90
    before = np.asarray(merged.assays["SCT"].layers["counts"].tocsc()
                        .sum(axis=0)).ravel()
    ratio_before = before[n:].mean() / before[:n].mean()
    prep_sct_find_markers(merged, verbose=False)
    after = np.asarray(merged.assays["SCT"].layers["counts"].tocsc()
                       .sum(axis=0)).ravel()
    ratio_after = after[n:].mean() / after[:n].mean()
    assert ratio_before > 3.0, f"fixture is not depth-skewed ({ratio_before:.2f})"
    assert ratio_after < 1.5, (
        f"depth gap not closed: {ratio_before:.2f}x before, {ratio_after:.2f}x after"
    )


def test_rejects_models_fitted_from_different_umi_assays():
    """R stops rather than pick one: a single corrected matrix cannot come from two."""
    merged = _two_batch_object()
    models = merged.assays["SCT"].misc["SCTModel.list"]
    models["model2"]["umi_assay"] = "OTHER"
    with pytest.raises(ValueError, match="Multiple UMI assays"):
        prep_sct_find_markers(merged, verbose=False)


def test_models_follow_their_cells_through_an_add_cell_ids_merge():
    """`merge(add_cell_ids=...)` renames every cell, and the models must follow.

    This is the common case, not an edge one: 10x barcodes repeat across runs,
    so prefixing is how you merge two batches without collisions. If the
    recorded cell attributes keep their pre-merge names they no longer address
    the assay, every model silently matches zero cells, and the re-correction
    writes an all-zero column for each of them — a wrong answer that raises
    nothing.
    """
    a = _one_batch(1.0, 11, "c")     # deliberately the SAME cell names
    b = _one_batch(4.0, 12, "c")
    merged = a.merge(b, add_cell_ids=["A", "B"])

    assay_cells = set(merged.assays["SCT"].cells())
    models = merged.assays["SCT"].misc["SCTModel.list"]
    assert len(models) == 2
    for name, m in models.items():
        recorded = set(m["cell_attributes"].index)
        assert recorded <= assay_cells, (
            f"{name} records cells the merged assay does not have: "
            f"{sorted(recorded - assay_cells)[:3]}"
        )
    # Every cell is covered by exactly one model, so none can be dropped.
    covered = set().union(*(set(m["cell_attributes"].index) for m in models.values()))
    assert covered == assay_cells

    prep_sct_find_markers(merged, verbose=False)
    counts = merged.assays["SCT"].layers["counts"].tocsc()
    col_sums = np.asarray(counts.sum(axis=0)).ravel()
    assert (col_sums > 0).all(), (
        f"{int((col_sums == 0).sum())} cells were corrected to all zeros — "
        f"their model matched no cells"
    )
