"""Tests for the pseudobulk DESeq2 branch of find_markers (test_use='deseq2')."""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("pydeseq2")

from truecell import create_truecell_object, find_markers  # noqa: E402
from truecell.preprocessing import normalize_data  # noqa: E402


@pytest.fixture
def pseudobulk_obj():
    """120 cells, 40 genes, 6 donors (3 per condition). gene-0 up in A, gene-1 up in B."""
    rng = np.random.default_rng(0)
    n_genes, per = 40, 20
    cond_of = {"d1": "A", "d2": "A", "d3": "A", "d4": "B", "d5": "B", "d6": "B"}
    blocks, cells, donor_col, cond_col = [], [], [], []
    for d, cond in cond_of.items():
        X = rng.poisson(2.0, size=(n_genes, per)).astype(float)
        X[0 if cond == "A" else 1] += rng.poisson(20, size=per)
        blocks.append(X)
        cells += [f"{d}_c{j}" for j in range(per)]
        donor_col += [d] * per
        cond_col += [cond] * per
    obj = create_truecell_object(
        counts=sp.csc_matrix(np.hstack(blocks)),
        feature_names=[f"gene-{i}" for i in range(n_genes)],
        cell_names=cells,
    )
    obj.meta_data["donor"] = donor_col
    obj.idents = cond_col
    normalize_data(obj)
    return obj


def test_deseq2_recovers_planted_de_genes(pseudobulk_obj):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence pydeseq2 small-panel convergence notes
        res = find_markers(
            pseudobulk_obj, ident_1="A", ident_2="B", test_use="deseq2",
            sample_col="donor", min_pct=0.0, logfc_threshold=0.0,
        )

    assert list(res.columns) == ["p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj"]
    # gene-0 is up in A -> positive LFC and significant.
    assert res.loc["gene-0", "avg_log2FC"] > 0
    assert res.loc["gene-0", "p_val_adj"] < 0.05
    # gene-1 is up in B -> down in A.
    assert res.loc["gene-1", "avg_log2FC"] < 0
    assert res.loc["gene-1", "p_val_adj"] < 0.05
    # A non-DE gene should not be significant.
    assert res.loc["gene-5", "p_val_adj"] > 0.05
    # Sorted by p_val ascending; a planted gene ranks first.
    assert res.index[0] in {"gene-0", "gene-1"}
    assert res["p_val"].is_monotonic_increasing


def test_deseq2_per_cell_refuses_counts_where_every_gene_has_a_zero(pseudobulk_obj):
    """Without sample_col every cell is a replicate, as in Seurat's DESeq2DETest.

    Every gene here has a zero in some cell, so DESeq2's median-of-ratios size
    factors do not exist, and estimateSizeFactors stops. So does this.
    """
    with pytest.raises(ValueError, match="at least one zero"):
        find_markers(pseudobulk_obj, ident_1="A", ident_2="B", test_use="deseq2",
                     min_pct=0.0)


def test_deseq2_unknown_sample_col_raises(pseudobulk_obj):
    with pytest.raises(KeyError, match="not found in meta_data"):
        find_markers(
            pseudobulk_obj, ident_1="A", ident_2="B",
            test_use="deseq2", sample_col="nope",
        )


def test_deseq2_warns_on_too_few_replicates(pseudobulk_obj):
    # A = 2 donors, B = 1 donor -> fewer than 2 replicates in group B.
    keep = [c for c, d in zip(pseudobulk_obj.cell_names(), pseudobulk_obj.meta_data["donor"])
            if d in ("d1", "d2", "d4")]
    sub = pseudobulk_obj.subset(cells=keep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with pytest.warns(RuntimeWarning, match="replicate"):
            res = find_markers(
                sub, ident_1="A", ident_2="B", test_use="deseq2",
                sample_col="donor", min_pct=0.0, logfc_threshold=0.0,
            )
    assert len(res) > 0
