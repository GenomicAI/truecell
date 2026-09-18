"""find_markers(test_use="deseq2") against Seurat 5.5.1's FindMarkers(test.use = "DESeq2").

`tests/data/r_deseq2_reference.json` is written by
`tests/data/make_deseq2_reference.R`. It holds a synthetic negative-binomial count
matrix: 8 samples per group, 40 planted DE genes, one planted Cook's outlier and
one all-zero gene. It also holds Seurat's own FindMarkers result on that matrix,
and DESeq2 1.52.0's intermediates from DESeq2DETest's steps repeated by hand. The
by-hand p-values are identical to Seurat's, so those intermediates are the ones
Seurat's test used.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import spearmanr

pytest.importorskip("pydeseq2")

from truecell import _deseq2, create_truecell_object, find_markers
from truecell.preprocessing import normalize_data

REF = json.loads((Path(__file__).parent / "data" / "r_deseq2_reference.json").read_text())
INTERNALS = REF["internals"]


def _values(key):
    return np.array([np.nan if v is None else v for v in INTERNALS[key]], dtype=float)


def _object(counts, genes, cells, groups):
    obj = create_truecell_object(counts=sp.csc_matrix(np.asarray(counts, dtype=float)),
                                 feature_names=list(genes), cell_names=list(cells))
    obj.idents = list(groups)
    normalize_data(obj)
    return obj


@pytest.fixture(scope="module")
def markers():
    obj = _object(REF["counts"], REF["genes"], REF["samples"], REF["groups"])
    return find_markers(obj, ident_1="A", ident_2="B", test_use="deseq2")


@pytest.fixture(scope="module")
def seurat():
    s = REF["seurat_markers"]
    return pd.DataFrame(
        {"p_val": np.array([np.nan if v is None else v for v in s["p_val"]], dtype=float),
         "avg_log2FC": s["avg_log2FC"], "pct.1": s["pct_1"], "pct.2": s["pct_2"],
         "p_val_adj": np.array([np.nan if v is None else v for v in s["p_val_adj"]], dtype=float)},
        index=s["gene"])


def test_the_reference_is_seurats_own_test():
    assert REF["by_hand_matches_seurat"] is True
    assert (REF["seurat"], REF["deseq2"]) == ("5.5.1", "1.52.0")


def test_local_trend_is_the_regression_deseq2_fits():
    """On R's own gene-wise estimates, the trend equals locfit evaluated at each gene."""
    nz = ~np.array(INTERNALS["all_zero"], dtype=bool)
    base, genewise = _values("base_mean")[nz], _values("disp_gene_est")[nz]
    got = _deseq2.local_dispersion_trend(base, genewise, base)
    np.testing.assert_allclose(np.log(got), np.log(_values("trend_direct")[nz]),
                               rtol=0, atol=1e-6)


def test_the_genes_tested_are_seurats(markers, seurat):
    """DESeq2 has no log-fold-change pre-filter; min.pct still drops the all-zero gene."""
    assert set(markers.index) == set(seurat.index)
    assert "gene-400" not in markers.index


def test_fold_change_and_detection_are_seurats(markers, seurat):
    """FindMarkers computes both on the data layer, whatever layer the test reads."""
    m = markers.loc[seurat.index]
    np.testing.assert_allclose(m["avg_log2FC"], seurat["avg_log2FC"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(m["pct.1"].to_numpy(), seurat["pct.1"].to_numpy())
    np.testing.assert_array_equal(m["pct.2"].to_numpy(), seurat["pct.2"].to_numpy())


def test_p_values_track_seurats_and_call_the_same_genes(markers, seurat):
    m = markers.loc[seurat.index]
    ok = seurat["p_val"].notna()
    assert spearmanr(m.loc[ok, "p_val"], seurat.loc[ok, "p_val"])[0] > 0.9999
    called = set(m.index[m["p_val_adj"] < 0.05])
    assert called == set(seurat.index[seurat["p_val_adj"] < 0.05])
    assert len(called) >= 20                     # the planted genes reach the call


def test_a_cooks_outlier_gets_p_1_where_seurat_gives_na(markers, seurat):
    na = set(seurat.index[seurat["p_val"].isna()])
    assert na == {"gene-050", "gene-328"}         # gene-050 is the planted outlier
    assert (markers.loc[sorted(na), "p_val"] == 1.0).all()


def test_bonferroni_counts_every_feature(markers):
    n = len(REF["genes"])
    np.testing.assert_array_equal(markers["p_val_adj"].to_numpy(),
                                  np.minimum(markers["p_val"].to_numpy() * n, 1.0))


def test_the_fit_tracks_deseq2_step_by_step():
    """Size factors exactly; dispersions to the optimiser's difference, measured at
    6e-3 (MAP) in log units. The Cook's NaNs land on the same genes."""
    genes, samples, groups = REF["genes"], np.array(REF["samples"]), np.array(REF["groups"])
    counts = pd.DataFrame(np.array(REF["counts"], dtype=np.int64), index=genes, columns=samples)
    order = list(samples[groups == "A"]) + list(samples[groups == "B"])
    feats = REF["tested_features"]
    dds, stats = _deseq2.fit(counts.loc[feats, order].T, np.isin(order, samples[groups == "A"]))

    np.testing.assert_allclose(dds.obs["size_factors"].to_numpy(),
                               np.array(INTERNALS["size_factors"], dtype=float), rtol=1e-12)
    log_map = np.log(dds.var["MAP_dispersions"].to_numpy(dtype=float))
    assert np.nanmax(np.abs(log_map - np.log(_values("disp_map")))) < 0.02
    assert int(dds.var["_outlier_genes"].sum()) == int(np.nansum(_values("disp_outlier")))
    nan_pattern = np.isnan(stats.p_values.to_numpy(dtype=float)).tolist()
    assert nan_pattern == np.isnan(_values("pvalue")).tolist()


def test_the_fit_runs_in_one_process(monkeypatch):
    """DESeq2 runs in one R process; pydeseq2 starts a joblib worker per core.

    Those workers outlive the call. On the benchmark's PBMC 3k DE call they took the
    process tree from under 1 GB to 5.1 GB to save 1.6 s, for the same output. So
    every parallel section pydeseq2 opens must ask for one job.
    """
    import pydeseq2.default_inference as inference

    jobs = []
    real_parallel = inference.Parallel

    def recording_parallel(*args, **kwargs):
        jobs.append(kwargs.get("n_jobs"))
        return real_parallel(*args, **kwargs)

    monkeypatch.setattr(inference, "Parallel", recording_parallel)
    genes, samples, groups = REF["genes"], np.array(REF["samples"]), np.array(REF["groups"])
    counts = pd.DataFrame(np.array(REF["counts"], dtype=np.int64), index=genes, columns=samples)
    order = list(samples[groups == "A"]) + list(samples[groups == "B"])
    _deseq2.fit(counts.loc[REF["tested_features"], order].T,
                np.isin(order, samples[groups == "A"]))
    assert jobs, "pydeseq2 opened no parallel section, so nothing was checked"
    assert set(jobs) == {1}, jobs


LOW = REF["low_count"]


def _low(key):
    return np.array([np.nan if v is None else v for v in LOW[key]], dtype=float)


@pytest.fixture(scope="module")
def low_count_fit():
    """The low-count matrix: half its fitted means under 0.5, dispersions up to the cap."""
    counts = pd.DataFrame(np.array(LOW["counts"], dtype=np.int64), index=LOW["genes"],
                          columns=LOW["samples"]).T
    return _deseq2.fit(counts, np.array(LOW["groups"]) == "A")


def test_the_wald_test_floors_fitted_means_as_deseq2_does(low_count_fit):
    """nbinomWaldTest floors each fitted mean at 0.5 before weighting it.

    On a matrix where half the fitted means sit below that, truecell's standard
    errors match R's only with the floor: median gap 8e-5 with it, 0.017 without.
    Per cell on PBMC 3k the floor is what brings the Bonferroni calls to Seurat's.
    """
    assert LOW["frac_means_below_floor"] > 0.3           # the floor is exercised
    _, stats = low_count_fit
    se = stats.SE.to_numpy(dtype=float) / np.log(2)
    ok = np.isfinite(se) & np.isfinite(_low("lfc_se"))
    assert np.median(np.abs(se[ok] - _low("lfc_se")[ok])) < 1e-3
    assert np.nanmax(np.abs(stats.statistics.to_numpy(dtype=float) - _low("stat"))) < 0.05
    genes, n = np.array(LOW["genes"]), len(LOW["genes"])

    def called(p):
        return set(genes[np.nan_to_num(p, nan=1.0) * n < 0.05])

    assert called(stats.p_values.to_numpy(dtype=float)) == called(_low("pvalue"))


def test_dispersions_reach_deseq2s_cap_of_max_10_n(low_count_fit):
    """DESeq2 caps dispersions at max(10, number of samples), 60 here.

    Low means are where the local trend climbs past the cap: R has 98 genes above
    10 and 53 at 60. truecell's final dispersions sit at most 0.048 from R's in
    log; a cap of 10 would put them 1.8 away. pydeseq2 enforces the same maximum
    itself, so this pins the behaviour, not the argument truecell passes.
    """
    dds, _ = low_count_fit
    r_disp = _low("dispersion")
    assert LOW["max_disp"] == 60 and (r_disp > 10).sum() >= 50   # the cap is exercised
    log_gap = np.abs(np.log(dds.var["dispersions"].to_numpy(dtype=float)) - np.log(r_disp))
    assert np.nanmax(log_gap) < 0.1


FLAT = REF["flat"]


def _flat(key):
    return np.array([np.nan if v is None else v for v in FLAT[key]], dtype=float)


def test_genewise_dispersions_on_a_flat_likelihood_stay_out_of_the_trend_fit():
    """Genes whose likelihood barely changes as their dispersion falls towards zero.

    DESeq2 reports them below 1e-6, which leaves them out of the trend fit, while
    pydeseq2's L-BFGS-B stops near 1e-5, inside it. On this matrix, before truecell
    moved such genes to the minimum, 5 genes landed on the other side of that cut,
    the trend sat 0.56 log units from R's at the median, and the genes called
    differed. Now it is 1 gene, 0.02, and the same genes.
    """
    r_genewise = _flat("disp_gene_est")
    assert (r_genewise < 1e-6).sum() > 100                        # the case is exercised
    counts = pd.DataFrame(np.array(FLAT["counts"], dtype=np.int64), index=FLAT["genes"],
                          columns=FLAT["samples"]).T
    dds, stats = _deseq2.fit(counts, np.array(FLAT["groups"]) == "A")

    genewise = dds.var["genewise_dispersions"].to_numpy(dtype=float)
    assert ((genewise > 1e-6) != (r_genewise > 1e-6)).sum() <= 2
    trend = dds.var["fitted_dispersions"].to_numpy(dtype=float)
    assert np.median(np.abs(np.log(trend) - np.log(_flat("disp_fit")))) < 0.1
    genes, n = np.array(FLAT["genes"]), len(FLAT["genes"])

    def called(p):
        return set(genes[np.nan_to_num(p, nan=1.0) * n < 0.05])

    assert called(stats.p_values.to_numpy(dtype=float)) == called(_flat("pvalue"))


def test_sample_col_sums_counts_per_sample_then_runs_the_same_test():
    """The pseudobulk route is the per-sample sum, handed to the per-replicate test."""
    rng = np.random.default_rng(7)
    donors = [f"d{i}" for i in range(8)]
    cells, donor_of, group_of, blocks = [], [], [], []
    for i, donor in enumerate(donors):
        group = "A" if i < 4 else "B"
        lam = rng.gamma(2.0, 3.0, size=60)
        lam[:6] *= 3.0 if group == "A" else 1.0
        block = rng.poisson(lam[:, None] * rng.uniform(0.8, 1.2), size=(60, 5))
        blocks.append(block)
        cells += [f"{donor}_c{j}" for j in range(5)]
        donor_of += [donor] * 5
        group_of += [group] * 5
    genes = [f"g{k}" for k in range(60)]
    per_cell = np.hstack(blocks)
    obj = _object(per_cell, genes, cells, group_of)
    obj.meta_data["donor"] = donor_of
    by_donor = find_markers(obj, ident_1="A", ident_2="B", test_use="deseq2",
                            sample_col="donor", min_pct=0.0)

    summed = np.column_stack([b.sum(axis=1) for b in blocks])
    pseudo = _object(summed, genes, donors, ["A"] * 4 + ["B"] * 4)
    per_sample = find_markers(pseudo, ident_1="A", ident_2="B", test_use="deseq2", min_pct=0.0)
    np.testing.assert_allclose(by_donor["p_val"].sort_index(), per_sample["p_val"].sort_index(),
                               rtol=1e-10, atol=0)


def test_the_pydeseq2_steps_the_port_relies_on_still_exist():
    """A pydeseq2 release that renames a step fails here, not deep inside find_markers."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    for name in ("fit_size_factors", "fit_genewise_dispersions", "fit_dispersion_prior",
                 "fit_MAP_dispersions", "fit_LFC", "calculate_cooks", "cooks_outlier"):
        assert callable(getattr(DeseqDataSet, name, None)), name
    for name in ("run_wald_test", "_cooks_filtering"):
        assert callable(getattr(DeseqStats, name, None)), name
