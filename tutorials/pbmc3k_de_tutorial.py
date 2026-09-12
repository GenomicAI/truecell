"""The differential-expression test suite — truecell against Seurat 5.5.1.

Wave 3's first tutorial, and the last big untested surface in the library:
``find_markers`` offers eight statistical tests and **none of them had ever been
compared to R**. Their unit tests assert self-consistency on synthetic fixtures,
which is exactly the shape of coverage that let the CLR and SCTransform defects
survive.

The comparison runs both tools on the **same cells** — Python clusters pbmc3k and
writes the cluster-0-vs-cluster-1 assignment to ``figures_de/groups.csv``, which
the R side reads — so nothing here is measuring a clustering difference. What is
left is the test.

What it found
-------------
Two defects, both fixed here. The numbers below are on the clusters the tutorial
tests now, 703 and 480 cells, except where a passage says otherwise.

1. **``avg_log2FC`` put the pseudocount in the wrong place.** Seurat 5's
   ``log1pdata.mean.fxn`` is ``log2((sum(expm1(x)) + 1) / n)`` — one pseudocount
   added to the group's *total*, worth ``1/n`` on the mean scale. truecell computed
   ``log2(mean(expm1(x)) + 1)``, adding a whole count to the *mean*. That floors
   every fold change near zero: a gene detected in 0 % of one cluster and 24 % of
   the other read **-1.26** where Seurat reads **-9.93**.

   This is the most-read column in any DE table, and it is worse than a display
   problem: ``logfc_threshold`` **filters on it**, so the error changed which
   genes came back at all. At Seurat's own default of 0.1, truecell returned 4,897
   genes where Seurat returned 12,990 (Jaccard 0.377); at 0.25, 2,299 against
   11,907 (Jaccard **0.193**). Fewer than one gene in five agreed.

   Telling: where both groups actually express the gene the two formulas nearly
   agree (Spearman 0.990) — the error is concentrated in sparse, marker-like
   genes, which is precisely what differential expression is looking for. After
   the fix, **6.2e-15 across all 13,714 genes**.

2. **``negbinom`` was a different test.** Seurat's ``GLMDETest`` fits
   ``MASS::glm.nb`` — dispersion estimated by **maximum likelihood** — and reads
   the **Wald** p-value off the group coefficient. truecell used a fixed
   method-of-moments dispersion and a **likelihood-ratio** test: a different
   estimator *and* a different statistic. It read HLA-DRA at 5.5e-128 against
   R's 1.1e-321. The first fix called statsmodels' ``NegativeBinomial``, whose
   BFGS fit failed on some genes, and on different ones in different statsmodels
   versions: 49/50 on the top 50 under 0.15.0. truecell now fits glm.nb's
   estimator itself (``truecell/_glm_nb.py``). Above 5 % detection the p-values
   agree to a median below 1e-9 decades, and no gene lands on the other side of
   ``p_val_adj`` = 0.05; the same bits come out under statsmodels 0.14.6 and 0.15.0.

Added later: ``poisson``
------------------------
The ninth test, and the other half of Seurat's ``GLMDETest`` — ``glm(family =
"poisson")`` on the counts layer, Wald p-value off the group coefficient. It
lands at **50/50 on the top 50**, ``avg_log2FC`` exact to 6.2e-15, p-value
Spearman **0.9999989** on genes detected above 5 %, and **zero** disagreements
on which genes clear ``p_val_adj < 0.05``.

**The residual is R's, and this is the third time that has been true here.**
truecell's p-values sit within ~6 significant figures of Seurat's rather than
being bit-identical, and the gap grows with significance (median |Δlog10 p|
1.7e-6 below ``-log10 p = 2``, rising to 9.8e-5 above 150) — the signature of
tail amplification, not of a wrong statistic. At z ≈ 37 a shift of 0.005 in z
moves p by 20 %. Chasing it down on the earlier partition of 692 and 515 cells:
on GPX1, R's default ``glm.control(epsilon = 1e-8)`` stops at **iteration 5**
with z = 37.002168 and p = 1.0568e-299, while ``epsilon = 1e-14`` takes **6** and
gives z = 36.997126, p = 1.27368e-299 — which is truecell's answer, matching R's
own converged coefficient to **14 significant figures**. Re-running the top 200
genes at both tolerances closed 9/10 of the median gap (5.9e-5 → 6.6e-6) and
57/58 of the worst case (8.1e-2 → 1.4e-3). truecell is the more converged of the
two, so no "fix" was applied; see the Visium tutorial for the first instance of a
difference being Seurat's.

**Divergence on the gene set, and it is exact.** R returns 11,387 genes to
truecell's 13,714. All **2,327** of the difference fail Seurat's
``GLMDETest`` ``min.cells = 3``-in-*both*-groups gate, which flags them with a
sentinel p-value of 2 and deletes the rows; 437 of those are not expressed at
all. truecell returns them with ``p_val = 1`` instead — no evidence rather than
no row — which keeps the frame's gene set the same across every ``test_use``.
Verified gene-by-gene, not assumed: the two sets coincide exactly.

Changed later: ``deseq2``
-------------------------
``deseq2`` used to sum counts per sample and required ``sample_col``, while
Seurat's ``DESeq2DETest`` gives DESeq2 one column per cell, so its row here was a
divergence (22/50). It now runs Seurat's test, every cell a replicate, and
``sample_col`` sums each sample's cells first for a sample-level test. pydeseq2
needed four of DESeq2's choices: the local dispersion trend, gene-wise
dispersions kept out of the trend fit where the likelihood is flat, no Cook's
refit, and the 0.5 floor on fitted means in the Wald standard error
(``truecell/_deseq2.py``). Per cell here: **50/50** on the top 50,
p-value Spearman **0.9999991** on genes detected above 5 %, and the same **712**
genes at ``p_val_adj < 0.05``.

Differences left standing, and why
----------------------------------
* **``mast`` is a hand-rolled hurdle model**, not a call to the MAST package —
  which is not installable as a Python dependency. Spearman 0.946 on p-values and
  the same top 50 genes.
* **Seurat rounds ``myAUC`` to three decimals** inside ``DifferentialAUC``, so
  the ROC comparison is exact only to 5e-4. That is R's rounding, not a
  divergence — worth stating, because it looks like one.
* **Genes expressed in neither group get ``p = 1`` from truecell.** Seurat does
  the same when it runs ``wilcox`` through presto, as it does for these
  references; base R's ``wilcox.test``, which it falls back to without presto,
  returns ``NaN`` for them. A test that cannot be run has no evidence against the
  null, so 1 is the more useful answer.

Usage
-----
    python tutorials/pbmc3k_de_tutorial.py
    Rscript tutorials/pbmc3k_de_verify.R      # writes figures_de/r_<test>.csv
    python tutorials/pbmc3k_de_tutorial.py --report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from truecell import create_truecell_object, percentage_feature_set
from truecell.clustering import find_clusters
from truecell.datasets import pbmc3k
from truecell.markers import find_markers
from truecell.neighbors import find_neighbors
from truecell.preprocessing import find_variable_features, normalize_data, scale_data
from truecell.reduction import run_pca
from tutorials.bands import (
    Band, check_bands, check_shared_groups, render_verdicts,
)

FIGURES = Path(__file__).parent / "figures_de"
IDENT_1, IDENT_2 = "0", "1"
TOP_N = 50

# truecell test -> Seurat's spelling of the same test.
TEST_MAP = {
    "wilcox": "wilcox",
    "t": "t",
    "bimod": "bimod",
    "LR": "LR",
    "negbinom": "negbinom",
    "poisson": "poisson",
    "roc": "roc",
    "mast": "MAST",
    "deseq2": "DESeq2",
}

# Seurat rounds myAUC to 3 dp in DifferentialAUC, so the ROC comparison cannot be
# tighter than that however correct both sides are. Everything else is compared
# on its own terms; no blanket tolerance.
AUC_TOLERANCE = 5e-4
# avg_log2FC is pure arithmetic on the shared matrix — it should be exact.
LOG2FC_TOLERANCE = 1e-12

# --- Declared bands --------------------------------------------------------
# The parity table below used to live only in de_vignette.md, so nothing failed
# when a number moved: `deseq2`'s top-50 overlap was described as "25/50" in
# prose and had drifted to 22 without anyone noticing, and the same silence
# would have covered a real regression in any other row.
#
# Every p-value test is a port of the same statistic over the same cells, so
# its top-50 band is exact. `deseq2` joined them when it began testing cells as
# replicates, as Seurat's `DESeq2DETest` does. Before that it tested pseudobulk
# samples, and its band was a divergence measurement of 15-32.
RANKED_TESTS = ("wilcox", "t", "bimod", "LR", "negbinom", "poisson", "mast", "deseq2")

_PARITY_TOP50 = (
    "The same statistic on the same cells: the 50 most significant genes must "
    "be the same 50. A single dropped gene here is a regression, not scatter. "
    "The first seven have read 50/50 on three cluster assignments; at deseq2's "
    "cut the 50th and 51st genes sit 4.99 decades apart in truecell and 4.92 in "
    "Seurat.")

BANDS: dict[str, Band] = {
    **{f"{t} top50": Band(TOP_N, TOP_N, _PARITY_TOP50, fmt=".0f")
       for t in RANKED_TESTS},
    **{f"{t} rho>5%": Band(low, 1.0, why) for t, low, why in (
        ("wilcox", 0.9999,
         "Identical rank-sum statistic; measured 1.0 to nine decimal places."),
        ("t", 0.9999, "Identical Welch t; measured exactly 1.0."),
        ("bimod", 0.9999,
         "Identical likelihood-ratio test; measured 1.0 to nine decimal places."),
        ("LR", 0.9999,
         "Identical logistic-regression LRT; measured 1.0 to nine decimal places."),
        ("negbinom", 0.9999,
         "truecell fits glm.nb's own estimator, theta and coefficients by "
         "maximum likelihood: 0.9999994. It read 0.9217, against a floor of "
         "0.88, while it ran statsmodels' BFGS fit."),
        ("poisson", 0.9999,
         "Identical Poisson GLM Wald test; measured 0.9999989. The residual is "
         "R's glm.control(epsilon = 1e-8) stopping an iteration early, not a "
         "difference in the statistic — see the convergence note in the module "
         "docstring."),
        ("mast", 0.99,
         "truecell's hurdle model is hand-rolled rather than a call to the MAST "
         "package, so this is the closest a reimplementation gets: 0.9993."),
        ("deseq2", 0.9999, "DESeq2's Wald test as DESeq2DETest runs it: 0.9999991."),
    )},
    "max |dlog2FC| (parity tests)": Band(
        0, LOG2FC_TOLERANCE,
        "avg_log2FC is arithmetic on the shared matrix with no statistics in "
        "it, and every test reports Seurat's definition, deseq2 included, so "
        "all must agree to floating point: 6.2e-15 measured.", fmt=".2e"),
    "roc max |dAUC|": Band(
        0, AUC_TOLERANCE,
        "Seurat rounds myAUC to three decimals inside DifferentialAUC, so this "
        "cannot be tighter than half a unit in the last place however correct "
        "both sides are.", fmt=".2e"),
}


def build(data_dir=None):
    """pbmc3k through the standard pipeline, clustered."""
    counts, genes, cells = pbmc3k(data_dir=data_dir)
    obj = create_truecell_object(counts=counts, assay="RNA", min_cells=3,
                               min_features=200, project="pbmc3k_de",
                               feature_names=genes, cell_names=cells)
    percentage_feature_set(obj, pattern=r"^MT-", col_name="percent.mt")
    md = obj.meta_data
    keep = (md["nFeature_RNA"] > 200) & (md["nFeature_RNA"] < 2500) & (md["percent.mt"] < 5)
    obj = obj.subset(cells=list(md.index[keep]))

    normalize_data(obj)
    find_variable_features(obj, selection_method="vst", nfeatures=2000)
    scale_data(obj, features=obj.assays["RNA"]._all_feature_names)
    run_pca(obj, n_pcs=50, features=obj.assays["RNA"].variable_features,
            reduction_name="pca")
    find_neighbors(obj, dims=range(10), k_param=20)
    find_clusters(obj, resolution=0.5, algorithm=1, random_seed=0)
    return obj


def shared_groups(obj) -> pd.Series:
    """The two clusters both tools will test, written out for the R side.

    Exported rather than re-derived because Louvain numbering is not guaranteed
    to agree across implementations, and a clustering difference would masquerade
    as a DE difference.
    """
    idents = pd.Series([str(i) for i in obj.idents],
                       index=list(obj.assays["RNA"].cells()), name="group")
    return idents[idents.isin([IDENT_1, IDENT_2])]


def run_tests(obj, groups: pd.Series, tests=None) -> dict[str, pd.DataFrame]:
    sub = obj.subset(cells=list(groups.index))
    sub.idents = list(groups.values)

    out = {}
    for test in (tests or TEST_MAP):
        # deseq2 too tests every cell as a replicate, as Seurat's DESeq2DETest does.
        out[test] = find_markers(sub, IDENT_1, IDENT_2, test_use=test,
                                 logfc_threshold=0, min_pct=0)
    return out


def read_exact(path) -> pd.DataFrame | None:
    """An R table written as C99 hex floats, or None if it was not written.

    See the `write_exact` note in ``pbmc3k_de_verify.R``: neither R's `write.csv`
    nor its `sprintf("%.17g")` round-trips a float64, so the ordinary CSV cannot
    settle whether two values are *identical* — only whether they agree to about
    15 digits. `%a` transcribes the IEEE-754 bits, and `float.fromhex` reads them
    back exactly.
    """
    path = Path(path)
    if not path.exists():
        return None
    raw = pd.read_csv(path, index_col=0, dtype=str)
    out = {}
    for col in raw.columns:
        vals = raw[col]
        if vals.dropna().str.startswith(("0x", "-0x", "inf", "-inf", "nan")).all():
            out[col] = vals.map(
                lambda s: float("nan") if pd.isna(s) else float.fromhex(s))
        else:
            out[col] = vals
    return pd.DataFrame(out, index=raw.index)


def compare_adjusted_p(py: pd.DataFrame, r: pd.DataFrame, shared,
                       exact: pd.DataFrame | None) -> dict:
    """Agreement on ``p_val_adj`` — the column a manual annotation is read off.

    Reported separately from the raw p-value because it is a different question.
    A raw p-value is an intermediate; ``p_val_adj`` is what a person looks at and
    thresholds, so what matters is whether the two tools put a gene on the same
    side of 0.05, not whether the mantissas agree.

    One trap this deliberately reports around: Seurat clamps ``p_val_adj`` at 1,
    and on this contrast **11,858 of 13,712 genes** land there in both tools. A
    bare "fraction identical" is therefore ~0.88 before any of the interesting
    genes are considered, and would read as agreement where it is mostly just the
    clamp. The unclamped subset is scored on its own line for that reason.
    """
    res: dict = {}
    if "p_val_adj" not in py or "p_val_adj" not in r:
        return res

    a = py.loc[shared, "p_val_adj"]
    b = (exact.loc[shared, "p_val_adj"] if exact is not None
         else r.loc[shared, "p_val_adj"])
    ok = a.notna() & b.notna()
    res["padj_n_compared"] = int(ok.sum())
    res["padj_source"] = "hex" if exact is not None else "csv_15_digits"

    clamped = ok & (a == 1.0) & (b == 1.0)
    res["padj_both_clamped_at_1"] = int(clamped.sum())
    free = ok & ~clamped
    res["padj_n_unclamped"] = int(free.sum())

    if exact is not None:
        # Only meaningful against the hex tables; through the 15-digit CSV this
        # measures R's formatter, not the computation.
        res["padj_identical"] = float((a[ok] == b[ok]).mean())
        if free.any():
            res["padj_identical_unclamped"] = float((a[free] == b[free]).mean())

    with np.errstate(all="ignore"):
        for sf in (3, 6):
            res[f"padj_agree_{sf}sf"] = float(
                np.isclose(a[ok], b[ok], rtol=10.0 ** -sf, atol=0).mean())

    # The number the reviewer's question is actually about: would the two tools
    # put the same genes in front of someone annotating clusters?
    for cut in (0.05, 0.01):
        same = (a[ok] < cut) == (b[ok] < cut)
        res[f"padj_same_call_at_{cut}"] = float(same.mean())
        res[f"padj_disagreements_at_{cut}"] = int((~same).sum())
    return res


def compare(py: pd.DataFrame, r: pd.DataFrame, test: str,
            exact: pd.DataFrame | None = None) -> dict:
    """One test's agreement with Seurat, on the genes both scored."""
    from scipy.stats import kendalltau, spearmanr

    shared = py.index.intersection(r.index)
    res: dict = {"n_python": len(py), "n_r": len(r), "n_shared": len(shared)}

    if "avg_log2FC" in py and "avg_log2FC" in r:
        d = np.abs(py.loc[shared, "avg_log2FC"] - r.loc[shared, "avg_log2FC"])
        res["log2fc_max_abs_diff"] = float(d.max())
        res["log2fc_exact"] = bool(d.max() <= LOG2FC_TOLERANCE)

        # Rank agreement, alongside the bound above rather than instead of it:
        # the two fail differently. A uniform scale error leaves every rank
        # perfect and blows up the max; a handful of swapped mid-table genes
        # leaves the max tiny and moves the ranks. Someone ranking markers by
        # fold change is reading the second.
        x, y = py.loc[shared, "avg_log2FC"], r.loc[shared, "avg_log2FC"]
        m = x.notna() & y.notna()
        if m.sum() > 2:
            res["log2fc_spearman"] = float(spearmanr(x[m], y[m])[0])
            res["log2fc_kendall"] = float(kendalltau(x[m], y[m])[0])
            for k in (10, 25, 50):
                top_py = set(x[m].abs().sort_values(ascending=False).head(k).index)
                top_r = set(y[m].abs().sort_values(ascending=False).head(k).index)
                res[f"log2fc_top{k}_overlap"] = len(top_py & top_r)

    res.update(compare_adjusted_p(py, r, shared, exact))

    if test == "roc":
        d = np.abs(py.loc[shared, "myAUC"] - r.loc[shared, "myAUC"])
        # Half a unit in the third decimal is the most Seurat's rounding can move
        # an AUC, but the subtraction lands a few ULPs either side of it:
        # 0.488 - 0.4875 is 0.0005000000000000004. Rounding at 1e-12 keeps the
        # bound exactly Seurat's rather than widening it.
        worst = round(float(d.max()), 12)
        res["auc_max_abs_diff"] = worst
        res["auc_within_seurat_rounding"] = bool(worst <= AUC_TOLERANCE)
        return res

    pp, pr = py.loc[shared, "p_val"], r.loc[shared, "p_val"]
    # R writes NaN where a test could not be run and 0 where it underflowed;
    # neither carries a rank, so both are excluded rather than imputed.
    ok = pp.notna() & pr.notna() & (pp > 0) & (pr > 0)
    res["n_compared"] = int(ok.sum())
    res["r_nan"] = int(pr.isna().sum())
    res["r_underflow_zero"] = int((pr == 0).sum())
    if ok.sum() > 2:
        res["p_spearman"] = float(spearmanr(pp[ok], pr[ok])[0])
        res["p_max_log10_ratio"] = float(np.abs(np.log10(pp[ok] / pr[ok])).max())
    # Ranked on the same `ok` genes the correlation uses. Including R's underflow
    # zeros here would rank 168 genes that are all exactly 0 against each other,
    # and the arbitrary tie-break made a perfectly-agreeing wilcox read 0/50.
    top_py = set(pp[ok].sort_values().head(TOP_N).index)
    top_r = set(pr[ok].sort_values().head(TOP_N).index)
    res[f"top{TOP_N}_overlap"] = len(top_py & top_r)

    # Where the gene is actually expressed, agreement should be strong; the tail
    # is near-empty rows where every NB/hurdle fit is ill-conditioned.
    expressed = shared[(py.loc[shared, "pct.1"] > 0.05) | (py.loc[shared, "pct.2"] > 0.05)]
    e = expressed[ok.reindex(expressed, fill_value=False)]
    if len(e) > 2:
        res["p_spearman_expressed"] = float(spearmanr(pp[e], pr[e])[0])
        res["n_expressed"] = len(e)
    return res


def run_full(data_dir=None, verbose=True):
    FIGURES.mkdir(exist_ok=True)
    obj = build(data_dir)
    groups = shared_groups(obj)
    groups.to_csv(FIGURES / "groups.csv", header=True)

    results = run_tests(obj, groups)
    for test, df in results.items():
        df.to_csv(FIGURES / f"py_{test}.csv")

    summary = {"n_cells": len(groups),
               "group_sizes": groups.value_counts().to_dict(),
               "n_genes": int(len(next(iter(results.values()))))}
    (FIGURES / "py_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"\n  {summary['n_cells']} cells "
              f"(cluster {IDENT_1}: {summary['group_sizes'].get(IDENT_1)}, "
              f"cluster {IDENT_2}: {summary['group_sizes'].get(IDENT_2)}), "
              f"{summary['n_genes']} genes")
        for test, df in results.items():
            print(f"    {test:9s} {len(df):6d} genes")
    return obj, results


def report_concordance():
    if not (FIGURES / "py_wilcox.csv").exists():
        raise SystemExit("Run the tutorial first — figures_de/py_*.csv are missing.")
    missing = [t for t in TEST_MAP if not (FIGURES / f"r_{t.lower()}.csv").exists()]
    if missing:
        raise SystemExit(
            f"Missing R output for: {', '.join(missing)}\n"
            "Run: Rscript tutorials/pbmc3k_de_verify.R"
        )
    rows = []
    for test in TEST_MAP:
        # `float_precision="round_trip"` because pandas' default CSV *reader* is
        # not correctly rounded — it misparses about a third of random doubles by
        # an ULP. `to_csv` was never the problem; it already writes the shortest
        # round-trippable form. Without this the Python table is perturbed on the
        # way back in, and an exactness comparison measures the parser.
        py = pd.read_csv(FIGURES / f"py_{test}.csv", index_col=0,
                         float_precision="round_trip")
        r = pd.read_csv(FIGURES / f"r_{test.lower()}.csv", index_col=0,
                        float_precision="round_trip")
        # Before anything is compared: was this R table computed on the groups
        # the Python side just wrote? Nothing else here can tell the difference
        # between a port that regressed and an R run left over from a previous
        # clustering, and the second reads exactly like the first.
        check_shared_groups(py, r, source=f"figures_de/r_{test.lower()}.csv")
        exact = read_exact(FIGURES / f"r_{test.lower()}_exact.csv")
        rows.append({"test": test, **compare(py, r, test, exact)})
    table = pd.DataFrame(rows).set_index("test")
    _print_report(table)
    verdicts = check_bands(BANDS, measure_bands(table))
    table.attrs["band_verdicts"] = verdicts
    table.attrs["bands_hold"] = render_verdicts(
        verdicts, "Declared bands — every row of the parity table.")
    return table


def measure_bands(table: pd.DataFrame) -> dict[str, float]:
    """Pull the numbers :data:`BANDS` judges out of the concordance table.

    Kept separate from :func:`compare` so the bands can be checked against a
    table read from anywhere — including a deliberately damaged one in the
    tests, which is the only way to know the assertions can fail.
    """
    measured: dict[str, float] = {}
    for test in table.index:
        if f"{test} top50" in BANDS:
            measured[f"{test} top50"] = table.loc[test].get(
                f"top{TOP_N}_overlap", float("nan"))
        if f"{test} rho>5%" in BANDS:
            measured[f"{test} rho>5%"] = table.loc[test].get(
                "p_spearman_expressed", float("nan"))
    # One band over every test: all of them report Seurat's fold change, deseq2
    # included. It was excluded by name while it reported a pseudobulk one.
    measured["max |dlog2FC| (parity tests)"] = float(
        table["log2fc_max_abs_diff"].max())
    if "auc_max_abs_diff" in table.columns:
        measured["roc max |dAUC|"] = float(table.loc["roc", "auc_max_abs_diff"])
    return measured


def _print_report(table: pd.DataFrame) -> None:
    print(f"\n  {'test':10s} {'shared':>7s} {'max|dFC|':>10s} "
          f"{'p spearman':>11s} {'expressed':>10s} {'top' + str(TOP_N):>7s}")
    print("  " + "-" * 62)
    for test, row in table.iterrows():
        fc = row.get("log2fc_max_abs_diff", float("nan"))
        sp = row.get("p_spearman", float("nan"))
        ex = row.get("p_spearman_expressed", float("nan"))
        top = row.get(f"top{TOP_N}_overlap", float("nan"))
        if test == "roc":
            print(f"  {test:10s} {int(row['n_shared']):7d} {fc:10.2e} "
                  f"{'AUC ' + format(row['auc_max_abs_diff'], '.1e'):>11s} "
                  f"{'(Seurat rounds to 3dp)':>19s}")
            continue
        print(f"  {test:10s} {int(row['n_shared']):7d} {fc:10.2e} "
              f"{sp:11.6f} {ex:10.4f} {int(top):4d}/{TOP_N}")

    _print_reader_facing(table)


def _print_reader_facing(table: pd.DataFrame) -> None:
    """The two columns a person actually reads off a marker table.

    The block above scores the statistic. This one scores what someone doing the
    annotation sees: the fold change they rank by, and the adjusted p they
    threshold on. Both were unreported until an expert reviewer pointed out that
    a set-overlap metric answers neither.
    """
    # Every cell is formatted through `cell`, because not every column exists for
    # every row: `roc` has no `p_val_adj`, and a table too short for a rank
    # correlation (the band tests use a two-gene stub) has no rho or tau. A row
    # that cannot be scored should print a dash, not raise.
    def cell(value, spec: str, width: int) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return f"{'—':>{width}}"
        return format(value, spec)

    print(f"\n  {'test':10s} {'logFC rho':>10s} {'logFC tau':>10s} "
          f"{'top50':>7s} {'padj 6sf':>9s} {'call@.05':>9s} {'differ':>7s}")
    print("  " + "-" * 68)
    for test, row in table.iterrows():
        top = row.get("log2fc_top50_overlap", float("nan"))
        print(f"  {test:10s} "
              f"{cell(row.get('log2fc_spearman'), '10.6f', 10)} "
              f"{cell(row.get('log2fc_kendall'), '10.6f', 10)} "
              f"{cell(top, '4.0f', 4)}/50 "
              f"{cell(row.get('padj_agree_6sf'), '9.4f', 9)} "
              f"{cell(row.get('padj_same_call_at_0.05'), '9.4f', 9)} "
              f"{cell(row.get('padj_disagreements_at_0.05'), '7.0f', 7)}")
    print("\n  `call@.05` is the fraction of genes both tools place on the same "
          "side of\n  adjusted p = 0.05, and `differ` counts the ones they do "
          "not — the number\n  that matters if the table is being read to "
          "annotate clusters by hand.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--report", action="store_true",
                        help="compare against figures_de/r_<test>.csv")
    args = parser.parse_args()
    if args.report:
        table = report_concordance()
        # Exit non-zero so `--report` can be used as a check rather than read as
        # a table. A band that only prints is the situation this replaced.
        if not table.attrs.get("bands_hold", True):
            raise SystemExit(1)
        return
    run_full(data_dir=args.data_dir)
    print(f"\n  Wrote {FIGURES}")
    print("  Next: Rscript tutorials/pbmc3k_de_verify.R")


if __name__ == "__main__":
    main()
