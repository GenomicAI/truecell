"""The DE tutorial's declared bands, and the guard that has to pass first.

`de_vignette.md` carries a parity table — top-50 overlap, p-value Spearman,
max |Δlog2FC| — that nothing checked. `deseq2`'s row had drifted from 25/50 to
22/50 unnoticed, and the same silence covered every other row. These tests drive
`measure_bands` and `BANDS` on hand-built tables, so each assertion can be shown
to fail rather than assumed to.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tutorials.pbmc3k_de_tutorial as de  # noqa: E402
from tutorials.bands import check_bands  # noqa: E402


def _clean_table() -> pd.DataFrame:
    """The concordance table as it reads on a good run, measured 2026-09-15.

    Re-measured once `find_clusters` ran Seurat's own optimiser, which changed the
    two clusters the tutorial tests to 703 and 480 cells, from 692 and 515, and
    again once `compare` read R's values from the hex tables: the fold-change
    bound went from 6.22e-15, R's 15-digit CSV, to 1.78e-15.
    """
    rows = {
        "wilcox":   (50, 1.000000, 1.78e-15, np.nan),
        "t":        (50, 1.000000, 1.78e-15, np.nan),
        "bimod":    (50, 1.000000, 1.78e-15, np.nan),
        "LR":       (50, 1.000000, 1.78e-15, np.nan),
        "negbinom": (50, 0.999999, 1.78e-15, np.nan),
        "poisson":  (50, 0.999999, 1.78e-15, np.nan),
        "roc":      (np.nan, np.nan, 1.78e-15, 5.0e-4),
        "mast":     (50, 0.999285, 1.78e-15, np.nan),
        "deseq2":   (50, 0.999999, 1.78e-15, np.nan),
    }
    return pd.DataFrame(
        [{"test": k, f"top{de.TOP_N}_overlap": v[0], "p_spearman_expressed": v[1],
          "log2fc_max_abs_diff": v[2], "auc_max_abs_diff": v[3]}
         for k, v in rows.items()]).set_index("test")


def _holds(table) -> list[str]:
    """Names of the bands that fail on ``table``."""
    return [v.name for v in check_bands(de.BANDS, de.measure_bands(table))
            if not v.ok]


# ---------------------------------------------------------------------------
# The bands, against the numbers actually measured
# ---------------------------------------------------------------------------

def test_every_band_holds_on_the_measured_run():
    """The declared bands must admit the run they were derived from.

    A band that fails on a good run is worse than no band: it trains the reader
    to ignore the section.
    """
    assert _holds(_clean_table()) == []


def test_every_band_is_fed_by_measure_bands():
    """No band may be declared without something computing it.

    `check_bands` scores an absent name NaN and NaN fails, so this would show up
    as a mystery failure at `--report` time; here it names the band.
    """
    measured = de.measure_bands(_clean_table())
    assert set(de.BANDS) <= set(measured), (
        f"declared but never measured: {sorted(set(de.BANDS) - set(measured))}")


def test_a_single_dropped_gene_fails_the_parity_band():
    """49/50 is a regression, not scatter — the band has to say so.

    The exact bands are the ones most likely to be dismissed as brittle, so this
    pins the intent: for a test that shares Seurat's statistic and cells, one
    gene is a failure.
    """
    table = _clean_table()
    table.loc["wilcox", f"top{de.TOP_N}_overlap"] = 49
    assert _holds(table) == ["wilcox top50"]


def test_deseq2_is_held_to_the_exact_band_too():
    """deseq2 runs Seurat's per-cell test now, so one dropped gene fails it."""
    table = _clean_table()
    table.loc["deseq2", f"top{de.TOP_N}_overlap"] = 49
    assert _holds(table) == ["deseq2 top50"]


def test_a_return_to_the_pseudobulk_test_fails_every_deseq2_band():
    """The pseudobulk deseq2 read 22/50, rho 0.195 and |dlog2FC| 3.47 here.

    Its old bands were built to admit exactly those numbers. A regression that
    summed cells again, or reported DESeq2's own fold change, must now fail all
    three bands it touches.
    """
    table = _clean_table()
    table.loc["deseq2", [f"top{de.TOP_N}_overlap", "p_spearman_expressed",
                         "log2fc_max_abs_diff"]] = [22, 0.195146, 3.47]
    assert _holds(table) == ["deseq2 top50", "deseq2 rho>5%",
                             "max |dlog2FC| (parity tests)"]


def test_the_fold_change_band_covers_every_test():
    """No row is excluded from the fold-change band any more.

    deseq2 was excluded by name while it reported a fold change on summed counts.
    It reports Seurat's now, so a regression in it has to fail the same band as
    one in any other test.
    """
    measured = de.measure_bands(_clean_table())
    assert measured["max |dlog2FC| (parity tests)"] == pytest.approx(1.78e-15, rel=1e-9, abs=0)

    for test in ("mast", "deseq2"):
        table = _clean_table()
        table.loc[test, "log2fc_max_abs_diff"] = 1e-6
        assert "max |dlog2FC| (parity tests)" in _holds(table), test


def test_the_auc_band_is_r_s_rounding_and_not_a_free_tolerance():
    """5e-4 is half a unit in Seurat's third decimal, not a chosen slack."""
    assert de.BANDS["roc max |dAUC|"].high == de.AUC_TOLERANCE
    table = _clean_table()
    table.loc["roc", "auc_max_abs_diff"] = 1e-3
    assert "roc max |dAUC|" in _holds(table)


def test_an_auc_exactly_at_seurat_s_rounding_passes_the_band():
    """``0.488 - 0.4875`` is 0.0005000000000000004, a few ULPs past the bound it
    sits on, and PBMC 3k has six such genes. ``compare`` rounds that away rather
    than widening the band, so a run that meets Seurat's rounding exactly passes."""
    assert 0.488 - 0.4875 > de.AUC_TOLERANCE
    py = pd.DataFrame({"myAUC": [0.4875, 0.25]}, index=["edge", "other"])
    r = pd.DataFrame({"myAUC": [0.488, 0.25]}, index=["edge", "other"])
    res = de.compare(py, r, "roc")
    assert res["auc_max_abs_diff"] == de.AUC_TOLERANCE
    assert res["auc_within_seurat_rounding"]
    table = _clean_table()
    table.loc["roc", "auc_max_abs_diff"] = res["auc_max_abs_diff"]
    assert "roc max |dAUC|" not in _holds(table)


def test_compare_reads_r_s_values_from_the_hex_table():
    """R's 15-digit CSV cannot say whether two doubles are the same one.

    ``0.1 + 0.2`` is one ULP above ``0.3``, which is what ``write.csv`` prints for
    it. Read through the CSV that looks like a disagreement; the hex table, which
    carries the bits, shows the two tools agree exactly.
    """
    genes = ["A", "B", "C"]
    py = pd.DataFrame({"p_val": [(0.1 + 0.2) * 1e-3, 0.5, 0.9],
                       "avg_log2FC": [0.1 + 0.2, -1.0, 2.0],
                       "pct.1": [0.5, 0.4, 0.3], "pct.2": [0.3, 0.2, 0.1],
                       "p_val_adj": [1.0, 1.0, 1.0]}, index=genes)
    csv = py.copy()
    for column in ("p_val", "avg_log2FC"):
        csv[column] = [float(f"{v:.15g}") for v in py[column]]
    assert csv.loc["A", "avg_log2FC"] != py.loc["A", "avg_log2FC"]

    through_csv = de.compare(py, csv, "wilcox")
    assert through_csv["r_values"] == "csv_15_digits"
    assert through_csv["log2fc_max_abs_diff"] > 0
    assert through_csv["p_max_log10_ratio"] > 0

    through_hex = de.compare(py, csv, "wilcox", exact=py.copy())
    assert through_hex["r_values"] == "hex"
    assert through_hex["log2fc_max_abs_diff"] == 0
    assert through_hex["p_max_log10_ratio"] == 0


def test_a_missing_column_fails_rather_than_disappearing():
    """Dropping the measurement must not be a way to pass.

    `compare` only emits `top50_overlap` when enough genes survive the NaN and
    underflow filters, so this is reachable without anyone editing a band.
    """
    table = _clean_table().drop(columns=[f"top{de.TOP_N}_overlap"])
    failed = _holds(table)
    assert "wilcox top50" in failed and "deseq2 top50" in failed


# ---------------------------------------------------------------------------
# The handoff guard, in the report path
# ---------------------------------------------------------------------------

def _marker_csv(path, pct1, pct2, genes=("A", "B")):
    # myAUC rides along for the roc tables; the other tests ignore the column.
    pd.DataFrame({"p_val": [0.1] * len(genes), "avg_log2FC": [0.0] * len(genes),
                  "myAUC": [0.5] * len(genes),
                  "pct.1": pct1, "pct.2": pct2,
                  "p_val_adj": [0.1] * len(genes)},
                 index=list(genes)).rename_axis("gene").to_csv(path)


def test_report_concordance_refuses_a_stale_r_run(tmp_path, monkeypatch):
    """End to end: a mismatched pct column stops the report, not just a helper.

    The helper being right is not enough — the defect was that nothing called
    it. Damaging only the R side of one test is exactly the 19-July-vs-25-July
    situation, in miniature.
    """
    from tutorials.bands import StaleReferenceError

    monkeypatch.setattr(de, "FIGURES", tmp_path)
    for test in de.TEST_MAP:
        _marker_csv(tmp_path / f"py_{test}.csv", [0.500, 0.400], [0.300, 0.200])
        _marker_csv(tmp_path / f"r_{test.lower()}.csv", [0.500, 0.400], [0.300, 0.200])
    _marker_csv(tmp_path / "r_mast.csv", [0.500, 0.470], [0.300, 0.200])

    with pytest.raises(StaleReferenceError, match="not computed on the current"):
        de.report_concordance()


def test_report_concordance_accepts_a_reference_within_r_s_rounding(tmp_path,
                                                                    monkeypatch):
    """The guard must not fire on R's three-decimal pct columns.

    Seurat rounds to three decimals and truecell now does too, but a Python table
    written before it did carries unrounded rates. The guard is there to catch a
    stale reference, so it has to let a rounding difference through.
    """
    monkeypatch.setattr(de, "FIGURES", tmp_path)
    for test in de.TEST_MAP:
        _marker_csv(tmp_path / f"py_{test}.csv", [0.50049, 0.4], [0.3, 0.2])
        _marker_csv(tmp_path / f"r_{test.lower()}.csv", [0.500, 0.4], [0.3, 0.2])

    table = de.report_concordance()
    assert list(table.index) == list(de.TEST_MAP)
