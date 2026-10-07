"""Composition / differential-abundance testing across conditions.

Answers "is cluster/cell-type X over-represented in condition A vs B?" — the
enrichment step every spatial/atlas Seurat analysis ends with. Reports a
*directional* log2 proportion ratio (unambiguous), a per-group Fisher exact
test (group vs rest), BH-adjusted q-values, and the overall chi-square.

The Fisher test counts every cell as a replicate, which is only right for a single
sample per condition. With several donors per condition, pass ``sample_col`` and
the test compares per-sample proportions instead.

Depends only on scipy (a core dependency); BH is computed inline so no
statsmodels requirement.
"""
from __future__ import annotations

import warnings
from math import comb
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def _finish(df: pd.DataFrame, ref: str, test: str) -> pd.DataFrame:
    """BH-adjust, label, direction and order a table of per-group tests."""
    df["padj"] = _bh(df["p"].to_numpy())
    bins = [-np.inf, 0.001, 0.01, 0.05, np.inf]
    df["sig"] = pd.cut(df["padj"], bins, labels=["***", "**", "*", "ns"])
    df["enriched_in"] = np.where(df["log2_ratio"] > 0, test, ref)
    return df.sort_values("log2_ratio", ascending=False).reset_index(drop=True)


def _across_samples(md, group_by, split_by, sample_col, ref, test) -> pd.DataFrame:
    """The per-sample version: one observation per sample, not one per cell.

    Each group's proportion in each sample is compared between the two levels with a
    two-sided Mann-Whitney U test. A cell with no sample is left out, up front: left to
    ``astype(str)`` it would become a sample called "nan" on pandas 2 and vanish on
    pandas 3.
    """
    keep = md[sample_col].notna()
    cond = md.loc[keep, split_by].astype(str)
    sample = md.loc[keep, sample_col].astype(str)
    group = md.loc[keep, group_by].astype(str)
    # A sample is one donor, so it sits in one condition; if it does not, the
    # comparison is not between samples at all.
    spans = cond.groupby(sample).nunique()
    spans = spans[spans > 1].index.tolist()
    if spans:
        raise ValueError(
            f"sample_col='{sample_col}' must sit inside one level of split_by='{split_by}', "
            f"but {spans[:5]} have cells in both."
        )
    counts = pd.crosstab(sample, group)
    cells = pd.crosstab(group, cond)                          # the cells used, group x level
    props = counts.div(counts.sum(axis=1), axis=0)
    level = cond.groupby(sample).first().reindex(props.index)
    in_ref, in_test = (level == ref).to_numpy(), (level == test).to_numpy()
    n_ref, n_test = int(in_ref.sum()), int(in_test.sum())
    if not (n_ref and n_test):
        empty = ref if not n_ref else test
        raise ValueError(
            f"No sample in split_by level '{empty}': sample_col='{sample_col}' is missing "
            f"for every one of its cells."
        )

    # The smallest p an exact two-sided test can give, with every sample of one level
    # above every sample of the other: 2 / C(n1 + n2, n1). Past 0.05 nothing can come
    # out significant, however large the difference.
    p_min = 2 / comb(n_ref + n_test, n_ref)
    if p_min >= 0.05:
        warnings.warn(
            f"{n_test} samples against {n_ref}: the smallest p-value a Mann-Whitney test "
            f"can give is {p_min:.3g}, so no group can come out significant. The log2 "
            f"ratios are still a ranking.",
            stacklevel=3,
        )

    rows = []
    for grp in counts.columns:
        x_test, x_ref = props.loc[in_test, grp].to_numpy(), props.loc[in_ref, grp].to_numpy()
        p = float(stats.mannwhitneyu(x_test, x_ref, alternative="two-sided").pvalue)
        # Identical in every sample: scipy returns NaN, and one NaN would turn every
        # adjusted p-value into NaN. There is nothing to tell the levels apart by.
        p = 1.0 if np.isnan(p) else p
        prop_test, prop_ref = x_test.mean(), x_ref.mean()
        with np.errstate(divide="ignore"):
            log2 = np.log2(prop_test / prop_ref) if prop_ref else np.nan
        rows.append({
            "group": grp, f"n_{ref}": cells.loc[grp, ref], f"n_{test}": cells.loc[grp, test],
            f"prop_{ref}": prop_ref, f"prop_{test}": prop_test,
            "log2_ratio": log2, "p": p,
        })
    df = _finish(pd.DataFrame(rows), ref, test)
    df.attrs.update(reference=ref, test=test, method="mannwhitneyu", sample_col=sample_col,
                    n_samples={ref: n_ref, test: n_test})
    return df


def composition_test(
    seurat,
    group_by: str,
    split_by: str,
    reference: Optional[str] = None,
    sample_col: str | None = None,
) -> pd.DataFrame:
    """Directional abundance test of ``group_by`` categories across ``split_by``.

    Mirrors the enrichment table an analyst builds by hand: for a two-level
    ``split_by`` (e.g. condition), each ``group_by`` category (e.g. cluster) gets
    a ``log2(prop_test / prop_reference)`` and a Fisher exact p (that category vs
    all others). p-values are BH-adjusted. The overall chi-square p is stored in
    ``df.attrs['chisq_p']``.

    **Cells are not replicates.** Without ``sample_col`` the Fisher test counts every
    cell as an independent observation. With thousands of cells nearly every
    difference is then significant, including one that is only donor-to-donor
    variation, because the unit that replicates is the donor, not the cell. In a
    simulation with no condition effect at all (12 donors, 6 against 6, each with its
    own cell-type proportions) at least one cell type came out significant in 35 of 40
    runs. Read the p-values as a ranking, or as a test for one sample per condition.
    For several donors per condition pass ``sample_col``.

    Parameters
    ----------
    group_by   : categorical metadata column tested for enrichment (rows).
    split_by   : metadata column with exactly two levels (the conditions).
    reference  : which ``split_by`` level is the denominator (default: the first
                 sorted level). log2 > 0 ⇒ enriched in the *other* level.
    sample_col : metadata column naming the sample (donor) each cell came from. The
                 test then has one observation per sample: each group's proportion
                 in each sample is compared between the two levels with a two-sided
                 Mann-Whitney U test, BH-adjusted across groups. Every sample must sit
                 in one level of ``split_by``, and a cell with no sample is left out.
                 ``prop_<level>`` is then the mean of the per-sample proportions,
                 ``odds_ratio`` and ``chisq_p`` are not reported (they are cell-level),
                 and ``df.attrs['n_samples']`` counts the samples per level. With few
                 samples the test cannot reach significance whatever the data: three
                 against three cannot give a p-value below 0.1, and a warning says so.

    Returns
    -------
    A DataFrame ordered by ``log2_ratio``, with the columns ``group``,
    ``n_<ref>``, ``n_<test>``, ``prop_<ref>``, ``prop_<test>``, ``log2_ratio``,
    ``odds_ratio``, ``p``, ``padj``, ``sig`` and ``enriched_in``.
    """
    md = seurat.meta_data
    for col in (group_by, split_by) + ((sample_col,) if sample_col is not None else ()):
        if col not in md.columns:
            raise KeyError(f"'{col}' not in meta_data.")
    tab = pd.crosstab(md[group_by].astype(str), md[split_by].astype(str))
    conds = list(tab.columns)
    if len(conds) != 2:
        raise ValueError(
            f"split_by='{split_by}' must have exactly 2 levels, found {conds}."
        )
    ref = reference if reference is not None else conds[0]
    if ref not in conds:
        raise ValueError(f"reference '{ref}' not a level of {split_by}: {conds}.")
    test = [c for c in conds if c != ref][0]
    if sample_col is not None:
        return _across_samples(md, group_by, split_by, sample_col, ref, test)

    n_ref, n_test = tab[ref].sum(), tab[test].sum()
    rows = []
    for grp in tab.index:
        a, b = tab.loc[grp, test], tab.loc[grp, ref]          # this group
        c, d = n_test - a, n_ref - b                          # all other groups
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        prop_test = a / n_test if n_test else np.nan
        prop_ref = b / n_ref if n_ref else np.nan
        with np.errstate(divide="ignore"):
            log2 = np.log2(prop_test / prop_ref) if prop_ref else np.nan
        rows.append({
            "group": grp, f"n_{ref}": b, f"n_{test}": a,
            f"prop_{ref}": prop_ref, f"prop_{test}": prop_test,
            "log2_ratio": log2, "odds_ratio": odds, "p": p,
        })
    df = _finish(pd.DataFrame(rows), ref, test)
    df.attrs["chisq_p"] = float(stats.chi2_contingency(tab.to_numpy())[1])
    df.attrs["reference"] = ref
    df.attrs["test"] = test
    df.attrs["method"] = "fisher_exact"
    return df
