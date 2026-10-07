"""``composition_test`` and its unit of replication.

The Fisher test per group counts every cell as an independent observation. With
thousands of cells nearly every difference is significant, including one that is
only donor-to-donor variation: in a simulation with no condition effect at all
(12 donors, 6 against 6, each with its own cell-type proportions) the default
reported a significant cell type in 35 of 40 runs. ``sample_col`` gives the test one
observation per sample, which is what replicates.

The function reads only ``meta_data``, so the simulations use a stand-in for the
object and run in a fraction of a second.
"""
import warnings
from math import comb
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from truecell import composition_test


def _simulate(seed, n_types=6, per_donor=600, n_donors=12, effect=None):
    """Donors with their own cell-type proportions around uniform; half in each condition."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_donors):
        p = rng.dirichlet(np.full(n_types, 15.0))
        if effect is not None and d >= n_donors // 2:
            p = p * effect
            p = p / p.sum()
        rows += [(f"d{d}", "A" if d < n_donors // 2 else "B", f"T{t}")
                 for t in rng.choice(n_types, per_donor, p=p)]
    md = pd.DataFrame(rows, columns=["donor", "cond", "ct"],
                      index=[f"c{i}" for i in range(len(rows))])
    return SimpleNamespace(meta_data=md)


def test_cells_as_replicates_find_differences_that_are_only_donor_variation():
    """The premise: with no condition effect, the default reports one almost every time,
    and `sample_col` about as often as its nominal 5%."""
    cell = sample = 0
    for seed in range(30):
        obj = _simulate(seed)
        cell += bool((composition_test(obj, "ct", "cond")["padj"] < 0.05).any())
        sample += bool((composition_test(obj, "ct", "cond", sample_col="donor")["padj"] < 0.05).any())
    # The simulation has to be able to show the problem, or the second assertion is empty.
    assert cell >= 20, cell
    assert sample <= 4, sample


def test_a_real_difference_between_donors_is_still_found():
    """Not merely conservative: a cell type three times as common in one condition."""
    found = 0
    for seed in range(10):
        obj = _simulate(100 + seed, effect=np.array([3, 1, 1, 1, 1, 1]))
        res = composition_test(obj, "ct", "cond", sample_col="donor").set_index("group")
        found += bool(res.loc["T0", "padj"] < 0.05) and res.loc["T0", "enriched_in"] == "B"
    assert found >= 9, found


def test_the_sample_level_table_is_the_mann_whitney_of_per_sample_proportions():
    rng = np.random.default_rng(3)
    rows = []
    # 4 donors a side; group "x" is rarer in B, "y" commoner, "z" the same
    for d in range(8):
        cond = "A" if d < 4 else "B"
        w = {"x": 0.5 if cond == "A" else 0.2, "y": 0.2 if cond == "A" else 0.5, "z": 0.3}
        rows += [(f"d{d}", cond, g) for g in rng.choice(list(w), 80 + 10 * d, p=list(w.values()))]
    md = pd.DataFrame(rows, columns=["donor", "cond", "ct"])
    res = composition_test(SimpleNamespace(meta_data=md), "ct", "cond", sample_col="donor",
                           reference="A").set_index("group")

    props = pd.crosstab(md["donor"], md["ct"], normalize="index")
    a, b = props.loc[[f"d{i}" for i in range(4)]], props.loc[[f"d{i}" for i in range(4, 8)]]
    for group in "xyz":
        want = mannwhitneyu(b[group], a[group], alternative="two-sided").pvalue
        assert res.loc[group, "p"] == pytest.approx(want, rel=1e-12)
        assert res.loc[group, "prop_A"] == pytest.approx(a[group].mean())      # mean of the samples,
        assert res.loc[group, "prop_B"] == pytest.approx(b[group].mean())      # not the pooled cells
        assert res.loc[group, "log2_ratio"] == pytest.approx(np.log2(b[group].mean() / a[group].mean()))
        cells = md[md["ct"] == group]
        assert res.loc[group, "n_A"] == (cells["cond"] == "A").sum()
        assert res.loc[group, "n_B"] == (cells["cond"] == "B").sum()
    assert res.loc["x", "enriched_in"] == "A" and res.loc["y", "enriched_in"] == "B"
    assert (res["padj"] >= res["p"] - 1e-12).all()
    assert list(res["log2_ratio"]) == sorted(res["log2_ratio"], reverse=True)


def test_the_sample_level_table_leaves_out_what_only_cells_can_give():
    res = composition_test(_simulate(0), "ct", "cond", sample_col="donor")
    assert "odds_ratio" not in res.columns
    assert "chisq_p" not in res.attrs
    assert res.attrs["method"] == "mannwhitneyu" and res.attrs["sample_col"] == "donor"
    assert res.attrs["n_samples"] == {"A": 6, "B": 6}
    assert {"group", "log2_ratio", "p", "padj", "sig", "enriched_in"} <= set(res.columns)


def test_the_default_is_the_same_cell_level_test_as_before():
    res = composition_test(_simulate(0), "ct", "cond")
    assert "odds_ratio" in res.columns and "chisq_p" in res.attrs
    assert res.attrs["method"] == "fisher_exact" and "sample_col" not in res.attrs


def test_a_sample_in_both_levels_is_an_error():
    obj = _simulate(0)
    obj.meta_data.loc[obj.meta_data["donor"] == "d0", "cond"] = ["A", "B"] * 300
    with pytest.raises(ValueError, match=r"must sit inside one level.*\['d0'\]"):
        composition_test(obj, "ct", "cond", sample_col="donor")


def test_an_unknown_sample_column_is_a_keyerror():
    with pytest.raises(KeyError, match="'nope' not in meta_data"):
        composition_test(_simulate(0), "ct", "cond", sample_col="nope")


@pytest.mark.parametrize("n_donors,warns", [(6, True), (7, True), (8, False), (12, False)])
def test_too_few_samples_to_reach_significance_warns(n_donors, warns):
    """Three against three cannot give a p below 0.1; four against four can give 0.029."""
    assert (2 / comb(n_donors, n_donors // 2) >= 0.05) == warns        # the premise, from the formula
    obj = _simulate(0, n_donors=n_donors, per_donor=100)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        composition_test(obj, "ct", "cond", sample_col="donor")
    few = [w for w in seen if "smallest p-value a Mann-Whitney test" in str(w.message)]
    assert bool(few) == warns


def test_a_group_the_same_in_every_sample_gets_p_one_and_does_not_poison_the_rest():
    """scipy returns NaN when every value is identical, and one NaN makes every
    adjusted p-value NaN."""
    rng = np.random.default_rng(5)
    rows = []
    for d in range(8):
        cond = "A" if d < 4 else "B"
        rows += [(f"d{d}", cond, "same")] * 20                       # exactly 20% of every sample
        rows += [(f"d{d}", cond, g) for g in rng.choice(["x", "y"], 80)]
    md = pd.DataFrame(rows, columns=["donor", "cond", "ct"])
    assert pd.crosstab(md["donor"], md["ct"], normalize="index")["same"].nunique() == 1   # the premise
    res = composition_test(SimpleNamespace(meta_data=md), "ct", "cond",
                           sample_col="donor").set_index("group")
    assert res.loc["same", "p"] == 1.0
    assert not res["p"].isna().any() and not res["padj"].isna().any()
