"""negbinom against MASS::glm.nb, on PBMC 3k genes statsmodels could not fit reliably.

`tests/data/r_negbinom_reference.json` is written by
`tests/data/make_negbinom_reference.R` from the counts in
`tests/data/negbinom_reference_counts.json`: clusters 0 and 1 of the DE tutorial,
1,207 cells. Eleven of the genes are among the 39 whose p-value moved by more than
2 % between statsmodels 0.14.6 and 0.15.0, while negbinom ran statsmodels' BFGS
fit: in one version or the other, each collapsed theta or stopped unconverged.
Three more fit well in both, and one, CACNA2D3, separates the groups.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from truecell import _glm_nb, create_truecell_object, find_markers
from truecell.preprocessing import normalize_data

DATA = Path(__file__).parent / "data"
COUNTS = json.loads((DATA / "negbinom_reference_counts.json").read_text())
REFERENCE = json.loads((DATA / "r_negbinom_reference.json").read_text())
R = {f["gene"]: f for f in REFERENCE["fits"]}
R_LATENT = {f["gene"]: f for f in REFERENCE["latent"]}
SEPARATED = "CACNA2D3"
FITTED = [g for g in COUNTS["genes"] if g != SEPARATED]
GROUP = np.array(COUNTS["group"], dtype=float)
X = np.column_stack([np.ones(GROUP.size), GROUP])


def _counts(gene):
    return np.array(COUNTS["counts"][COUNTS["genes"].index(gene)], dtype=float)


@pytest.mark.parametrize("gene", FITTED)
def test_the_fit_is_glm_nbs(gene):
    """Theta, the standard error and the p-value within 2e-5 of R's; measured at most 8e-6."""
    fit = _glm_nb.fit(_counts(gene), X)
    ref = R[gene]
    assert fit.theta == pytest.approx(ref["theta"], rel=2e-5, abs=0)
    assert fit.se[1] == pytest.approx(ref["se"], rel=2e-5, abs=0)
    # R's coefficient is Group2 over Group1; the group column here marks group 1.
    assert fit.coef[1] == pytest.approx(-ref["coef_group2"], rel=0, abs=1e-9)
    assert _glm_nb.wald_pvalue(_counts(gene), X) == pytest.approx(ref["p_val"], rel=2e-5, abs=0)


def test_a_separated_gene_has_no_evidence_in_either():
    """Every count in one group is zero, so the coefficient runs off towards infinity.

    glm.nb and truecell stop at different points on the way, so the numbers
    differ, but both give a p-value near 1.
    """
    assert R[SEPARATED]["p_val"] > 0.9
    assert _glm_nb.wald_pvalue(_counts(SEPARATED), X) > 0.9


def _object():
    counts = sp.csc_matrix(np.array(COUNTS["counts"], dtype=float))
    obj = create_truecell_object(counts=counts, feature_names=COUNTS["genes"],
                                 cell_names=[f"c{i}" for i in range(GROUP.size)])
    normalize_data(obj)
    obj.idents = np.where(GROUP == 1, "0", "1").tolist()
    obj.meta_data[COUNTS["latent_name"]] = COUNTS["latent"]
    return obj


def test_find_markers_reports_glm_nbs_p_values():
    res = find_markers(_object(), ident_1="0", ident_2="1", test_use="negbinom",
                       min_pct=0, logfc_threshold=0)
    for gene in FITTED:
        assert res.loc[gene, "p_val"] == pytest.approx(R[gene]["p_val"], rel=2e-5, abs=0), gene


@pytest.mark.parametrize("gene", COUNTS["latent_genes"])
def test_a_covariate_is_fitted_as_glm_nb_fits_it(gene):
    """GENE ~ group + log total count, as GLMDETest passes latent.vars.

    With only the group in the model, the fitted means are the two group means
    whatever theta or the weights are. A covariate ends that, so this is where
    theta's estimation, the negative binomial weights and the alternation between
    them are checked.

    One gene, given the covariate, varies no more than Poisson counts do. glm.nb
    stops its theta at the iteration limit and warns; truecell's runs on to its
    bound, which moves the p-value by 6e-4. Such a gene is held to that.
    """
    ref = R_LATENT[gene]
    design = np.column_stack([np.ones(GROUP.size), GROUP, COUNTS["latent"]])
    fit = _glm_nb.fit(_counts(gene), design)
    p = _glm_nb.wald_pvalue(_counts(gene), design)
    if ref["theta_warning"]:
        assert fit.theta > ref["theta"]
        assert p == pytest.approx(ref["p_val"], rel=2e-3, abs=0)
        return
    assert fit.theta == pytest.approx(ref["theta"], rel=2e-5, abs=0)
    assert fit.se[1] == pytest.approx(ref["se"], rel=2e-5, abs=0)
    # R's IRLS stops at glm.control's 1e-8 change in deviance, so 6e-8 apart here.
    assert fit.coef[1] == pytest.approx(-ref["coef_group2"], rel=0, abs=1e-6)
    assert p == pytest.approx(ref["p_val"], rel=2e-5, abs=0)


def test_find_markers_passes_latent_vars_to_the_fit():
    res = find_markers(_object(), ident_1="0", ident_2="1", test_use="negbinom",
                       latent_vars=[COUNTS["latent_name"]], min_pct=0, logfc_threshold=0)
    for gene in COUNTS["latent_genes"]:
        ref = R_LATENT[gene]
        rel = 2e-3 if ref["theta_warning"] else 2e-5
        assert res.loc[gene, "p_val"] == pytest.approx(ref["p_val"], rel=rel, abs=0), gene


def test_negbinom_does_not_run_statsmodels_negative_binomial(monkeypatch):
    """Its BFGS fit is what made these genes move with the statsmodels version."""
    import statsmodels.discrete.discrete_model as dm

    from truecell.markers import _negbinom_pvalue

    def refuse(*args, **kwargs):
        raise AssertionError("statsmodels' NegativeBinomial.fit was called")

    monkeypatch.setattr(dm.NegativeBinomial, "fit", refuse)
    p = _negbinom_pvalue(_counts("EIF2AK4"), GROUP, None)
    assert p == pytest.approx(R["EIF2AK4"]["p_val"], rel=2e-5, abs=0)
