"""Seurat's DESeq2 test, ``FindMarkers(test.use = "DESeq2")``, run on pydeseq2.

Seurat's ``DESeq2DETest`` calls ``estimateSizeFactors``, then
``estimateDispersions(fitType = "local")``, then ``nbinomWaldTest``, and reads
``results(alpha = 0.05)$pvalue``. pydeseq2 fits the same model. It differs in
four ways that change the answer, and this module closes all four:

* **The dispersion trend.** pydeseq2 ships only the parametric and mean trends.
  Seurat asks for DESeq2's local one. The local trend is implemented below, and
  written into the slot pydeseq2's later steps read.
* **Gene-wise dispersions at the lower bound.** When a gene's likelihood barely
  changes as its dispersion falls towards zero, pydeseq2's L-BFGS-B stops near
  1e-5, while DESeq2 reports a value below 1e-6. Only genes above 1e-6 enter the
  trend fit. On the CD14 monocytes of Seurat's ifnb pseudobulk vignette, 192
  genes entered pydeseq2's fit that DESeq2 left out. They pulled the trend 0.5 log
  units low at the median, and truecell called 1,429 genes to Seurat's 1,387.
  Where the adjusted likelihood at the minimum dispersion is at least as high as
  at L-BFGS-B's estimate, this uses the minimum.
* **Cook's outliers.** pydeseq2's ``deseq2()`` replaces outlying counts and
  refits by default. ``nbinomWaldTest`` does not, so neither does this. The
  Cook's cutoff that sets a p-value to NA is applied as ``results()`` applies it.
* **The Wald standard error.** DESeq2 floors each fitted mean at 0.5 (its
  ``minmu``) before building the weights the standard error comes from.
  pydeseq2's Wald test recomputes the means from the fold changes without the
  floor, which inflates the standard error of every low-count gene. On PBMC 3k,
  per cell, the floor takes the genes called at Bonferroni 0.05 from 652 to
  Seurat's 712, the same 712.

It keeps DESeq2's refusal too. Size factors are medians of ratios to each
gene's geometric mean, which does not exist for a gene with a zero. When every
gene has one, ``estimateSizeFactors`` stops, and this raises rather than falling
back to a different estimator the way pydeseq2 does.

The local trend is DESeq2's ``localDispersionFit``: a local regression of log
gene-wise dispersion on log mean. It is implemented from the method as published
(Cleveland & Devlin 1988; Loader 1999): a local quadratic, a tricube kernel, a
nearest-neighbour span of 0.7, and the gene means as prior weights. It is
evaluated directly at every gene. DESeq2 fits it with locfit, whose default
evaluation interpolates between fitted vertices, so the two differ by locfit's
interpolation error, 6e-3 in log dispersion on the reference fixture. locfit is
GPL-licensed, and nothing here derives from its source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: DESeq2's ``minDisp``.
MIN_DISP = 1e-8
#: Nearest-neighbour span of the local trend, as a fraction of the genes it is fitted on.
SPAN = 0.7


def local_dispersion_trend(base_mean, genewise_dispersion, eval_mean,
                           min_disp: float = MIN_DISP) -> np.ndarray:
    """DESeq2's local dispersion trend, evaluated at ``eval_mean``.

    Fitted, as ``estimateDispersionsFit`` fits it, on the genes whose gene-wise
    dispersion exceeds ``100 * min_disp``. At an evaluation point x on the log-mean
    axis, the span h is the distance to the ``floor(0.7 n)``-th nearest of those n
    genes. Each gene is weighted by its mean times the tricube of its distance
    over h, and the trend at x is the intercept of the weighted quadratic centred
    on x.
    """
    base_mean = np.asarray(base_mean, dtype=float)
    disp = np.asarray(genewise_dispersion, dtype=float)
    use = disp > 100 * min_disp
    if not use.any():
        raise ValueError(
            "Every gene-wise dispersion estimate is within two orders of magnitude of "
            "the minimum, so no dispersion trend can be fitted; DESeq2's "
            "estimateDispersionsFit stops here too."
        )
    x = np.log(base_mean[use])
    y = np.log(disp[use])
    prior = base_mean[use]
    n = x.size
    k = max(int(np.floor(SPAN * n)), 1)

    at = np.log(np.asarray(eval_mean, dtype=float)).ravel()
    fitted = np.empty(at.size)
    for j, x0 in enumerate(at):
        dx = x - x0
        dist = np.abs(dx)
        h = np.partition(dist, k - 1)[k - 1]
        if h > 0:
            u = dist / h
            kernel = np.where(u < 1, (1 - u ** 3) ** 3, 0.0)
        else:
            kernel = (dist == 0).astype(float)
        root = np.sqrt(prior * kernel)
        design = np.column_stack([np.ones(n), dx, dx * dx])
        coef = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        fitted[j] = coef[0]
    return np.exp(fitted).reshape(np.shape(eval_mean))


def cox_reid_loss(dds, dispersion: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """The loss pydeseq2 minimises for a gene-wise dispersion, one value per gene.

    The negative binomial negative log-likelihood of each gene's counts at its
    fitted means, plus half the log determinant of the Cox-Reid information, at
    ``dispersion[g]`` for gene g. Genes are taken ``chunk`` at a time, so a
    per-cell matrix is never expanded whole.
    """
    from scipy.special import gammaln

    design = dds.obsm["design_matrix"].to_numpy(dtype=float)
    mu_hat = dds.layers["_mu_hat"]
    loss = np.full(dds.n_vars, np.nan)
    for start in range(0, dds.n_vars, chunk):
        cols = slice(start, start + chunk)
        y = np.asarray(dds.X[:, cols], dtype=float)
        mu = mu_hat[:, cols]
        alpha = np.asarray(dispersion, dtype=float)[cols]
        size = 1.0 / alpha
        log_binom = gammaln(y + size) - gammaln(y + 1) - gammaln(size)
        with np.errstate(invalid="ignore", divide="ignore"):
            nll = dds.n_obs * size * np.log(alpha) + (
                -log_binom + (y + size) * np.log(size + mu) - y * np.log(mu)).sum(axis=0)
            weights = mu / (1 + mu * alpha)
            information = np.einsum("si,sg,sj->gij", design, weights, design)
            loss[cols] = nll + 0.5 * np.linalg.slogdet(information)[1]
    return loss


def fit(counts: pd.DataFrame, group1: np.ndarray):
    """Run ``DESeq2DETest``'s steps on pydeseq2, and return ``(dds, stats)``.

    ``counts`` is samples × genes, integer. ``group1`` marks the samples in group
    1, and every other sample is group 2. ``stats.p_values`` holds the Wald
    p-values, NaN where ``results()``' Cook's cutoff removes them, as R gives NA.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    values = counts.to_numpy()
    if (values == 0).any(axis=0).all():
        raise ValueError(
            "Every gene contains at least one zero, so DESeq2 cannot compute size "
            "factors (estimateSizeFactors stops with the same complaint). Test fewer, "
            "larger groups of cells, or aggregate them with sample_col."
        )
    group1 = np.asarray(group1, dtype=bool)
    metadata = pd.DataFrame(
        {"condition": np.where(group1, "group1", "group2")}, index=counts.index)
    # max_disp is DESeq2's maxDisp, max(10, samples). pydeseq2 enforces that maximum
    # itself; passing it keeps the rule from depending on that.
    dds = DeseqDataSet(
        counts=counts, metadata=metadata, design="~condition", refit_cooks=False,
        max_disp=max(10.0, float(counts.shape[0])), min_disp=MIN_DISP, quiet=True,
    )
    dds.fit_size_factors(fit_type="ratio")
    dds.fit_genewise_dispersions()

    # L-BFGS-B stops short of the minimum on a flat likelihood; see the module docstring.
    non_zero = dds.var["non_zero"].to_numpy()
    genewise = dds.var["genewise_dispersions"].to_numpy(dtype=float).copy()
    minimum = np.full_like(genewise, MIN_DISP)
    at_minimum = non_zero & (genewise > MIN_DISP) & (
        cox_reid_loss(dds, minimum) <= cox_reid_loss(dds, genewise))
    genewise[at_minimum] = MIN_DISP
    dds.var["genewise_dispersions"] = genewise

    means = dds.var["_normed_means"].to_numpy()[non_zero]
    trend = np.full(dds.n_vars, np.nan)
    trend[non_zero] = local_dispersion_trend(
        means, dds.var["genewise_dispersions"].to_numpy()[non_zero], means)
    dds.var["fitted_dispersions"] = trend
    dds.uns["disp_function_type"] = "local"

    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()

    stats = DeseqStats(dds, contrast=["condition", "group1", "group2"], alpha=0.05,
                       quiet=True)
    # nbinomWaldTest floors the fitted means at minmu before weighting them;
    # DeseqStats.run_wald_test does not, so run its Wald test with the floor.
    design = stats.design_matrix.to_numpy(dtype=float)
    lfc = stats.LFC.to_numpy(dtype=float)
    mu = np.exp(design @ lfc.T) * dds.obs["size_factors"].to_numpy(dtype=float)[:, None]
    pvalues, statistics, se = stats.inference.wald_test(
        design_matrix=design, disp=dds.var["dispersions"].to_numpy(dtype=float), lfc=lfc,
        mu=np.maximum(mu, dds.min_mu), ridge_factor=np.diag(np.repeat(1e-6, design.shape[1])),
        contrast=stats.contrast_vector, lfc_null=0.0, alt_hypothesis=None,
    )
    stats.p_values = pd.Series(np.asarray(pvalues, dtype=float), index=dds.var_names)
    stats.statistics = pd.Series(np.asarray(statistics, dtype=float), index=dds.var_names)
    stats.SE = pd.Series(np.asarray(se, dtype=float), index=dds.var_names)
    stats._cooks_filtering()
    return dds, stats


def wald_pvalues(counts: pd.DataFrame, group1: np.ndarray) -> pd.Series:
    """Wald p-values for group 1 against group 2, as ``DESeq2DETest`` computes them."""
    _, stats = fit(counts, group1)
    return pd.Series(stats.p_values.to_numpy(dtype=float), index=counts.columns,
                     name="pvalue")
