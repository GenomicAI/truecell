"""Seurat's ``negbinom`` test: a negative binomial GLM and the Wald test on it.

Seurat's ``GLMDETest`` fits ``MASS::glm.nb(GENE ~ group)`` and reads the group
coefficient's p-value off ``summary()``. Besides the coefficients the model has
one more unknown, the negative binomial's theta, and ``glm.nb`` estimates both
by maximum likelihood, alternating between them. So does this:

* For a fixed theta, the coefficients come from iteratively reweighted least
  squares with a log link, stopped as ``glm.control`` stops it: at
  ``|dev - dev_old| / (|dev| + 0.1) < 1e-8``, or after 25 iterations. The first
  fit is Poisson, started at ``mu = y + 0.1`` as the ``poisson`` family starts.
* For fixed means, theta maximises the negative binomial log-likelihood, by
  Newton's method in log theta with step halving, within [1e-8, 1e8].
* The two alternate until neither the log-likelihood nor theta moves, at most 25
  times. The Wald statistic divides each coefficient by its standard error from
  the inverse information at the fit, with dispersion 1, and the p-value is
  two-sided normal.

It replaces ``statsmodels``' ``NegativeBinomial.fit``. That BFGS fit collapsed
theta, or stopped unconverged, on a few dozen of PBMC 3k's genes, and which ones
moved with the numpy, scipy and statsmodels versions. Between statsmodels 0.14.6
and 0.15.0, 39 genes' p-values moved by more than 2 %, 8 of them by more than a
decade, and the DE tutorial's top 50 fell to 49 in one of the two. This fit gives
the same bits in both, and on the genes they disagreed on it matches R's
``glm.nb`` to six significant figures.

Where a gene's counts vary no more than Poisson counts do, theta runs out to its
bound. ``glm.nb`` stops short of it at its own iteration limit, and warns; that
moves such a gene's p-value by up to 1 %.

Written from the model and ``glm.control``'s documented defaults; nothing here
derives from MASS's source.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import norm

#: ``glm.control``'s defaults, which ``glm.nb`` also uses to stop alternating.
EPSILON = 1e-8
MAXIT = 25
_LOG_THETA = (float(np.log(1e-8)), float(np.log(1e8)))


class NegBinFit(NamedTuple):
    coef: np.ndarray
    se: np.ndarray
    theta: float
    loglik: float


def _loglik(theta: float, y: np.ndarray, mu: np.ndarray, lgamma_y1: np.ndarray) -> float:
    return float(np.sum(gammaln(y + theta) - gammaln(theta) - lgamma_y1
                        + theta * np.log(theta / (theta + mu)) + y * np.log(mu / (theta + mu))))


def _theta(y: np.ndarray, mu: np.ndarray, theta: float, lgamma_y1: np.ndarray) -> float:
    """ML theta for fixed means: Newton in log theta, halving any step that does not descend."""
    t = float(np.clip(np.log(theta), *_LOG_THETA))
    f = -_loglik(np.exp(t), y, mu, lgamma_y1)
    for _ in range(50):
        th = np.exp(t)
        a = y + th
        score = np.sum(digamma(a) - digamma(th) + np.log(th) + 1 - np.log(th + mu) - a / (th + mu))
        curvature = np.sum(polygamma(1, a) - polygamma(1, th) + 1 / th - 2 / (th + mu)
                           + a / (th + mu) ** 2)
        grad = -th * score                        # of -loglik, with respect to log theta
        hess = -(th * th * curvature + th * score)
        step = float(np.clip(-grad / hess if hess > 0 else -np.sign(grad), -5, 5))
        for _ in range(30):
            t_new = float(np.clip(t + step, *_LOG_THETA))
            f_new = -_loglik(np.exp(t_new), y, mu, lgamma_y1) if t_new != t else np.inf
            if f_new <= f:
                break
            step /= 2
        else:
            break                                 # no descent left at this precision
        converged = abs(t_new - t) < 1e-9
        t, f = t_new, f_new
        if converged:
            break
    return float(np.exp(t))


def _deviance(y: np.ndarray, mu: np.ndarray, theta: float | None, positive: np.ndarray) -> float:
    ylogy = np.zeros_like(y)
    ylogy[positive] = y[positive] * np.log(y[positive] / mu[positive])
    if theta is None:
        return 2 * float(np.sum(ylogy - (y - mu)))
    return 2 * float(np.sum(ylogy - (y + theta) * np.log((y + theta) / (mu + theta))))


def _irls(y: np.ndarray, X: np.ndarray, theta: float | None, coef: np.ndarray | None):
    """Coefficients for a fixed theta, or for the Poisson model when theta is None."""
    if coef is None:
        mu = y + 0.1
        eta = np.log(mu)
    else:
        eta = X @ coef
        mu = np.exp(eta)
    positive = y > 0
    dev_old = np.inf
    for _ in range(MAXIT):
        w = mu if theta is None else mu / (1 + mu / theta)
        z = eta + (y - mu) / mu
        xtw = X.T * w
        coef = np.linalg.solve(xtw @ X, xtw @ z)
        eta = X @ coef
        mu = np.exp(eta)
        dev = _deviance(y, mu, theta, positive)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < EPSILON:
            break
        dev_old = dev
    return coef, mu


def fit(y, X) -> NegBinFit:
    """Fit ``y ~ X`` as ``glm.nb`` does: counts ``y``, design ``X`` with its intercept column."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    lgamma_y1 = gammaln(y + 1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        coef, mu = _irls(y, X, None, None)
        theta = _theta(y, mu, 1.0, lgamma_y1)
        loglik = _loglik(theta, y, mu, lgamma_y1)
        for _ in range(MAXIT):
            coef, mu = _irls(y, X, theta, coef)
            theta_new = _theta(y, mu, theta, lgamma_y1)
            loglik_new = _loglik(theta_new, y, mu, lgamma_y1)
            converged = (abs(loglik_new - loglik) <= 1e-10 * (abs(loglik_new) + 1)
                         and abs(np.log(theta_new / theta)) <= 1e-8)
            theta, loglik = theta_new, loglik_new
            if converged:
                break
        w = mu / (1 + mu / theta)
        cov = np.linalg.inv((X.T * w) @ X)
    return NegBinFit(coef=coef, se=np.sqrt(np.diag(cov)), theta=theta, loglik=loglik)


def wald_pvalue(y, X, coef: int = 1) -> float:
    """Two-sided Wald p-value for ``X[:, coef]``, as ``summary(glm.nb(...))$coef[coef + 1, 4]``."""
    result = fit(y, X)
    return float(2 * norm.sf(abs(result.coef[coef] / result.se[coef])))
