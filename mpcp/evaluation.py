"""Evaluation metrics for posterior intensity samples (Section 5 and Appendix C.5)."""
import numpy as np
from scipy.special import logsumexp


def integrate_on_grid(intensities, grid):
    """Integrate intensities over a regular grid with the trapezoidal rule.

    Args:
        intensities: array (..., n_points), e.g. (B, n_points) posterior samples.
        grid: (n_points, d) points of a regular grid, in any order.

    Returns:
        array (...) of integrals.
    """
    intensities = np.asarray(intensities)
    grid = np.asarray(grid).reshape(len(grid), -1)
    n_points, d = grid.shape
    if intensities.shape[-1] != n_points:
        raise ValueError(f"intensities has {intensities.shape[-1]} points but the grid has {n_points}.")

    axes = [np.unique(grid[:, k]) for k in range(d)]
    if np.prod([len(a) for a in axes]) != n_points:
        raise ValueError("The grid is not a regular grid.")

    # Sort so that the first coordinate varies fastest, then integrate over
    # the first coordinate, the second, ...
    order = np.lexsort(grid.T)
    vals = np.take(intensities, order, axis=-1)
    vals = vals.reshape(intensities.shape[:-1] + tuple(len(a) for a in reversed(axes)))
    for k in range(d):
        vals = np.trapz(vals, x=axes[k], axis=-1)
    return vals


def test_log_likelihood(integral, log_intensities_test):
    """Monte Carlo estimate of the test log-likelihood log p(x_test | x_{1:n}).

    Args:
        integral: (B,) samples of the integral of lambda over the domain.
        log_intensities_test: (B, n_test) samples of log lambda at the test points.

    Returns:
        log of the posterior mean of exp(-integral) prod_i lambda(x_i).
    """
    log_likelihoods = -integral + np.sum(log_intensities_test, axis=1)
    B = log_likelihoods.shape[0]
    return logsumexp(log_likelihoods) - np.log(B)


def l2_error(intensities, true_intensity):
    """Squared error of the posterior mean, averaged over grid points.

    Args:
        intensities: (n_seeds, B, n_points) posterior samples for each dataset.
        true_intensity: (n_points,) true intensity on the same grid.

    Returns:
        (n_seeds,) L2 error per dataset.
    """
    means = np.mean(intensities, axis=1)
    return np.mean((means - true_intensity) ** 2, axis=1)


def posterior_integrals(intensities, grid):
    """Posterior samples of gamma = int lambda, (n_seeds, B), from intensities (n_seeds, B, n_points)."""
    return np.stack([integrate_on_grid(samples, grid) for samples in intensities])


def interval_coverage(integral_samples, true_integral, q=0.9):
    """Coverage and length of central q-level credible intervals for gamma = int lambda.

    Args:
        integral_samples: (n_seeds, B) posterior samples of gamma for each dataset
            (see ``posterior_integrals``).
        true_integral: the true value of gamma.
        q: credible level.

    Returns:
        (fraction of datasets whose interval covers the true value,
         (n_seeds,) interval lengths).
    """
    alpha = 1 - q
    lower = np.quantile(integral_samples, alpha / 2, axis=1)
    upper = np.quantile(integral_samples, 1 - alpha / 2, axis=1)

    coverage = (true_integral >= lower) & (true_integral <= upper)
    return coverage.mean(), upper - lower


def mean_and_se(values):
    """Mean and standard error of the mean."""
    values = np.asarray(values)
    return values.mean(), values.std() / np.sqrt(len(values))
