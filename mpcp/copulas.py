"""Bivariate copulas used in the predictive update.

Each copula is a function ``(u, v, theta) -> (log h-function, log density)``,
where the h-function is H(u, v) = dC(u, v)/dv (the conditional CDF) and the
density is c(u, v). Each copula also has a ``transform`` that maps the
unconstrained hyperparameter optimised by L-BFGS onto the copula parameter.
"""
import jax.numpy as jnp
from jax.nn import sigmoid, softplus
from jax.scipy.special import ndtri
from jax.scipy.stats import norm


def frank_copula(u, v, theta):
    """Frank copula with parameter theta > 0 (the default in MPCP, see Section 4.2)."""
    eps = 1e-6
    tiny = 1e-30

    u = jnp.clip(u, eps, 1.0 - eps)
    v = jnp.clip(v, eps, 1.0 - eps)

    # For theta > 0, A = 1 - exp(-theta u), B = 1 - exp(-theta v) and
    # D = 1 - exp(-theta) are positive.
    A = -jnp.expm1(-theta * u)
    B = -jnp.expm1(-theta * v)
    D = -jnp.expm1(-theta)

    E = jnp.exp(-theta)
    Ev = jnp.exp(-theta * v)

    # Stable version of den = D - A * B, avoiding cancellation near u, v ~ 1:
    # D - AB = D exp(-theta v) + B [exp(-theta u) - exp(-theta)], with
    # exp(-theta u) - exp(-theta) = exp(-theta) expm1(theta (1 - u)).
    delta_u = E * jnp.expm1(theta * (1.0 - u))

    den = D * Ev + B * delta_u
    den = jnp.maximum(den, tiny)

    log_den = jnp.log(den)

    # h-function: dC(u, v)/dv = exp(-theta v) A / den
    logcop_dist = -theta * v + jnp.log(A) - log_den

    logcop_dist = jnp.clip(
        logcop_dist,
        jnp.log(eps),
        jnp.log1p(-eps),
    )

    # Density: c(u, v) = theta D exp(-theta (u + v)) / den^2
    logcop_dens = (
        jnp.log(theta)
        + jnp.log(D)
        - theta * (u + v)
        - 2.0 * log_den
    )

    return logcop_dist, logcop_dens


def gaussian_copula(u, v, rho):
    """Gaussian copula with correlation rho in (0, 1) (Hahn et al., 2018; Fong et al., 2023)."""
    # Clip to avoid CDF values of exactly 0 or 1 (numerical stability).
    eps = 1e-6
    u = jnp.clip(u, eps, 1 - eps)
    v = jnp.clip(v, eps, 1 - eps)

    pu = ndtri(u)
    pv = ndtri(v)
    z = (pu - rho * pv) / jnp.sqrt(1 - rho**2)

    logcop_dist = norm.logcdf(z)
    logcop_dist = jnp.clip(logcop_dist, jnp.log(eps), jnp.log(1 - eps))
    logcop_dens = -0.5 * jnp.log(1 - rho**2) + (0.5 / (1 - rho**2)) * (-(rho**2) * (pu**2 + pv**2) + 2 * rho * pu * pv)

    return logcop_dist, logcop_dens


# name -> (copula, transform from the unconstrained hyperparameter to the copula
# parameter). MPCP starts the optimisation at hyperparameter 10, i.e. a
# strongly peaked update (theta ~ 10 for Frank, rho ~ 1 for Gaussian).
COPULAS = {
    "frank": (frank_copula, softplus),
    "gaussian": (gaussian_copula, sigmoid),
}
