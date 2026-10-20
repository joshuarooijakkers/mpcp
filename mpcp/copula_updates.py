"""Recursive bivariate copula updates of the predictive density (eqs. 5 and 7).

All densities live on the unit cube [0, 1]^d and are initialised at the
uniform density p_0. ``make_copula_updates`` builds the jitted functions for a
given copula, for

* computing the overhead terms v_{1:n} from the observed data (fitting),
* the prequential log-likelihood and its gradient (bandwidth selection), and
* evaluating p_n at test points, averaged over data permutations.
"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
from jax import checkpoint, jit, value_and_grad, vmap
from jax.lax import scan
from jax.scipy.special import logsumexp

from mpcp.copulas import COPULAS


def log_alpha(i):
    """log of the update weight alpha_{i+1} = (2 - 1/(i+1)) / (i+2), for 0-based i."""
    return jnp.log(2. - (1 / (i + 1))) - jnp.log(i + 2)


def init_marginals_single(y_test):
    """Uniform initial density at one point: (log conditional CDFs, log joint pdf)."""
    d = jnp.shape(y_test)[0]

    logcdf_init_conditionals = jnp.log(y_test)       # log F(y) = log y
    logpdf_init_joints = jnp.sum(jnp.zeros(d))       # log 1 = 0

    return logcdf_init_conditionals, logpdf_init_joints


init_marginals = vmap(init_marginals_single, (0))


@lru_cache(maxsize=None)
def make_copula_updates(copula="frank"):
    """Build the (jitted) copula-update functions for the copula named ``copula``."""
    copula_fn, transform = COPULAS[copula]

    def update_copula_single(logcdf_conditionals, logpdf_joints, u, v, logalpha, theta):
        """Update the conditional CDFs and joint pdf at one point given one datum (eqs. 5, 7)."""
        d = jnp.shape(logcdf_conditionals)[0]

        logcop_distribution, logcop_dens = copula_fn(u, v, theta)

        # Products of copula densities over the first j covariates
        logcop_dens_prod = jnp.cumsum(logcop_dens)

        # Shifted by one covariate for the conditional CDFs
        logcop_dens_prod_staggered = jnp.concatenate((jnp.zeros(1), logcop_dens_prod[0:d - 1]))

        log1alpha = jnp.log1p(-jnp.exp(logalpha))

        logcdf_conditionals = jnp.logaddexp((log1alpha + logcdf_conditionals), (logalpha + logcop_dens_prod_staggered + logcop_distribution))\
            - jnp.logaddexp(log1alpha, (logalpha + logcop_dens_prod_staggered))

        logpdf_joints = jnp.logaddexp(log1alpha, (logalpha + logcop_dens_prod[-1])) + logpdf_joints

        return logcdf_conditionals, logpdf_joints

    update_copula = vmap(update_copula_single, (0, 0, 0, None, None, None))

    # --- Overhead terms v_{1:n} ---

    def update_pn(carry, i):
        """Store v_i (the conditional CDFs of datum i under p_{i-1}) and update all data points."""
        vn, logcdf_conditionals_yn, logpdf_joints_yn, theta = carry

        logalpha = log_alpha(i)

        u = jnp.exp(logcdf_conditionals_yn)
        v = jnp.exp(logcdf_conditionals_yn[i])

        vn = vn.at[i].set(v)

        logcdf_conditionals_yn, logpdf_joints_yn = update_copula(logcdf_conditionals_yn, logpdf_joints_yn, u, v, logalpha, theta)

        carry = vn, logcdf_conditionals_yn, logpdf_joints_yn, theta
        return carry, i

    def update_pn_loop(theta, y):
        """Compute v_{1:n} for one ordering of the data y (n, d)."""
        n = jnp.shape(y)[0]
        d = jnp.shape(y)[1]

        vn = jnp.zeros((n, d))

        logcdf_conditionals_yn, logpdf_joints_yn = init_marginals(y)

        carry = vn, logcdf_conditionals_yn, logpdf_joints_yn, theta
        rng = jnp.arange(n)
        carry, rng = scan(update_pn, carry, rng)

        return carry[0]

    # (theta, y_perm (n_perm, n, d)) -> v_{1:n} per permutation, (n_perm, n, d)
    update_pn_loop_perm = jit(vmap(update_pn_loop, (None, 0)))

    # --- Prequential log-likelihood, for selecting the copula parameters ---

    def update_pn_opt(carry, i):
        logcdf_conditionals_yn, logpdf_joints_yn, theta = carry
        logalpha = log_alpha(i)
        u = jnp.exp(logcdf_conditionals_yn)
        v = jnp.exp(logcdf_conditionals_yn[i])

        preq_loglik_i = logpdf_joints_yn[i]  # log p_{i-1}(y_i), before the update with y_i

        logcdf_conditionals_yn, logpdf_joints_yn = update_copula(
            logcdf_conditionals_yn, logpdf_joints_yn, u, v, logalpha, theta
        )
        return (logcdf_conditionals_yn, logpdf_joints_yn, theta), preq_loglik_i

    def update_pn_loop_opt(theta, y):
        n, _ = y.shape
        logcdf_conditionals_yn, logpdf_joints_yn = init_marginals(y)
        carry = (logcdf_conditionals_yn, logpdf_joints_yn, theta)

        _, preq_loglik = scan(
            checkpoint(update_pn_opt),  # recompute each step in the backward pass
            carry,
            jnp.arange(n)
        )
        return preq_loglik

    update_pn_loop_perm_opt = vmap(update_pn_loop_opt, (None, 0))

    def negpreq_jointloglik_perm(hyperparam, y_perm):
        """Negative prequential log-likelihood, averaged over data and permutations."""
        theta = transform(hyperparam)
        preq_loglik = jnp.mean(update_pn_loop_perm_opt(theta, y_perm), axis=0)
        return -jnp.mean(preq_loglik)

    fun_grad_jll_perm = jit(value_and_grad(negpreq_jointloglik_perm))

    def fun_grad_jll_perm_sp(hyperparam, z):
        """``fun_grad_jll_perm`` with NumPy outputs, for scipy.optimize.minimize."""
        value, grad = fun_grad_jll_perm(hyperparam, z)
        return (np.array(value), np.array(grad))

    # --- Evaluating p_n at test points ---

    def update_ptest_single(carry, i):
        """Update p(y) at one test point with the overhead term v_i."""
        vn, logcdf_conditionals_ytest, logpdf_joints_ytest, theta = carry

        logalpha = log_alpha(i)

        u = jnp.exp(logcdf_conditionals_ytest)
        v = vn[i]

        logcdf_conditionals_ytest, logpdf_joints_ytest = update_copula_single(logcdf_conditionals_ytest, logpdf_joints_ytest, u, v, logalpha, theta)

        carry = vn, logcdf_conditionals_ytest, logpdf_joints_ytest, theta
        return carry, i

    def update_ptest_single_scan(carry, rng):
        """Apply ``update_ptest_single`` for the indices in ``rng``."""
        return scan(update_ptest_single, carry, rng)

    def update_ptest_single_loop(vn, theta, y_test):
        """p_n at one test point for one data ordering."""
        n = jnp.shape(vn)[0]

        logcdf_conditionals_ytest, logpdf_joints_ytest = init_marginals_single(y_test)

        carry = vn, logcdf_conditionals_ytest, logpdf_joints_ytest, theta
        rng = jnp.arange(n)
        carry, rng = update_ptest_single_scan(carry, rng)
        vn, logcdf_conditionals_ytest, logpdf_joints_ytest, theta = carry

        return logcdf_conditionals_ytest, logpdf_joints_ytest

    def update_ptest_single_loop_perm_av(vn_perm, theta, y_test):
        """p_n at one test point, averaged over the data permutations."""
        n_perm = jnp.shape(vn_perm)[0]

        update_ptest_single_loop_perm = vmap(update_ptest_single_loop, (0, None, None))
        logcdf_conditionals, logpdf_joints = update_ptest_single_loop_perm(vn_perm, theta, y_test)

        logcdf_conditionals = logsumexp(logcdf_conditionals, axis=0) - jnp.log(n_perm)
        logpdf_joints = logsumexp(logpdf_joints, axis=0) - jnp.log(n_perm)
        return logcdf_conditionals, logpdf_joints

    # (vn_perm, theta, y_test (n_test, d)) -> permutation-averaged (log CDFs, log pdf)
    update_ptest_loop_perm_av = jit(vmap(update_ptest_single_loop_perm_av, (None, None, 0)))

    return SimpleNamespace(
        transform=transform,
        update_copula=update_copula,
        update_pn_loop_perm=update_pn_loop_perm,
        fun_grad_jll_perm=fun_grad_jll_perm,
        fun_grad_jll_perm_sp=fun_grad_jll_perm_sp,
        update_ptest_single_scan=update_ptest_single_scan,
        update_ptest_loop_perm_av=update_ptest_loop_perm_av,
    )
