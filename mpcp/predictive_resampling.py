"""Predictive resampling (Algorithm 1) of the density component.

Under predictive resampling the conditional CDFs v of the imputed points
X_{n+1:N} are i.i.d. U[0, 1] (Appendix A.2), so a posterior draw of p_N only
requires uniform random numbers and the copula update.
"""
from functools import lru_cache
from types import SimpleNamespace

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.lax import fori_loop

from mpcp.copula_updates import log_alpha, make_copula_updates


@lru_cache(maxsize=None)
def make_predictive_resampling(copula="frank"):
    """Build the (jitted) predictive-resampling functions for the copula named ``copula``."""
    updates = make_copula_updates(copula)
    update_copula = updates.update_copula
    update_ptest_single_scan = updates.update_ptest_single_scan

    def predictive_resample_single_loop(key, logcdf_conditionals, logpdf_joints, rho, n, T):
        """One forward-sampling chain of T steps at one test point; returns log p_N there."""
        d = jnp.shape(logcdf_conditionals)[0]

        key, subkey = random.split(key)
        a_rand = random.uniform(subkey, shape=(T, d))

        # Place the uniforms after n empty rows so the update index runs over n, ..., n + T - 1
        vT = jnp.concatenate((jnp.zeros((n, d)), a_rand), axis=0)

        inputs = vT, logcdf_conditionals, logpdf_joints, rho
        rng = jnp.arange(n, n + T)
        outputs, rng = update_ptest_single_scan(inputs, rng)
        vT, logcdf_conditionals, logpdf_joints, rho = outputs

        return logpdf_joints

    # vmap over test points, then over the B posterior samples (one key each)
    predictive_resample_loop = vmap(predictive_resample_single_loop, (None, 0, 0, None, None, None))
    predictive_resample_loop_B = jit(vmap(predictive_resample_loop, (0, None, None, None, None, None)), static_argnums=(4, 5))

    def pr_1step_conv(i, inputs):
        """One forward-sampling step that also records the change in p(y) (step t = n + i)."""
        logcdf_conditionals, logpdf_joints, logpdf_joints_init, pdiff, rho, n, a_rand = inputs

        logalpha = log_alpha(n + i)

        u = jnp.exp(logcdf_conditionals)
        v = a_rand[i]

        logcdf_conditionals, logpdf_joints = update_copula(logcdf_conditionals, logpdf_joints, u, v, logalpha, rho)

        # Mean absolute difference from the initial joint density over the test points
        pdiff = pdiff.at[i].set(jnp.mean(jnp.abs(jnp.exp(logpdf_joints) - jnp.exp(logpdf_joints_init))))

        return logcdf_conditionals, logpdf_joints, logpdf_joints_init, pdiff, rho, n, a_rand

    def pr_loop_conv(key, logcdf_conditionals, logpdf_joints, rho, n, T):
        """One forward-sampling chain over all test points; returns the (T,) distance trajectory."""
        d = jnp.shape(logcdf_conditionals)[1]

        key, subkey = random.split(key)
        a_rand = random.uniform(subkey, shape=(T, d))

        inputs = logcdf_conditionals, logpdf_joints, logpdf_joints, jnp.zeros(T), rho, n, a_rand
        outputs = fori_loop(0, T, pr_1step_conv, inputs)
        return outputs[3]

    pr_loop_conv_B = jit(vmap(pr_loop_conv, (0, None, None, None, None, None)), static_argnums=(4, 5))

    return SimpleNamespace(
        predictive_resample_loop_B=predictive_resample_loop_B,
        pr_loop_conv_B=pr_loop_conv_B,
    )
