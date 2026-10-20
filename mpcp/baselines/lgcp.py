"""VLGCP and LGCP baselines: log-Gaussian Cox processes on binned counts.

Both discretise the unit cube into a regular grid of bins and model the bin
counts as Poisson with log-intensity given by a GP with a squared exponential
kernel and constant mean (Moller et al., 1998). VLGCP fits a variational GP
(gpflow ``VGP``); LGCP samples the GP and its hyperparameters with HMC
(``fit_lgcp``).
"""
import time

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def bin_counts(events, bins_per_dim):
    """Counts of ``events`` (n, d) on the unit cube in a regular grid of bins.

    Returns:
        (bin centres (M, d), counts (M, 1), volume of one bin), with
        M = bins_per_dim ** d. In 2D the first coordinate varies fastest
        (as ``np.meshgrid``'s default), otherwise the last; the order does not
        change the model, but it is kept as in the paper runs for identical
        floating-point results.
    """
    d = events.shape[1]
    edges = [np.linspace(0.0, 1.0, bins_per_dim + 1) for _ in range(d)]
    centers = [(e[:-1] + e[1:]) / 2.0 for e in edges]
    bin_volume = np.prod([e[1] - e[0] for e in edges])

    counts, _ = np.histogramdd(events, bins=edges)
    if d == 2:
        grids = np.meshgrid(*centers, indexing="xy")
        counts = counts.T
    else:
        grids = np.meshgrid(*centers, indexing="ij")
    X = np.vstack([g.ravel() for g in grids]).T
    Y = counts.ravel().reshape(-1, 1).astype(np.float64)
    return X, Y, bin_volume


def _binned_model(events, bins_per_dim, lengthscales):
    X, Y, bin_volume = bin_counts(events, bins_per_dim)
    n = events.shape[0]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
    mean_function = gpflow.mean_functions.Constant(np.log(n / X.shape[0]))
    likelihood = gpflow.likelihoods.Poisson(invlink=tf.exp, binsize=bin_volume)
    return (X, Y), kernel, likelihood, mean_function


def _minimize(model, maxiter=None):
    """L-BFGS on the training loss after an untimed one-iteration warm-up.

    The warm-up (which also compiles the loss) takes one optimiser step and is
    not included in the returned time.
    """
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, variables=model.trainable_variables,
                 method="L-BFGS-B", options=dict(maxiter=1))

    options = {} if maxiter is None else dict(options=dict(maxiter=maxiter))
    start = time.perf_counter()
    opt.minimize(model.training_loss, variables=model.trainable_variables,
                 method="L-BFGS-B", **options)
    return time.perf_counter() - start


def fit_vlgcp(events, bins_per_dim, lengthscales):
    """Fit a variational LGCP to ``events`` (n, d) on the unit cube.

    Returns:
        (fitted ``gpflow.models.VGP``, optimisation time in seconds).
    """
    data, kernel, likelihood, mean_function = _binned_model(events, bins_per_dim, lengthscales)
    model = gpflow.models.VGP(data=data, kernel=kernel, likelihood=likelihood, mean_function=mean_function)
    return model, _minimize(model)


def fit_lgcp(events, bins_per_dim, lengthscales, map_maxiter=None, num_samples=1000,
             num_burnin_steps=300, num_leapfrog_steps=10, step_size=0.01,
             num_adaptation_steps=10, target_accept_prob=0.75, adaptation_rate=0.1):
    """Fit an LGCP to ``events`` (n, d) on the unit cube with HMC (gpflow ``GPMC``).

    Gamma(2, 0.5) priors are placed on the kernel lengthscales and variance
    and a Normal(log(n / M), 2) prior on the constant mean. HMC is started at
    the MAP estimate.

    Returns:
        (model, sampling helper, ``run_chain`` -- a function that runs the HMC
        chain and returns the samples of each parameter --, MAP optimisation
        time in seconds). The chain is run by the caller so that
        it can be timed together with prediction.
    """
    data, kernel, likelihood, mean_function = _binned_model(events, bins_per_dim, lengthscales)
    n = events.shape[0]
    M = data[0].shape[0]

    kernel.lengthscales.prior = tfp.distributions.Gamma(
        concentration=np.float64(2.0), rate=np.float64(0.5)
    )
    kernel.variance.prior = tfp.distributions.Gamma(
        concentration=np.float64(2.0), rate=np.float64(0.5)
    )
    mean_function.c.prior = tfp.distributions.Normal(
        loc=np.float64(np.log(n / M)), scale=np.float64(2.0)
    )

    model = gpflow.models.GPMC(data=data, kernel=kernel, likelihood=likelihood, mean_function=mean_function)
    map_time = _minimize(model, map_maxiter)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=target_accept_prob, adaptation_rate=adaptation_rate
    )

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=None,
        )

    return model, hmc_helper, run_chain_fn, map_time


def lgcp_f_samples(model, hmc_helper, samples, x):
    """One draw of f at ``x`` (unit cube) for each HMC sample, shape (num_samples, len(x), 1).

    Leaves the model parameters at the last HMC sample.
    """
    f_samples = []
    for i in range(len(samples[0])):
        for var, var_samples in zip(hmc_helper.current_state, samples):
            var.assign(var_samples[i])
        f_samples.append(model.predict_f_samples(x, num_samples=1))
    return tf.concat(f_samples, axis=0)
