"""VBPP baseline (Lloyd et al., 2015): a Cox process with a squared GP link.

``VBPPWithSampling`` extends ``vbpp.model.VBPP`` from https://github.com/st--/vbpp
(v0.1.1, commit 26c2a71; Apache License 2.0, Copyright (C) 2022 ST John and
Secondmind Ltd 2017-2020). ``elbo`` below is upstream's ``VBPP.elbo`` with one
added line.
"""
import time

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.config import default_float
from vbpp.model import VBPP


class VBPPWithSampling(VBPP):
    """``vbpp.model.VBPP`` (st--/vbpp, commit 26c2a71) with two changes.

    1. ``elbo`` adds a 1e-6 jitter to Kuu before it is used in the integral,
       data and KL terms (otherwise identical to upstream).
    2. New method ``predict_lambda_samples`` draws posterior samples of the
       intensity lambda(x) = f(x)^2.
    """

    def elbo(self, events):
        """Evidence Lower Bound (ELBo) for the log likelihood."""
        Kuu = gpflow.covariances.Kuu(self.inducing_variable, self.kernel)
        jitter = 1e-6 * tf.eye(self.inducing_variable.num_inducing, dtype=default_float())
        K = Kuu + jitter

        integral_term = self._elbo_integral_term(K)
        data_term = self._elbo_data_term(events, K)
        kl_div = self.prior_kl(K)

        elbo = self.num_observations * integral_term + data_term - kl_div
        return elbo

    def predict_lambda_samples(self, Xnew, n_samples):
        """Draw samples from the posterior of the intensity lambda(x) = f(x)^2.

        :param Xnew: N x D array of input locations
        :param n_samples: number of samples to draw
        :return: (n_samples, N) tensor of intensity samples
        """
        mean_f, cov_f = self.predict_f(Xnew, full_cov=True)
        mean_f = tf.reshape(mean_f, [-1])  # (N,)

        # predict_f with full_cov=True returns cov shaped (1, N, N) for
        # this single-output model -- squeeze the leading dim if present
        if cov_f.shape.ndims == 3:
            cov_f = cov_f[0]

        N = tf.shape(mean_f)[0]
        jitter = 1e-6 * tf.eye(N, dtype=default_float())
        L = tf.linalg.cholesky(cov_f + jitter)

        eps = tf.random.normal((n_samples, N), dtype=default_float())
        f_samples = mean_f[None, :] + tf.matmul(eps, L, transpose_b=True)  # (n_samples, N)

        lambda_samples = tf.square(f_samples)
        return lambda_samples


def fit_vbpp(events, domain, inducing_points, lengthscales):
    """Fit VBPP to ``events`` (n, d) in the box ``domain`` (d, 2).

    Uses fixed inducing points, a whitened q(u) initialised at N(0, I), a
    squared exponential kernel and beta0 = sqrt(n / |domain|), and maximises
    the ELBO with L-BFGS.

    Returns:
        (fitted model, optimisation time in seconds, excluding a first warm-up
        evaluation of the ELBO).
    """
    domain = np.asarray(domain)
    n = events.shape[0]
    M = inducing_points.shape[0]

    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
    feature = gpflow.inducing_variables.InducingPoints(inducing_points)
    gpflow.set_trainable(feature, False)

    q_mu = np.zeros(M)
    q_S = np.eye(M)
    beta0 = np.sqrt(n / np.prod(domain[:, 1] - domain[:, 0]))

    model = VBPPWithSampling(
        feature, kernel, domain, q_mu, q_S, beta0=beta0, num_events=n, whiten=True
    )

    def objective_closure():
        return -model.elbo(events)

    _ = objective_closure()  # warm-up, not timed

    start = time.perf_counter()
    gpflow.optimizers.Scipy().minimize(objective_closure, model.trainable_variables)
    return model, time.perf_counter() - start
