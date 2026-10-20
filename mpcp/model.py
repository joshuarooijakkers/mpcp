"""The Martingale Posterior Cox Process (MPCP)."""
import time

import numpy as np
import jax.numpy as jnp
from jax import block_until_ready, vmap
from jax.random import PRNGKey, permutation, split
from scipy.optimize import minimize

from mpcp.copula_updates import make_copula_updates
from mpcp.predictive_resampling import make_predictive_resampling
from mpcp.utils import normalize


class MPCP:
    """Martingale Posterior Cox Process.

    The intensity is factorised as lambda = gamma * s (Section 4): the density s
    is learned with copula-based predictive updates on the unit cube and its
    posterior is obtained by predictive resampling, while the total count gamma
    has the conjugate posterior Gamma(n, 1) under the prior pi(gamma) ~ 1/gamma.

    Args:
        n_perm: number of random data permutations to average over (M in the paper).
        seed: seed for drawing the data permutations.
        copula: ``"frank"`` (default, used in the paper) or ``"gaussian"``.
        verbose: print timings.

    Typical use, for events ``x`` (n, d) in a box ``domain`` (d, 2)::

        model = MPCP().fit(normalize(x, domain))
        intensities = model.sample_intensity(x_grid, domain, B=1000, T=2000)

    The lower-level steps are ``predict`` (permutation-averaged p_n at test
    points) and ``predictive_resample`` (posterior draws of p_N at those points).
    """

    def __init__(self, n_perm=10, seed=20, copula="frank", verbose=False):
        self.n_perm = n_perm
        self.seed = seed
        self.copula = copula
        self.verbose = verbose

        self._updates = make_copula_updates(copula)
        self._resampling = make_predictive_resampling(copula)

        self.vn_perm = None
        self.theta_opt = None
        self.timings = {}

    def _log(self, message):
        if self.verbose:
            print(message)

    def fit(self, y, theta=None):
        """Fit to data ``y`` (n, d) on the unit cube [0, 1]^d.

        Selects one copula parameter per dimension by maximising the
        prequential log-likelihood (unless ``theta`` is given) and computes the overhead
        terms v_{1:n} for each permutation. Returns ``self``; the time spent
        (excluding JIT compilation) is stored in ``self.timings["fit"]``.
        """
        updates = self._updates

        key = PRNGKey(self.seed)
        key, *subkey = split(key, self.n_perm + 1)
        subkey = jnp.array(subkey)
        y_perm = vmap(permutation, (0, None))(subkey, y)

        d = jnp.shape(y)[1]

        if theta is None:
            hyperparam_init = 10 * jnp.ones(d)

            # JIT-compile before timing
            start = time.time()
            val = updates.fun_grad_jll_perm(hyperparam_init, y_perm)[0]
            block_until_ready((val, updates.update_pn_loop_perm(hyperparam_init, y_perm)))
            self._log('Optimization and fitting compilation time: {}s'.format(round(time.time() - start, 3)))

            start = time.time()
            opt = minimize(fun=updates.fun_grad_jll_perm_sp, x0=np.array(hyperparam_init),
                           args=(y_perm), jac=True, method='L-BFGS-B')
            opt_time = time.time() - start
            self._log('Optimization time: {}s'.format(round(opt_time, 3)))

            if not opt.success:
                print('Optimization failed:', opt.message)
            theta_opt = updates.transform(opt.x)
        else:
            opt_time = 0.0
            theta_opt = theta

        start = time.time()
        vn_perm = updates.update_pn_loop_perm(theta_opt, y_perm).block_until_ready()
        fit_time = time.time() - start
        self._log('Fit time: {}s'.format(round(fit_time, 3)))

        self.theta_opt = theta_opt
        self.vn_perm = vn_perm
        self.n = int(jnp.shape(y)[0])
        self.timings["fit"] = opt_time + fit_time
        return self

    def predict(self, y_test):
        """Permutation-averaged p_n at test points ``y_test`` (n_test, d) on the unit cube.

        Returns the log conditional CDFs (n_test, d) and log joint densities
        (n_test,). They are kept for ``predictive_resample``.
        """
        start = time.time()
        logcdf_conditionals, logpdf_joints = self._updates.update_ptest_loop_perm_av(
            self.vn_perm,
            self.theta_opt,
            y_test
        )
        logcdf_conditionals = logcdf_conditionals.block_until_ready()
        self.timings["predict"] = time.time() - start
        self._log('Prediction time: {}s'.format(round(self.timings["predict"], 3)))

        self.logcdf_conditionals = logcdf_conditionals
        self.logpdf_joints = logpdf_joints
        return logcdf_conditionals, logpdf_joints

    def predictive_resample(self, B=1000, T=5000, seed=100):
        """B posterior draws of p_N, N = n + T, at the points of the last ``predict`` call.

        Returns the log joint densities, shape (B, n_test).
        """
        key = PRNGKey(seed)
        key, *subkey = split(key, B + 1)
        subkey = jnp.array(subkey)

        n = jnp.shape(self.vn_perm)[1]

        start = time.time()
        logpdf_joints_pr = self._resampling.predictive_resample_loop_B(
            subkey, self.logcdf_conditionals, self.logpdf_joints,
            self.theta_opt, n, T
        ).block_until_ready()
        self.timings["resample"] = time.time() - start
        self._log('Predictive resampling time: {}s'.format(round(self.timings["resample"], 3)))

        return logpdf_joints_pr

    def check_convergence(self, B=1, T=10000, seed=100):
        """Run B forward-sampling chains of length T from the last ``predict`` call.

        Returns (B, T): the mean absolute difference between p_{n+t} and p_n
        over the test points, for t = 1, ..., T.
        """
        key = PRNGKey(seed)
        key, *subkey = split(key, B + 1)
        subkey = jnp.array(subkey)

        n = jnp.shape(self.vn_perm)[1]
        return self._resampling.pr_loop_conv_B(subkey, self.logcdf_conditionals, self.logpdf_joints,
                                               self.theta_opt, n, T)

    def sample_intensity(self, x, domain, B=1000, T=2000, seed=200, rng=None):
        """Posterior samples of the intensity lambda(x), shape (B, len(x)).

        ``x`` are points in the original units of the box ``domain`` (d, 2).
        Each sample is gamma * p_N(x) / |domain|, with p_N a predictive-resampling
        draw (``T`` steps, JAX ``seed``) and gamma ~ Gamma(n, 1) drawn from the
        NumPy generator ``rng``.
        """
        if rng is None:
            rng = np.random.default_rng()
        domain = np.asarray(domain)

        self.predict(normalize(x, domain))
        logpdf_joints_pr = self.predictive_resample(B=B, T=T, seed=seed)

        pdf_predictive = np.exp(logpdf_joints_pr).astype(np.float32)
        gamma_samples = rng.gamma(self.n, 1.0, size=B).astype(np.float32)
        return pdf_predictive * gamma_samples[:, np.newaxis] / np.prod(domain[:, 1] - domain[:, 0])

    def sample_log_intensity(self, x, domain, B=1000, T=2000, seed=200, rng=None):
        """Like ``sample_intensity`` but returns log lambda(x) (a float32 JAX array), computed in log space."""
        if rng is None:
            rng = np.random.default_rng()
        domain = np.asarray(domain)

        self.predict(normalize(x, domain))
        logpdf_joints_pr = self.predictive_resample(B=B, T=T, seed=seed)

        gamma_samples = rng.gamma(self.n, 1.0, size=B).astype(np.float32)
        return (logpdf_joints_pr + np.log(gamma_samples)[:, np.newaxis]
                - np.sum(np.log(domain[:, 1] - domain[:, 0])))
