"""Fitting MPCP and the baselines on a ``Task``, with the timings reported in the paper.

Each ``run_<method>(task, cfg, seed)`` fits on ``task.x_train`` and returns a
``Fit`` with posterior intensity samples on ``task.grid`` (in original units),
the runtime in seconds and a function giving log-intensity samples at test
points. Draws happen in the order of the paper runs: first the grid, then each
test set in turn.

Runtime excludes JIT compilation / warm-up steps: for MPCP it is fitting
(bandwidth selection and overhead terms) plus sampling on the grid; for the
baselines optimisation plus sampling on the grid (including HMC for LGCP).
"""
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from omegaconf import ListConfig, OmegaConf

from mpcp import MPCP, denormalize


@dataclass
class Fit:
    intensities: np.ndarray                       # (B, G)
    runtime: float                                # seconds
    log_intensity: Callable[[np.ndarray], np.ndarray]  # test points (n, d) -> (B, n)


def _lengthscales(cfg):
    """Kernel lengthscales as a float or a plain list (gpflow cannot take an OmegaConf list)."""
    ls = cfg.lengthscales
    return OmegaConf.to_container(ls) if isinstance(ls, ListConfig) else ls


def _log_volume(domain):
    return np.sum(np.log(domain[:, 1] - domain[:, 0]))


def run_mpcp(task, cfg, seed):
    """MPCP; ``seed`` seeds the NumPy generator of the Gamma(n, 1) count draws."""
    rng = np.random.default_rng(seed)
    model = MPCP(n_perm=cfg.n_perm, seed=cfg.perm_seed, copula=cfg.copula)
    model.fit(task.unit(task.x_train))

    sample = dict(domain=task.domain, B=cfg.num_samples, T=cfg.T, seed=cfg.resample_seed, rng=rng)
    start = time.perf_counter()
    intensities = model.sample_intensity(task.grid, **sample)
    runtime = model.timings["fit"] + time.perf_counter() - start

    return Fit(intensities, runtime, lambda x: model.sample_log_intensity(x, **sample))


def run_vbpp(task, cfg, seed):
    """VBPP on the original domain (``cfg.normalize = false``) or on the unit cube."""
    from mpcp import create_test_grid
    from mpcp.baselines.vbpp import fit_vbpp

    if cfg.normalize:
        to_model = task.unit
        domain = np.array([[0.0, 1.0]] * task.domain.shape[0])
        log_volume = _log_volume(task.domain)
    else:
        to_model = lambda x: x  # noqa: E731
        domain = task.domain
        log_volume = 0.0

    inducing_points = create_test_grid(domain, n_per_dim=cfg.inducing_per_dim)
    model, opt_time = fit_vbpp(to_model(task.x_train), domain, inducing_points, _lengthscales(cfg))

    start = time.perf_counter()
    intensities = model.predict_lambda_samples(to_model(task.grid), n_samples=cfg.num_samples).numpy()
    if cfg.normalize:
        intensities = denormalize(intensities, task.domain)
    runtime = opt_time + time.perf_counter() - start

    def log_intensity(x):
        return np.log(model.predict_lambda_samples(to_model(x), n_samples=cfg.num_samples)) - log_volume

    return Fit(intensities, runtime, log_intensity)


def _binned_log_intensity(model, task, num_samples):
    def log_intensity(x):
        f = model.predict_f_samples(task.unit(x), num_samples=num_samples)
        return np.log(np.exp(f.numpy().squeeze(-1))) - _log_volume(task.domain)
    return log_intensity


def run_vlgcp(task, cfg, seed):
    """Variational LGCP on binned counts on the unit cube."""
    from mpcp.baselines.lgcp import fit_vlgcp

    model, opt_time = fit_vlgcp(task.unit(task.x_train), cfg.bins_per_dim, _lengthscales(cfg))

    start = time.perf_counter()
    f_samples = model.predict_f_samples(task.unit(task.grid), num_samples=cfg.num_samples)
    intensities = denormalize(np.exp(f_samples.numpy().squeeze(-1)), task.domain)
    runtime = opt_time + time.perf_counter() - start

    return Fit(intensities, runtime, _binned_log_intensity(model, task, cfg.num_samples))


def run_lgcp(task, cfg, seed):
    """LGCP on binned counts on the unit cube, sampled with HMC.

    The test log-likelihood uses the model at the last HMC sample, as in the paper runs.
    """
    from mpcp.baselines.lgcp import fit_lgcp, lgcp_f_samples

    model, hmc_helper, run_chain, map_time = fit_lgcp(
        task.unit(task.x_train), cfg.bins_per_dim, _lengthscales(cfg),
        map_maxiter=cfg.map_maxiter, num_samples=cfg.num_samples, num_burnin_steps=cfg.num_burnin_steps,
        num_leapfrog_steps=cfg.num_leapfrog_steps, step_size=cfg.step_size,
        num_adaptation_steps=cfg.num_adaptation_steps, target_accept_prob=cfg.target_accept_prob,
        adaptation_rate=cfg.adaptation_rate,
    )

    start = time.perf_counter()
    samples = run_chain()
    f_samples = lgcp_f_samples(model, hmc_helper, samples, task.unit(task.grid))
    intensities = denormalize(np.exp(f_samples.numpy().squeeze(-1)), task.domain)
    runtime = map_time + time.perf_counter() - start

    return Fit(intensities, runtime, _binned_log_intensity(model, task, cfg.num_samples))


METHODS = {"mpcp": run_mpcp, "vbpp": run_vbpp, "vlgcp": run_vlgcp, "lgcp": run_lgcp}
