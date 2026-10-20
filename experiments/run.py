"""Fit one method on one dataset for a list of seeds and save the results.

Examples (from the repository root)::

    python experiments/run.py dataset=lambda1 method=mpcp
    python experiments/run.py dataset=bei method=vbpp seeds=[0]
    python experiments/run.py -m +experiment=vbpp_ablation

Results go to ``logs/<dataset>/<run_name>/``:

* ``intensities.npy``  (n_seeds, B, G) posterior intensity samples on the grid
* ``grid.npy``         (G, d) grid, in original units
* ``test_loglik.npy``  (n_seeds,) test log-likelihood
* ``runtime.npy``      (n_seeds,) runtime in seconds
* ``seeds.npy``        (n_seeds,) dataset seeds
* ``config.yaml``      the full configuration

All seeds run in one process, so JIT compilation is shared across seeds (as in
the paper runs, the first MPCP seed includes some compilation time).
"""
import os
import time

# Several jobs (e.g. a multirun over methods) may share one process and GPU:
# let JAX and TensorFlow allocate GPU memory as needed instead of up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from datasets import REPO_ROOT, Dataset
from methods import METHODS
from mpcp.evaluation import integrate_on_grid, test_log_likelihood


def evaluate_seed(method, task, method_cfg, seed):
    """Fit, then compute the test log-likelihood averaged over the task's test sets."""
    fit = method(task, method_cfg, seed)
    integral = integrate_on_grid(fit.intensities, task.grid)
    logliks = [test_log_likelihood(integral, fit.log_intensity(x_test)) for x_test in task.test_sets]
    return fit, sum(logliks) / len(logliks)


def _tensorflow():
    """Import TensorFlow, letting it use the GPU.

    TensorFlow loads the CUDA libraries installed with jax[cuda12]; they are
    only found once JAX has initialised its GPU backend, so JAX is
    initialised first (otherwise TensorFlow silently falls back to the CPU).
    """
    import jax
    jax.devices()
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:  # already initialised
            pass
    return tf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dataset = Dataset(cfg.dataset, REPO_ROOT / cfg.paths.data)
    seeds = list(cfg.seeds) if cfg.seeds is not None else list(range(dataset.n_seeds))
    method = METHODS[cfg.method.name]

    out_dir = REPO_ROOT / cfg.paths.logs / dataset.name / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    np.save(out_dir / "grid.npy", dataset.grid)
    np.save(out_dir / "seeds.npy", np.array(seeds))

    B = cfg.method.num_samples
    intensities = np.lib.format.open_memmap(
        out_dir / "intensities.npy", mode="w+", dtype=np.float64, shape=(len(seeds), B, len(dataset.grid))
    )
    test_loglik = np.full(len(seeds), np.nan)
    runtime = np.full(len(seeds), np.nan)

    print(f"{cfg.method.name} on {dataset.name}, seeds {seeds} -> {out_dir}")
    for i, seed in enumerate(seeds):
        stream_seed = cfg.rng_seed + seed
        if cfg.method.name != "mpcp":
            tf = _tensorflow()
            tf.random.set_seed(stream_seed)

        start = time.perf_counter()
        fit, test_loglik[i] = evaluate_seed(method, dataset.task(seed), cfg.method, stream_seed)
        intensities[i] = fit.intensities
        runtime[i] = fit.runtime

        intensities.flush()
        np.save(out_dir / "test_loglik.npy", test_loglik)
        np.save(out_dir / "runtime.npy", runtime)
        print(f"seed {seed}: test log-lik {test_loglik[i]:.3f}, runtime {runtime[i]:.2f}s "
              f"(wall {time.perf_counter() - start:.1f}s)", flush=True)

    print(f"test log-lik {np.mean(test_loglik):.3f} ({np.std(test_loglik) / np.sqrt(len(seeds)):.3f}), "
          f"runtime {np.mean(runtime):.2f}s ({np.std(runtime) / np.sqrt(len(seeds)):.2f})")


if __name__ == "__main__":
    main()
