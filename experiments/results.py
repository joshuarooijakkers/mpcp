"""Loading saved runs from ``logs/`` (shared by tables.py and figures.py)."""
from functools import lru_cache

import numpy as np
from omegaconf import OmegaConf

from datasets import REPO_ROOT, Dataset

LOGS = REPO_ROOT / "logs"
FIGS = REPO_ROOT / "figs"
CONF = REPO_ROOT / "experiments" / "conf"


@lru_cache(maxsize=None)
def dataset(name):
    """The ``Dataset`` for ``conf/dataset/<name>.yaml``."""
    return Dataset(OmegaConf.load(CONF / "dataset" / f"{name}.yaml"))


def load_run(dataset_name, run_name):
    """Results of ``run.py`` (or dpmm_lambda1.R) in logs/<dataset>/<run>, or None if absent.

    ``intensities`` is memory-mapped; seeds that did not finish are dropped.
    """
    run_dir = LOGS / dataset_name / run_name
    if not (run_dir / "test_loglik.npy").exists():
        return None
    run = {
        "intensities": np.load(run_dir / "intensities.npy", mmap_mode="r"),
        "test_loglik": np.load(run_dir / "test_loglik.npy"),
        "runtime": np.load(run_dir / "runtime.npy"),
        "seeds": np.load(run_dir / "seeds.npy"),
    }
    done = ~np.isnan(run["test_loglik"])
    if not done.all():
        run = {k: v[done] for k, v in run.items()}
    return run


def seed_intensities(dataset_name, run_name, seed):
    """Posterior intensity samples (B, G) of one seed of a run."""
    run = load_run(dataset_name, run_name)
    if run is None:
        raise FileNotFoundError(f"No results in {LOGS / dataset_name / run_name}; run experiments/run.py first.")
    index = np.flatnonzero(run["seeds"] == seed)
    if len(index) == 0:
        raise ValueError(f"Seed {seed} was not run for {dataset_name}/{run_name}.")
    return np.asarray(run["intensities"][index[0]])


def missing(dataset_name, run_name):
    print(f"  (skipping {dataset_name}/{run_name}: no results in {LOGS / dataset_name / run_name})")
