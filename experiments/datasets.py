"""Datasets and evaluation protocol of the experiments (Section 5, Appendix C)."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from mpcp import create_test_grid, load_points, normalize

REPO_ROOT = Path(__file__).resolve().parents[1]


def lambda1(x):
    """True intensity lambda_1 of the 1D synthetic experiment, on [0, 50]."""
    return 2 * np.exp(-x / 15) + np.exp(-((x - 25) / 10) ** 2)


@dataclass
class Task:
    """One train/test problem.

    Attributes:
        x_train: (n, d) training events, in original units.
        test_sets: test event sets; the test log-likelihood is averaged over them.
        grid: (G, d) regular grid on which the intensity is sampled and integrated.
        domain: (d, 2) box [min, max] per dimension.
        true_intensity: (G,) true intensity on the grid (simulated data only).
        dtype: dtype of normalised inputs passed to the methods.
    """
    x_train: np.ndarray
    test_sets: List[np.ndarray]
    grid: np.ndarray
    domain: np.ndarray
    true_intensity: Optional[np.ndarray] = None
    dtype: str = "float64"

    def unit(self, points):
        """Map points to the unit cube."""
        return normalize(points, self.domain).astype(self.dtype)


class Dataset:
    """A dataset configured by a Hydra ``dataset`` config (see experiments/conf/dataset).

    Two protocols:

    * ``simulated``: ``n_seeds`` simulated datasets; the method is fit on dataset
      ``seed`` and the test log-likelihood averaged over all other datasets.
    * ``split``: one dataset, split 50/50 into train and test with
      ``random_state = split_offset + seed``.
    """

    def __init__(self, cfg, data_dir=REPO_ROOT / "data"):
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.name = cfg.name
        self.n_seeds = cfg.n_seeds
        self.domain = np.array(cfg.domain, dtype=float)
        self.grid, self.true_intensity = self._grid()

        if cfg.protocol == "split":
            self._points = self._load(cfg.file)

        # Process the covariates in reverse order (covariate-ordering ablation)
        if cfg.get("reverse_covariates", False):
            self.domain = self.domain[::-1]
            self.grid = self.grid[:, ::-1]

    def _load(self, relative_path):
        points = load_points(self.data_dir / relative_path)
        columns = self.cfg.get("columns")
        return points if columns is None else points[:, list(columns)]

    def _grid(self):
        grid_cfg = self.cfg.grid
        if grid_cfg.kind == "linspace":
            grid = np.linspace(grid_cfg.low, grid_cfg.high, grid_cfg.n)[:, None]
            return grid, lambda1(grid[:, 0])
        if grid_cfg.kind == "file":
            values = load_points(self.data_dir / grid_cfg.file)
            return values[:, :-1], values[:, -1]
        if grid_cfg.kind == "regular":
            return create_test_grid(self.domain, grid_cfg.n_per_dim), None
        raise ValueError(f"Unknown grid kind {grid_cfg.kind}")

    def _simulated(self, seed):
        return self._load(self.cfg.files.format(seed=seed))

    def task(self, seed):
        """The train/test problem for ``seed``."""
        reverse = self.cfg.get("reverse_covariates", False)
        order = (lambda p: p[:, ::-1]) if reverse else (lambda p: p)

        if self.cfg.protocol == "simulated":
            x_train = self._simulated(seed)
            test_sets = [self._simulated(s) for s in range(self.n_seeds) if s != seed]
        elif self.cfg.protocol == "split":
            x_train, x_test = train_test_split(
                self._points, test_size=0.5, random_state=self.cfg.split_offset + seed, shuffle=True
            )
            test_sets = [x_test]
        else:
            raise ValueError(f"Unknown protocol {self.cfg.protocol}")

        return Task(
            x_train=order(x_train),
            test_sets=[order(t) for t in test_sets],
            grid=self.grid,
            domain=self.domain,
            true_intensity=self.true_intensity,
            dtype=self.cfg.get("dtype", "float64"),
        )
