"""Data loading and mapping between a box domain and the unit cube."""
import os

import numpy as np


def load_points(file_path):
    """Load a CSV of points with a header row as an (n, d) array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    return np.loadtxt(file_path, delimiter=',', skiprows=1, ndmin=2)


def normalize(data, domain):
    """Map points in the box ``domain`` (d, 2) of [min, max] rows to the unit cube."""
    mins = domain[:, 0]
    ranges = domain[:, 1] - mins
    if np.any(ranges == 0):
        raise ValueError("Domain min and max cannot be equal for any dimension.")
    return (data - mins) / ranges


def denormalize(unit_intensities, domain):
    """Convert intensities on the unit cube to intensities on the box ``domain``."""
    original_volume = np.prod(domain[:, 1] - domain[:, 0])
    if original_volume == 0:
        raise ValueError("Domain volume cannot be zero.")
    return unit_intensities / original_volume


def create_test_grid(domain, n_per_dim):
    """Regular grid with ``n_per_dim`` points per dimension (including the bounds), shape (n_per_dim^d, d)."""
    d = domain.shape[0]
    axes = [np.linspace(start, end, n_per_dim) for start, end in domain]
    grid = np.meshgrid(*axes, indexing='ij')
    return np.stack(grid, axis=-1).reshape(-1, d)
