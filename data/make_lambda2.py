"""Simulate the lambda2 datasets (data/lambda2/) of the 2D synthetic experiment.

Writes samples_<seed>.csv for seeds 0-49, each a realisation of the Poisson
process with intensity lambda2 on [0, 10]^2 (simulated by thinning), and
true.csv with lambda2 on a 50 x 50 grid. Run from the repository root:

    python data/make_lambda2.py [--out data/lambda2]
"""
import argparse
import os

import numpy as np
import pandas as pd


def intensity(x1, x2):
    """lambda2(x1, x2) = 2 cos^5(pi (x1 - x2) / 20) + 0.2 sin^2(pi (x1 + x2) / 10)."""
    return (
        2.0 * np.cos(np.pi * (x1 - x2) / 20.0) ** 5
        + 0.2 * np.sin(np.pi * (x1 + x2) / 10.0) ** 2
    )


def simulate_ipp(seed, domain=(0.0, 10.0, 0.0, 10.0)):
    """One realisation of the inhomogeneous Poisson process, by thinning."""
    rng = np.random.default_rng(seed)

    x_min, x_max, y_min, y_max = domain
    area = (x_max - x_min) * (y_max - y_min)

    # Upper bound on the intensity over [0, 10]^2
    lambda_max = 2.2

    # Homogeneous Poisson candidates
    n_candidates = rng.poisson(lambda_max * area)
    x1_candidates = rng.uniform(x_min, x_max, n_candidates)
    x2_candidates = rng.uniform(y_min, y_max, n_candidates)

    # Thinning
    lambdas = intensity(x1_candidates, x2_candidates)
    keep = rng.uniform(0.0, 1.0, n_candidates) < (lambdas / lambda_max)

    return pd.DataFrame({"x1": x1_candidates[keep], "x2": x2_candidates[keep]})


def save_true_intensity_grid(out_dir, grid_size=50):
    """lambda2 on a grid_size x grid_size grid."""
    x1 = np.linspace(0.0, 10.0, grid_size)
    x2 = np.linspace(0.0, 10.0, grid_size)
    X1, X2 = np.meshgrid(x1, x2)

    true_df = pd.DataFrame({
        "x1": X1.ravel(),
        "x2": X2.ravel(),
        "intensity": intensity(X1.ravel(), X2.ravel()),
    })
    true_df.to_csv(os.path.join(out_dir, "true.csv"), index=False)


def generate_data(n_seeds=50, out_dir="data/lambda2"):
    os.makedirs(out_dir, exist_ok=True)
    for seed in range(n_seeds):
        simulate_ipp(seed).to_csv(os.path.join(out_dir, f"samples_{seed}.csv"), index=False)
    save_true_intensity_grid(out_dir=out_dir, grid_size=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/lambda2")
    generate_data(out_dir=parser.parse_args().out)
