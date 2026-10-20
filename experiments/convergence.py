"""Convergence of predictive resampling (Figure 9, Appendix C.2).

Fits MPCP to one simulated dataset and runs B forward-sampling chains of T
steps from p_n on the evaluation grid. For each chain it records the mean
absolute change |p_{n+t}(x) - p_n(x)| over the grid, scaled by a Gamma(n, 1)
draw of the total count, i.e. a grid estimate of L_N in eq. (9) on the unit
cube. Saves ``logs/<dataset>/convergence.npy`` with shape (B, T).

Example::

    python experiments/convergence.py dataset=lambda1
    python experiments/convergence.py dataset=lambda2
"""
import hydra
import numpy as np
from omegaconf import DictConfig

from datasets import REPO_ROOT, Dataset
from mpcp import MPCP


@hydra.main(version_base=None, config_path="conf", config_name="convergence")
def main(cfg: DictConfig):
    dataset = Dataset(cfg.dataset, REPO_ROOT / cfg.paths.data)
    task = dataset.task(cfg.seed)
    rng = np.random.default_rng(cfg.rng_seed + cfg.seed)

    m = cfg.method
    model = MPCP(n_perm=m.n_perm, seed=m.perm_seed, copula=m.copula)
    model.fit(task.unit(task.x_train))
    model.predict(task.unit(task.grid))
    gamma_samples = rng.gamma(model.n, 1.0, size=cfg.num_chains).astype(np.float32)

    pdiff = model.check_convergence(B=cfg.num_chains, T=cfg.num_steps, seed=cfg.chain_seed)
    distance = pdiff * gamma_samples[:, np.newaxis]

    out_dir = REPO_ROOT / cfg.paths.logs / dataset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "convergence.npy", distance)
    print(f"Saved {distance.shape} trajectories to {out_dir / 'convergence.npy'}")


if __name__ == "__main__":
    main()
