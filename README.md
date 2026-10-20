# Martingale Posterior Cox Processes

📄 **Paper:** [Martingale Posterior Cox Processes](https://arxiv.org/abs/2610.24787)\
👥 **Authors:** [Joshua Rooijakkers](https://joshuarooijakkers.github.io/) and [Edwin Fong](https://edfong.github.io/)

The Martingale Posterior Cox Process (MPCP) is a Bayesian approach to
estimating the intensity of an inhomogeneous Poisson process. It does not
require placing a prior on latent functions, but only on the count of events.
The posterior is obtained by predictive resampling: unobserved events are
imputed one at a time from a sequence of predictive densities, which are
updated in closed form with a bivariate (Frank) copula, like an online kernel
density estimator. Sampling requires no MCMC or variational approximation,
runs in parallel on a GPU, and provably converges to a random intensity
function.

<p align="center">
  <img src="assets/mpcp_explainer_dark.gif" width="540" alt="Animation of the MPCP method: events are observed over time; starting from a flat intensity, the events are processed in a random order and the intensity is updated one event at a time; predictive resampling then simulates new events until the curve converges, giving one posterior sample; repeating this gives an intensity estimate with uncertainty that captures the true intensity.">
</p>

## Installation

```bash
git clone https://github.com/joshuarooijakkers/mpcp.git
cd mpcp
conda env create -f environment.yml    # Python 3.10, JAX with CUDA 12, and this package
conda activate mpcp
```

Alternatively, with pip in an existing Python 3.10 environment:

```bash
pip install -e .                              # MPCP only (JAX on CPU)
pip install -e ".[baselines,experiments]"     # + GP baselines and experiment scripts
pip install "jax[cuda12]==0.6.2"              # optional: run JAX on an NVIDIA GPU
```

The paper's results were computed with the versions pinned in `pyproject.toml`
(JAX 0.6.2, NumPy 1.26.4, gpflow 2.11.1, TensorFlow 2.21.0) on an NVIDIA RTX
3000 GPU. The figure scripts need a LaTeX installation (matplotlib's
`text.usetex`). The DPMM baseline is written in R and needs the packages
`dirichletprocess` and `reticulate`.

## Repository layout

```
mpcp/                       the package
  model.py                  MPCP class
  copulas.py                Frank and Gaussian copulas
  copula_updates.py         recursive copula updates and bandwidth selection
  predictive_resampling.py  predictive resampling
  evaluation.py             grid integrals, test log-likelihood, L2 error, interval coverage (Fig. 4)
  utils.py                  data loading, (de)normalisation, grids
  baselines/                VBPP, VLGCP and LGCP (gpflow / TensorFlow)
experiments/
  run.py                    fit one method on one dataset (Hydra)
  convergence.py            predictive-resampling trajectories
  tables.py, figures.py     tables and figures from the saved results
  dpmm_lambda1.R            DPMM baseline (R)
  conf/                     Hydra configs: dataset/, method/, dataset_method/, experiment/
  naturalearth_10m/         country borders for the earthquake maps (Natural Earth, public domain)
data/                       datasets and the scripts that create them (see data/README.md)
```

## Acknowledgements

The VBPP baseline (`mpcp/baselines/vbpp.py`) builds on
[st--/vbpp](https://github.com/st--/vbpp) (Apache License 2.0) by ST John: it
subclasses `vbpp.model.VBPP` to add a small jitter to the ELBO and a method
for sampling the intensity. The bei data are from the R package
[spatstat.data](https://cran.r-project.org/package=spatstat.data), and the
earthquake data from the U.S. Geological Survey
[ANSS Comprehensive Catalog](https://earthquake.usgs.gov/data/comcat/).

## Citation

```bibtex
@article{rooijakkers2026martingale,
  title   = {Martingale Posterior Cox Processes},
  author  = {Rooijakkers, Joshua and Fong, Edwin},
  journal = {arXiv preprint arXiv:2610.24787},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
