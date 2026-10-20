# Data

All datasets used in the paper are included. Files are CSVs of points with a
header row; columns are the coordinates.

| Path | Used for | Source | How it was created |
|---|---|---|---|
| `lambda1/lambda1_{0..49}.csv` | λ1, 1D synthetic (Table 1, Figs. 3, 4, 8, 9, 12) | 50 realisations of the Poisson process with intensity λ1(x) = 2 exp(−x/15) + exp(−((x − 25)/10)²) on [0, 50] | See note below |
| `lambda2/samples_{0..49}.csv`, `lambda2/true.csv` | λ2, 2D synthetic (Tables 1 and 3, Figs. 4, 9, 11, 12) | 50 realisations of the Poisson process with intensity λ2(x, x′) = 2 cos⁵(π(x − x′)/20) + 0.2 sin²(π(x + x′)/10) on [0, 10]², simulated by thinning; `true.csv` is λ2 on a 50 × 50 grid | `python data/make_lambda2.py` |
| `bei.csv` | bei (Table 2, Figs. 13, 15) | Locations of 3,604 *Beilschmiedia pendula* trees on Barro Colorado Island (Hubbell and Foster, 1983), from the R package [spatstat.data](https://cran.r-project.org/package=spatstat.data) | `Rscript data/get_bei.R` |
| `earthquake.csv` | earthquake (Table 2, Figs. 5, 14, 15) | Earthquakes with magnitude > 4 in 2000–2009 in 122–150° E, 22–46° N, from the U.S. Geological Survey [ANSS Comprehensive Catalog](https://earthquake.usgs.gov/data/comcat/); columns are decimal year, longitude, latitude | `python data/get_earthquake.py` |

The scripts are run from the repository root. Each reproduces the included
file exactly (checked in October 2026; the earthquake catalogue may be revised
later).

**λ1.** λ1 is the 1D benchmark intensity of Adams et al. (2009). The code that
simulated these 50 datasets is not part of this repository, so the CSVs
themselves are the reference data.
