"""Make the result figures of the paper from the runs in ``logs/``; PDFs go to ``figs/``.

Usage (from the repository root)::

    python experiments/figures.py              # all figures
    python experiments/figures.py lambda1 bei

Figures (paper numbering) and the runs they need:

    lambda1              Fig. 3   lambda1: mpcp, lgcp, dpmm, vbpp, vlgcp
    coverage             Fig. 4   lambda1 and lambda2 runs of all methods
    earthquake_mean      Fig. 5   earthquake: mpcp, vbpp, vlgcp
    copula               Fig. 8   lambda1: mpcp, mpcp_gaussian (+experiment=gaussian_copula)
    convergence          Fig. 9   experiments/convergence.py for lambda1 and lambda2
    lambda2              Fig. 11  lambda2: mpcp, lgcp, vbpp, vlgcp
    lengths              Fig. 12  as coverage
    bei                  Fig. 13  bei: mpcp, vbpp, vlgcp
    earthquake_variance  Fig. 14  as earthquake_mean

Figures 1, 2, 6, 7, 10 and 15 (illustrations and ablation plots) are not
produced here; the numbers behind Figure 15 are printed by tables.py.
Requires a LaTeX installation (text.usetex).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import MaxNLocator

from datasets import REPO_ROOT
from results import FIGS, LOGS, dataset, load_run, seed_intensities
from mpcp.evaluation import integrate_on_grid, interval_coverage, posterior_integrals

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["text.usetex"] = True

COLORS = {
    "MPCP": "royalblue",
    "LGCP": "orange",
    "DPMM": "teal",
    "VBPP": "crimson",
    "VLGCP": "green",
}
Q_LEVELS = np.arange(0.04, 0.97, 0.04)


def truncated_cmap(name, start):
    return LinearSegmentedColormap.from_list(f"{name}_truncated", plt.get_cmap(name)(np.linspace(start, 1, 256)))


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / name}.pdf")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def lambda1(seed=12):
    """Posterior mean and 90% pointwise interval for lambda1 (Figure 3)."""
    ds = dataset("lambda1")
    x_grid = ds.grid[:, 0]
    events = ds.task(seed).x_train.flatten()
    methods = ["MPCP", "LGCP", "DPMM", "VBPP", "VLGCP"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, methods):
        samples = seed_intensities("lambda1", name.lower(), seed)
        lower = np.percentile(samples, 5, axis=0)
        upper = np.percentile(samples, 95, axis=0)

        ax.fill_between(x_grid, lower, upper, color=COLORS[name], alpha=0.2)
        ax.plot(x_grid, ds.true_intensity, color="black", linestyle="--", linewidth=3, alpha=0.7)
        ax.plot(x_grid, np.mean(samples, axis=0), color=COLORS[name], linewidth=3)
        ax.vlines(events, ymin=0, ymax=0.2, linewidth=1.0, alpha=1.0, color="black")

        ax.set_title(name, fontsize=22)
        ax.set_xlabel(r"$x$", fontsize=22)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.tick_params(axis="both", labelsize=20)

    axes[0].set_ylabel("Intensity", fontsize=24)
    plt.tight_layout()
    save(fig, "lambda1_comparison")


def lambda2(seed=3):
    """Posterior mean and variance for lambda2 (Figure 11)."""
    ds = dataset("lambda2")
    events = ds.task(seed).x_train
    x, y = ds.grid[:, 0], ds.grid[:, 1]
    names = ["MPCP", "LGCP", "VBPP", "VLGCP", "True"]

    means, variances = {}, {}
    for name in names[:-1]:
        samples = seed_intensities("lambda2", name.lower(), seed)
        means[name] = np.mean(samples, axis=0)
        variances[name] = np.var(samples, axis=0)
    means["True"] = ds.true_intensity

    cmap_mean = truncated_cmap("viridis", 0.20)
    cmap_var = truncated_cmap("inferno", 0.30)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True)
    for j, name in enumerate(names):
        panels = [(axes[0, j], means[name], cmap_mean, 2.5)]
        if name != "True":
            panels.append((axes[1, j], variances[name], cmap_var, 0.5))
        for ax, values, cmap, vmax in panels:
            ax.tricontourf(x, y, values, levels=20, cmap=cmap, vmin=0, vmax=vmax)
            ax.scatter(events[:, 0], events[:, 1], s=14, c="black", alpha=1.0, linewidths=0)
        axes[0, j].set_title(name, fontsize=22, pad=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
    for spine in axes[1, 4].spines.values():
        spine.set_visible(False)
    axes[0, 0].set_ylabel("Mean", fontsize=24, labelpad=15)
    axes[1, 0].set_ylabel("Variance", fontsize=24, labelpad=15)

    plt.tight_layout()
    save(fig, "lambda2_comparison")


def _count_intervals():
    """Coverage and mean length of q-level intervals for gamma, per dataset and method."""
    results = {}
    for name, methods in [("lambda1", ["MPCP", "LGCP", "DPMM", "VBPP", "VLGCP"]),
                          ("lambda2", ["MPCP", "LGCP", "VBPP", "VLGCP"])]:
        ds = dataset(name)
        true_integral = integrate_on_grid(ds.true_intensity, ds.grid)
        for method in methods:
            run = load_run(name, method.lower())
            if run is None:
                raise FileNotFoundError(f"No results for {name}/{method.lower()}; run experiments/run.py first.")
            integrals = posterior_integrals(run["intensities"], ds.grid)
            cov_len = [interval_coverage(integrals, true_integral, q) for q in Q_LEVELS]
            results[name, method] = (np.array([c for c, _ in cov_len]), np.array([l.mean() for _, l in cov_len]))
    return results


def _two_panel_legend(fig, axes, order, ncol):
    handles, labels = axes[0].get_legend_handles_labels()
    handle_dict = dict(zip(labels, handles))
    fig.legend([handle_dict[label] for label in order], order, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=ncol, fontsize=14, frameon=False)


def coverage_and_lengths(which=("coverage", "lengths")):
    """Coverage (Figure 4) and length (Figure 12) of the credible intervals for gamma."""
    results = _count_intervals()
    xticks = np.arange(0, 1.01, 0.2)
    q_plot = np.concatenate(([0], Q_LEVELS, [1]))

    for kind in which:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for ax, name, title in zip(axes, ["lambda1", "lambda2"], [r"$\lambda_1$", r"$\lambda_2$"]):
            # Drawing order as in the paper (MPCP last, so it is on top)
            order = ["VBPP", "VLGCP", "LGCP", "DPMM", "MPCP"] if kind == "coverage" else ["LGCP", "DPMM", "VBPP", "VLGCP", "MPCP"]
            methods = [m for m in order if (name, m) in results]
            for method in methods:
                coverage, length = results[name, method]
                if kind == "coverage":
                    if method == "MPCP":
                        ax.plot(q_plot, q_plot, "--", label="Ideal", color="black", alpha=0.5, linewidth=2.5)
                    ax.plot(q_plot, np.r_[0, coverage, 1], label=method, color=COLORS[method], linewidth=2.5)
                else:
                    ax.plot(Q_LEVELS, length, label=method, color=COLORS[method], linewidth=2.5)

            ax.set_title(title, fontsize=18)
            ax.set_xlabel("Level", fontsize=16)
            ax.set_ylabel("Coverage" if kind == "coverage" else "Interval length", fontsize=16, labelpad=10)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.1f}" for x in xticks], fontsize=14)
            ax.tick_params(axis="both", labelsize=14, width=2.5)
            ax.set_xlim(0, 1)
            if kind == "coverage":
                ax.tick_params(axis="y", labelleft=True)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)

        if kind == "coverage":
            _two_panel_legend(fig, axes, ["MPCP", "LGCP", "DPMM", "VBPP", "VLGCP", "Ideal"], ncol=6)
            plt.subplots_adjust(wspace=0.35, bottom=0.20)
            save(fig, "coverage")
        else:
            _two_panel_legend(fig, axes, ["MPCP", "LGCP", "DPMM", "VBPP", "VLGCP"], ncol=5)
            plt.subplots_adjust(wspace=0.35, bottom=0.20)
            save(fig, "interval_lengths")


def coverage():
    coverage_and_lengths(("coverage",))


def lengths():
    coverage_and_lengths(("lengths",))


def copula(seed=12):
    """MPCP with the Frank and the Gaussian copula on lambda1 (Figure 8)."""
    ds = dataset("lambda1")
    x_grid = ds.grid[:, 0]
    events = ds.task(seed).x_train.flatten()
    colors = {"Frank": "royalblue", "Gaussian": "mediumorchid"}
    runs = {"Frank": "mpcp", "Gaussian": "mpcp_gaussian"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharex=True, sharey=True)
    for ax, (name, run_name) in zip(axes, runs.items()):
        samples = seed_intensities("lambda1", run_name, seed)
        ax.fill_between(x_grid, np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0),
                        color=colors[name], alpha=0.2)
        ax.plot(x_grid, ds.true_intensity, color="black", linestyle="--", linewidth=3, alpha=0.7)
        ax.plot(x_grid, np.mean(samples, axis=0), color=colors[name], linewidth=3)

        ax.set_title(f"MPCP ({name})", fontsize=22)
        ax.set_xlabel(r"$x$", fontsize=22)
        ax.set_ylabel("Intensity", fontsize=24)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.tick_params(axis="both", labelsize=20)
        ax.vlines(events, ymin=0, ymax=0.2, linewidth=1.0, alpha=1.0, color="black")

    axes[1].tick_params(axis="y", labelleft=True)
    plt.subplots_adjust(wspace=0.3)
    save(fig, "copula_comparison")


def convergence(chains=slice(10, 15)):
    """Trajectories of the distance L_N during predictive resampling (Figure 9)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 4.5), sharex=True)
    for ax, name, title in zip(axes, ["lambda1", "lambda2"], [r"$\lambda_1$", r"$\lambda_2$"]):
        path = LOGS / name / "convergence.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run experiments/convergence.py dataset={name} first.")
        distance = np.load(path)[chains]
        T = distance.shape[1]
        for trajectory in distance:
            ax.plot(np.arange(T), trajectory, color="black", alpha=0.5, linewidth=2.5)

        ax.set_title(title, fontsize=24)
        ax.set_xlabel(r"$N - n$", fontsize=20)
        ax.set_ylabel(r"Distance", fontsize=22, labelpad=10)
        ax.set_xlim(0, T)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        ax.tick_params(axis="both", which="major", labelsize=16, width=2.0)

    plt.subplots_adjust(wspace=0.35)
    save(fig, "convergence")


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def bei(seed=0):
    """Posterior mean and variance on the bei data (Figure 13)."""
    ds = dataset("bei")
    x_train = ds.task(seed).x_train
    x_plot, domain = ds.grid, ds.domain
    titles = ["MPCP", "VBPP", "VLGCP"]
    samples = [seed_intensities("bei", t.lower(), seed) for t in titles]

    means = [np.mean(s, axis=0) for s in samples]
    variances = [np.maximum(np.var(s, axis=0), 1e-10) for s in samples]

    vmin = min(np.min(v[v > 0]) for v in variances)
    vmax = max(np.max(v) for v in variances)
    variance_levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)

    fig, axes = plt.subplots(2, 3, figsize=(15, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.08, hspace=0.05)
    for j, title in enumerate(titles):
        axes[0, j].tricontourf(x_plot[:, 0], x_plot[:, 1], means[j], levels=np.linspace(0, 0.02, 10),
                               cmap=truncated_cmap("viridis", 0.2), extend="both")
        axes[1, j].tricontourf(x_plot[:, 0], x_plot[:, 1], variances[j], levels=variance_levels,
                               cmap=truncated_cmap("inferno", 0.3), norm=LogNorm(vmin=vmin, vmax=vmax), extend="both")
        for ax in axes[:, j]:
            ax.scatter(x_train[:, 0], x_train[:, 1], s=5, c="black", alpha=1.0)
        axes[0, j].set_title(title, fontsize=22, pad=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(domain[0])
        ax.set_ylim(domain[1])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_linewidth(3)
    axes[0, 0].set_ylabel("Mean", fontsize=22, labelpad=20)
    axes[1, 0].set_ylabel("Variance", fontsize=22, labelpad=20)
    save(fig, "bei_mean_variance")


def _earthquake_maps(statistic, seed=0):
    """Mean or variance of the earthquake intensity at four times (Figures 5 and 14)."""
    import geopandas as gpd

    ds = dataset("earthquake")
    x_train = ds.task(seed).x_train
    x_plot = ds.grid
    methods = ["MPCP", "VBPP", "VLGCP"]
    samples = {m: seed_intensities("earthquake", m.lower(), seed) for m in methods}

    if statistic == "mean":
        reduce = lambda s: np.mean(s, axis=0)  # noqa: E731
        levels, norm, cmap = np.linspace(0, 7, 15), None, truncated_cmap("viridis", 0.20)
    else:
        reduce = lambda s: np.maximum(np.var(s, axis=0), 1e-6)  # noqa: E731
        levels, norm, cmap = np.geomspace(1e-3, 1000, 15), LogNorm(vmin=1e-3, vmax=1000), truncated_cmap("inferno", 0.3)

    timestamps = [2000.5, 2003.5, 2006.5, 2009.5]   # middle of each year
    lon_values, lat_values = np.unique(x_plot[:, 1]), np.unique(x_plot[:, 2])
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    (lon_min, lon_max), (lat_min, lat_max) = ds.domain[1], ds.domain[2]

    world = gpd.read_file(REPO_ROOT / "experiments" / "naturalearth_10m" / "ne_10m_admin_0_countries.shp")
    world_crop = world.to_crs("EPSG:4326").cx[lon_min:lon_max, lat_min:lat_max]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12), sharex=True, sharey=True)
    for row, method in enumerate(methods):
        for col, t in enumerate(timestamps):
            ax = axes[row, col]
            time_mask = np.isclose(x_plot[:, 0], t)
            x_grid = x_plot[time_mask]
            values = reduce(samples[method][:, time_mask])

            intensity_grid = np.full((len(lat_values), len(lon_values)), np.nan)
            for i, (lon, lat) in enumerate(zip(x_grid[:, 1], x_grid[:, 2])):
                intensity_grid[np.where(lat_values == lat)[0][0], np.where(lon_values == lon)[0][0]] = values[i]

            ax.contourf(lon_grid, lat_grid, intensity_grid, levels=levels, norm=norm, extend="both", cmap=cmap)
            world_crop.plot(ax=ax, color="white", edgecolor="black", linewidth=1.0, zorder=1, alpha=0.5)

            year = int(t)
            in_year = (x_train[:, 0] >= year) & (x_train[:, 0] < year + 1)
            ax.scatter(x_train[in_year, 1], x_train[in_year, 2], s=14, c="black", alpha=1.0, edgecolors="none")

            if row == 0:
                ax.set_title(f"{year}", fontsize=30, pad=8)
            if col == 0:
                ax.text(-0.08, 0.5, method, transform=ax.transAxes, fontsize=30,
                        rotation=90, va="center", ha="center")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
            ax.set_aspect("equal")
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)

    plt.tight_layout()
    save(fig, f"earthquake_{statistic}")


def earthquake_mean():
    """Figure 5."""
    _earthquake_maps("mean")


def earthquake_variance():
    """Figure 14."""
    _earthquake_maps("variance")


FIGURES = {
    "lambda1": lambda1, "coverage": coverage, "earthquake_mean": earthquake_mean, "copula": copula,
    "convergence": convergence, "lambda2": lambda2, "lengths": lengths, "bei": bei,
    "earthquake_variance": earthquake_variance,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", choices=list(FIGURES), default=list(FIGURES))
    for figure in parser.parse_args().figures:
        FIGURES[figure]()
