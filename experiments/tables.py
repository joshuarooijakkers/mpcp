"""Print the tables of the paper from the results in ``logs/``.

Usage (from the repository root)::

    python experiments/tables.py              # all tables
    python experiments/tables.py table1 table3

Tables: ``table1`` (synthetic data), ``table2`` (real data), ``table3``
(covariate ordering) and ``ablation`` (the numbers behind Figure 15).
Entries are mean +- standard error across the seeds that were run (column
"Seeds"); missing runs are skipped.
"""
import argparse

import numpy as np

from results import dataset, load_run, missing
from mpcp.evaluation import l2_error, mean_and_se


def pm(values, fmt="{:.3g}"):
    mean, se = mean_and_se(values)
    return f"{fmt.format(mean)} ± {se:.2g}"


def print_table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def table1():
    """L2, test log-likelihood and runtime on the synthetic data (Table 1)."""
    for name, methods in [("lambda1", ["lgcp", "dpmm", "vlgcp", "vbpp", "mpcp"]),
                          ("lambda2", ["lgcp", "vlgcp", "vbpp", "mpcp"])]:  # no 2D DPMM
        ds = dataset(name)
        rows = []
        for method in methods:
            run = load_run(name, method)
            if run is None:
                missing(name, method)
                continue
            rows.append([method.upper(), pm(l2_error(run["intensities"], ds.true_intensity), "{:.2f}"),
                         pm(run["test_loglik"], "{:.1f}"), pm(run["runtime"]), len(run["seeds"])])
        print_table(f"Table 1 ({name})", ["", "L2", "L_test", "Runtime (s)", "Seeds"], rows)


def table2():
    """Test log-likelihood and runtime on the real data (Table 2)."""
    for name in ["bei", "earthquake"]:
        rows = []
        for method in ["vlgcp", "vbpp", "mpcp"]:
            run = load_run(name, method)
            if run is None:
                missing(name, method)
                continue
            rows.append([method.upper(), pm(run["test_loglik"], "{:.1f}"), pm(run["runtime"]), len(run["seeds"])])
        print_table(f"Table 2 ({name})", ["", "L_test", "Runtime (s)", "Seeds"], rows)


def table3():
    """MPCP on lambda2 with the covariates in the original and reversed order (Table 3)."""
    ds = dataset("lambda2")
    rows = []
    for label, run_name in [("(x, x')", "mpcp"), ("(x', x)", "mpcp_reversed")]:
        run = load_run("lambda2", run_name)
        if run is None:
            missing("lambda2", run_name)
            continue
        rows.append([label, pm(l2_error(run["intensities"], ds.true_intensity), "{:.2f}"),
                     pm(run["test_loglik"], "{:.1f}")])
    print_table("Table 3 (lambda2, covariate order)", ["Order", "L2", "L_test"], rows)


def ablation():
    """Test log-likelihood of VBPP and VLGCP against sqrt(M) (Figure 15)."""
    for name in ["bei", "earthquake2d"]:
        rows = []
        mpcp = load_run(name, "mpcp")
        if mpcp is not None:
            rows.append(["MPCP", "-", pm(mpcp["test_loglik"], "{:.1f}")])
        for method in ["vbpp", "vlgcp"]:
            for M in range(8, 25):
                run = load_run(name, f"{method}_M{M}")
                if run is not None:
                    rows.append([method.upper(), M, pm(run["test_loglik"], "{:.1f}")])
        if len(rows) <= 1:
            missing(name, "vbpp_M*/vlgcp_M*")
        print_table(f"Ablation ({name})", ["", "sqrt(M)", "L_test"], rows)


TABLES = {"table1": table1, "table2": table2, "table3": table3, "ablation": ablation}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="*", choices=list(TABLES), default=list(TABLES))
    for table in parser.parse_args().tables:
        TABLES[table]()
