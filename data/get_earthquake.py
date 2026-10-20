"""Download the earthquake data (data/earthquake.csv) from the USGS ANSS ComCat.

Events of type "earthquake" with magnitude > 4 in 2000-2009 in the box
122-150 E, 22-46 N (in and around Japan), as (x1, x2, x3) = (decimal year,
longitude, latitude).
Run from the repository root:

    python data/get_earthquake.py [--out data/earthquake.csv]

The catalogue is revised over time, so a fresh download can differ slightly
from the data used in the paper (8,955 events); a download in October 2026
reproduced data/earthquake.csv exactly.
"""
import argparse
import io
import urllib.parse
import urllib.request

import pandas as pd

QUERY = {
    "format": "csv",
    "starttime": "2000-01-01",
    "endtime": "2010-01-01",
    "minlongitude": 122, "maxlongitude": 150,
    "minlatitude": 22, "maxlatitude": 46,
    "minmagnitude": 4,
    "eventtype": "earthquake",   # excludes e.g. the 2006 and 2009 nuclear tests
    "orderby": "time-asc",
}
URL = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(QUERY)


def to_decimal_year(time):
    """Year plus the fraction of that year elapsed at ``time``."""
    start_of_year = pd.to_datetime(time.dt.year.astype(str) + "-01-01")
    start_of_next_year = pd.to_datetime((time.dt.year + 1).astype(str) + "-01-01")
    return time.dt.year + (time - start_of_year) / (start_of_next_year - start_of_year)


def main(out):
    with urllib.request.urlopen(URL) as response:
        df = pd.read_csv(io.StringIO(response.read().decode()))

    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(None)   # UTC, timezone-naive
    df = df[df["mag"] > 4]
    df = df[(df["time"].dt.year >= 2000) & (df["time"].dt.year <= 2009)]
    df = df.sort_values("time")

    earthquake = pd.DataFrame({"x1": to_decimal_year(df["time"]), "x2": df["longitude"], "x3": df["latitude"]})
    earthquake.to_csv(out, index=False)
    print(f"Saved {len(earthquake)} events to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/earthquake.csv")
    main(parser.parse_args().out)
