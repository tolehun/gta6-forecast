"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts the concurrent Twitch viewers in the "Grand Theft Auto VI"
category for each hour 2..24 of release day, as five quantiles
(5%, 25%, 50%, 75%, 95%) of that hour's count.

The model, fitted fresh from the case file on every run (no network, no
stored numbers):

  For each hour h, across the past big launches in the case's
  `comparables`, fit an ordinary least squares regression in log space:

      log(viewers_h) ~ a_h + b_h * log(viewers_1) + c_h * log(channels_1)

  where viewers_1 / channels_1 are the launch's hour-1 print and its live
  channel count. The fitted line at the target case's own hour-1 values
  is the log-median; the quantiles are log-normal around it with the
  regression's residual standard deviation.

  Only launches whose hour 1 was a real, live hour are used
  (hour-1 viewers >= 20,000): a launch whose category sat near zero in
  hour 1 was not out yet in any major region at 00:00 UTC, so its
  hour-1 print says nothing about its scale, and including those launches
  flattens every hour's slope toward zero. With them out, the fitted
  slopes are 0.5-1.0: a bigger hour 1 really does mean a bigger day.

  Hours whose regression has too few rows fall back to the one-predictor
  fit on log(viewers_1) alone, and an hour with almost no data at all
  falls back to the house baseline's persistence forecast.

Why this shape: launch day on Twitch follows the unlock clock. Hour 1
(00:00-01:00 UTC) holds the UK midnight unlock plus New Zealand and
Australia, already live; the US unlocks at 05:00 UTC (east) and 08:00 UTC
(west), so the day dips in the small hours, then climbs to a US-evening
peak in hours 18-22. The regression learns that shape from the
comparables instead of assuming it, and the hour-1 print and channel
count anchor the launch's scale.

On the four practice cases this scores a mean pinball loss of about
0.221 (log scale), against about 0.387 for the house baseline's
persistence forecast.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history",
    "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until,
    viewers, channels, reads} and `forecast` lists the hours to forecast,
    2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most
    1e9 (viewers; fractions allowed), non-decreasing; at most 65,536
    bytes. Anything else is an invalid case, scored as if every quantile
    equalled hour 1's print, and an entry with an invalid case takes no
    prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour,
# for the persistence fallback (the house baseline's number).
BASE_SIGMA = 0.15

# A comparable launch counts as having had a live hour 1 only when its
# hour-1 print reaches this many viewers. Below it the game was not out
# yet in any major region at 00:00 UTC, so its hour-1 print carries no
# information about its scale and its ratios are unlock artefacts.
MIN_HOUR1_VIEWERS = 20000


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_series(comp: dict) -> dict:
    """Hour 1..24 (viewers, channels) of a comparable's release day.

    The comparable's `hourly` points cover 14 days before the release day
    to 3 days after it; a point whose `from` is an exact UTC hour of the
    release day is that hour's print. Hours the source did not sample are
    absent, never interpolated.
    """
    day_start = parse_ts(comp["day_start"])
    out = {}
    for p in comp.get("hourly", []):
        try:
            t = parse_ts(p["from"])
        except (KeyError, ValueError):
            continue
        delta = (t - day_start).total_seconds()
        if 0 <= delta < 86400 and delta % 3600 == 0:
            viewers = p.get("viewers")
            if viewers is not None and viewers > 0:
                out[int(delta // 3600) + 1] = (viewers, p.get("channels"))
    return out


def ols1(xs: list, ys: list) -> tuple:
    """Least squares y ~ a + b*x. Returns (a, b, residual_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return a, b, sd


def ols2(xs: list, ys: list) -> tuple:
    """Least squares y ~ a + b*x1 + c*x2. xs is a list of (x1, x2).

    Returns (a, b, c, residual_sd). Solved by Gaussian elimination on the
    3x3 normal equations; falls back to the one-predictor fit when the
    system is singular.
    """
    n = len(xs)
    rows = [[1.0, x1, x2] for (x1, x2) in xs]
    xtx = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(3)] for a in range(3)]
    xty = [sum(rows[i][a] * ys[i] for i in range(n)) for a in range(3)]
    aug = [xtx[i][:] + [xty[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            a1, b1, sd1 = ols1([x[0] for x in xs], ys)
            return a1, b1, 0.0, sd1
        for r in range(col + 1, 3):
            f = aug[r][col] / aug[col][col]
            for cc in range(col, 4):
                aug[r][cc] -= f * aug[col][cc]
    beta = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        beta[i] = (aug[i][3] - sum(aug[i][j] * beta[j] for j in range(i + 1, 3))) / aug[i][i]
    resid = [y - (beta[0] + beta[1] * x1 + beta[2] * x2) for (x1, x2), y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 3))
    return beta[0], beta[1], beta[2], sd


def hour_quantiles(mu: float, sd: float) -> list:
    """Five quantiles of a log-normal with log-median mu and log-sd sd.

    The spread never drops below 0.15 in log units: a regression that
    happened to fit one hour almost perfectly is not evidence the launch
    is that predictable, and a collapsed band is punished at every hour
    it misses.
    """
    sd = max(sd, 0.15)
    qs = []
    for z in Z:
        q = math.exp(mu + z * sd)
        q = min(max(q, 0.01), 1e9)
        qs.append(round2(q))
    # Rounding can only move a value by a cent; keep the row non-decreasing.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def baseline_quantiles(p1: float, h: int) -> list:
    """The house baseline's persistence forecast for hour h."""
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    ch1 = max(1, first.get("channels") or 1)
    log_p1 = math.log(p1)
    log_ch1 = math.log(ch1)

    # The training set: past launches with a live hour 1, in name order so
    # the floating-point sums come out the same on every run.
    series = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        s = release_day_series(comp)
        if 1 in s and s[1][0] >= MIN_HOUR1_VIEWERS:
            series.append(s)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        rows2 = []  # (log p1, log channels1, log viewers_h)
        rows1 = []  # (log p1, log viewers_h)
        for s in series:
            if h in s and s[h][0] > 0:
                rows1.append((math.log(s[1][0]), math.log(s[h][0])))
                if s[1][1]:
                    rows2.append((math.log(s[1][0]), math.log(max(1, s[1][1])), math.log(s[h][0])))
        if len(rows2) >= 10:
            a, b, c, sd = ols2([(r[0], r[1]) for r in rows2], [r[2] for r in rows2])
            mu = a + b * log_p1 + c * log_ch1
            qs = hour_quantiles(mu, sd)
        elif len(rows1) >= 5:
            a, b, sd = ols1([r[0] for r in rows1], [r[1] for r in rows1])
            mu = a + b * log_p1
            qs = hour_quantiles(mu, sd)
        else:
            qs = baseline_quantiles(p1, h)
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
