# GTA VI launch-day Twitch forecast

[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tolehun/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tolehun/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on [make.fun](https://make.fun).

`main.py` (Python 3.13, standard library only) forecasts the concurrent
Twitch viewers in the "Grand Theft Auto VI" category for each hour 2–24
of release day (2026-11-19 UTC), as five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's count.

The model is fitted fresh from the case file on every run: for each
hour, an OLS regression in log space of that hour's viewers on the
launch's hour-1 viewers and hour-1 channel count, over the past big
launches in the case's `comparables` whose hour 1 was a real, live hour
(hour-1 viewers ≥ 20,000). The fitted line at the target's own hour-1
values is the log-median; the quantiles are log-normal around it with
the regression's residual standard deviation.

On the four practice cases this scores a mean pinball loss of about
0.221 on the log scale, against about 0.387 for the house baseline.
