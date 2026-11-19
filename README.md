# makefun-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/thanhvu261090/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/thanhvu261090/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC as five quantiles (5%, 25%, 50%, 75%, 95%), from the house's recorded reads of the trophy page. Everything is modelled in holder counts, not rates: a trophy's holders only ever increase, while its rate moves because the tracked population jumps at a refresh (dilution) and because holders keep earning (progression). When the page is frozen, the forecast anticipates the next refresh as a count process — newcomers' holdings of each trophy are Poisson with a mean set by how fast that trophy is earned and by the launch-night aggregate of about 0.032 other trophies per newcomer — mixed over a small grid of population multiples, so rare trophies get the narrow, asymmetric bands a small integer count of newcomers really implies instead of a wide symmetric spread. When the page has refreshed before, the population and the counts are projected in log time, the newcomers' holdings at future refreshes are estimated from the observed refresh jumps shrunk toward the same type prior, and the spread combines the fit's residual with the Poisson noise on the counts and the population uncertainty.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Runs are deterministic: two runs on the same case write byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=1xmptn07&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
