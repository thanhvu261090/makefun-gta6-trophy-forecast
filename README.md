# makefun-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/thanhvu261090/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/thanhvu261090/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC as five quantiles (5%, 25%, 50%, 75%, 95%), from the house's recorded reads of the trophy page. When the page's rates are moving it extrapolates each trophy's logit rate against log time since the first read; when the page is frozen it anticipates the next refresh, which brings in new players who have earned each trophy at only a fraction of the early cohort's rate, and dilutes the rates accordingly.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Runs are deterministic: two runs on the same case write byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=1xmptn07&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
