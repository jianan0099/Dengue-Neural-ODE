# Climate-driven dengue early warning in India using neural ordinary differential equations

## Layout

```
fit_location_specs.py   fit one location / spec across seeds (train, then test)
export_ensemble.py      collect seed runs into ensemble forecast workbooks
beta_climate.py         per-seed beta, S and Re table for a fitted cell

dataloader/             cases + climate data per location
epi_bases/              fixed epi rates and beta / rho priors per location
models/                 SEIRS RHS, beta network, ODE module, segment fitter, losses
baseline_models/        seasonal_naive, sarima, glm, seirs_season
model_analysis/         ensembles, scoring, neural-vs-baseline figure
utils/                  paths, CLI args, epi base loading, helpers
```

Locations: `Goa`, `Bengaluru`, `SanJuan`. Climate features: `T2M`, `PRECTOTCORR_SUM`,
`QV2M`, plus `_lag1` / `_lag2` versions.

## Requirements

Install the dependencies with:

```
pip install -r requirements.txt
```

## Example

`example.md` is an end-to-end walkthrough on Bengaluru with three seeds: fitting the
neural ODE, exporting the ensemble, running the baselines, building the beta / Re table,
and plotting the comparison. Every command runs from the project root.
