# Example: Bengaluru, three seeds

Run everything from the project root. Each step writes into a local folder that the next step reads.

## 1. Run the neural ODE

Train first, then continue every seed across the test period.

```bash
FEATS=T2M,PRECTOTCORR_SUM,QV2M,T2M_lag1,PRECTOTCORR_SUM_lag1,QV2M_lag1,T2M_lag2,PRECTOTCORR_SUM_lag2,QV2M_lag2

python fit_location_specs.py --location Bengaluru --start_date 2022-01-01 --test_steps 24 \
    --feats $FEATS  --seed_start 0 --n_inits 3 --alt_scenarios climatology forecast --mode train

python fit_location_specs.py --location Bengaluru --start_date 2022-01-01 --test_steps 24 \
    --feats $FEATS  --seed_start 0 --n_inits 3 --alt_scenarios climatology forecast --mode test
```

Output:
```
data_cache/Bengaluru_2022-01-01_t24/run_data.csv          the record every step reads
data_cache/Bengaluru_2022-01-01_t24/train_stats.json
checkpoints_local/Bengaluru_2022-01-01_t24/epi_v1/PSl0l1l2-Ql0l1l2-TMl0l1l2_h3_p0.1_w18/
    train/se_0  train/se_1  train/se_2                    final_fitter.pt + segment_diagnostics.xlsx each
    test/se_0   test/se_1   test/se_2
```

## 2. Export the ensemble

```bash
python export_ensemble.py \
    --run_root checkpoints_local/Bengaluru_2022-01-01_t24/epi_v1/PSl0l1l2-Ql0l1l2-TMl0l1l2_h3_p0.1_w18 \
    --alias bengaluru_example --n_seeds 3 --block 3
```


Output:
```
neural_epi_results/bengaluru_example/
    observed.xlsx  climatology.xlsx  forecast.xlsx
    identity.json
```


## 3. Run the baselines under the same epi setting

```bash
python -m baseline_models.run_baselines \
    --data_rel_path Bengaluru_2022-01-01_t24/run_data.csv --epi_tag epi_v1
```

Output:
```
baseline_results/Bengaluru_2022-01-01_t24/
    selection/form_summary_h6.xlsx
    test/forecasts_h6.xlsx
    test/summary_h6.xlsx
```

## 4. The beta and Re table

```python
from beta_climate import save_beta_climate_table

save_beta_climate_table(
    "checkpoints_local/Bengaluru_2022-01-01_t24/epi_v1/PSl0l1l2-Ql0l1l2-TMl0l1l2_h3_p0.1_w18",
    "results", n_seeds=3)
```

Writes `results/Bengaluru_2022-01-01_t24_beta_climate.csv`.

## 5. Neural ODE vs baselines figure

Reads the ensembles from step 2 and the baselines from step 3.

```python
from model_analysis.plot_compare import plot_compare

plot_compare("Bengaluru_2022-01-01_t24", "results/Bengaluru_2022-01-01_t24_compare.png", scenario="observed")
```

Left: reported cases with every model's forecast at 1, 3 and 6 months ahead.
Right: RMSE and MAE over lead times 1 to 6. `scenario` picks `observed`, `climatology` or `forecast` climate.