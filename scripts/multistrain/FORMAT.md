# Multistrain output formats

Two kinds of files come out of the multistrain pipeline:

1. **Aggregated parquet**: per-sample trajectories summed across strains. This is the file sent to the dashboard.
2. **Submission CSV**: hubverse quantile (and for hosp, PMF) forecasts built from the aggregated parquet.

Both are produced only by the scripts in this directory (commands at the end). If you get a file with a different layout, it was not built by these scripts.

## 1. Aggregated parquet (dashboard file)

**Filename:** `trajectories_aggregated_<config stem>.parquet`, where `<config stem>` is the aggregation config filename without `.yaml`.
For example, `202616_hosp_aug_smc_wrmse.yaml` → `trajectories_aggregated_202616_hosp_aug_smc_wrmse.parquet`.

One row per (`sample_id`, `location`, `date`).

| column | dtype | description |
|---|---|---|
| `sample_id` | int64 | Aggregated sample index, `0 .. n_samples-1` |
| `sim_id_<strain>` | int64 | Trajectory of `<strain>` used for this sample. One column per strain |
| `location` | string | epydemix population name, see below |
| `date` | datetime64[us] | Week end date (Saturday) |
| `target_<strain>` | float64 | Strain value. One column per strain. May be NaN before the strain's fitting window starts |
| `target_total` | float64 | Sum over strains (NaN treated as 0) |
| `target_baseline_k<k>` | float64 | Only when the config has a `baseline`: `target_total` plus baseline noise with mean from `baseline.observed_means` and dispersion/concentration `k` (`negative_binomial` for counts; `beta` for proportions is a stub that raises `NotImplementedError`). Drawn with `random_seed`. One column per `k` |
| `reference_date` | datetime64[us] | Saturday of the submission epiweek. Same for every row |
| `horizon` | int64 | `(date - reference_date)` in weeks. Negative before `reference_date`, `0` at it |
| `epiweek` | string | CDC epiweek of `date`, `YYYYww` |

The file covers the whole trajectory, not only the forecast horizons. Filter on `horizon` when you need the forecast weeks.

Column order is: `sample_id`, the `sim_id_<strain>` columns, `location`, `date`, the `target_<strain>` columns, `target_total`, the `target_baseline_k<k>` columns (if any), `reference_date`, `horizon`, `epiweek`.

### Strain labels

Strain labels come from `sources[].strain` in the aggregation config. They are fixed per pathogen:

| pathogen | labels | strain columns |
|---|---|---|
| flu (hosp, ED, metrocast) | `h1`, `h3`, `b` | `sim_id_h1`, `sim_id_h3`, `sim_id_b`, `target_h1`, `target_h3`, `target_b` |
| BPHC | `a`, `b` | `sim_id_a`, `sim_id_b`, `target_a`, `target_b` |

### Location vocabulary

`location` is the epydemix population name that comes out of the model run:

| target | examples |
|---|---|
| FluSight (hosp, ED) | `United_States`, `United_States_Massachusetts` (older runs) or `United_States__Massachusetts` (epydemix ≥ 1.2) |
| Metrocast | `metrocast_location_boston`, `metrocast_location_denver` |
| BPHC | `metrocast_location_boston-all` |

Submission files convert these to hub location ids (next section).

## 2. Submission CSV

**Filename:** `<reference_date>-<model_name>.csv`, where `model_name` is `<team>-<model>` from the submission config (e.g. `2026-04-25-MOBS-EpyStrain_Flu.csv`).

The 8 hubverse columns, sorted by `location`, `target`, `horizon`, `output_type`, `output_type_id`:

| column | description |
|---|---|
| `reference_date` | Saturday of the submission epiweek, `YYYY-MM-DD` |
| `horizon` | Weeks from `reference_date` |
| `target_end_date` | `reference_date + horizon` weeks, `YYYY-MM-DD` |
| `location` | Hub location id (see profiles) |
| `target` | Target name (see profiles) |
| `output_type` | `quantile`, `pmf`, or `sample` |
| `output_type_id` | Quantile level (e.g. `0.025`), rate-trend category for `pmf`, or sample id for `sample` |
| `value` | Quantile value (rounded, see profiles), probability for `pmf`, or trajectory value for `sample` |

### Profiles

The config's `profile` picks the hub format. Everything that differs between hubs is in this table (`SUBMISSION_PROFILES` in `epymodelingsuite/multistrain/formatter.py`).

| profile | `location` | `target` | decimals | horizons | quantiles | pmf |
|---|---|---|---|---|---|---|
| `flusight_hosp` | FIPS code (`US`, `01`, … `72`) | `wk inc flu hosp` | 0 | -1..3 | 23 (FluSight) | yes |
| `flusight_ed` | FIPS code | `wk inc flu prop ed visits` | 3 | -1..3 | 23 (FluSight) | no |
| `metrocast` | metrocast id (`boston`, `denver`, …) | `Flu ED visits pct` | 3 | 0..3 | 9 (Metrocast) | no |
| `bphc_ed` | metrocast id (`boston-all`) | `wk inc ed signal` | 3 | -1..3 | 23 (FluSight) | no |

- FluSight quantiles: `0.01, 0.025, 0.05, 0.1, 0.15, …, 0.9, 0.95, 0.975, 0.99`.
- Metrocast quantiles: `0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975`.
- `flusight_hosp` pmf rows: target `wk flu hosp rate change`, horizons 0..3, `output_type_id` in `large_decrease`, `decrease`, `stable`, `increase`, `large_increase`. The baseline is the observed value at `reference_date - 1 week` from the surveillance file, so this profile needs `surveillance` in its config. The other profiles don't.
- `sample` rows are only written when the config has `trajectory_samples`. Up to `trajectory_samples.n_samples` trajectories per location are picked (`select_trajectory_indices` in `epymodelingsuite/output/trajectory_samples.py`) from the same `aggregated.target_column` as the quantiles, over the profile's horizons. Trajectories missing any horizon are never picked, and fewer complete trajectories means fewer samples (with a warning). Values follow the single-strain FluSight output, not the profile's decimals: `wk inc flu hosp` is rounded to integers, ED percentages and proportions are clipped to their range, and `wk inc ed signal` is left as is. Sample ids are `<state abbreviation>00`, `01`, … for `flusight_*` and `1`, `2`, … per location for `metrocast` / `bphc_ed`, so samples are independent across locations. Selection is unseeded unless `trajectory_samples.seed` is set.

## Commands

Scripts run from anywhere with `uv` and no conda env. They use the checked-out `epymodelingsuite` source.

```bash
REPO=~/Developer/epymodelingsuite

# 1. Per-strain GCS runs -> aggregated parquet (needs gcloud auth)
uv run $REPO/scripts/multistrain/aggregate.py --config 202616_hosp.yaml --output outputs/202616_hosp

# 2. Aggregated parquet -> submission CSV
uv run $REPO/scripts/multistrain/submission.py --config 202616_hosp_submission.yaml \
    --aggregated outputs/202616_hosp/trajectories_aggregated_202616_hosp.parquet --output submissions/202616

# 3. Aggregated parquet -> comparison plots (optional)
uv run $REPO/scripts/multistrain/plot.py --config 202616_hosp_plot.yaml \
    --aggregated outputs/ --output outputs/202616_hosp/plots
```

### Config examples

Aggregation (`aggregation:` root, see `epymodelingsuite/schema/aggregation.py`):

```yaml
aggregation:
  bucket: gs://gs_mobs_jessica/pipeline/flu
  submission_week: 202616
  random_seed: 1337
  sources:
  - {strain: h1, experiment: 202616/hosp_aug_smc_wrmse_202550-202615_h1, trajectory_file: trajectories_projection_transitions.csv.gz, target_column: hospitalizations}
  - {strain: h3, experiment: 202616/hosp_aug_smc_wrmse_202550-202615_h3, trajectory_file: trajectories_projection_transitions.csv.gz, target_column: hospitalizations}
  - {strain: b,  experiment: 202616/hosp_aug_smc_wrmse_202550-202615_b,  trajectory_file: trajectories_projection_transitions.csv.gz, target_column: hospitalizations}
  sampling: {method: random, n_samples: 1000}
  aggregate_method: sum
```

Submission (`submission:` root, see `epymodelingsuite/schema/submission.py`):

```yaml
submission:
  submission_week: 202616
  profile: flusight_hosp
  model_name: MOBS-EpyStrain_Flu
  aggregated:
    target_column: target_total
  surveillance:                       # flusight_hosp only
    directory: ../flu-forecast-epydemix/common-data/surveillance
    fit_fname: flu_hosp_25_202616_prelim.csv
    location_column: location_code    # column holding hub location ids (FIPS here)
  trajectory_samples:                 # optional, adds 'sample' rows
    n_samples: 100
    seed: 1337
```

Plot (`plot:` root, see `epymodelingsuite/schema/plot.py`). `surveillance` and `single_strain` are optional.
Each label under `aggregated` is read from `--aggregated` joined with its `trajectory_file`, and all labels are overlaid in one plot:

```yaml
plot:
  submission_week: 202616
  profile: flusight_hosp               # panels are labelled with this profile's location ids
  aggregated:
    multistrain:
      trajectory_file: 202616_hosp/trajectories_aggregated_202616_hosp.parquet
      target_column: target_total
    multistrain_baseline:
      trajectory_file: 202616_hosp/trajectories_aggregated_202616_hosp.parquet
      target_column: target_baseline_k10
  subplots_per_row: 4                  # default 4
  surveillance:
    directory: ../flu-forecast-epydemix/common-data/surveillance
    fit_fname: flu_hosp_25_202616_prelim.csv
    recent_fname: flu_hosp_25_202617_prelim.csv
    location_column: location_code
  season_start_week: 202540
  season_end_week: 202620
  focus_start_week: 202550
  focus_end_week: 202620
  strain_fit_starts:
  - {label: h1/h3/b, week: 202550}
```

Surveillance `location_column` must hold the hub location ids of the profile: `location_code` in `flu_hosp_*` files, `location` in `flu_prop-ed_*` and `metrocast_*` files. ED and metrocast files also need `date_column: date` and `target_column: value`.
