# Postprocessing output formats

The postprocessing scripts produce two kinds of files:

1. **Dashboard parquet**: long-form strain trajectories, totals and optional baselines. This is the file sent to the dashboard.
2. **Submission CSV**: hubverse quantile (and for hosp, PMF) forecasts built from the aggregated parquet.

The scripts in this directory produce these files (commands at the end).
The dashboard trajectory contract is shared output infrastructure in `epymodelingsuite/output/dashboard_trajectories.py`; it does not require aggregation or sampling.

## 1. Dashboard trajectories (long-form Parquet)

**Filename:** `dashboard_trajectories_<config stem>.parquet` for `aggregate.py` output.
Single-strain and multistrain dashboard files use the same fixed eleven columns.
One row per (`sample_id`, `location`, `date`, `age_group`, `target`, `series`).

| column | Parquet type | description |
|---|---|---|
| `sample_id` | int64 | Sample index, `0 .. n_samples-1` per target; the same sample identifies the same source trajectories across ages and targets |
| `sim_id` | int64, nullable | Original trajectory ID on strain rows; null on total/baseline rows |
| `location` | string | epydemix population name, not hub location ID |
| `date` | timestamp[us] | Week end date, Saturday at midnight, no timezone |
| `age_group` | string | Model age-group label, or `all` for all ages; required and non-blank |
| `target` | string | Non-blank output variable identifier, such as `hospitalizations`, `ed_signal`, or a hub target name |
| `series` | string | Strain label, `total`, `baseline`, or `baseline_k<k>` |
| `value` | float64, nullable | Value of this series; missingness follows the rules below |
| `reference_date` | timestamp[us] | Forecast reference Saturday, constant within each target |
| `horizon` | int64 | `(date - reference_date)` in weeks; negative before the reference date |
| `epiweek` | string | CDC epiweek of `date`, `YYYYww` |

The file covers the whole trajectory. Filter on `horizon` for forecast weeks and on `series == "total"` for the strain sum within an age group. Select `age_group == "all"` for all ages.
All series must have a row for every sample/location/date/age_group/target, including strain rows whose value is NaN.
No target-specific values are stored in column names.

### Target names

Set `aggregation.target` explicitly. Use the model variable name (for example, `hospitalizations` or `ed_signal`) or a hub target name. The following hub names remain valid examples, not a required vocabulary.
`sources[].target_column` still identifies the internal input variable, such as `hospitalizations` or `ed_prop`.

| profile | target |
|---|---|
| FluSight hosp | `wk inc flu hosp` |
| FluSight ED | `wk inc flu prop ed visits` |
| Metrocast | `Flu ED visits pct` |
| BPHC ED | `wk inc ed signal` |

The validator requires a non-empty target name but does not hard-code a hub's changing task vocabulary.
Use one target per aggregation run. Multiple targets may be combined in a dashboard file; their series and values are checked separately. A source simulation ID must stay constant for each location/sample/strain across dates, age groups and targets.
Values retain the source variable's units; formatting does not rescale them.

### Age groups

Use the model's age-group labels without renaming them. For BPHC these are `0-4`,
`5-17`, `18-49`, `50-64`, and `65+`. Labels are not restricted to these five values.
The reserved label `all` means all ages. Null and blank age labels are invalid.

For `aggregate.py`, add `age_group_columns` to each source to map age labels to
input CSV columns. `target_column` supplies the `all` rows. Omitting
`age_group_columns` keeps the existing all-age output. For example:

```yaml
aggregation:
  bucket: gs://your-bucket/pipeline/flu
  submission_week: 202643
  target: hospitalizations
  random_seed: 1337
  sources:
    - strain: single
      experiment: your-experiment
      trajectory_file: trajectories_projection_transitions.csv.gz
      target_column: hospitalizations
      age_group_columns:
        "0-4": hospitalizations_0-4
        "5-17": hospitalizations_5-17
        "18-49": hospitalizations_18-49
        "50-64": hospitalizations_50-64
        "65+": hospitalizations_65+
  sampling: {method: random, n_samples: 1000}
  aggregate_method: sum
```

Replace the example column names with those in the input file. All sources must
specify the same age labels, but their input column names can differ. Each age
must use a distinct column, separate from `target_column`. Missing input columns
raise an error. Sampling happens once per strain and keeps the same source
trajectory across all ages. Aggregation sums strains separately for each age and
retains the supplied all-age values; it does not calculate all ages from the age
columns. `age_group_columns` cannot be combined with `baseline`, whose current
inputs describe only all-age noise. Submission and plot settings need no change.

A target may contain only `all`, only age-specific rows, or both. Within a target,
every sample/location/date/series must contain the same set of age labels.
Keep rows with missing values; do not replace missing values with zero.

When `all` and age-specific rows coexist, the age groups must form a complete,
non-overlapping partition. For each sample/location/date/target/series, `all` must
equal the sum of those ages (rtol `1e-10`, atol `1e-12`). If any age value is
missing, `all` must also be missing. The validator checks values and row coverage;
it does not parse age-label boundaries. Do not combine age-specific rates with an
all-age rate unless their values are additive in the stored units.

`series=total` sums strains within one age group. `age_group=all` combines ages
within one series. These are separate dimensions. The formatter does not compute
either sum. The producer supplies them and the validator checks them.

Use the same `sample_id` and strain `sim_id` for a trajectory across ages and
targets. For example, a producer can derive age-specific `ed_signal` by multiplying
age-specific `hospitalizations` by that trajectory's shared `alpha`. This is a
model-derived age breakdown, not a forecast fitted to age-specific ED data.

### Series and provenance

| run | strain series | derived series |
|---|---|---|
| Single strain | `single` | `total` |
| Flu multistrain | `h1`, `h3`, `b` | `total` |
| BPHC multistrain | `a`, `b` | `total` |

Strain labels come from `sources[].strain`. Custom labels are allowed except reserved total/baseline names; `single` must be used alone.
`total` sums the available strain values, ignoring NaN. If all strains are NaN, `total` is also NaN (`sum(skipna=True, min_count=1)`). A real zero remains zero, even when the other strains are NaN. Baseline series preserve exactly the missing values of `total`. For a single-strain run, retain both `single` and `total` rows, including missing values. Rows are not dropped; plotting may hide missing values or display them as zero without changing the saved data.
Baseline configurations add `baseline` (beta) or `baseline_k<k>` (negative binomial), with the same meaning as the previous `target_baseline*` columns. Omit `baseline` in the aggregation config to omit these rows.

The original simulation ID is on each strain row. To recover the contributors to a total/baseline row, select the strain rows with the same sample/location/date/age_group/target. No synthetic simulation ID is assigned to sums.

Example (weekly metadata omitted):

| sample_id | sim_id | location | date | age_group | target | series | value |
|---|---|---|---|---|---|---|---|
| 0 | 42 | United_States | 2026-10-10 | all | hospitalizations | single | 120 |
| 0 | null | United_States | 2026-10-10 | all | hospitalizations | total | 120 |

### File metadata

Parquet key/value metadata carries `schema_version` (required, currently the UTF-8 string `"1"`)
and `sources` (optional, a UTF-8 JSON array). These are not additional table columns.
Each source identifies a `target`/`strain` pair and may include `experiment`, `run_id`, and
`trajectory_file`. Omit unavailable fields; local producers need no experiment or run ID.
Source entries must refer to strain rows, with at most one entry per target/strain pair.

```json
[{"target": "wk inc flu hosp", "strain": "single", "experiment": "202640/hosp",
  "run_id": "20261007-182529-061e374e",
  "trajectory_file": "gs://bucket/202640/hosp/20261007-182529-061e374e/outputs/20261013/trajectories.csv.gz"}]
```

Aggregation records the resolved run ID and exact input URI, including the output directory.
`latest`, `any`, `*`, and `unknown` are not valid provenance run IDs. A source pattern must
match exactly one input file; ambiguous matches fail before downloading.
Callers supply provenance explicitly; the shared output functions do not inspect cloud environment variables.

### Shared schema, formatter and validator

`epymodelingsuite.output.dashboard_trajectories` provides:

- `DASHBOARD_TRAJECTORY_SCHEMA`: fixed Arrow column order and types above. Conditional null rules are checked by the validator.
- `format_dashboard_trajectories(df, strains, reference_date)`: requires `age_group` on long rows, recomputes weekly metadata, safely casts types and validates. Does not sample or aggregate. Returns a new Arrow-backed DataFrame; invalid data raises `ValueError` before writing.
- `dashboard_trajectories_from_wide(df, strains, reference_date, target)`: adapter for existing aggregate results/legacy files; preserves strain values (including NaN), source IDs and an existing `age_group` column. For legacy wide data without an age column, adds `age_group="all"`, then calls the shared formatter.
- `validate_dashboard_trajectories(df, strains)`: returns errors, or an empty list on success, without changing the input. Accepts string/large_string as equivalent logical strings.
- `write_dashboard_trajectories(df, path, sources=None)`: validates data and optional provenance, and writes versioned Parquet. If `sources` is omitted, preserves `df.attrs["sources"]` when present.
- `read_dashboard_trajectories(path)`: validates the version, data, and provenance and returns long rows. File metadata is available in `df.attrs["schema_version"]` and optional `df.attrs["sources"]`.

Use the shared writer rather than `DataFrame.to_parquet` to include required file metadata.
This schema is still under development before its first merge, so these changes
remain version 1. The reader requires the current eleven-column schema. To update
an earlier all-age export without `age_group`, load it with `pd.read_parquet`, add `age_group="all"`,
and order the columns with `DASHBOARD_TRAJECTORY_SCHEMA.names`. Save to a new path
with `write_dashboard_trajectories`. Pass any original `sources` metadata explicitly.
This validates the data, including the shared source-ID mapping across targets,
before writing. The row validator alone does not validate file metadata.

Validation checks exact columns/order/types, required values, unique keys, complete series per key, non-negative IDs, contiguous sample IDs, and stable source IDs within each location/sample/strain across ages and targets. Dates must be Saturdays at midnight with a single reference date per target and matching horizon/epiweek. Non-missing values must be finite and non-negative. Totals must match the sum of available strains (rtol `1e-10`, atol `1e-12`), including NaN exactly when all strains are NaN. Baseline series must be missing exactly where total is missing, and otherwise be at least total; beta and negative-binomial baseline series cannot be mixed within a target.

`aggregate.py` uses the adapter and writes only long-form dashboard output. Submission and plot readers also accept legacy wide files. Existing `target_column: target_total` selects `series: total`; `target_column: target_baseline_k10` selects `series: baseline_k10`. The shorter `target_column: total` is also accepted for long files. If a file contains multiple targets, set `aggregated.target` to the stored variable identifier to avoid mixing them. Submission and plot readers select `age_group="all"` and fail if it is absent, so age rows cannot be mixed into all-age quantiles. Hub submission target names still come from the submission profile.

A future direct single-strain output producer can supply long rows to `format_dashboard_trajectories`, without invoking aggregation. Dispatcher integration is not yet implemented. Hub submission schema/validation remain separate.

```python
from epymodelingsuite.output.dashboard_trajectories import (
    format_dashboard_trajectories,
    read_dashboard_trajectories,
    write_dashboard_trajectories,
)

# rows contains sample_id, sim_id, location, date, age_group, target, series, value.
dashboard = format_dashboard_trajectories(rows, ["single"], "2026-10-10")
write_dashboard_trajectories(dashboard, "dashboard.parquet")  # No cloud metadata needed.
dashboard = read_dashboard_trajectories("dashboard.parquet")
```

### Location vocabulary

| target | examples |
|---|---|
| FluSight (hosp, ED) | `United_States`, `United_States_Massachusetts` (older runs) or `United_States__Massachusetts` |
| Metrocast | `metrocast_location_boston`, `metrocast_location_denver` |
| BPHC | `metrocast_location_boston-all` |

Submission files convert these to hub location IDs.

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
uv run $REPO/scripts/postprocess/aggregate.py --config 202616_hosp.yaml --output outputs/202616_hosp

# 2. Aggregated parquet -> submission CSV
uv run $REPO/scripts/postprocess/submission.py --config 202616_hosp_submission.yaml \
    --aggregated outputs/202616_hosp/dashboard_trajectories_202616_hosp.parquet --output submissions/202616

# 3. Aggregated parquet -> comparison plots (optional)
uv run $REPO/scripts/postprocess/plot.py --config 202616_hosp_plot.yaml \
    --aggregated outputs/ --output outputs/202616_hosp/plots
```

### Config examples

Aggregation (`aggregation:` root, see `epymodelingsuite/schema/aggregation.py`):

```yaml
aggregation:
  bucket: gs://gs_mobs_jessica/pipeline/flu
  submission_week: 202616
  target: wk inc flu hosp
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
    target: wk inc flu hosp          # must match the stored target; required for multiple targets
    target_column: total
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
      trajectory_file: 202616_hosp/dashboard_trajectories_202616_hosp.parquet
      target_column: target_total
    multistrain_baseline:
      trajectory_file: 202616_hosp/dashboard_trajectories_202616_hosp.parquet
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
