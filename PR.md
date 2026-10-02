# Fix DataFrame fragmentation warnings in output generation

Closes https://github.com/mobs-lab/epymodelingsuite/issues/215

Wide result frames can already be fragmented when they reach the output dispatcher. Inserting metadata columns individually then raises pandas `PerformanceWarning`, including during simulation quantile exports and calibration trajectory exports.

Replace the `DataFrame.insert()` calls in `dispatcher/output.py` with batched concatenation. A shared `prepend_metadata_columns()` helper builds metadata together, aligns it to the existing index, and rejects duplicate columns. FluSight fields inserted within the table are also concatenated in a batch. Existing output column order and values are preserved, and the helper leaves its input unchanged.

## Validation

Using pandas 3.0.1:

- The four wide-output regression cases failed on the original implementation with fragmentation warnings.
- **164 tests pass** with `PerformanceWarning` treated as an error, including 11 new regression cases and the existing simulation output pipeline test.
- Regression coverage includes simulation/calibration quantiles and trajectories, posterior generations, scalar/array/Series metadata, empty and duplicate indexes, duplicate-column rejection, unchanged source frames, and FluSight/Metrocast hospitalization and prop-ED formatting.
- `git diff --check` and Ruff formatting pass. Production Ruff diagnostic counts are unchanged from the base branch (103 existing findings).

```sh
python -m pytest \
  tests/dispatcher/test_output_metadata.py \
  tests/dispatcher/test_output.py \
  tests/dispatcher/test_output_location.py \
  tests/dispatcher/test_flusight_rate_trends.py \
  tests/integration/test_pipeline_integration.py::TestSimulationPipelineE2E::test_simulation_output_generation \
  -q -p no:sugar -W error::pandas.errors.PerformanceWarning
```

The full repository test suite was not run. This change batches metadata additions; it does not change how upstream result builders construct their data frames.
