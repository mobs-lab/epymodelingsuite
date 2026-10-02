# Saving output without retaining all results

The existing `generate_calibration_outputs`, `generate_simulation_outputs` and
`dispatch_output_generator` APIs still return `OutputObject` dictionaries.
Their calibration quantile calculations now share summaries between tables,
Hub forecasts, and single/grid plots. epydemix is unchanged.

New output logic lives in `epymodelingsuite/output/`: variable selection in
`quantiles.py`, shared preparation in `shared_quantiles.py`, and file saving
in `streaming.py`. Plot data preparation stays in `visualization/`. The
existing dispatcher imports these implementations and re-exports the save APIs.

For file output, configure `tabular_output_types: [CSVBytes]` and figure formats
`PNG`, `PDF`, or `SVG`, then pass an iterator to `write_outputs`:

```python
from epymodelingsuite.output.streaming import write_outputs

saved = write_outputs(
    results=result_iterator,
    output_config=output_config,
    directory="output",
    chunk_rows=10000,
)
# saved: dict[str, list[pathlib.Path]]
```

Yield one `CalibrationOutput` or `SimulationOutput` at a time, from the runner
or from a file-loading iterator. Do not mix these types. Passing a preloaded
list cannot free the results still held by that list. DataFrame, MPLFigure and
Parquet saving requests are rejected before consuming the iterator.

Raw results live for one input; trajectories are converted in bounded row
chunks without a full stack or a full output table. Temporary pandas shards
retain column types and identifiers, including leading zeros. Final gzip CSVs
use the input order and pandas' union of columns. Individual images use the
existing names: the last successful image wins for duplicate populations.

Hub quantiles, prop ED fitting/rescaling, metadata and quantile grid data still
grow with the number of locations, but contain summaries rather than raw
draws. Posterior grid samples are staged on disk and read one location at a
time by the existing histogram renderer. The final grid figure itself still
needs memory proportional to its number of panels.

Each completed file replaces its destination atomically. This is not a
transaction covering every file. `OutputWriteError.completed` lists the files
already published before a failure; other existing destination files remain
unchanged. Temporary files are cleaned up on success and on exceptions.
Existing per-output warnings/skips are retained, while disk failures propagate.

Surveillance reads are shared within an invocation. Parser options remain
part of the cache key: the existing plot and Hub date/identifier inference can
require two representations of the same CSV. No process-global result cache
is introduced, and neither summaries nor the raw input arrays are downcast.

## Multiple CPU cores

Use one trusted pickle containing one result object per file:

```python
import pickle
from epymodelingsuite.output.streaming import write_outputs_from_files

if __name__ == "__main__":  # Required when launching spawn workers from a script.
    # Save each runner result separately; do not pickle a list of all results.
    with open("result-0.pickle", "wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)

    saved = write_outputs_from_files(
        paths=result_paths,
        output_config=output_config,
        directory="output",
        workers=2,
    )
```

Only unpickle files you trust: pickle can execute code. This is an explicit
one-result file contract; there is no universal result-file loader in the
existing repository. Files containing an all-location list are rejected.

`workers=1` runs directly without a process pool. More workers use
`ProcessPoolExecutor` with `spawn` and Matplotlib's Agg backend. Only paths and
configuration cross the process boundary. At most `workers` tasks are submitted
at once; workers return staged paths. The parent assembles CSVs in input order
and draws the global grids, regardless of completion order. A worker error
names the input, stops new submissions, and waits for active workers before
cleaning staging files. This API does not launch simulation/calibration pools.

Workers each need memory for one result plus their plotting libraries. Increase
the count only within the machine's RAM budget. Global grids and final writes
remain serial; process startup and I/O can dominate small jobs. The benchmark
reports the RSS sum of the parent and all children, including shared pages that
can be counted more than once. It also records numeric-library thread settings.

```bash
python scripts/benchmark_output.py --mode files --format CSVBytes --workers 2 \
  --locations 32 --cases tabular hub plots --no-grid --repeat 3
python scripts/benchmark_output.py --mode stream --format CSVBytes \
  --locations 16 --draws 8000 --cases tabular --repeat 3
```

## Measured results

Three fresh-process runs per setting; medians on the development machine
(16 available CPUs). NumPy 2.4.2, pandas 3.0.1, Matplotlib 3.10.8 and epydemix
1.3.2. Numeric-library thread environment variables were unset.

32 locations, each with 2,000 draws, 52 dates, eight float64 variables, 23 levels,
and about 1% NaNs. Outputs include quantile tables, Hub forecasts, metadata and
three single-location PNGs per location; grids are disabled in this measurement.
All workers use the same one-result files. Output time includes file reading,
pool startup, processing and final assembly; input-file creation is separate.

| Workers | Output time | Process-tree peak RSS |
| --- | ---: | ---: |
| 1 | 19.19 s | 267.4 MiB |
| 2 | 12.20 s | 725.5 MiB |
| 4 | 8.36 s | 1,244.6 MiB |

CSV content hashes match at all worker counts. Actual spawn tests also compare
PNG bytes with the serial path and force reversed completion order.

For memory scaling, the same tabular configuration with 8,000 draws per location:

| Locations | Preloaded results and CSVBytes return | Lazy input and file saving |
| --- | ---: | ---: |
| 4 | 357.9 MiB | 239.2 MiB |
| 16 | 845.1 MiB | 239.9 MiB |

CSV hashes match for each location count. RSS includes imports, synthetic input
generation, output and hash verification, sampled every 10 ms. The lazy benchmark
constructs its synthetic inputs during consumption; its output time therefore
also includes construction and must not be compared directly with the preloaded
path's output-only time. Worker RSS includes duplicated/shared library pages.
These measurements describe these inputs, rather than a fixed speed or memory
guarantee. Quantile grids, large surveillance files and figures can still grow.

Tests cover generation/scenario/NaN isolation, shared computation, exact CSV
content (including heterogeneous columns and nonnumeric trajectory metadata),
PNG equivalence, posterior grid samples released between locations, duplicate
populations, failed-projection alignment, lazy input release, bounded row chunks
and task submissions, real spawn processes, and write/worker failures.

Final validation after moving new output logic into `output/` (based on PR275
`bb2f9e5`): `python -m pytest -q` — 1,280 passed, five nightly tests deselected,
and two existing deprecation warnings (254.26 seconds). New modules and tests pass
Ruff E/F/I and formatting checks; `git diff --check` passes.
