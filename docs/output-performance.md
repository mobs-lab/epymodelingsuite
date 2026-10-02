# Saving output without retaining all results

The existing `generate_calibration_outputs`, `generate_simulation_outputs` and
`dispatch_output_generator` APIs still return `OutputObject` dictionaries.
Their calibration quantile calculations now share summaries between tables,
Hub forecasts, and single/grid plots. epydemix is unchanged.

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
