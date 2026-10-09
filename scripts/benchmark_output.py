"""Measure output time and sampled RSS in a fresh process per repetition.

Compare the same inputs against another checkout with --source-tree. Stage
timings are inclusive (plotting includes quantiles/serialization), not additive.
Metadata is included in every case, as required by the current generator.
Example: python scripts/benchmark_output.py --cases tabular hub plots --repeat 3
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import psutil


def positive_int(value):
    """Parse a positive integer command-line argument.

    Parameters
    ----------
    value : str
        Argument text supplied by argparse.

    Returns
    -------
    int
        Parsed integer greater than zero.

    Raises
    ------
    argparse.ArgumentTypeError
        The parsed integer is not positive.
    ValueError
        The text cannot be parsed as an integer.
    """
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def run_once(args):
    """Benchmark one synthetic output workload in a fresh child process.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed workload, source-tree and format options.

    Returns
    -------
    dict
        JSON-compatible dependency versions, preparation/output timings, parent call counts, output keys and table hash.


    Notes
    -----
    Synthetic input preparation is measured separately from output processing.
    Instrumentation covers the child process running this workload.
    """
    sys.path.insert(0, str(args.source_tree.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    from epydemix.calibration import CalibrationResults

    from epymodelingsuite.dispatcher import output
    from epymodelingsuite.schema.dispatcher import CalibrationOutput
    from epymodelingsuite.schema.output import OutputConfig
    from epymodelingsuite.utils.location import get_location_codebook
    from epymodelingsuite.visualization import generators

    started = time.perf_counter()
    dates = pd.date_range("2024-01-06", periods=args.dates, freq="W-SAT").to_list()
    levels = np.linspace(0.01, 0.99, args.levels).tolist()
    rng = np.random.default_rng(123)
    calibrations = []
    codebook = get_location_codebook()
    populations = codebook.loc[codebook.ISO.str.startswith("US-"), "location_name_epydemix"].tolist()
    if args.locations > len(populations):
        msg = f"Use at most {len(populations)} locations to avoid duplicate-location plot keys."
        raise ValueError(msg)
    for location in range(args.locations):
        values = rng.normal(100, 15, (args.draws, args.dates, args.variables))
        values[rng.random(values.shape) < 0.01] = np.nan
        projections = [
            {
                "date": dates,
                "hospitalizations": draw[:, 0],
                **{f"S_to_I_{i}": draw[:, i] for i in range(1, args.variables)},
            }
            for draw in values
        ]
        selected = [{"date": dates, "data": p["hospitalizations"]} for p in projections]
        results = CalibrationResults(selected_trajectories={0: selected}, projections={"baseline": projections})
        calibrations.append(
            CalibrationOutput(
                primary_id=location,
                seed=123,
                population=populations[location],
                results=results,
            )
        )
    config = {"tabular_output_types": [args.format]}
    if "tabular" in args.cases:
        config["quantiles"] = {"selections": levels, "compartments": ["hospitalizations"], "calibration": True}
    if "hub" in args.cases:
        config["flusight_format"] = {"reference_date": "2024-01-13", "quantiles": levels}
    if "trajectories" in args.cases:
        config["trajectories"] = {"compartments": ["hospitalizations"]}
    if "plots" in args.cases:
        config["plots"] = {
            "reference_date": "2024-01-13",
            "figure_output_types": ["PNG"],
            "quantiles": {"single": True, "grid": True, "quantiles": levels},
        }
    config = OutputConfig.model_validate({"output": config})
    preparation_seconds = time.perf_counter() - started
    timings = {}
    counts = {}

    def measured(name, fn):
        """Wrap a callable to accumulate inclusive parent-process time and calls.

        Parameters
        ----------
        name : str
            Counter and timing label.
        fn : callable
            Operation to instrument.

        Returns
        -------
        callable
            Wrapper forwarding arguments, return values and exceptions.
        """

        def call(*positional, **keywords):
            """Measure a call, including elapsed time when it raises.

            Parameters
            ----------
            *positional : tuple
                Positional arguments passed through to the measured function.
            **keywords : dict
                Keyword arguments passed through to the measured function.

            Returns
            -------
            object
                The measured function's return value.
            """
            before = time.perf_counter()
            try:
                return fn(*positional, **keywords)
            finally:
                counts[name] = counts.get(name, 0) + 1
                timings[name] = timings.get(name, 0) + time.perf_counter() - before

        return call

    with (
        patch.object(np, "nanquantile", measured("nanquantile", np.nanquantile)),
        patch.object(np, "quantile", measured("quantile", np.quantile)),
        patch.object(pd, "concat", measured("concat", pd.concat)),
        patch.object(output, "format_tabular_object", measured("tabular_format", output.format_tabular_object)),
        patch.object(
            generators, "figure_to_output_object", measured("figure_format", generators.figure_to_output_object)
        ),
        patch.object(
            output, "generate_single_quantile_plots", measured("single_plots", output.generate_single_quantile_plots)
        ),
        patch.object(output, "generate_quantile_grid_plot", measured("grid_plots", output.generate_quantile_grid_plot)),
    ):
        before = time.perf_counter()
        outputs = output.generate_calibration_outputs(calibrations=calibrations, output_config=config)
        output_seconds = time.perf_counter() - before
    tables = hashlib.sha256()
    for name, objects in outputs.items():
        for obj in objects:
            if isinstance(obj.data, pd.DataFrame):
                tables.update(name.encode())
                tables.update(obj.data.to_csv(index=False, date_format="%Y-%m-%d").encode())
            elif obj.name.endswith(".csv.gz"):
                tables.update(name.encode())
                tables.update(gzip.decompress(obj.data))
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib", "epydemix")}
    dependency = importlib.metadata.distribution("epydemix").read_text("direct_url.json")
    return {
        "source_tree": str(args.source_tree.resolve()),
        "versions": versions,
        "epydemix_source": json.loads(dependency) if dependency else None,
        "preparation_seconds": preparation_seconds,
        "output_seconds": output_seconds,
        "stage_seconds_inclusive": timings,
        "calls": counts,
        "table_sha256": tables.hexdigest(),
        "output_keys": list(outputs),
    }


def main():
    """Run fresh-process repetitions and print benchmark results as JSON.

    Returns
    -------
    None
        No value is returned.

    Raises
    ------
    RuntimeError
        A child fails or repeated runs produce different table hashes.

    Notes
    -----
    Parses sys.argv. Samples child-process RSS. Child mode writes a single result file instead of printing medians.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--locations", type=positive_int, default=1)
    parser.add_argument("--draws", type=positive_int, default=2000)
    parser.add_argument("--dates", type=positive_int, default=52)
    parser.add_argument("--variables", type=positive_int, default=8)
    parser.add_argument("--levels", type=positive_int, default=23)
    parser.add_argument("--cases", nargs="+", choices=["tabular", "hub", "plots", "trajectories"], default=["tabular"])
    parser.add_argument("--format", choices=["DataFrame", "CSVBytes"], default="DataFrame")
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--sample-ms", type=positive_int, default=10)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (args.source_tree / "epymodelingsuite/dispatcher/output.py").is_file():
        parser.error("--source-tree must contain epymodelingsuite/dispatcher/output.py")
    if args.child_result:
        args.child_result.write_text(json.dumps(run_once(args)))
        return
    runs = []
    with tempfile.TemporaryDirectory(prefix="benchmark-output-") as directory:
        for i in range(args.repeat):
            result_path = Path(directory) / f"{i}.json"
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child-result", str(result_path)]
            with (Path(directory) / f"{i}.log").open("w+") as log:
                before = time.perf_counter()
                child = subprocess.Popen(command, stdout=log, stderr=log)
                peak = 0
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    processes = [process]
                    try:
                        processes += process.children(recursive=True)
                    except psutil.NoSuchProcess:
                        pass
                    rss = 0
                    for proc in processes:
                        try:
                            rss += proc.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, rss)
                    time.sleep(args.sample_ms / 1000)
                if child.returncode:
                    log.seek(0)
                    raise RuntimeError(log.read())
            run = json.loads(result_path.read_text())
            run.update(wall_seconds=time.perf_counter() - before, sampled_peak_rss_bytes=peak)
            runs.append(run)
    if len({run["table_sha256"] for run in runs}) != 1:
        raise RuntimeError("Repeated runs produced different tables")
    print(
        json.dumps(
            {
                "input": {k: v for k, v in vars(args).items() if k not in ("source_tree", "child_result")},
                "thread_environment": {
                    k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
                "median_output_seconds": statistics.median(r["output_seconds"] for r in runs),
                "median_peak_rss_bytes": statistics.median(r["sampled_peak_rss_bytes"] for r in runs),
                "rss_note": "Sampled child process tree RSS sum; shared pages may be counted more than once.",
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
