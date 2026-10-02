"""File output without retaining every result or every output object.

CSVBytes, PNG, PDF and SVG configurations are supported. The input iterator
must itself load/yield one result at a time. Internal pickle shards preserve
identifiers and pandas dtypes; they are produced and read only by this call.
"""

import logging
import multiprocessing
import os
import pickle
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..schema.dispatcher import CalibrationOutput, SimulationOutput
from ..schema.output import FigureOutputTypeEnum, OutputConfig, TabularOutputTypeEnum, get_metrocast_quantiles
from ..utils.surveillance import surveillance_cache
from ..visualization.generators import (
    POSTERIOR_METADATA_COLUMNS,
    _check_incomplete_generations,
    _collect_location_plot_data,
    _format_plot_notes,
    _load_surveillance_sources,
    _quantile_plot_needs,
    generate_categorical_plots,
    generate_posterior_grid_plot,
    generate_quantile_grid_plot,
)
from .shared_quantiles import prepare_shared_quantiles

logger = logging.getLogger(__name__)


class OutputWriteError(RuntimeError):
    """A failed write, with the files already published by this invocation."""

    def __init__(self, message, completed):
        """Capture a write failure and a snapshot of published paths.

        Parameters
        ----------
        message : str
            Description of the failed input or output assembly.
        completed : dict[str, list[Path]]
            Mutable record of files successfully published so far.

        Returns
        -------
        None
            No value is returned.

        Notes
        -----
        The completed mapping and its lists are copied so later mutations do not alter the error.
        """
        super().__init__(message)
        self.completed = {name: list(paths) for name, paths in completed.items()}


def _dump(value, path):
    """Write an internally staged value as a pickle shard.

    Parameters
    ----------
    value : object
        Pickleable value to serialize using the highest protocol.
    path : Path
        Destination file inside the staging directory.

    Returns
    -------
    None
        No value is returned.

    Notes
    -----
    An existing shard is overwritten. Serialization and filesystem errors propagate.
    """
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path):
    """Load a trusted input or internally generated pickle shard.

    Parameters
    ----------
    path : str or Path
        File to deserialize; pickle may execute code, so inputs must be trusted.

    Returns
    -------
    object
        The deserialized object.

    Notes
    -----
    Filesystem and deserialization errors propagate to the caller.
    """
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def _validate_config(config, chunk_rows):
    """Validate supported file formats and the row chunk limit.

    Parameters
    ----------
    config : OutputConfig
        Validated output configuration.
    chunk_rows : int
        Positive maximum rows per trajectory or CSV write chunk.

    Returns
    -------
    None
        No value is returned.

    Raises
    ------
    ValueError
        chunk_rows is not a positive integer, tables are not exactly CSVBytes, or figure formats are unsupported.
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    if config.output.tabular_output_types != [TabularOutputTypeEnum.CSVBytes]:
        raise ValueError(
            "File output requires tabular_output_types=[CSVBytes]; use the existing API for DataFrame output"
        )
    plots = config.output.plots
    if plots and (
        not plots.figure_output_types
        or any(
            kind not in (FigureOutputTypeEnum.PNG, FigureOutputTypeEnum.PDF, FigureOutputTypeEnum.SVG)
            for kind in plots.figure_output_types
        )
    ):
        raise ValueError("File output supports PNG, PDF and SVG figures; use the existing API for MPLFigure")


class _Shards(dict):
    """A generator's output sink: serialize and discard each object immediately."""

    def __init__(self, directory):
        """Create a per-result staging directory and empty output indexes.

        Parameters
        ----------
        directory : str or Path
            New directory whose parent already exists.

        Returns
        -------
        None
            No value is returned.

        Raises
        ------
        OSError
            The directory already exists or cannot be created.
        """
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir()
        self.tables = defaultdict(list)
        self.images = {}
        self.error = None
        self.serial = 0

    def frame(self, name, frame):
        """Stage a nonempty table while preserving pandas column types.

        Parameters
        ----------
        name : str
            Logical table name used during final assembly.
        frame : pd.DataFrame
            Table to pickle; empty frames are ignored.

        Returns
        -------
        None
            No value is returned.

        Notes
        -----
        Only the shard path is retained. Serialization and filesystem errors propagate.
        """
        if frame.empty:
            return
        path = self.directory / f"table-{self.serial}.pickle"
        self.serial += 1
        _dump(frame, path)
        self.tables[name].append(path)

    def __setitem__(self, name, objects):
        """Serialize generator outputs immediately and index their shard paths.

        Parameters
        ----------
        name : str
            Logical output name; repeated image filenames overwrite earlier ones.
        objects : iterable of OutputObject
            DataFrame tables or byte-valued figure outputs to stage.

        Returns
        -------
        None
            No value is returned.

        Raises
        ------
        ValueError
            An image name includes directory components.
        Exception
            Serialization or file writing fails; the same exception is saved on the sink.

        Notes
        -----
        Hub rows are separated into quantiles and rate trends. Failures are recorded
        for check() and re-raised so legacy warning handlers cannot hide disk errors.
        """
        try:
            for obj in objects:
                if obj.output_type == TabularOutputTypeEnum.DataFrame:
                    if name == "output_hub_formatted":
                        self.frame("_hub_hosp", obj.data[obj.data.output_type == "quantile"])
                        self.frame("_hub_trends", obj.data[obj.data.output_type != "quantile"])
                    else:
                        self.frame(name, obj.data)
                else:
                    if Path(obj.name).name != obj.name:
                        raise ValueError(f"Invalid output filename: {obj.name}")
                    path = self.directory / obj.name
                    path.write_bytes(obj.data)
                    self.images.setdefault(name, {})[obj.name] = path
        except Exception as exc:
            # Plot generators retain their warning/skip policy. A disk failure
            # must nevertheless fail the save invocation after their return.
            self.error = exc
            raise

    def check(self):
        """Re-raise a staging error swallowed by a legacy generator.

        Returns
        -------
        None
            No value is returned.

        Raises
        ------
        Exception
            A previous output assignment failed.
        """
        if self.error is not None:
            raise self.error


def _trajectory_frames(item, config, chunk_rows):
    """Convert raw draws into bounded trajectory table chunks.

    Parameters
    ----------
    item : CalibrationOutput or SimulationOutput
        One result to process; calibration projections may be filtered in place.
    config : TrajectoriesOutput
        Enabled compartment/transition selections.
    chunk_rows : int
        Positive maximum rows per trajectory or CSV write chunk.

    Yields
    ------
    tuple[str, pd.DataFrame]
        Logical table name and a frame with at most chunk_rows rows.

    Raises
    ------
    ValueError
        A selected simulation variable has a different length from its dates.

    Notes
    -----
    No all-variable stack is created. Malformed calibration draws are logged
    and skipped; simulation length mismatches raise an error.
    """
    calibration = isinstance(item, CalibrationOutput)
    if calibration:
        draws = item.results.projections.get("baseline", [])
        if not draws:
            return
        keys = list(draws[0])
        if any(any(key not in draw for key in keys) for draw in draws):
            logger.warning("Skipping projection trajectories for primary_id=%s: missing variables", item.primary_id)
            return
        try:
            shapes = {key: np.shape(draws[0][key]) for key in keys}
            if any(np.shape(draw[key]) != shapes[key] for draw in draws for key in keys):
                raise ValueError("trajectory shapes differ")
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping projection trajectories for primary_id=%s: %s", item.primary_id, exc)
            return
    else:
        draws = item.results.trajectories
    for kind in ("compartments", "transitions"):
        selection = getattr(config, kind)
        if not selection:
            continue
        pending, count = [], 0
        name = f"trajectories_{'projection_' if calibration else ''}{kind}"
        for sim_id, draw in enumerate(draws):
            if calibration:
                columns = [key for key in keys if ("_to_" in key) == (kind == "transitions")]
                if kind == "transitions":
                    columns = ["date", *columns]
                if isinstance(selection, list) and all(key in keys for key in selection):
                    columns = ["date", *selection]
                # Preserve Series alignment for nonnumeric metadata as well as
                # the date column's original position in all-column outputs.
                index = pd.Series(draw[keys[0]]).index
                for key in keys[1:]:
                    index = index.union(pd.Series(draw[key]).index, sort=False)
                series = [pd.Series(draw[key], name=key) for key in columns]
                dates = index
            else:
                values, dates = getattr(draw, kind), draw.dates
                columns = (
                    selection if isinstance(selection, list) and all(c in values for c in selection) else list(values)
                )
                if any(len(values[key]) != len(dates) for key in columns):
                    raise ValueError(
                        f"Trajectory/date lengths differ for primary_id={item.primary_id}, sim_id={sim_id}"
                    )
            for start in range(0, len(dates), chunk_rows):
                stop = start + chunk_rows
                frame = (
                    pd.concat([column.reindex(index[start:stop]) for column in series], axis=1)
                    if calibration
                    else pd.DataFrame({key: values[key][start:stop] for key in columns})
                )
                if calibration:
                    frame.insert(0, "sim_id", sim_id)
                    frame.insert(0, "primary_id", item.primary_id)
                    frame.insert(2, "seed", item.seed)
                    frame.insert(3, "population", item.population)
                else:
                    frame.insert(0, "primary_id", item.primary_id)
                    frame.insert(1, "sim_id", sim_id)
                    frame.insert(2, "date", dates[start:stop])
                    frame.insert(3, "seed", item.seed)
                    frame.insert(4, "population", item.population)
                pending.append(frame)
                count += len(frame)
                if count >= chunk_rows:
                    combined = pd.concat(pending, ignore_index=True)
                    yield name, combined.iloc[:chunk_rows]
                    remainder = combined.iloc[chunk_rows:].copy()
                    pending, count = ([remainder], len(remainder)) if not remainder.empty else ([], 0)
        if pending:
            yield name, pd.concat(pending, ignore_index=True)


def _stage_result(item, config, directory, chunk_rows):
    """Stage outputs for one result and return its manifest path.

    Parameters
    ----------
    item : CalibrationOutput or SimulationOutput
        One result to process; calibration projections may be filtered in place.
    config : OutputConfig
        Validated output configuration.
    directory : str or Path
        New per-result directory inside the temporary root.
    chunk_rows : int
        Positive maximum rows per trajectory or CSV write chunk.

    Returns
    -------
    Path
        Pickled manifest indexing table/image shards and compact grid summaries.

    Raises
    ------
    TypeError
        The input is neither CalibrationOutput nor SimulationOutput.
    Exception
        Output generation or shard writing fails.

    Notes
    -----
    Global grids and prop ED assembly are deferred to the parent. Posterior
    samples are staged on disk. Calibration filtering mutates the source results.
    """
    from ..dispatcher.output import filter_failed_projections, generate_calibration_outputs, generate_simulation_outputs

    if not isinstance(item, (CalibrationOutput, SimulationOutput)):
        raise TypeError("Expected a CalibrationOutput or SimulationOutput")
    sink = _Shards(directory)
    local = config.model_copy(deep=True)
    local.output.tabular_output_types = [TabularOutputTypeEnum.DataFrame]
    local.output.trajectories = None
    summary = {
        "population": item.population,
        "plot_data": None,
        "posterior": None,
        "notes": ("", ""),
        "reference": None,
    }
    if isinstance(item, CalibrationOutput):
        filter_failed_projections(item.results)
        prepared = prepare_shared_quantiles([item], config.output)
        plots = config.output.plots
        plot_data = {}
        if plots:
            summary["reference"] = str(item.start_date_reference) if item.start_date_reference else None
            sources = config.output.options.surveillance if config.output.options else None
            surveillance = _load_surveillance_sources(sources, plots.quantiles.outputs)
            grid = plots.quantiles.grid
            if grid is not False and grid.enabled:
                try:
                    data = _collect_location_plot_data(
                        item,
                        plots,
                        surveillance,
                        **_quantile_plot_needs(plots.quantiles.outputs),
                        quantiles=prepared,
                    )
                    plot_data[id(item)] = data
                    summary["plot_data"] = data
                except Exception as exc:
                    logger.warning("Failed to prepare grid for %s: %s", item.population, exc)
            if plots.posterior.grid:
                summary["notes"] = _format_plot_notes([note] if (note := _check_incomplete_generations(item)) else [])
                try:
                    posterior = item.results.get_posterior_distribution()
                    if not posterior.empty:
                        path = sink.directory / "posterior.pickle"
                        _dump(posterior, path)
                        summary["posterior"] = path
                        summary["parameters"] = set(posterior.columns) - POSTERIOR_METADATA_COLUMNS
                except (ValueError, AttributeError) as exc:
                    logger.warning("Failed to get posterior for %s: %s", item.population, exc)
            local.output.plots.quantiles.grid = False
            local.output.plots.posterior.grid = False
            local.output.plots.categorical = None
        hub = config.output.flusight_format
        if hub and hub.prop_ed:
            local.output.flusight_format.prop_ed = None
            levels = get_metrocast_quantiles() if hub.metrocast else hub.quantiles
            strategy = hub.prop_ed.strategy
            if strategy in ("calibration_window", "transition"):
                try:
                    if strategy == "calibration_window":
                        draws = item.results.get_selected_trajectories()
                        frame = prepared.calibration(
                            item.results,
                            dates=draws[0].get("date") if draws else None,
                            quantiles=levels,
                            variables=["data"],
                            ignore_nan=True,
                        )
                        name = "_prop_calibration"
                    else:
                        draws = item.results.projections.get("baseline", [])
                        frame = prepared.projection(
                            item.results,
                            dates=draws[0].get("date") if draws else None,
                            quantiles=levels,
                            variables=[hub.prop_ed.transition_name],
                            ignore_nan=True,
                        )
                        name = "_prop_projection"
                    frame.insert(0, "primary_id", item.primary_id)
                    frame.insert(1, "seed", item.seed)
                    frame.insert(2, "population", item.population)
                    sink.frame(name, frame)
                except Exception as exc:
                    logger.warning("Failed to prepare prop ED for primary_id=%s: %s", item.primary_id, exc)
        generate_calibration_outputs(
            calibrations=[item],
            output_config=local,
            _sink=sink,
            _filtered=True,
            _prepared=prepared,
            _plot_data=plot_data,
        )
    else:
        generate_simulation_outputs(simulations=[item], output_config=local, _sink=sink)
    sink.check()
    if config.output.trajectories:
        for name, frame in _trajectory_frames(item, config.output.trajectories, chunk_rows):
            sink.frame(name, frame)
    part = {
        "tables": dict(sink.tables),
        "images": sink.images,
        "summary": summary,
        "kind": "calibration" if isinstance(item, CalibrationOutput) else "simulation",
    }
    path = sink.directory / "part.pickle"
    _dump(part, path)
    return path


def _concat(paths):
    """Load and concatenate a small group of summary table shards.

    Parameters
    ----------
    paths : sequence of Path
        Trusted internal pandas pickle shards in desired row order.

    Returns
    -------
    pd.DataFrame
        Concatenated rows, or an empty frame when no paths are supplied.

    Notes
    -----
    All supplied frames are loaded together; use for summaries, not raw trajectory tables.
    """
    return pd.concat([_load(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()


class _PosteriorFiles(Mapping):
    def __init__(self, paths):
        """Store posterior shard paths without loading their samples.

        Parameters
        ----------
        paths : mapping[str, Path]
            Population names mapped to trusted internal posterior shards.

        Returns
        -------
        None
            No value is returned.
        """
        self.paths = paths

    def __iter__(self):
        """Iterate population names without loading posterior frames.

        Returns
        -------
        iterator of str
            Keys from the underlying path mapping.
        """
        return iter(self.paths)

    def __len__(self):
        """Return the number of staged posterior populations.

        Returns
        -------
        int
            Number of keys in the underlying path mapping.
        """
        return len(self.paths)

    def __getitem__(self, key):
        """Load one population's posterior frame without caching it.

        Parameters
        ----------
        key : str
            Population name to look up.

        Returns
        -------
        pd.DataFrame
            A freshly deserialized posterior sample frame.

        Raises
        ------
        KeyError
            No shard is registered for the population.
        """
        return _load(self.paths[key])


def _publish(parts, config, staging, destination, chunk_rows, completed):
    """Assemble staged summaries and publish final output files.

    Parameters
    ----------
    parts : iterable of Path
        Manifest paths in original input order.
    config : OutputConfig
        Validated output configuration.
    staging : str or Path
        Temporary root shared with workers; removed by the caller after they stop.
    destination : Path
        Existing output directory on the same filesystem as staging.
    chunk_rows : int
        Positive maximum rows per trajectory or CSV write chunk.
    completed : dict[str, list[Path]]
        Mutable record of files successfully published so far.

    Returns
    -------
    None
        No value is returned.

    Notes
    -----
    Mutates completed after each successful replacement. Duplicate images use
    the last input. Global grids and Hub assembly run serially; replacement is
    atomic per file, not for the entire run. Filesystem failures propagate.
    """
    from ..dispatcher.output import make_prop_ed_flusightforecast

    tables, images = defaultdict(list), {}
    locations, plot_data, posteriors, notes, parameters = {}, {}, {}, {}, set()
    reference = None
    calibration_run = False
    for path in parts:
        part = _load(path)
        calibration_run = part["kind"] == "calibration"
        for name, paths in part["tables"].items():
            tables[name].extend(paths)
        for name, formats in part["images"].items():
            images.setdefault(name, {}).update(formats)  # Latest same-name figure wins, in input order.
        summary = part["summary"]
        population = summary["population"]
        reference = reference or summary["reference"]
        if summary["plot_data"] is not None:
            data = summary["plot_data"]
            if data.calibration_quantiles is not None or data.projection_quantiles is not None:
                locations.setdefault(population, SimpleNamespace(population=population))
                plot_data[id(locations[population])] = data
        notes[population] = summary["notes"]
        if summary["posterior"] is not None:
            posteriors[population] = summary["posterior"]
            parameters.update(summary["parameters"])
    final = _Shards(Path(staging) / "global")
    metadata = _concat(tables.pop("model_metadata", []))
    # Legacy metadata builds a single dict of lists (rather than concatenating
    # per-result frames): e.g. [13, None] must infer float64 and write 13.0.
    metadata = pd.DataFrame({column: metadata[column].tolist() for column in metadata})
    hub = config.output.flusight_format if calibration_run else None
    hospitalizations = _concat(tables.pop("_hub_hosp", []))
    trends = _concat(tables.pop("_hub_trends", []))
    hub_frames = [hospitalizations]
    if hub and hub.prop_ed:
        try:
            prop, factors = make_prop_ed_flusightforecast(
                hospitalizations,
                hub.prop_ed,
                config.output.options.surveillance if config.output.options else {},
                _concat(tables.pop("_prop_calibration", [])),
                _concat(tables.pop("_prop_projection", [])),
                hub.reference_date,
                metrocast=hub.metrocast,
            )
            hub_frames.append(prop)
            if not factors.empty and not metadata.empty:
                metadata = metadata.merge(factors, on="population")
        except (ValueError, AssertionError, KeyError, IndexError) as exc:
            logger.warning("Failed to generate prop ED forecasts: %s, skipping prop ED output", exc)
    hub_frames.append(trends)
    final.frame("model_metadata", metadata)
    hub_data = (
        pd.concat([f for f in hub_frames if not f.empty], ignore_index=True)
        if any(not f.empty for f in hub_frames)
        else pd.DataFrame()
    )
    final.frame("output_hub_formatted", hub_data)
    plots = config.output.plots if calibration_run else None
    if plots:
        sources = config.output.options.surveillance if config.output.options else None
        surveillance = _load_surveillance_sources(sources, plots.quantiles.outputs)
        generate_quantile_grid_plot(
            list(locations.values()), plots, final, sources, prepared_data=plot_data, surveillance_data=surveillance
        )
        generate_posterior_grid_plot(
            [],
            plots,
            final,
            reference,
            prepared_posteriors=_PosteriorFiles(posteriors),
            prepared_parameters=parameters,
            prepared_notes=notes,
        )
        if plots.categorical:
            generate_categorical_plots(plots, final, hub_data)
    final.check()
    for name, paths in final.tables.items():
        tables[name].extend(paths)
    for name, formats in final.images.items():
        images.setdefault(name, {}).update(formats)
    for name, paths in tables.items():
        if not paths or name.startswith("_"):
            continue
        # One row per shard is sufficient for concat's column/dtype union;
        # complete trajectory frames are loaded only one shard at a time.
        samples = []
        for path in paths:
            frame = _load(path)
            samples.append(frame.iloc[:1].copy())
            del frame
        schema = pd.concat(samples, ignore_index=True)
        del samples
        target = destination / f"{name}.csv.gz"
        temporary = Path(staging) / target.name
        first = True
        import gzip

        with gzip.open(temporary, "wt", newline="") as stream:
            for path in paths:
                frame = _load(path).reindex(columns=schema.columns).astype(schema.dtypes.to_dict())
                for start in range(0, len(frame), chunk_rows):
                    frame.iloc[start : start + chunk_rows].to_csv(
                        stream, index=False, header=first, date_format="%Y-%m-%d"
                    )
                    first = False
        os.replace(temporary, target)
        completed[name] = [target]
    for name, formats in images.items():
        completed[name] = []
        for filename, path in formats.items():
            target = destination / filename
            os.replace(path, target)
            completed[name].append(target)


def write_outputs(
    *,
    results: Iterable[CalibrationOutput | SimulationOutput],
    output_config: OutputConfig,
    directory: str | Path,
    chunk_rows: int = 10000,
) -> dict[str, list[Path]]:
    """Consume results lazily and save outputs to files.

    Parameters
    ----------
    results : iterable of CalibrationOutput or SimulationOutput
        One homogeneous result stream; mixed calibration/simulation inputs are rejected.
    output_config : OutputConfig
        Configure exactly CSVBytes and file figure formats PNG, PDF or SVG.
    directory : str or Path
        Output directory, created if needed. Matching output files are replaced.
    chunk_rows : int, optional
        Positive trajectory/CSV chunk limit, default 10000 rows.

    Returns
    -------
    dict[str, list[Path]]
        Logical output names mapped to successfully published file paths.

    Raises
    ------
    ValueError
        The configuration or chunk limit is invalid, before inputs are consumed.
    OutputWriteError
        Input consumption, generation or publication fails; completed lists already published files.

    Notes
    -----
    A preloaded input list retains its raw results. Filtering may mutate
    calibration projections. Temporary shards are cleaned on exit. Publication
    is atomic per file; unrelated existing files remain in place.
    """
    _validate_config(output_config, chunk_rows)
    destination = Path(directory)
    completed = {}
    current = "input"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".output-", dir=destination) as staging, surveillance_cache():
            parts = []
            kind = None
            index = 0
            for item in results:
                current = f"input {index}, primary_id={getattr(item, 'primary_id', '?')}"
                if kind is not None and type(item) is not kind:
                    raise TypeError("Do not mix calibration and simulation results in one output invocation")
                kind = type(item)
                parts.append(_stage_result(item, output_config, Path(staging) / str(index), chunk_rows))
                del item
                index += 1
            current = "final output assembly"
            _publish(parts, output_config, staging, destination, chunk_rows, completed)
    except Exception as exc:
        raise OutputWriteError(f"Output writing failed during {current}: {exc}", completed) from exc
    return completed


def _stage_file(index, path, config, staging, chunk_rows):
    """Load and stage one trusted result in a spawn-safe worker.

    Parameters
    ----------
    index : int
        Original input position, used for staging directory names.
    path : str or Path
        Trusted pickle containing exactly one result.
    config : OutputConfig
        Validated output configuration.
    staging : str or Path
        Temporary root shared with workers; removed by the caller after they stop.
    chunk_rows : int
        Positive maximum rows per trajectory or CSV write chunk.

    Returns
    -------
    Path
        Per-result manifest path; raw result arrays never cross IPC.

    Raises
    ------
    RuntimeError
        Loading or staging fails; the message identifies the input.

    Notes
    -----
    Uses the noninteractive Agg backend and a worker-local surveillance cache.
    """
    import matplotlib

    matplotlib.use("Agg")
    identifier = f"input {index} ({path})"
    try:
        item = _load(path)
        identifier += f", primary_id={getattr(item, 'primary_id', '?')}"
        with surveillance_cache():
            return _stage_result(item, config, Path(staging) / str(index), chunk_rows)
    except Exception as exc:
        raise RuntimeError(f"{identifier}: {exc}") from exc


def _file_parts(paths, config, staging, chunk_rows, workers):
    """Stage file inputs with at most workers unfinished jobs.

    Parameters
    ----------
    paths : iterable of str or Path
        Trusted pickle paths, each containing one calibration or simulation output.
    config : OutputConfig
        Validated output configuration.
    staging : str or Path
        Temporary root shared with workers; removed by the caller after they stop.
    chunk_rows : int
        Positive maximum rows per trajectory or CSV write chunk.
    workers : int
        Positive worker count; one runs directly without creating a process pool.

    Yields
    ------
    tuple[int, Path]
        Input index and manifest path, in completion order for parallel workers.

    Notes
    -----
    One worker runs directly. Multiple workers use spawn. Failure or generator
    closure cancels pending jobs and waits for active jobs before returning.
    """
    inputs = iter(enumerate(paths))
    if workers == 1:
        for index, path in inputs:
            yield index, _stage_file(index, path, config, staging, chunk_rows)
        return
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    pending = {}

    def submit_next():
        """Submit the next input and record its future in the pending set.

        Returns
        -------
        bool
            True if a job was submitted; False when the input iterator is exhausted.
        """
        try:
            index, path = next(inputs)
        except StopIteration:
            return False
        future = executor.submit(_stage_file, index, str(path), config, str(staging), chunk_rows)
        pending[future] = index
        return True

    try:
        for _ in range(workers):
            if not submit_next():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Inspect all finished tasks before submitting any new work. A failed
            # task stops input consumption even when other tasks have succeeded.
            finished = [(pending.pop(future), future.result()) for future in done]
            yield from finished
            for _ in finished:
                submit_next()
    finally:
        for future in pending:
            future.cancel()
        # Workers must stop before TemporaryDirectory removes their staging area.
        executor.shutdown(wait=True, cancel_futures=True)


def write_outputs_from_files(
    *,
    paths: Iterable[str | Path],
    output_config: OutputConfig,
    directory: str | Path,
    workers: int = 1,
    chunk_rows: int = 10000,
) -> dict[str, list[Path]]:
    """Save trusted one-result pickle files, optionally using multiple cores.

    Parameters
    ----------
    paths : iterable of str or Path
        Trusted pickle paths, each containing one calibration or simulation output.
    output_config : OutputConfig
        Configure exactly CSVBytes and file figure formats PNG, PDF or SVG.
    directory : str or Path
        Output directory, created if needed. Matching output files are replaced.
    workers : int, optional
        Positive maximum concurrent jobs, default 1 (no process pool).
    chunk_rows : int, optional
        Positive trajectory/CSV chunk limit, default 10000 rows.

    Returns
    -------
    dict[str, list[Path]]
        Logical output names mapped to successfully published file paths.

    Raises
    ------
    ValueError
        Configuration, workers or chunk_rows is invalid before paths are consumed.
    OutputWriteError
        Loading, staging, mixed input kinds or publication fails; completed lists published files.

    Notes
    -----
    Each trusted pickle contains one CalibrationOutput or SimulationOutput,
    not a list. Pickle can execute code. Multiple workers use spawn and Agg;
    call from an importable module under an if __name__ == "__main__" guard.
    The parent assembles in input order and renders global grids. Worker memory
    scales with workers. Temporary files are cleaned after workers stop; final
    replacement is atomic per file, not for the whole invocation.
    """
    _validate_config(output_config, chunk_rows)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    destination = Path(directory)
    completed = {}
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".output-", dir=destination) as staging, surveillance_cache():
            parts, kinds = {}, set()
            with closing(_file_parts(paths, output_config, staging, chunk_rows, workers)) as staged:
                for index, path in staged:
                    parts[index] = path
                    kinds.add(_load(path)["kind"])
                    if len(kinds) > 1:
                        raise TypeError("Do not mix calibration and simulation results in one output invocation")
            _publish(
                [parts[index] for index in sorted(parts)], output_config, staging, destination, chunk_rows, completed
            )
    except Exception as exc:
        raise OutputWriteError(f"Output writing from files failed: {exc}", completed) from exc
    return completed
