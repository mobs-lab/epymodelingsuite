"""Real spawn workers, bounded submissions, ordering and failure propagation."""

import gzip
from concurrent.futures import Future
from unittest.mock import patch

import pytest

from epymodelingsuite.output import streaming
from epymodelingsuite.output.streaming import OutputWriteError, write_outputs_from_files

from .test_output_quantiles import calibration
from .test_output_streaming import config


def _delayed_result(seconds, index):
    """Delay deserialization of a test result to force real worker completion out of input order."""
    import time

    time.sleep(seconds)
    item = calibration()
    item.primary_id = index
    return item


class _DelayedPickle:
    def __init__(self, seconds, index):
        self.seconds, self.index = seconds, index

    def __reduce__(self):
        return _delayed_result, (self.seconds, self.index)


def test_real_spawn_workers_match_serial_tables_and_figures(tmp_path):
    """Compare actual spawn workers with the no-pool serial path for exact CSV/PNG output and duplicate locations."""
    paths = []
    for index in range(3):
        item = calibration()
        item.primary_id = index
        if index == 1:
            item.population = "United_States_Texas"
        # Last California input overwrites the first one's single images.
        for draw in item.results.projections["baseline"]:
            draw["hospitalizations"] += index * 20
        path = tmp_path / f"{index}.pickle"
        streaming._dump(item, path)
        paths.append(path)
    settings = config(plots=True, prop=True)
    with patch.object(streaming, "ProcessPoolExecutor", side_effect=AssertionError("serial must not create a pool")):
        serial = write_outputs_from_files(paths=iter(paths), output_config=settings, directory=tmp_path / "serial")
    parallel = write_outputs_from_files(
        paths=iter(paths), output_config=settings, directory=tmp_path / "parallel", workers=2
    )
    assert serial.keys() == parallel.keys()
    for name in serial:
        for left, right in zip(serial[name], parallel[name], strict=True):
            assert left.name == right.name
            if left.name.endswith(".csv.gz"):
                assert gzip.decompress(left.read_bytes()) == gzip.decompress(right.read_bytes())
            elif left.name.endswith(".png"):
                assert left.read_bytes() == right.read_bytes()
    assert not list((tmp_path / "parallel").glob(".output-*"))


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_workers_fail_before_consuming_paths(tmp_path, workers):
    """Reject zero, negative, boolean and noninteger worker counts without advancing file inputs."""

    def paths():
        raise AssertionError("must not consume")
        yield

    with pytest.raises(ValueError, match="workers"):
        write_outputs_from_files(paths=paths(), output_config=config(), directory=tmp_path, workers=workers)


def test_real_worker_failure_names_input_and_cleans_staging(tmp_path):
    """Identify a failed input from a real worker and leave no published files or staging directory."""
    missing = tmp_path / "missing.pickle"
    with pytest.raises(OutputWriteError, match="input 0.*missing.pickle") as caught:
        write_outputs_from_files(paths=[missing], output_config=config(), directory=tmp_path / "output", workers=2)
    assert caught.value.completed == {}
    assert not list((tmp_path / "output").iterdir())


def test_real_workers_can_finish_in_reverse_order(tmp_path):
    """Force real workers to finish in reverse order while final CSV rows retain the original input order."""
    paths = [tmp_path / "slow.pickle", tmp_path / "fast.pickle"]
    for index, path in enumerate(paths):
        streaming._dump(_DelayedPickle(2 if index == 0 else 0, index), path)
    staging = tmp_path / "staged"
    staging.mkdir()
    parts = list(streaming._file_parts(paths, config(), staging, 100, 2))
    assert [index for index, _ in parts] == [1, 0]
    output = tmp_path / "saved"
    output.mkdir()
    completed = {}
    streaming._publish([path for _, path in sorted(parts)], config(), staging, output, 100, completed)
    import pandas as pd

    frame = pd.read_csv(completed["quantiles_calibration"][0])
    assert frame.primary_id.drop_duplicates().tolist() == [0, 1]


def test_submissions_are_bounded_and_reverse_completion_keeps_input_order(tmp_path):
    """Bound pending jobs by worker count and preserve input indexes despite reversed fake-future completion."""
    outstanding, peak, consumed, returned = set(), [], [], []

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def submit(self, function, index, *args):
            future = Future()
            future.index = index
            future.set_result(tmp_path / f"{index}.pickle")
            outstanding.add(future)
            peak.append(len(outstanding))
            return future

        def shutdown(self, **kwargs):
            assert kwargs == {"wait": True, "cancel_futures": True}

    def reverse_wait(pending, **kwargs):
        newest = max(pending, key=lambda future: future.index)
        outstanding.remove(newest)
        returned.append(newest.index)
        return {newest}, set(pending) - {newest}

    def inputs():
        for index in range(5):
            consumed.append(index)
            yield tmp_path / f"{index}.pickle"

    with patch.object(streaming, "ProcessPoolExecutor", Executor), patch.object(streaming, "wait", reverse_wait):
        staged = streaming._file_parts(inputs(), config(), tmp_path, 100, 2)
        first = next(staged)
        assert consumed == [0, 1]
        parts = [first, *staged]
    assert max(peak) == 2
    assert returned != sorted(returned)
    assert [path.name for _, path in sorted(parts)] == [f"{index}.pickle" for index in range(5)]


def test_failed_future_stops_submission_and_unsupported_file_is_named(tmp_path):
    """Stop consuming paths after a failed future and identify a pickle containing an unsupported result list."""
    consumed = []

    class Executor:
        def __init__(self, **_):
            pass

        def submit(self, function, index, *args):
            future = Future()
            future.set_exception(RuntimeError("worker failure"))
            return future

        def shutdown(self, **_):
            pass

    def inputs():
        for index in range(10):
            consumed.append(index)
            yield "unused"

    with patch.object(streaming, "ProcessPoolExecutor", Executor), pytest.raises(RuntimeError, match="worker failure"):
        list(streaming._file_parts(inputs(), config(), tmp_path, 100, 2))
    assert consumed == [0, 1]
    invalid = tmp_path / "all-results.pickle"
    streaming._dump([calibration()], invalid)
    with pytest.raises(OutputWriteError, match="all-results.pickle.*Expected a CalibrationOutput"):
        write_outputs_from_files(paths=[invalid], output_config=config(), directory=tmp_path / "output")
