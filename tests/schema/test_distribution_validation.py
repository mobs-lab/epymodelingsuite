"""Tests for distribution validation through shared configuration schemas."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.logging import LogCaptureFixture

from epymodelingsuite.config_loader import load_calibration_config_from_file
from epymodelingsuite.schema.calibration import CalibrationConfig
from epymodelingsuite.schema.sampling import SamplingConfig


def _calibration_config_with_prior(prior: dict) -> dict:
    """Create a minimal calibration configuration containing one prior."""
    return {
        "modelset": {
            "population_names": ["US-CA"],
            "calibration": {
                "strategy": {"name": "SMC"},
                "observed_data_path": "data.csv",
                "comparison": [
                    {
                        "observed_date_column": "date",
                        "observed_value_column": "value",
                        "simulation": ["I_to_R"],
                    }
                ],
                "fitting_window": {"start_date": "2024-01-01", "end_date": "2024-02-01"},
                "parameters": {"alpha": {"prior": prior}},
            },
        }
    }


def test_calibration_prior_error_includes_configuration_path() -> None:
    """Calibration errors identify the exact prior containing invalid scale."""
    config = _calibration_config_with_prior(
        {"type": "scipy", "name": "uniform", "args": [1.0, 0.0]},
    )

    with pytest.raises(
        ValueError,
        match=r"(?s)modelset\.calibration\.parameters\.alpha\.prior.*scale must be a finite scalar greater than 0",
    ):
        CalibrationConfig(**config)


def test_sampling_distribution_uses_shared_validation() -> None:
    """Sampling distributions receive the same early validation as priors."""
    config = {
        "modelset": {
            "population_names": ["US-CA"],
            "sampling": {
                "samplers": [
                    {
                        "strategy": "LHS",
                        "n_samples": 10,
                        "parameters": ["alpha"],
                    }
                ],
                "parameters": {
                    "alpha": {
                        "distribution": {
                            "type": "scipy",
                            "name": "uniform",
                            "kwargs": {"loc": 1.0, "scale": 0.0},
                        }
                    }
                },
            },
        }
    }

    with pytest.raises(
        ValueError,
        match=r"(?s)modelset\.sampling\.parameters\.alpha\.distribution.*scale must be a finite scalar greater than 0",
    ):
        SamplingConfig(**config)


def test_loader_logs_filename_and_prior_error(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """The loader logs one actionable error with filename and prior path."""
    config_path = tmp_path / "invalid-calibration.yml"
    config_path.write_text(
        """
modelset:
  population_names: [US-CA]
  calibration:
    strategy:
      name: SMC
    observed_data_path: data.csv
    comparison:
      - observed_date_column: date
        observed_value_column: value
        simulation: [I_to_R]
    fitting_window:
      start_date: 2024-01-01
      end_date: 2024-02-01
    parameters:
      alpha:
        prior:
          type: scipy
          name: uniform
          args: [1.0, 0.0]
"""
    )

    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match=r"invalid-calibration\.yml"):
        load_calibration_config_from_file(str(config_path))

    assert "modelset.calibration.parameters.alpha.prior" in caplog.text  # noqa: S101
    assert "upper bound = loc + scale" in caplog.text  # noqa: S101
