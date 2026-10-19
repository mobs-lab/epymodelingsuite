"""Tests for calibration schema validation."""

import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from epydemix.calibration import ABCSampler
from pydantic import ValidationError

from epymodelingsuite.config_loader import load_calibration_config_from_file
from epymodelingsuite.schema.calibration import (
    CalibrationConfig,
    CalibrationStrategy,
    ComparisonSpec,
    ProjectionSpec,
)


class TestCalibrationStrategyMaxTime:
    """Tests for CalibrationStrategy with max_time option."""

    def test_max_time_as_string_minutes(self):
        """Test that max_time string is converted to timedelta for minutes."""
        strategy = CalibrationStrategy(
            name="rejection",
            options={"num_particles": 100, "max_time": "30m"},
        )
        assert isinstance(strategy.options["max_time"], timedelta)
        assert strategy.options["max_time"] == timedelta(minutes=30)

    def test_max_time_as_string_hours(self):
        """Test that max_time string is converted to timedelta for hours."""
        strategy = CalibrationStrategy(
            name="smc",
            options={"num_particles": 500, "num_generations": 10, "max_time": "4h"},
        )
        assert isinstance(strategy.options["max_time"], timedelta)
        assert strategy.options["max_time"] == timedelta(hours=4)

    def test_max_time_as_string_days(self):
        """Test that max_time string is converted to timedelta for days."""
        strategy = CalibrationStrategy(
            name="rejection",
            options={"num_particles": 100, "max_time": "2D"},
        )
        assert isinstance(strategy.options["max_time"], timedelta)
        assert strategy.options["max_time"] == timedelta(days=2)

    def test_max_time_as_string_weeks(self):
        """Test that max_time string is converted to timedelta for weeks."""
        strategy = CalibrationStrategy(
            name="smc",
            options={"num_particles": 100, "max_time": "W"},
        )
        assert isinstance(strategy.options["max_time"], timedelta)
        assert strategy.options["max_time"] == timedelta(weeks=1)

    def test_max_time_as_timedelta(self):
        """Test that max_time as timedelta is kept as-is."""
        td = timedelta(hours=2)
        strategy = CalibrationStrategy(
            name="rejection",
            options={"num_particles": 100, "max_time": td},
        )
        assert isinstance(strategy.options["max_time"], timedelta)
        assert strategy.options["max_time"] == td

    def test_max_time_compound_duration(self):
        """Test that compound max_time string is converted correctly."""
        strategy = CalibrationStrategy(
            name="smc",
            options={"num_particles": 500, "max_time": "1h30m"},
        )
        assert isinstance(strategy.options["max_time"], timedelta)
        assert strategy.options["max_time"] == timedelta(hours=1, minutes=30)

    def test_max_time_invalid_string_raises_error(self):
        """Test that invalid max_time string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid max_time value"):
            CalibrationStrategy(
                name="rejection",
                options={"num_particles": 100, "max_time": "M"},  # Month is variable-length
            )

    def test_max_time_empty_string_raises_error(self):
        """Test that empty max_time string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid max_time value"):
            CalibrationStrategy(
                name="rejection",
                options={"num_particles": 100, "max_time": ""},
            )

    def test_max_time_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid max_time value"):
            CalibrationStrategy(
                name="rejection",
                options={"num_particles": 100, "max_time": "invalid_format"},
            )

    def test_strategy_without_max_time(self):
        """Test that strategy works without max_time option."""
        strategy = CalibrationStrategy(
            name="smc",
            options={"num_particles": 500, "num_generations": 10},
        )
        assert "max_time" not in strategy.options
        assert strategy.options["num_particles"] == 500

    def test_strategy_with_other_options(self):
        """Test that max_time doesn't interfere with other options."""
        strategy = CalibrationStrategy(
            name="smc",
            options={
                "num_particles": 500,
                "num_generations": 10,
                "epsilon_schedule": [0.1, 0.05, 0.01],
                "max_time": "2h",
            },
        )
        assert strategy.options["num_particles"] == 500
        assert strategy.options["num_generations"] == 10
        assert strategy.options["epsilon_schedule"] == [0.1, 0.05, 0.01]
        assert strategy.options["max_time"] == timedelta(hours=2)

    def test_max_time_zero_duration(self):
        """Test that zero duration is handled correctly."""
        strategy = CalibrationStrategy(
            name="rejection",
            options={"num_particles": 100, "max_time": "0m"},
        )
        assert strategy.options["max_time"] == timedelta(0)


class TestCalibrationConfigWithMaxTime:
    """Tests for full CalibrationConfig with max_time."""

    @pytest.fixture
    def base_calibration_config(self):
        """Create a minimal calibration configuration for testing."""
        return {
            "modelset": {
                "population_names": ["US-CA"],
                "calibration": {
                    "strategy": {
                        "name": "SMC",
                        "options": {"num_particles": 100, "num_generations": 5, "max_time": "4h"},
                    },
                    "distance_function": "rmse",
                    "observed_data_path": "data/test.csv",
                    "comparison": [
                        {
                            "observed_date_column": "date",
                            "observed_value_column": "value",
                            "simulation": ["I_to_R"],
                        }
                    ],
                    "fitting_window": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
                    "parameters": {
                        "beta": {
                            "prior": {"type": "scipy", "name": "uniform", "args": [0.1, 0.5]},
                        }
                    },
                },
            }
        }

    def test_full_config_with_max_time(self, base_calibration_config):
        """Test that full calibration config validates with max_time."""
        config = CalibrationConfig(**base_calibration_config)
        assert isinstance(config.modelset.calibration.strategy.options["max_time"], timedelta)
        assert config.modelset.calibration.strategy.options["max_time"] == timedelta(hours=4)

    def test_full_config_max_time_different_formats(self, base_calibration_config):
        """Test various max_time formats in full config."""
        # Test with minutes
        base_calibration_config["modelset"]["calibration"]["strategy"]["options"]["max_time"] = "30m"
        config = CalibrationConfig(**base_calibration_config)
        assert config.modelset.calibration.strategy.options["max_time"] == timedelta(minutes=30)

        # Test with days
        base_calibration_config["modelset"]["calibration"]["strategy"]["options"]["max_time"] = "2D"
        config = CalibrationConfig(**base_calibration_config)
        assert config.modelset.calibration.strategy.options["max_time"] == timedelta(days=2)

        # Test with weeks
        base_calibration_config["modelset"]["calibration"]["strategy"]["options"]["max_time"] = "W"
        config = CalibrationConfig(**base_calibration_config)
        assert config.modelset.calibration.strategy.options["max_time"] == timedelta(weeks=1)

    def test_full_config_without_max_time(self, base_calibration_config):
        """Test that full config works without max_time."""
        del base_calibration_config["modelset"]["calibration"]["strategy"]["options"]["max_time"]
        config = CalibrationConfig(**base_calibration_config)
        assert "max_time" not in config.modelset.calibration.strategy.options

    def test_full_config_invalid_max_time(self, base_calibration_config):
        """Test that invalid max_time in full config raises error."""
        base_calibration_config["modelset"]["calibration"]["strategy"]["options"]["max_time"] = "M"
        with pytest.raises(ValueError, match="Invalid max_time value"):
            CalibrationConfig(**base_calibration_config)


class TestCalibrationStrategyEnum:
    """Tests for CalibrationStrategy enum values."""

    def test_smc_strategy_with_max_time(self):
        """Test SMC strategy with max_time."""
        strategy = CalibrationStrategy(
            name="SMC",
            options={"num_particles": 500, "num_generations": 10, "max_time": "4h"},
        )
        assert strategy.name == "smc"
        assert strategy.options["max_time"] == timedelta(hours=4)

    def test_rejection_strategy_with_max_time(self):
        """Test rejection strategy with max_time."""
        strategy = CalibrationStrategy(
            name="rejection",
            options={"num_particles": 1000, "max_time": "2h"},
        )
        assert strategy.name == "rejection"
        assert strategy.options["max_time"] == timedelta(hours=2)

    def test_top_fraction_strategy_without_max_time(self):
        """Test top_fraction strategy (which doesn't use max_time)."""
        strategy = CalibrationStrategy(
            name="top_fraction",
            options={"top_fraction": 0.1, "Nsim": 1000},
        )
        assert strategy.name == "top_fraction"
        assert "max_time" not in strategy.options


class TestCalibrationStrategyName:
    """Strategy names are normalized to the names epydemix accepts."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("SMC", "smc"),
            ("smc", "smc"),
            ("Smc", "smc"),
            ("REJECTION", "rejection"),
            ("rejection", "rejection"),
            ("Top_Fraction", "top_fraction"),
            ("top_fraction", "top_fraction"),
        ],
    )
    def test_name_is_case_insensitive(self, raw: str, expected: str) -> None:
        strategy = CalibrationStrategy(name=raw)
        assert strategy.name == expected
        assert type(strategy.name) is str

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Input should be"):
            CalibrationStrategy(name="mcmc")

    @pytest.mark.parametrize("name", [e.value for e in CalibrationStrategy.CalibrationStrategyEnum])
    def test_every_exposed_name_accepted_by_epydemix(self, name: str) -> None:
        """Each enum value dispatches to the matching ABCSampler.run_<name> method."""
        sampler = MagicMock()
        ABCSampler.calibrate(sampler, strategy=CalibrationStrategy(name=name.upper()).name)
        getattr(sampler, f"run_{name}").assert_called_once()

    def test_tutorial_modelset_calibration_yaml_loads(self) -> None:
        path = Path(__file__).parents[2] / "tutorials" / "data" / "basic_modelset_calibration.yml"
        config = load_calibration_config_from_file(str(path))
        assert config.modelset.calibration.strategy.name == "smc"


class TestComparisonSpecLocationFormat:
    """Tests for ComparisonSpec observed_location_format validation."""

    def test_default_location_format_is_iso(self):
        """Test that default observed_location_format is ISO."""
        spec = ComparisonSpec(
            observed_date_column="date",
            observed_value_column="value",
            simulation=["I_to_R"],
        )
        assert spec.observed_location_format == "ISO"

    def test_iso_location_format_is_valid(self):
        """Test that ISO location format is accepted."""
        spec = ComparisonSpec(
            observed_date_column="date",
            observed_value_column="value",
            observed_location_format="ISO",
            simulation=["I_to_R"],
        )
        assert spec.observed_location_format == "ISO"

    def test_fips_location_format_is_valid(self):
        """Test that FIPS location format is accepted."""
        spec = ComparisonSpec(
            observed_date_column="date",
            observed_value_column="value",
            observed_location_format="FIPS",
            simulation=["I_to_R"],
        )
        assert spec.observed_location_format == "FIPS"

    def test_metrocast_location_id_format_is_valid(self):
        """Test that metrocast_location_id location format is accepted.

        This is a regression test to ensure metrocast location format
        is properly supported in calibration configs.
        """
        spec = ComparisonSpec(
            observed_date_column="date",
            observed_value_column="value",
            observed_location_format="metrocast_location_id",
            simulation=["I_to_R"],
        )
        assert spec.observed_location_format == "metrocast_location_id"

    def test_epydemix_population_format_is_valid(self):
        """Test that epydemix_population location format is accepted."""
        spec = ComparisonSpec(
            observed_date_column="date",
            observed_value_column="value",
            observed_location_format="epydemix_population",
            simulation=["I_to_R"],
        )
        assert spec.observed_location_format == "epydemix_population"

    def test_invalid_location_format_raises_error(self):
        """Test that invalid location format raises ValueError."""
        with pytest.raises(ValueError, match="observed_location_format must be one of"):
            ComparisonSpec(
                observed_date_column="date",
                observed_value_column="value",
                observed_location_format="invalid_format",
                simulation=["I_to_R"],
            )


class TestCalibrationModelsetPopulationNames:
    """Tests for CalibrationModelset population_names validation."""

    @pytest.fixture
    def base_config(self):
        """Return base calibration configuration dict for testing population_names."""
        return {
            "modelset": {
                "calibration": {
                    "strategy": {"name": "top_fraction", "options": {"Nsim": 100, "top_fraction": 0.1}},
                    "observed_data_path": "/tmp/data.csv",
                    "fitting_window": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
                    "comparison": [
                        {
                            "observed_value_column": "value",
                            "observed_date_column": "date",
                            "simulation": ["I_to_R"],
                        }
                    ],
                    "parameters": {
                        "beta": {"prior": {"type": "scipy", "name": "uniform", "args": [0.1, 0.5]}},
                    },
                },
            },
        }

    def test_all_states_keyword_is_valid(self, base_config):
        """Test that 'all-states' keyword is accepted as a valid population name."""
        base_config["modelset"]["population_names"] = ["all-states"]
        config = CalibrationConfig(**base_config)
        assert config.modelset.population_names == ["all-states"]

    def test_all_metrocast_keyword_is_valid(self, base_config):
        """Test that 'all-metrocast' keyword is accepted as a valid population name."""
        base_config["modelset"]["population_names"] = ["all-metrocast"]
        config = CalibrationConfig(**base_config)
        assert config.modelset.population_names == ["all-metrocast"]

    def test_deprecated_all_keyword_is_valid(self, base_config):
        """Test that deprecated 'all' keyword is still accepted (with deprecation warning)."""
        import warnings

        base_config["modelset"]["population_names"] = ["all"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = CalibrationConfig(**base_config)
            assert config.modelset.population_names == ["all"]
            # Should have raised a deprecation warning
            assert any("deprecated" in str(warning.message).lower() for warning in w)

    def test_iso_location_is_valid(self, base_config):
        """Test that ISO location codes are accepted."""
        base_config["modelset"]["population_names"] = ["US-MA", "US-CA"]
        config = CalibrationConfig(**base_config)
        assert config.modelset.population_names == ["US-MA", "US-CA"]

    def test_metrocast_location_is_valid(self, base_config):
        """Test that metrocast location names are accepted."""
        base_config["modelset"]["population_names"] = ["denver", "boston"]
        config = CalibrationConfig(**base_config)
        assert config.modelset.population_names == ["denver", "boston"]

    def test_mixed_keywords_and_locations(self, base_config):
        """Test that keywords can be mixed with specific locations."""
        base_config["modelset"]["population_names"] = ["all-states", "denver"]
        config = CalibrationConfig(**base_config)
        assert config.modelset.population_names == ["all-states", "denver"]

    def test_invalid_location_raises_error(self, base_config):
        """Test that invalid location names raise ValueError."""
        base_config["modelset"]["population_names"] = ["invalid_location_xyz"]
        with pytest.raises(ValueError):
            CalibrationConfig(**base_config)


class TestProjectionSpecTrajectories:
    def test_n_trajectories_required(self):
        with pytest.raises(ValidationError, match="n_trajectories"):
            ProjectionSpec()

    @pytest.mark.parametrize("n_trajectories", [0, -1])
    def test_n_trajectories_positive(self, n_trajectories):
        with pytest.raises(ValidationError, match="greater than 0"):
            ProjectionSpec(n_trajectories=n_trajectories)


def test_schema_validation_survives_optimized_mode():
    """Validators must not rely on assert, which `python -O` strips."""
    code = (
        "from pydantic import ValidationError\n"
        "from epymodelingsuite.schema.basemodel import Timespan\n"
        "try:\n"
        "    Timespan(start_date='2025-01-01', end_date='2025-02-01', delta_t=-1)\n"
        "except ValidationError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    assert subprocess.run([sys.executable, "-O", "-c", code], check=False).returncode == 0
