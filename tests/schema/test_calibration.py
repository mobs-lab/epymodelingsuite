"""Tests for calibration schema validation."""

from datetime import timedelta

import pytest

from epymodelingsuite.schema.calibration import (
    CalibrationConfig,
    CalibrationStrategy,
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
        assert strategy.name == "SMC"
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


class TestFittingWindowEpiweekConversion:
    """Tests for FittingWindow epiweek to date conversion."""

    def test_epiweek_to_date_basic(self):
        """Test basic epiweek to date conversion."""
        from datetime import date

        from epymodelingsuite.schema.calibration import FittingWindow

        fw = FittingWindow(start_epiweek=202548, end_epiweek=202601)

        # start_epiweek 202548 -> Sunday 2025-11-23
        assert fw.epiweek_start_date == date(2025, 11, 23)
        # end_epiweek 202601 -> Saturday 2026-01-10
        assert fw.epiweek_end_date == date(2026, 1, 10)

    def test_epiweek_start_is_sunday(self):
        """Test that epiweek_start_date returns Sunday of the week."""
        from epymodelingsuite.schema.calibration import FittingWindow

        fw = FittingWindow(start_epiweek=202501, end_epiweek=202504)

        # 2025 week 1 starts on Sunday 2024-12-29
        start = fw.epiweek_start_date
        assert start.weekday() == 6  # Sunday

    def test_epiweek_end_is_saturday(self):
        """Test that epiweek_end_date returns Saturday of the week."""
        from epymodelingsuite.schema.calibration import FittingWindow

        fw = FittingWindow(start_epiweek=202501, end_epiweek=202504)

        # 2025 week 4 ends on Saturday 2025-01-25
        end = fw.epiweek_end_date
        assert end.weekday() == 5  # Saturday

    def test_year_boundary_epiweek(self):
        """Test epiweek conversion across year boundary."""
        from datetime import date

        from epymodelingsuite.schema.calibration import FittingWindow

        fw = FittingWindow(start_epiweek=202552, end_epiweek=202602)

        # 2025 week 52 starts on Sunday 2025-12-21
        assert fw.epiweek_start_date == date(2025, 12, 21)
        # 2026 week 2 ends on Saturday 2026-01-17
        assert fw.epiweek_end_date == date(2026, 1, 17)

    def test_week_53_handling(self):
        """Test that years with week 53 are handled correctly."""
        from datetime import date

        from epymodelingsuite.schema.calibration import FittingWindow

        # 2025 has a week 53
        fw = FittingWindow(start_epiweek=202553, end_epiweek=202601)

        # Week 53 of 2025 starts on Sunday 2025-12-28
        assert fw.epiweek_start_date == date(2025, 12, 28)
        # Week 1 of 2026 ends on Saturday 2026-01-10
        assert fw.epiweek_end_date == date(2026, 1, 10)

    def test_date_fields_used_when_provided(self):
        """Test that start_date/end_date are used when provided directly."""
        from datetime import date

        from epymodelingsuite.schema.calibration import FittingWindow

        fw = FittingWindow(start_date=date(2025, 3, 1), end_date=date(2025, 6, 30))

        # Should return the provided dates, not compute from epiweeks
        assert fw.epiweek_start_date == date(2025, 3, 1)
        assert fw.epiweek_end_date == date(2025, 6, 30)

    def test_validation_both_epiweeks_required(self):
        """Test that both start_epiweek and end_epiweek must be provided together."""
        from epymodelingsuite.schema.calibration import FittingWindow

        with pytest.raises(ValueError, match="Must supply both start and end epiweek"):
            FittingWindow(start_epiweek=202548)

        with pytest.raises(ValueError, match="Must supply both start and end epiweek"):
            FittingWindow(end_epiweek=202601)

    def test_validation_cannot_mix_date_and_epiweek(self):
        """Test that date fields and epiweek fields cannot be mixed."""
        from datetime import date

        from epymodelingsuite.schema.calibration import FittingWindow

        with pytest.raises(ValueError, match="Cannot use both date fields and epiweek fields"):
            FittingWindow(
                start_date=date(2025, 1, 1),
                end_date=date(2025, 6, 30),
                start_epiweek=202501,
            )

    def test_validation_end_epiweek_after_start(self):
        """Test that end_epiweek must be >= start_epiweek."""
        from epymodelingsuite.schema.calibration import FittingWindow

        with pytest.raises(ValueError, match="start_epiweek cannot be after end_epiweek"):
            FittingWindow(start_epiweek=202510, end_epiweek=202505)
