"""Tests for output schema validation."""

from datetime import date

import pytest

from epymodelingsuite.schema.output import (
    CategoricalPlotConfig,
    FlusightForecastOutput,
    FlusightPropED,
    ObservedValuesConfig,
    OutputConfiguration,
    OutputOptions,
    PropEDStrategyEnum,
)


class TestFlusightPropEDValidation:
    """Test FlusightPropED schema validation."""

    def test_transition_strategy_without_ed_source(self):
        """Test that transition strategy works without ed_source specified."""
        # This should not raise validation errors
        prop_ed = FlusightPropED(
            strategy=PropEDStrategyEnum.transition,
            transition_name="ed_prop",
        )
        assert prop_ed.strategy == PropEDStrategyEnum.transition
        assert prop_ed.transition_name == "ed_prop"
        assert prop_ed.ed_source is None

    def test_transition_strategy_requires_transition_name(self):
        """Test that transition strategy requires transition_name."""
        with pytest.raises(ValueError, match="requires field 'transition_name'"):
            FlusightPropED(strategy=PropEDStrategyEnum.transition)

    def test_calibration_window_requires_ed_source(self):
        """Test that calibration_window strategy requires ed_source."""
        with pytest.raises(ValueError, match="requires field 'ed_source'"):
            FlusightPropED(
                strategy=PropEDStrategyEnum.calibration_window,
                num_fit_weeks=4,
            )

    def test_surveillance_window_requires_ed_source(self):
        """Test that surveillance_window strategy requires ed_source."""
        with pytest.raises(ValueError, match="requires field 'ed_source'"):
            FlusightPropED(
                strategy=PropEDStrategyEnum.surveillance_window,
                hosp_source="hosp",
                fit_start=date(2025, 1, 1),
                fit_end=date(2025, 2, 1),
            )


class TestOutputConfigurationSurveillanceValidation:
    """Test OutputConfiguration surveillance reference validation."""

    def test_prop_ed_transition_without_surveillance_options(self):
        """Test that prop_ed with transition strategy doesn't require surveillance options."""
        # This should raise an error because prop_ed without transition strategy
        # requires surveillance sources
        config = OutputConfiguration(
            flusight_format=FlusightForecastOutput(
                reference_date=date(2025, 1, 1),
                hospitalizations=None,
                prop_ed=FlusightPropED(
                    strategy=PropEDStrategyEnum.transition,
                    transition_name="ed_prop",
                ),
            ),
        )
        # Should not raise - transition strategy doesn't need surveillance sources
        # because ed_source is None and not checked when None
        assert config.flusight_format.prop_ed.strategy == PropEDStrategyEnum.transition

    def test_prop_ed_with_ed_source_validates_surveillance(self):
        """Test that ed_source is validated against available surveillance sources."""
        with pytest.raises(ValueError, match="not found in surveillance sources"):
            OutputConfiguration(
                options=OutputOptions(
                    surveillance={
                        "hosp": ObservedValuesConfig(
                            data_path="/tmp/data.csv",
                            value_column="value",
                            date_column="date",
                            location_column="location",
                            location_format="ISO",
                        ),
                    }
                ),
                flusight_format=FlusightForecastOutput(
                    reference_date=date(2025, 1, 1),
                    hospitalizations=None,
                    prop_ed=FlusightPropED(
                        strategy=PropEDStrategyEnum.calibration_window,
                        ed_source="nonexistent_source",  # This doesn't exist
                        num_fit_weeks=4,
                    ),
                ),
            )

    def test_prop_ed_transition_with_valid_surveillance(self):
        """Test that prop_ed transition strategy works with surveillance defined (even though not needed)."""
        config = OutputConfiguration(
            options=OutputOptions(
                surveillance={
                    "prop_ed": ObservedValuesConfig(
                        data_path="/tmp/data.csv",
                        value_column="value",
                        date_column="date",
                        location_column="location",
                        location_format="ISO",
                    ),
                }
            ),
            flusight_format=FlusightForecastOutput(
                reference_date=date(2025, 1, 1),
                hospitalizations=None,
                prop_ed=FlusightPropED(
                    strategy=PropEDStrategyEnum.transition,
                    transition_name="ed_prop",
                    # ed_source is None (not specified)
                ),
            ),
        )
        assert config.flusight_format.prop_ed.ed_source is None


class TestCategoricalPlotConfigValidation:
    """Test CategoricalPlotConfig schema validation."""

    def test_categorical_plot_config_defaults(self):
        """Test CategoricalPlotConfig default values."""
        config = CategoricalPlotConfig()
        assert len(config.categories) == 5
        assert config.categories == ["large_decrease", "decrease", "stable", "increase", "large_increase"]
        assert len(config.colors) == 5
        assert config.colors == ["#476a6f", "#519e8a", "#b7c3f3", "#dd7596", "#cf1259"]
        assert config.horizons == [0, 1, 2, 3]
        assert config.figsize is None

    def test_categorical_plot_config_custom_values(self):
        """Test CategoricalPlotConfig with custom values."""
        config = CategoricalPlotConfig(
            categories=["low", "medium", "high"],
            colors=["#000000", "#808080", "#FFFFFF"],
            horizons=[0, 1],
            figsize=(12, 10),
        )
        assert config.categories == ["low", "medium", "high"]
        assert config.colors == ["#000000", "#808080", "#FFFFFF"]
        assert config.horizons == [0, 1]
        assert config.figsize == (12, 10)

    def test_categorical_plot_config_color_validation(self):
        """Test validation that colors list matches categories list."""
        with pytest.raises(ValueError, match="colors list length .* must match categories list length"):
            CategoricalPlotConfig(
                categories=["a", "b", "c"],
                colors=["#000000", "#111111"],  # Wrong length (2 instead of 3)
            )

    def test_categorical_plot_config_color_validation_with_defaults(self):
        """Test that default categories and colors have matching lengths."""
        # This should not raise - defaults are designed to match
        config = CategoricalPlotConfig(
            categories=["large_decrease", "decrease", "stable", "increase", "large_increase"],
            # colors will use default which also has 5 elements
        )
        assert len(config.categories) == len(config.colors)


class TestObservedValuesConfigLocationFormat:
    """Tests for ObservedValuesConfig location_format validation."""

    def test_default_location_format_is_iso(self):
        """Test that default location_format is ISO."""
        config = ObservedValuesConfig(
            data_path="/tmp/data.csv",
            value_column="value",
            date_column="date",
            location_column="location",
        )
        assert config.location_format == "ISO"

    def test_iso_location_format_is_valid(self):
        """Test that ISO location format is accepted."""
        config = ObservedValuesConfig(
            data_path="/tmp/data.csv",
            value_column="value",
            date_column="date",
            location_column="location",
            location_format="ISO",
        )
        assert config.location_format == "ISO"

    def test_fips_location_format_is_valid(self):
        """Test that FIPS location format is accepted."""
        config = ObservedValuesConfig(
            data_path="/tmp/data.csv",
            value_column="value",
            date_column="date",
            location_column="location",
            location_format="FIPS",
        )
        assert config.location_format == "FIPS"

    def test_metrocast_location_id_format_is_valid(self):
        """Test that metrocast_location_id location format is accepted.

        This is a regression test to ensure metrocast location format
        is properly supported in output configs.
        """
        config = ObservedValuesConfig(
            data_path="/tmp/data.csv",
            value_column="value",
            date_column="date",
            location_column="location",
            location_format="metrocast_location_id",
        )
        assert config.location_format == "metrocast_location_id"

    def test_epydemix_population_format_is_valid(self):
        """Test that epydemix_population location format is accepted."""
        config = ObservedValuesConfig(
            data_path="/tmp/data.csv",
            value_column="value",
            date_column="date",
            location_column="location",
            location_format="epydemix_population",
        )
        assert config.location_format == "epydemix_population"

    def test_invalid_location_format_raises_error(self):
        """Test that invalid location format raises ValueError."""
        with pytest.raises(ValueError, match="location_format must be one of"):
            ObservedValuesConfig(
                data_path="/tmp/data.csv",
                value_column="value",
                date_column="date",
                location_column="location",
                location_format="invalid_format",
            )
