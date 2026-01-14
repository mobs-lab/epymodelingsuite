"""Tests for visualization core functions."""

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from epymodelingsuite.visualization.core import (
    plot_calibration_projection,
    plot_categorical_stacked_bars,
    plot_categorical_stacked_bars_multihorizon,
)


class TestCategoricalStackedBars:
    """Tests for categorical stacked bar plots."""

    def test_single_horizon_basic(self):
        """Test basic categorical plot for single horizon."""
        # Create sample data with 3 locations, 5 categories
        df = pd.DataFrame(
            {
                "location": ["US-CA", "US-CA", "US-CA", "US-TX", "US-TX", "US-TX", "US-NY", "US-NY", "US-NY"],
                "output_type_id": [
                    "stable",
                    "increase",
                    "decrease",
                    "stable",
                    "increase",
                    "decrease",
                    "stable",
                    "increase",
                    "decrease",
                ],
                "value": [0.5, 0.3, 0.2, 0.4, 0.4, 0.2, 0.6, 0.2, 0.2],
            }
        )
        categories = ["decrease", "stable", "increase"]
        colors = ["#519e8a", "#b7c3f3", "#dd7596"]

        fig, ax = plot_categorical_stacked_bars(
            df_categorical=df,
            categories=categories,
            colors=colors,
        )

        # Assert: figure created
        assert fig is not None
        assert ax is not None

        # Assert: correct number of bars (one per location)
        assert len(ax.patches) == 9  # 3 locations * 3 categories

        # Assert: y-axis limits
        assert ax.get_ylim() == (0, 1.0)

        # Assert: ylabel is empty
        assert ax.get_ylabel() == ""

        # Clean up
        plt.close(fig)

    def test_missing_categories(self):
        """Test handling when some locations lack certain categories."""
        # Create data with missing categories for some locations
        df = pd.DataFrame(
            {
                "location": ["US-CA", "US-CA", "US-TX"],  # US-TX missing some categories
                "output_type_id": ["stable", "increase", "stable"],
                "value": [0.7, 0.3, 1.0],
            }
        )
        categories = ["stable", "increase", "decrease"]
        colors = ["#b7c3f3", "#dd7596", "#519e8a"]

        fig, ax = plot_categorical_stacked_bars(
            df_categorical=df,
            categories=categories,
            colors=colors,
        )

        # Should not raise error - missing categories filled with 0.0
        assert fig is not None
        assert ax is not None

        # Clean up
        plt.close(fig)

    def test_empty_data_raises_error(self):
        """Test error raised on empty DataFrame."""
        df = pd.DataFrame()
        categories = ["stable", "increase"]
        colors = ["#b7c3f3", "#dd7596"]

        with pytest.raises(ValueError, match="Cannot create categorical plot: data is empty"):
            plot_categorical_stacked_bars(
                df_categorical=df,
                categories=categories,
                colors=colors,
            )

    def test_missing_columns_raises_error(self):
        """Test error raised when required columns missing."""
        df = pd.DataFrame(
            {
                "location": ["US-CA"],
                "value": [0.5],
                # Missing "output_type_id" column
            }
        )
        categories = ["stable"]
        colors = ["#b7c3f3"]

        with pytest.raises(ValueError, match="Required columns missing from data"):
            plot_categorical_stacked_bars(
                df_categorical=df,
                categories=categories,
                colors=colors,
            )

    def test_color_category_mismatch_raises_error(self):
        """Test error when colors/categories lists have different lengths."""
        df = pd.DataFrame(
            {
                "location": ["US-CA"],
                "output_type_id": ["stable"],
                "value": [0.5],
            }
        )
        categories = ["stable", "increase", "decrease"]
        colors = ["#b7c3f3", "#dd7596"]  # Only 2 colors for 3 categories

        with pytest.raises(ValueError, match="categories list length .* must match colors list length"):
            plot_categorical_stacked_bars(
                df_categorical=df,
                categories=categories,
                colors=colors,
            )

    def test_with_provided_axes(self):
        """Test plotting on provided axes."""
        df = pd.DataFrame(
            {
                "location": ["US-CA", "US-TX"],
                "output_type_id": ["stable", "stable"],
                "value": [0.5, 0.6],
            }
        )
        categories = ["stable"]
        colors = ["#b7c3f3"]

        # Create axes
        fig_ext, ax_ext = plt.subplots()

        # Plot on provided axes
        returned_fig, returned_ax = plot_categorical_stacked_bars(
            df_categorical=df,
            categories=categories,
            colors=colors,
            ax=ax_ext,
        )

        # When axes provided, returned fig should be None
        assert returned_fig is None
        assert returned_ax is ax_ext

        # Clean up
        plt.close(fig_ext)

    def test_multihorizon_vertical_stack(self):
        """Test multi-horizon plot with vertical stack (4x1 grid)."""
        # Create data for horizons 0-3
        data_rows = []
        for horizon in [0, 1, 2, 3]:
            for location in ["US-CA", "US-TX"]:
                for category in ["stable", "increase"]:
                    data_rows.append(
                        {
                            "location": location,
                            "horizon": horizon,
                            "output_type_id": category,
                            "value": 0.5,
                        }
                    )

        df = pd.DataFrame(data_rows)
        categories = ["stable", "increase"]
        colors = ["#b7c3f3", "#dd7596"]
        horizons = [0, 1, 2, 3]

        fig, axes = plot_categorical_stacked_bars_multihorizon(
            df_categorical=df,
            categories=categories,
            colors=colors,
            horizons=horizons,
        )

        # Assert: correct grid shape (4 rows, 1 col)
        assert axes.shape == (4, 1)

        # Assert: all subplots created
        for i in range(4):
            assert axes[i, 0] is not None

        # Assert: figure title
        assert "Rate-change trend forecasts" in fig._suptitle.get_text()

        # Clean up
        plt.close(fig)

    def test_multihorizon_with_reference_date(self):
        """Test multi-horizon plot with reference date for panel titles."""
        # Create data
        data_rows = []
        for horizon in [0, 1]:
            data_rows.append(
                {
                    "location": "US-CA",
                    "horizon": horizon,
                    "output_type_id": "stable",
                    "value": 0.5,
                }
            )

        df = pd.DataFrame(data_rows)
        categories = ["stable"]
        colors = ["#b7c3f3"]
        horizons = [0, 1]
        reference_date = datetime(2025, 11, 26)

        fig, axes = plot_categorical_stacked_bars_multihorizon(
            df_categorical=df,
            categories=categories,
            colors=colors,
            horizons=horizons,
            reference_date=reference_date,
        )

        # Assert: panel titles contain dates
        title_0 = axes[0, 0].get_title()
        title_1 = axes[1, 0].get_title()

        assert "2 week ahead" in title_0
        assert "3 week ahead" in title_1
        assert "2025-12-10" in title_0  # 2 weeks after 2025-11-26
        assert "2025-12-17" in title_1  # 3 weeks after 2025-11-26

        # Clean up
        plt.close(fig)

    def test_multihorizon_custom_figsize(self):
        """Test multi-horizon plot with custom figsize."""
        # Create minimal data
        df = pd.DataFrame(
            {
                "location": ["US-CA"],
                "horizon": [0],
                "output_type_id": ["stable"],
                "value": [1.0],
            }
        )
        categories = ["stable"]
        colors = ["#b7c3f3"]
        horizons = [0]
        custom_figsize = (15, 8)

        fig, axes = plot_categorical_stacked_bars_multihorizon(
            df_categorical=df,
            categories=categories,
            colors=colors,
            horizons=horizons,
            figsize=custom_figsize,
        )

        # Assert: figure has custom size
        assert fig.get_figwidth() == custom_figsize[0]
        assert fig.get_figheight() == custom_figsize[1]

        # Clean up
        plt.close(fig)


class TestPlotCalibrationProjection:
    """Tests for plot_calibration_projection function."""

    @pytest.fixture
    def sample_quantile_data(self):
        """Create sample quantile data for testing."""
        dates = pd.date_range("2024-10-01", "2024-12-31", freq="W-SAT")
        rows = []
        for date in dates:
            for q in [0.025, 0.25, 0.5, 0.75, 0.975]:
                rows.append({"date": date, "quantile": q, "hospitalizations": 100 + q * 50})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("xlabel_interval", ["W-SAT", "2W-SAT", "MS"])
    def test_xlabel_interval_with_valid_dates(self, sample_quantile_data, xlabel_interval):
        """Test xlabel_interval correctly converts matplotlib dates to datetime."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=sample_quantile_data,
            xlabel_interval=xlabel_interval,
        )

        assert fig is not None
        assert ax is not None

        # Verify x-axis has ticks set
        xticks = ax.get_xticks()
        assert len(xticks) > 0

        plt.close(fig)

    def test_xlabel_interval_with_projection_data(self, sample_quantile_data):
        """Test xlabel_interval works with projection data."""
        # Create projection data with later dates
        proj_dates = pd.date_range("2025-01-01", "2025-03-01", freq="W-SAT")
        proj_rows = []
        for date in proj_dates:
            for q in [0.025, 0.25, 0.5, 0.75, 0.975]:
                proj_rows.append({"date": date, "quantile": q, "hospitalizations": 150 + q * 50})
        proj_data = pd.DataFrame(proj_rows)

        fig, ax = plot_calibration_projection(
            calibration_quantiles=sample_quantile_data,
            projection_quantiles=proj_data,
            xlabel_interval="2W-SAT",
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_without_xlabel_interval(self, sample_quantile_data):
        """Test plot works without xlabel_interval (default behavior)."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=sample_quantile_data,
            xlabel_interval=None,
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_ylabel_sets_axis_label(self, sample_quantile_data):
        """Test ylabel parameter sets the y-axis label."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=sample_quantile_data,
            ylabel="Hospitalizations",
        )

        assert ax.get_ylabel() == "Hospitalizations"

        plt.close(fig)

    def test_ylabel_none_no_label(self, sample_quantile_data):
        """Test ylabel=None results in no y-axis label."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=sample_quantile_data,
            ylabel=None,
        )

        assert ax.get_ylabel() == ""

        plt.close(fig)
