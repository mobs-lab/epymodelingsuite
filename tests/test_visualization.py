"""Tests for visualization module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.schema.output import FigureOutputTypeEnum
from epymodelingsuite.visualization.core import (
    figure_to_output_object,
    plot_calibration_projection,
    plot_calibration_projection_grid,
    plot_posterior_histogram,
    plot_posterior_histogram_grid,
    plot_quantiles,
    plot_quantiles_grid,
    plot_surveillance_scatter,
)


@pytest.fixture
def quantile_df():
    """Create sample quantile DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    quantiles = [0.025, 0.5, 0.975]
    data = []
    for date in dates:
        for q in quantiles:
            data.append(
                {
                    "date": date,
                    "quantile": q,
                    "hospitalizations": 100 + q * 20,  # Higher quantile = higher value
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def surveillance_df():
    """Create sample surveillance DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [95, 102, 108, 115, 122],
        }
    )


@pytest.fixture
def posterior_df():
    """Create sample posterior DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "R0": rng.normal(1.5, 0.1, 100),
            "beta": rng.uniform(0.1, 0.3, 100),
            "start_date": rng.integers(-10, 10, 100),
        }
    )


class TestPlotQuantiles:
    """Tests for plot_quantiles function."""

    def test_basic_plot(self, quantile_df):
        """Test basic quantile plot."""
        fig, ax = plot_quantiles(
            df_quantiles=quantile_df,
            value_col="hospitalizations",
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_title(self, quantile_df):
        """Test plot with title."""
        fig, ax = plot_quantiles(
            df_quantiles=quantile_df,
            value_col="hospitalizations",
            title="Test Plot",
        )

        assert ax.get_title() == "Test Plot"
        plt.close(fig)

    def test_with_provided_axes(self, quantile_df):
        """Test plot on provided axes."""
        _, ax = plt.subplots()
        fig, returned_ax = plot_quantiles(
            df_quantiles=quantile_df,
            value_col="hospitalizations",
            ax=ax,
        )

        assert fig is None  # Should not create new figure
        assert returned_ax is ax
        plt.close()

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Quantile data is empty"):
            plot_quantiles(df_quantiles=empty_df, value_col="hospitalizations")

    def test_missing_column_raises(self, quantile_df):
        """Test that missing column raises ValueError."""
        with pytest.raises(ValueError, match="Missing columns"):
            plot_quantiles(df_quantiles=quantile_df, value_col="nonexistent")


class TestPlotQuantilesGrid:
    """Tests for plot_quantiles_grid function."""

    def test_basic_grid(self, quantile_df):
        """Test basic grid plot."""
        location_quantiles = {
            "US-CA": quantile_df.copy(),
            "US-TX": quantile_df.copy(),
            "US-NY": quantile_df.copy(),
        }

        fig, axes = plot_quantiles_grid(
            location_quantiles=location_quantiles,
            value_col="hospitalizations",
        )

        assert fig is not None
        assert axes.shape == (1, 4)  # 1 row, 4 panels per row (default)
        # First 3 panels should have titles
        assert axes[0, 0].get_title() == "US-CA"
        assert axes[0, 1].get_title() == "US-TX"
        assert axes[0, 2].get_title() == "US-NY"
        plt.close(fig)

    def test_custom_panels_per_row(self, quantile_df):
        """Test grid with custom panels per row."""
        location_quantiles = {
            "US-CA": quantile_df.copy(),
            "US-TX": quantile_df.copy(),
        }

        fig, axes = plot_quantiles_grid(
            location_quantiles=location_quantiles,
            value_col="hospitalizations",
            panels_per_row=2,
        )

        assert axes.shape == (1, 2)  # 1 row, 2 panels per row
        plt.close(fig)


class TestPlotCalibrationProjection:
    """Tests for plot_calibration_projection function."""

    def test_calibration_only(self, quantile_df):
        """Test plot with calibration data only."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=quantile_df,
            value_col="hospitalizations",
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_projection_only(self, quantile_df):
        """Test plot with projection data only."""
        fig, ax = plot_calibration_projection(
            projection_quantiles=quantile_df,
            value_col="hospitalizations",
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_both_periods(self, quantile_df):
        """Test plot with both calibration and projection."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=quantile_df.copy(),
            projection_quantiles=quantile_df.copy(),
            value_col="hospitalizations",
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_surveillance(self, quantile_df, surveillance_df):
        """Test plot with surveillance overlay."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=quantile_df,
            df_surveillance=surveillance_df,
            value_col="hospitalizations",
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_reference_date(self, quantile_df):
        """Test plot with reference date line."""
        fig, ax = plot_calibration_projection(
            calibration_quantiles=quantile_df,
            value_col="hospitalizations",
            reference_date="2024-01-03",
        )

        assert fig is not None
        assert ax is not None
        # Verify vertical line was added
        assert len(ax.get_lines()) > 0
        plt.close(fig)

    def test_neither_quantiles_raises(self):
        """Test that providing neither quantiles raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            plot_calibration_projection(value_col="hospitalizations")


class TestPlotCalibrationProjectionGrid:
    """Tests for plot_calibration_projection_grid function."""

    def test_basic_grid(self, quantile_df):
        """Test basic grid with multiple locations."""
        cal_quants = {
            "US-CA": quantile_df.copy(),
            "US-TX": quantile_df.copy(),
        }
        proj_quants = {
            "US-CA": quantile_df.copy(),
            "US-TX": quantile_df.copy(),
        }

        fig, axes = plot_calibration_projection_grid(
            location_calibration_quantiles=cal_quants,
            location_projection_quantiles=proj_quants,
            value_col="hospitalizations",
        )

        assert fig is not None
        assert axes.shape == (1, 4)  # 1 row, 4 panels per row (default)
        assert axes[0, 0].get_title() == "US-CA"
        assert axes[0, 1].get_title() == "US-TX"
        plt.close(fig)

    def test_calibration_only(self, quantile_df):
        """Test grid with calibration data only."""
        cal_quants = {
            "US-CA": quantile_df.copy(),
        }

        fig, axes = plot_calibration_projection_grid(
            location_calibration_quantiles=cal_quants,
            value_col="hospitalizations",
        )

        assert fig is not None
        plt.close(fig)

    def test_with_surveillance(self, quantile_df, surveillance_df):
        """Test grid with surveillance data."""
        cal_quants = {"US-CA": quantile_df.copy()}
        surv_dict = {"US-CA": surveillance_df.copy()}

        fig, axes = plot_calibration_projection_grid(
            location_calibration_quantiles=cal_quants,
            location_surveillance=surv_dict,
            value_col="hospitalizations",
        )

        assert fig is not None
        plt.close(fig)

    def test_no_locations_raises(self):
        """Test that providing no locations raises ValueError."""
        with pytest.raises(ValueError, match="No locations provided"):
            plot_calibration_projection_grid(value_col="hospitalizations")


class TestPlotSurveillanceScatter:
    """Tests for plot_surveillance_scatter function."""

    def test_basic_scatter(self, surveillance_df):
        """Test basic surveillance scatter plot."""
        fig, ax = plot_surveillance_scatter(df_surveillance=surveillance_df)

        assert fig is not None
        assert ax is not None
        # Check that scatter points were added
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_with_title(self, surveillance_df):
        """Test scatter plot with title."""
        fig, ax = plot_surveillance_scatter(
            df_surveillance=surveillance_df,
            title="Surveillance Data",
        )

        assert ax.get_title() == "Surveillance Data"
        plt.close(fig)

    def test_custom_color_and_size(self, surveillance_df):
        """Test scatter plot with custom color and size."""
        fig, ax = plot_surveillance_scatter(
            df_surveillance=surveillance_df,
            color="red",
            size=50.0,
        )

        assert fig is not None
        plt.close(fig)

    def test_with_provided_axes(self, surveillance_df):
        """Test scatter plot on provided axes."""
        _, ax = plt.subplots()
        fig, returned_ax = plot_surveillance_scatter(
            df_surveillance=surveillance_df,
            ax=ax,
        )

        assert fig is None  # Should not create new figure
        assert returned_ax is ax
        plt.close()

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Surveillance data is empty"):
            plot_surveillance_scatter(df_surveillance=empty_df)


class TestPlotPosteriorHistogram:
    """Tests for plot_posterior_histogram function."""

    def test_basic_histogram(self, posterior_df):
        """Test basic posterior histogram."""
        fig, ax = plot_posterior_histogram(
            df_posterior=posterior_df,
            parameter="R0",
        )

        assert fig is not None
        assert ax is not None
        # Check that histogram was created
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_multiple_parameters(self, posterior_df):
        """Test histogram for each parameter."""
        for param in ["R0", "beta", "start_date"]:
            fig, ax = plot_posterior_histogram(
                df_posterior=posterior_df,
                parameter=param,
            )
            assert fig is not None
            plt.close(fig)

    def test_custom_bins(self, posterior_df):
        """Test histogram with custom number of bins."""
        fig, ax = plot_posterior_histogram(
            df_posterior=posterior_df,
            parameter="R0",
            bins=50,
        )

        assert fig is not None
        plt.close(fig)

    def test_start_date_with_reference(self, posterior_df):
        """Test start_date parameter with reference date conversion."""
        fig, ax = plot_posterior_histogram(
            df_posterior=posterior_df,
            parameter="start_date",
            start_date_reference="2024-01-15",
        )

        assert fig is not None
        plt.close(fig)

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Posterior data is empty"):
            plot_posterior_histogram(df_posterior=empty_df, parameter="R0")

    def test_missing_parameter_raises(self, posterior_df):
        """Test that missing parameter raises ValueError."""
        with pytest.raises(ValueError, match="Parameter.*not in DataFrame"):
            plot_posterior_histogram(df_posterior=posterior_df, parameter="nonexistent")


class TestPlotPosteriorGrid:
    """Tests for plot_posterior_histogram_grid function."""

    def test_basic_grid(self, posterior_df):
        """Test basic posterior grid."""
        location_posteriors = {
            "US-CA": posterior_df.copy(),
            "US-TX": posterior_df.copy(),
        }

        fig, axes = plot_posterior_histogram_grid(
            location_posteriors=location_posteriors,
            parameters=["R0", "beta"],
        )

        assert fig is not None
        assert axes.shape == (2, 2)  # 2 locations x 2 parameters
        # Check row labels (locations)
        assert axes[0, 0].get_ylabel() == "US-CA"
        assert axes[1, 0].get_ylabel() == "US-TX"
        # Check column labels (parameters)
        assert axes[0, 0].get_title() == "R0"
        assert axes[0, 1].get_title() == "beta"
        plt.close(fig)

    def test_single_location(self, posterior_df):
        """Test grid with single location."""
        location_posteriors = {"US-CA": posterior_df.copy()}

        fig, axes = plot_posterior_histogram_grid(
            location_posteriors=location_posteriors,
            parameters=["R0", "beta", "start_date"],
        )

        assert fig is not None
        assert axes.shape == (1, 3)  # 1 location x 3 parameters
        plt.close(fig)

    def test_with_start_date_reference(self, posterior_df):
        """Test grid with start_date reference."""
        location_posteriors = {"US-CA": posterior_df.copy()}

        fig, axes = plot_posterior_histogram_grid(
            location_posteriors=location_posteriors,
            parameters=["start_date"],
            start_date_reference="2024-01-15",
        )

        assert fig is not None
        plt.close(fig)


class TestFigureToOutputObject:
    """Tests for figure_to_output_object function."""

    def test_png_output(self, quantile_df):
        """Test PNG output creation."""
        fig, _ = plot_quantiles(
            df_quantiles=quantile_df,
            value_col="hospitalizations",
        )

        output_obj = figure_to_output_object(fig, name="test_plot", output_format="png")

        assert output_obj.output_type == FigureOutputTypeEnum.PNG
        assert output_obj.data is not None
        assert isinstance(output_obj.data, bytes)
        plt.close(fig)

    def test_pdf_output(self, quantile_df):
        """Test PDF output creation."""
        fig, _ = plot_quantiles(
            df_quantiles=quantile_df,
            value_col="hospitalizations",
        )

        output_obj = figure_to_output_object(fig, name="test_plot", output_format="pdf")

        assert output_obj.output_type == FigureOutputTypeEnum.PDF
        assert output_obj.data is not None
        assert isinstance(output_obj.data, bytes)
        plt.close(fig)

    def test_svg_output(self, quantile_df):
        """Test SVG output creation."""
        fig, _ = plot_quantiles(
            df_quantiles=quantile_df,
            value_col="hospitalizations",
        )

        output_obj = figure_to_output_object(fig, name="test_plot", output_format="svg")

        assert output_obj.output_type == FigureOutputTypeEnum.SVG
        assert output_obj.data is not None
        assert isinstance(output_obj.data, bytes)  # SVG is returned as bytes
        plt.close(fig)

    def test_default_png(self, quantile_df):
        """Test default output type is PNG."""
        fig, _ = plot_quantiles(
            df_quantiles=quantile_df,
            value_col="hospitalizations",
        )

        output_obj = figure_to_output_object(fig, name="test_plot")

        assert output_obj.output_type == FigureOutputTypeEnum.PNG
        plt.close(fig)
