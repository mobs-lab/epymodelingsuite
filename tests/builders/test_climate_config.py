"""Tests for data-driven (climate) seasonality configs: schema validation and model build."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epymodelingsuite.config_loader import load_basemodel_config_from_file
from epymodelingsuite.dispatcher.builder import build_basemodel
from epymodelingsuite.schema.basemodel import BasemodelConfig, Seasonality

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CLIMATE_FIXTURE = FIXTURES_DIR / "minimal_basemodel_climate.yaml"

# Fixture params: beta=0.3, b1=1e-4, b3=0.01, s_min=0.2 over tests/data/climate_daily_test.csv
# (US-CA, 3 days). The data-driven multiplier is normalised to [s_min, 1], so the
# peak day reproduces the baseline beta (0.3) and the trough day equals 0.3 * s_min.
BASELINE_BETA = 0.3
S_MIN = 0.2


class TestClimateBaseModelSchemaValidation:
    """Data-driven seasonality config schema round-trips correctly."""

    def test_loads_without_error(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        assert isinstance(cfg, BasemodelConfig)

    def test_seasonality_method_is_data_driven(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        assert cfg.model.seasonality is not None
        assert cfg.model.seasonality.method == Seasonality.SeasonalityMethodEnum.data_driven

    def test_seasonality_target_is_beta(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        assert cfg.model.seasonality.target_parameter == "beta"

    def test_climate_coeffs_present(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        params = cfg.model.parameters
        assert "b1" in params
        assert "b3" in params
        assert "s_min" in params

    def test_missing_seasonality_data_path_raises(self):
        with pytest.raises(Exception, match="seasonality_data_path"):
            Seasonality(
                method="data_driven",
                target_parameter="beta",
                # seasonality_data_path intentionally omitted
            )


class TestClimateBaseModelBuild:
    """build_basemodel applies data-driven seasonality to beta."""

    @pytest.fixture(scope="class")
    def built(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        return build_basemodel(basemodel_config=cfg)

    def test_build_succeeds(self, built):
        assert built is not None

    def test_beta_is_time_varying_array(self, built):
        beta = built.model.get_parameter("beta")
        assert isinstance(beta, np.ndarray), "beta should be an ndarray after data-driven seasonality"
        assert beta.ndim >= 1

    def test_beta_length_matches_timespan(self, built):
        # 2024-01-01 to 2024-01-03 inclusive = 3 days
        beta = built.model.get_parameter("beta")
        assert beta.shape[0] == 3

    def test_all_beta_values_positive(self, built):
        beta = built.model.get_parameter("beta")
        assert np.all(beta > 0), f"Some beta values non-positive: {beta}"

    def test_beta_varies_across_days(self, built):
        # Climate data has different RH/temp each day so beta should not be constant
        beta = built.model.get_parameter("beta")
        assert not np.allclose(beta, beta[0]), "beta should vary day-to-day with climate forcing"

    def test_peak_beta_equals_baseline(self, built):
        # Normalisation rescales the multiplier to [s_min, 1]; the peak day reproduces baseline beta.
        beta = built.model.get_parameter("beta")
        assert np.isclose(beta.max(), BASELINE_BETA, rtol=1e-6), (
            f"Peak beta should equal baseline {BASELINE_BETA}, got {beta.max()}"
        )

    def test_trough_beta_equals_baseline_times_s_min(self, built):
        # The trough day gets the minimum multiplier s_min.
        beta = built.model.get_parameter("beta")
        assert np.isclose(beta.min(), BASELINE_BETA * S_MIN, rtol=1e-6), (
            f"Trough beta should equal baseline*s_min = {BASELINE_BETA * S_MIN}, got {beta.min()}"
        )

    def test_beta_never_below_floor(self, built):
        # Multiplier is bounded below by s_min, so beta never drops below baseline*s_min.
        beta = built.model.get_parameter("beta")
        assert np.all(beta >= BASELINE_BETA * S_MIN - 1e-9)
