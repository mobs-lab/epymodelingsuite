"""Tests for climate seasonality configs: schema validation and model build."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epymodelingsuite.config_loader import load_basemodel_config_from_file
from epymodelingsuite.dispatcher.builder import build_basemodel
from epymodelingsuite.schema.basemodel import BasemodelConfig, Seasonality

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CLIMATE_FIXTURE = FIXTURES_DIR / "minimal_basemodel_climate.yaml"


class TestClimateBaseModelSchemaValidation:
    """Seasonality config schema round-trips correctly."""

    def test_loads_without_error(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        assert isinstance(cfg, BasemodelConfig)

    def test_seasonality_method_is_climate(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        assert cfg.model.seasonality is not None
        assert cfg.model.seasonality.method == Seasonality.SeasonalityMethodEnum.climate

    def test_seasonality_target_is_beta(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        assert cfg.model.seasonality.target_parameter == "beta"

    def test_climate_coeffs_present(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        params = cfg.model.parameters
        assert "b1" in params
        assert "b2" in params
        assert "b3" in params

    def test_missing_climate_data_path_raises(self):
        with pytest.raises(Exception, match="climate_data_path"):
            Seasonality(
                method="climate",
                target_parameter="beta",
                # climate_data_path intentionally omitted
            )


class TestClimateBaseModelBuild:
    """build_basemodel applies climate seasonality to beta."""

    @pytest.fixture(scope="class")
    def built(self):
        cfg = load_basemodel_config_from_file(str(CLIMATE_FIXTURE))
        return build_basemodel(basemodel_config=cfg)

    def test_build_succeeds(self, built):
        assert built is not None

    def test_beta_is_time_varying_array(self, built):
        beta = built.model.get_parameter("beta")
        assert isinstance(beta, np.ndarray), "beta should be an ndarray after climate seasonality"
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

    def test_beta_baseline_preserved(self, built):
        # Fixture has beta=0.3, b2=1.0 (dominant term at rh_optimum).
        # At day 0 (RH=40=rh_optimum, T=5°C): factor = 0 + 1.0 + b3*(min_T - 5).
        # min_T = min(5, 10, 5) = 5, so factor = 1.0 exactly.
        beta = built.model.get_parameter("beta")
        assert np.isclose(beta[0], 0.3, rtol=1e-6), (
            f"Day 0 beta should be 0.3 (factor=1.0 when RH=rh_optimum and T=min_T), got {beta[0]}"
        )

    def test_clamped_beta_never_below_minimum(self, built):
        # Even if formula goes negative, our max(st, 1e-6) clamp ensures a floor.
        beta = built.model.get_parameter("beta")
        assert np.all(beta >= 1e-6 * 0.3 - 1e-12)


class TestClimateModelsetSchemaValidation:
    """Tutorials modelset_climate.yml validates and has expected priors."""

    MODELSET_YAML = FIXTURES_DIR.parent.parent / "tutorials" / "data" / "modelset_climate.yml"

    @pytest.fixture(scope="class")
    def raw(self):
        import yaml
        with open(self.MODELSET_YAML) as f:
            return yaml.safe_load(f)

    def test_modelset_file_exists(self):
        assert self.MODELSET_YAML.exists()

    def test_has_b1_b2_b3_priors(self, raw):
        params = raw["modelset"]["calibration"]["parameters"]
        assert "b1" in params
        assert "b2" in params
        assert "b3" in params

    def test_b3_prior_is_non_negative(self, raw):
        b3_args = raw["modelset"]["calibration"]["parameters"]["b3"]["prior"]["args"]
        assert b3_args[0] >= 0, "b3 lower bound should be >= 0"

    def test_basemodel_climate_yml_validates(self):
        tutorials_basemodel = FIXTURES_DIR.parent.parent / "tutorials" / "data" / "basemodel_climate.yml"
        cfg = load_basemodel_config_from_file(str(tutorials_basemodel))
        assert cfg.model.seasonality is not None
        assert cfg.model.seasonality.method == Seasonality.SeasonalityMethodEnum.climate
        assert "b1" in cfg.model.parameters
        assert "b2" in cfg.model.parameters
        assert "b3" in cfg.model.parameters
