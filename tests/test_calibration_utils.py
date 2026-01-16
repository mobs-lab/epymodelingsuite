"""Tests for calibration utility functions in epymodelingsuite.calibration module."""

import numpy as np
import pytest
from epydemix.population import load_epydemix_population

from epymodelingsuite.calibration import calc_beta


class TestCalcBeta:
    """Tests for calc_beta function."""

    @pytest.fixture
    def population(self):
        """Load a real population with contact matrices."""
        age_group_mapping = {
            "0-4": [str(i) for i in range(5)],
            "5-17": [str(i) for i in range(5, 18)],
            "18-49": [str(i) for i in range(18, 50)],
            "50-64": [str(i) for i in range(50, 65)],
            "65+": [str(i) for i in range(65, 84)] + ["84+"],
        }
        return load_epydemix_population("United_States_California", age_group_mapping=age_group_mapping)

    def test_returns_float(self, population):
        """Test that calc_beta returns a float."""
        result = calc_beta(
            model_pop=population,
            Rt=1.5,
            mu=1 / 3.0,
            p_asymptomatic=0.3,
            r_beta_asymp=0.5,
        )
        assert isinstance(result, float)

    def test_positive_result(self, population):
        """Test that calc_beta returns a positive value for valid inputs."""
        result = calc_beta(
            model_pop=population,
            Rt=1.5,
            mu=1 / 3.0,
            p_asymptomatic=0.3,
            r_beta_asymp=0.5,
        )
        assert result > 0

    def test_scales_linearly_with_Rt(self, population):
        """Test that beta scales linearly with Rt."""
        mu = 1 / 3.0
        p_asymptomatic = 0.3
        r_beta_asymp = 0.5

        beta_1 = calc_beta(population, Rt=1.0, mu=mu, p_asymptomatic=p_asymptomatic, r_beta_asymp=r_beta_asymp)
        beta_2 = calc_beta(population, Rt=2.0, mu=mu, p_asymptomatic=p_asymptomatic, r_beta_asymp=r_beta_asymp)

        # Doubling Rt should double beta
        assert np.isclose(beta_2 / beta_1, 2.0, rtol=1e-10)

    def test_scales_linearly_with_mu(self, population):
        """Test that beta scales linearly with recovery rate mu."""
        Rt = 1.5
        p_asymptomatic = 0.3
        r_beta_asymp = 0.5

        beta_1 = calc_beta(population, Rt=Rt, mu=0.1, p_asymptomatic=p_asymptomatic, r_beta_asymp=r_beta_asymp)
        beta_2 = calc_beta(population, Rt=Rt, mu=0.2, p_asymptomatic=p_asymptomatic, r_beta_asymp=r_beta_asymp)

        # Doubling mu should double beta
        assert np.isclose(beta_2 / beta_1, 2.0, rtol=1e-10)

    def test_with_no_asymptomatic(self, population):
        """Test calculation when p_asymptomatic=0 (all symptomatic)."""
        Rt = 1.5
        mu = 1 / 3.0

        # When p_asymptomatic=0, r_beta_asymp doesn't matter
        # denominator becomes: eigenvalue * (1 - 0 + 0 * r_beta_asymp) = eigenvalue * 1
        beta_1 = calc_beta(population, Rt=Rt, mu=mu, p_asymptomatic=0.0, r_beta_asymp=0.5)
        beta_2 = calc_beta(population, Rt=Rt, mu=mu, p_asymptomatic=0.0, r_beta_asymp=0.9)

        # Results should be identical since p_asymptomatic=0
        assert np.isclose(beta_1, beta_2, rtol=1e-10)

    def test_with_full_asymptomatic(self, population):
        """Test calculation when p_asymptomatic=1 (all asymptomatic)."""
        Rt = 1.5
        mu = 1 / 3.0
        r_beta_asymp = 0.5

        # When p_asymptomatic=1, denominator becomes: eigenvalue * (1 - 1 + 1 * r_beta_asymp) = eigenvalue * r_beta_asymp
        beta = calc_beta(population, Rt=Rt, mu=mu, p_asymptomatic=1.0, r_beta_asymp=r_beta_asymp)

        # Compare with no asymptomatic case
        beta_no_asymp = calc_beta(population, Rt=Rt, mu=mu, p_asymptomatic=0.0, r_beta_asymp=r_beta_asymp)

        # With full asymptomatic and r_beta_asymp=0.5, beta should be 2x larger
        # because denominator is halved (eigenvalue * 0.5 vs eigenvalue * 1.0)
        assert np.isclose(beta / beta_no_asymp, 1.0 / r_beta_asymp, rtol=1e-10)

    def test_formula_correctness(self, population):
        """Test that the formula matches expected: beta = Rt * mu / (eigenvalue * (1 - p_asymp + p_asymp * r_beta))."""
        Rt = 1.5
        mu = 1 / 3.0
        p_asymptomatic = 0.3
        r_beta_asymp = 0.5

        # Calculate expected value manually
        C = np.sum([c for _, c in population.contact_matrices.items()], axis=0)
        eigenvalue = np.linalg.eigvals(C).real.max()
        expected_beta = Rt * mu / (eigenvalue * (1 - p_asymptomatic + p_asymptomatic * r_beta_asymp))

        result = calc_beta(population, Rt=Rt, mu=mu, p_asymptomatic=p_asymptomatic, r_beta_asymp=r_beta_asymp)

        assert np.isclose(result, expected_beta, rtol=1e-10)

    def test_different_populations_give_different_results(self):
        """Test that different populations (different contact matrices) give different beta values."""
        age_group_mapping = {
            "0-4": [str(i) for i in range(5)],
            "5-17": [str(i) for i in range(5, 18)],
            "18-49": [str(i) for i in range(18, 50)],
            "50-64": [str(i) for i in range(50, 65)],
            "65+": [str(i) for i in range(65, 84)] + ["84+"],
        }

        pop_ca = load_epydemix_population("United_States_California", age_group_mapping=age_group_mapping)
        pop_ny = load_epydemix_population("United_States_New_York", age_group_mapping=age_group_mapping)

        Rt = 1.5
        mu = 1 / 3.0
        p_asymptomatic = 0.3
        r_beta_asymp = 0.5

        beta_ca = calc_beta(pop_ca, Rt=Rt, mu=mu, p_asymptomatic=p_asymptomatic, r_beta_asymp=r_beta_asymp)
        beta_ny = calc_beta(pop_ny, Rt=Rt, mu=mu, p_asymptomatic=p_asymptomatic, r_beta_asymp=r_beta_asymp)

        # Different populations should give different beta values (different contact matrices)
        assert beta_ca != beta_ny
