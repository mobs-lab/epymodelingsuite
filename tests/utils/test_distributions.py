"""Tests for distribution conversion utilities."""

from __future__ import annotations

import pytest

from epymodelingsuite.schema.common import Distribution
from epymodelingsuite.utils.distributions import distribution_to_scipy, validate_distribution


class TestDistributionToScipy:
    """Tests for distribution_to_scipy function."""

    def test_uniform_distribution(self):
        """Convert uniform distribution config to scipy."""
        dist_config = Distribution(type="scipy", name="uniform", args=[0, 1])
        scipy_dist = distribution_to_scipy(dist_config)

        # Verify it's a distribution-like object with rvs method
        assert hasattr(scipy_dist, "rvs")
        assert hasattr(scipy_dist, "pdf")
        # uniform(loc=0, scale=1) produces values in [0, 1]
        samples = scipy_dist.rvs(100)
        assert all(0 <= s <= 1 for s in samples)

    def test_norm_distribution(self):
        """Convert normal distribution config to scipy."""
        dist_config = Distribution(type="scipy", name="norm", args=[0, 1])
        scipy_dist = distribution_to_scipy(dist_config)

        assert hasattr(scipy_dist, "rvs")
        assert hasattr(scipy_dist, "pdf")
        # Check mean is approximately 0
        samples = scipy_dist.rvs(1000)
        assert -0.5 < samples.mean() < 0.5

    def test_distribution_with_kwargs(self):
        """Distribution with kwargs converts correctly."""
        # norm with loc and scale as kwargs
        dist_config = Distribution(type="scipy", name="norm", args=[], kwargs={"loc": 10, "scale": 2})
        scipy_dist = distribution_to_scipy(dist_config)

        assert hasattr(scipy_dist, "rvs")
        # Check mean is approximately 10
        samples = scipy_dist.rvs(1000)
        assert 9 < samples.mean() < 11

    def test_distribution_with_args_and_kwargs(self):
        """Distribution with both args and kwargs."""
        # beta distribution with shape params as args, loc/scale as kwargs
        dist_config = Distribution(type="scipy", name="beta", args=[2, 5], kwargs={"loc": 0, "scale": 1})
        scipy_dist = distribution_to_scipy(dist_config)

        assert hasattr(scipy_dist, "rvs")
        samples = scipy_dist.rvs(100)
        assert all(0 <= s <= 1 for s in samples)

    def test_invalid_type_raises_error(self):
        """Custom distributions cannot be converted to SciPy distributions."""
        dist_config = Distribution(type="custom", name="foo", args=[])

        with pytest.raises(ValueError, match="Cannot convert distribution type 'custom' to scipy"):
            distribution_to_scipy(dist_config)

    def test_invalid_name_rejected_during_configuration(self):
        """Invalid SciPy distribution names fail during model validation."""
        with pytest.raises(ValueError, match=r"Unknown scipy\.stats distribution 'not_a_distribution'"):
            Distribution(type="scipy", name="not_a_distribution", args=[])

    def test_non_distribution_scipy_name_rejected(self):
        """Other scipy.stats functions are not accepted as random variables."""
        with pytest.raises(ValueError, match="not a supported scalar continuous or discrete random variable"):
            Distribution(type="scipy", name="describe", args=[])

    def test_uniform_zero_scale_rejected_during_configuration(self):
        """A zero-scale uniform prior is rejected before calibration starts."""
        with pytest.raises(
            ValueError,
            match=r"scale must be a finite scalar greater than 0.*upper bound = loc \+ scale",
        ):
            Distribution(type="scipy", name="uniform", args=[1.0, 0.0])

    def test_negative_keyword_scale_rejected_during_configuration(self):
        """Scale validation supports keyword arguments."""
        with pytest.raises(ValueError, match="scale must be a finite scalar greater than 0"):
            Distribution(type="scipy", name="norm", kwargs={"loc": 0, "scale": -1})

    @pytest.mark.parametrize("scale", [float("inf"), float("nan")])
    def test_non_finite_scale_rejected(self, scale):
        """Continuous distributions require finite scale values."""
        with pytest.raises(ValueError, match="scale must be a finite scalar greater than 0"):
            Distribution(type="scipy", name="norm", kwargs={"scale": scale})

    def test_non_finite_location_rejected(self):
        """All supported distributions require finite scalar locations."""
        with pytest.raises(ValueError, match="loc must be a finite scalar"):
            Distribution(type="scipy", name="poisson", kwargs={"mu": 2, "loc": float("inf")})

    def test_invalid_discrete_parameters_rejected(self):
        """Distribution-specific discrete constraints are checked through support."""
        with pytest.raises(ValueError, match=r"Invalid parameters for scipy\.stats\.randint"):
            Distribution(type="scipy", name="randint", args=[1, 1])

    def test_invalid_continuous_shape_parameters_rejected(self):
        """Distribution-specific continuous shape constraints are checked through support."""
        with pytest.raises(ValueError, match=r"Invalid parameters for scipy\.stats\.beta"):
            Distribution(type="scipy", name="beta", args=[-1, 2])

    def test_validate_distribution_includes_runtime_context(self):
        """Defensive validation errors identify the parameter being converted."""
        dist_config = Distribution.model_construct(type="scipy", name="uniform", args=[1.0, 0.0], kwargs={})

        with pytest.raises(ValueError, match=r"calibration parameter 'alpha'.*scale"):
            validate_distribution(dist_config, context="calibration parameter 'alpha'")

    def test_default_type_is_scipy(self):
        """Distribution with default type (scipy) works correctly."""
        # Don't specify type - should default to "scipy"
        dist_config = Distribution(name="uniform", args=[0, 1])
        scipy_dist = distribution_to_scipy(dist_config)

        assert hasattr(scipy_dist, "rvs")
        samples = scipy_dist.rvs(100)
        assert all(0 <= s <= 1 for s in samples)

    def test_empty_kwargs_uses_default(self):
        """Distribution without kwargs uses empty dict default."""
        dist_config = Distribution(type="scipy", name="norm", args=[0, 1])
        scipy_dist = distribution_to_scipy(dist_config)

        assert hasattr(scipy_dist, "rvs")
        # Should work without kwargs
        samples = scipy_dist.rvs(100)
        assert len(samples) == 100

    def test_randint_discrete_distribution(self):
        """Convert randint (discrete uniform) distribution config to scipy."""
        # randint(low, high) produces integers in [low, high)
        dist_config = Distribution(type="scipy", name="randint", args=[0, 10])
        scipy_dist = distribution_to_scipy(dist_config)

        assert hasattr(scipy_dist, "rvs")
        assert hasattr(scipy_dist, "pmf")  # discrete distributions have pmf, not pdf
        samples = scipy_dist.rvs(100)
        # All samples should be integers in [0, 10)
        assert all(0 <= s < 10 for s in samples)
        assert all(isinstance(s, (int, type(samples[0]))) for s in samples)
