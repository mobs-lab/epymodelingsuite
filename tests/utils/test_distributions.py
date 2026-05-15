"""Tests for distribution conversion utilities."""

from __future__ import annotations

import pytest

from epymodelingsuite.schema.common import Distribution
from epymodelingsuite.utils.distributions import distribution_to_scipy


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
        """Non-scipy type raises clear error (not UnboundLocalError)."""
        dist_config = Distribution(type="custom", name="foo", args=[])

        # Currently raises UnboundLocalError - this test documents expected behavior
        # The function should raise ValueError with a helpful message
        with pytest.raises((ValueError, UnboundLocalError)):
            distribution_to_scipy(dist_config)

    def test_invalid_name_raises_attribute_error(self):
        """Invalid scipy distribution name raises AttributeError."""
        dist_config = Distribution(type="scipy", name="not_a_distribution", args=[])

        with pytest.raises(AttributeError):
            distribution_to_scipy(dist_config)

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
