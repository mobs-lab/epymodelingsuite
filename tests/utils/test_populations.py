"""Tests for population utilities in utils/populations.py."""

import pandas as pd
import pytest

from epymodelingsuite.utils.populations import (
    aggregate_population_by_age_groups,
    get_age_group_mapping,
    get_all_age_map,
    validate_age_groups,
)


class TestAggregatePopulationByAgeGroups:
    """Tests for aggregate_population_by_age_groups function."""

    def test_dataframe_input_simple(self):
        """Test aggregation with DataFrame input."""
        df = pd.DataFrame(
            {
                "age": [0, 1, 2, 3, 4, 5, 6, 7],
                "population": [100, 100, 100, 100, 100, 200, 200, 200],
            }
        )
        result = aggregate_population_by_age_groups(df, ["0-4", "5+"])
        assert result["0-4"] == 500
        # 5+ includes ages 5-83 and 84+, but we only have 5, 6, 7 in data
        assert result["5+"] == 600

    def test_dataframe_input_standard_age_groups(self):
        """Test aggregation with standard flu forecasting age groups."""
        # Create data for ages 0-84+
        ages = list(range(85)) + ["84+"]
        populations = [1000] * 85 + [500]  # 1000 per age, 500 for 84+
        df = pd.DataFrame({"age": ages, "population": populations})

        result = aggregate_population_by_age_groups(df, ["0-4", "5-17", "18-49", "50-64", "65+"])

        assert result["0-4"] == 5000  # ages 0-4
        assert result["5-17"] == 13000  # ages 5-17
        assert result["18-49"] == 32000  # ages 18-49
        assert result["50-64"] == 15000  # ages 50-64
        # 65+ includes 65-83 (19 ages) + 84+ entry
        assert result["65+"] == 19000 + 500

    def test_series_input(self):
        """Test aggregation with pandas Series input."""
        # Series indexed by position (0, 1, 2, ..., 84)
        data = [100] * 85  # 100 per single-year age
        series = pd.Series(data)

        result = aggregate_population_by_age_groups(series, ["0-4", "5+"])

        assert result["0-4"] == 500  # 5 ages * 100
        # 5+ includes ages 5-84 (80 ages)
        assert result["5+"] == 8000

    def test_dict_input(self):
        """Test aggregation with dict input."""
        pop_dict = dict.fromkeys(range(10), 100)  # ages 0-9, 100 each

        result = aggregate_population_by_age_groups(pop_dict, ["0-4", "5+"])

        assert result["0-4"] == 500
        # Only ages 5-9 present in dict, rest will be 0
        assert result["5+"] == 500

    def test_dict_input_with_84_plus(self):
        """Test dict input handles 84+ age correctly."""
        pop_dict = {84: 500}  # Only 84+ population

        result = aggregate_population_by_age_groups(pop_dict, ["0-64", "65+"])

        assert result["0-64"] == 0
        assert result["65+"] == 500

    def test_empty_dataframe_returns_zeros(self):
        """Test that empty DataFrame returns zeros for all age groups."""
        df = pd.DataFrame({"age": [], "population": []})

        result = aggregate_population_by_age_groups(df, ["0-4", "5+"])

        assert result["0-4"] == 0
        assert result["5+"] == 0

    def test_preserves_age_group_order(self):
        """Test that result preserves the order of input age groups."""
        df = pd.DataFrame(
            {
                "age": list(range(85)),
                "population": [100] * 85,
            }
        )
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        result = aggregate_population_by_age_groups(df, age_groups)

        assert list(result.keys()) == age_groups

    def test_dataframe_string_ages(self):
        """Test DataFrame with string age values."""
        df = pd.DataFrame(
            {
                "age": ["0", "1", "2", "84+"],
                "population": [100, 100, 100, 500],
            }
        )

        result = aggregate_population_by_age_groups(df, ["0-2", "3+"])

        assert result["0-2"] == 300
        assert result["3+"] == 500

    def test_missing_ages_in_dict_treated_as_zero(self):
        """Test that missing ages in dict are treated as zero population."""
        pop_dict = {0: 100, 2: 100}  # Missing age 1

        result = aggregate_population_by_age_groups(pop_dict, ["0-2", "3+"])

        assert result["0-2"] == 200  # 100 + 0 + 100
        assert result["3+"] == 0


class TestValidateAgeGroups:
    """Tests for validate_age_groups function."""

    def test_valid_standard_age_groups(self):
        """Test that standard flu forecasting age groups pass validation."""
        # Should not raise
        validate_age_groups(["0-4", "5-17", "18-49", "50-64", "65+"])

    def test_valid_simple_age_groups(self):
        """Test simple two-group validation."""
        validate_age_groups(["0-17", "18+"])

    def test_valid_single_year_groups(self):
        """Test single-year age groups pass validation."""
        validate_age_groups(["0-0", "1-1", "2+"])

    def test_invalid_first_group_not_starting_at_zero(self):
        """Test that age groups not starting at 0 raise error."""
        with pytest.raises(ValueError, match="first age group must start at '0'"):
            validate_age_groups(["5-17", "18+"])

    def test_invalid_last_group_missing_plus(self):
        """Test that last age group must end with +."""
        with pytest.raises(ValueError, match="last age group must end with '\\+'"):
            validate_age_groups(["0-17", "18-64"])

    def test_invalid_non_contiguous_groups(self):
        """Test that non-contiguous age groups raise error."""
        with pytest.raises(ValueError, match="contiguous"):
            validate_age_groups(["0-4", "10-17", "18+"])  # Gap between 4 and 10

    def test_invalid_overlapping_groups(self):
        """Test that overlapping age groups raise error."""
        with pytest.raises(ValueError, match="contiguous"):
            validate_age_groups(["0-10", "5-17", "18+"])  # 5-10 overlaps

    def test_invalid_format_missing_dash(self):
        """Test that age groups without dash raise error."""
        with pytest.raises(ValueError, match="format"):
            validate_age_groups(["04", "5+"])


class TestGetAgeGroupMapping:
    """Tests for get_age_group_mapping function."""

    def test_simple_two_groups(self):
        """Test mapping with two age groups."""
        result = get_age_group_mapping(["0-4", "5+"])

        assert result["0-4"] == ["0", "1", "2", "3", "4"]
        assert "5" in result["5+"]
        assert "84+" in result["5+"]

    def test_standard_flu_age_groups(self):
        """Test mapping with standard flu forecasting age groups."""
        result = get_age_group_mapping(["0-4", "5-17", "18-49", "50-64", "65+"])

        assert result["0-4"] == ["0", "1", "2", "3", "4"]
        assert result["5-17"] == [str(i) for i in range(5, 18)]
        assert result["18-49"] == [str(i) for i in range(18, 50)]
        assert result["50-64"] == [str(i) for i in range(50, 65)]
        assert result["65+"] == [str(i) for i in range(65, 84)] + ["84+"]

    def test_plus_group_includes_84_plus(self):
        """Test that open-ended groups include ages up to 83 plus '84+'."""
        result = get_age_group_mapping(["0-64", "65+"])

        assert "65+" in result
        assert "83" in result["65+"]
        assert "84+" in result["65+"]
        assert len(result["65+"]) == 20  # 65-83 (19 ages) + "84+"

    def test_single_year_age_groups(self):
        """Test mapping with single-year age groups."""
        result = get_age_group_mapping(["0-0", "1-1", "2+"])

        assert result["0-0"] == ["0"]
        assert result["1-1"] == ["1"]
        assert "2" in result["2+"]

    def test_returns_dict_of_string_lists(self):
        """Test that all values in mapping are lists of strings."""
        result = get_age_group_mapping(["0-4", "5+"])

        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, list)
            assert all(isinstance(age, str) for age in value)

    def test_preserves_input_order(self):
        """Test that mapping preserves order of input age groups."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        result = get_age_group_mapping(age_groups)

        assert list(result.keys()) == age_groups


class TestGetAllAgeMap:
    """Tests for get_all_age_map function."""

    def test_returns_85_age_groups(self):
        """Test that all age map contains 85 age groups (0-84+)."""
        result = get_all_age_map()

        assert len(result) == 85

    def test_single_year_groups_have_one_element(self):
        """Test that each single-year group maps to a single age."""
        result = get_all_age_map()

        assert result["0-0"] == ["0"]
        assert result["1-1"] == ["1"]
        assert result["50-50"] == ["50"]
        assert result["83-83"] == ["83"]

    def test_84_plus_group(self):
        """Test that 84+ group maps correctly."""
        result = get_all_age_map()

        assert "84+" in result
        assert result["84+"] == ["84+"]

    def test_all_ages_covered(self):
        """Test that all ages from 0-84+ are covered."""
        result = get_all_age_map()

        # Check first few
        for i in range(10):
            assert f"{i}-{i}" in result

        # Check some middle ages
        assert "40-40" in result
        assert "60-60" in result

        # Check last regular age and 84+
        assert "83-83" in result
        assert "84+" in result

    def test_returns_dict_of_string_lists(self):
        """Test that return type is correct."""
        result = get_all_age_map()

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, list)
            assert all(isinstance(age, str) for age in value)
