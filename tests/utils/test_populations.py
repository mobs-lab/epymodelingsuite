"""Tests for population utilities in utils/populations.py."""

import pandas as pd

from epymodelingsuite.utils.populations import aggregate_population_by_age_groups


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
        pop_dict = {i: 100 for i in range(10)}  # ages 0-9, 100 each

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
