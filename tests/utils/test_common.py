"""Tests for common utility functions."""

from datetime import timedelta

import pytest

from flumodelingsuite.utils import parse_timedelta
from flumodelingsuite.utils.common import parse_transition_name, strip_agegroup_suffix, to_set


class TestParseTimedelta:
    """Tests for parse_timedelta function."""

    def test_parse_minutes_lowercase(self):
        """Test parsing minutes with lowercase 'm'."""
        result = parse_timedelta("30m")
        assert result == timedelta(minutes=30)

    def test_parse_hours_uppercase(self):
        """Test parsing hours with uppercase 'H'."""
        result = parse_timedelta("2H")
        assert result == timedelta(hours=2)

    def test_parse_hours_lowercase(self):
        """Test parsing hours with lowercase 'h'."""
        result = parse_timedelta("4h")
        assert result == timedelta(hours=4)

    def test_parse_minutes_uppercase_t(self):
        """Test parsing minutes with uppercase 'T' (time)."""
        result = parse_timedelta("15T")
        assert result == timedelta(minutes=15)

    def test_parse_seconds(self):
        """Test parsing seconds."""
        result = parse_timedelta("45s")
        assert result == timedelta(seconds=45)

    def test_parse_days_lowercase(self):
        """Test parsing days with lowercase 'd'."""
        result = parse_timedelta("3d")
        assert result == timedelta(days=3)

    def test_parse_days_uppercase(self):
        """Test parsing days with uppercase 'D'."""
        result = parse_timedelta("3D")
        assert result == timedelta(days=3)

    def test_parse_week_single(self):
        """Test parsing single week with 'W'."""
        result = parse_timedelta("W")
        assert result == timedelta(weeks=1)

    def test_parse_weeks_multiple(self):
        """Test parsing multiple weeks."""
        result = parse_timedelta("2W")
        assert result == timedelta(weeks=2)

    def test_parse_compound_duration(self):
        """Test parsing compound duration like '1h30m'."""
        result = parse_timedelta("1h30m")
        assert result == timedelta(hours=1, minutes=30)

    def test_parse_compound_days_hours(self):
        """Test parsing compound duration like '2D3H'."""
        result = parse_timedelta("2D3H")
        assert result == timedelta(days=2, hours=3)

    def test_parse_with_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        result = parse_timedelta("  30m  ")
        assert result == timedelta(minutes=30)

    def test_parse_fractional_hours(self):
        """Test parsing fractional hours."""
        result = parse_timedelta("1.5H")
        assert result == timedelta(hours=1, minutes=30)

    def test_parse_week_with_anchor_sunday(self):
        """Test parsing week with Sunday anchor."""
        result = parse_timedelta("W-SUN")
        assert result == timedelta(weeks=1)

    def test_parse_week_with_anchor_monday(self):
        """Test parsing week with Monday anchor."""
        result = parse_timedelta("W-MON")
        assert result == timedelta(weeks=1)

    def test_invalid_month_raises_error(self):
        """Test that variable-length month duration raises ValueError."""
        with pytest.raises(ValueError, match="not a fixed-length duration"):
            parse_timedelta("M")

    def test_invalid_year_raises_error(self):
        """Test that variable-length year duration raises ValueError."""
        with pytest.raises(ValueError, match="not a fixed-length duration"):
            parse_timedelta("A")

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized duration"):
            parse_timedelta("")

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized duration"):
            parse_timedelta("invalid")

    def test_zero_minutes(self):
        """Test parsing zero duration."""
        result = parse_timedelta("0m")
        assert result == timedelta(0)

    def test_large_duration(self):
        """Test parsing large duration."""
        result = parse_timedelta("1000H")
        assert result == timedelta(hours=1000)

    def test_microseconds(self):
        """Test parsing microseconds."""
        result = parse_timedelta("1000us")
        assert result == timedelta(microseconds=1000)

    def test_milliseconds(self):
        """Test parsing milliseconds."""
        result = parse_timedelta("500ms")
        assert result == timedelta(milliseconds=500)

    def test_nanoseconds(self):
        """Test parsing nanoseconds."""
        result = parse_timedelta("1000ns")
        # Note: timedelta precision is microseconds, so 1000ns = 1us
        assert result == timedelta(microseconds=1)


class TestToSet:
    """Tests for to_set function."""

    def test_none_input(self):
        """Test converting None to empty set."""
        assert to_set(None) == set()

    def test_list_input(self):
        """Test converting list to set."""
        assert to_set([1, 2, 3]) == {1, 2, 3}

    def test_set_input(self):
        """Test converting set to set."""
        assert to_set({1, 2}) == {1, 2}

    def test_empty_list(self):
        """Test converting empty list to empty set."""
        assert to_set([]) == set()


class TestStripAgegroupSuffix:
    """Tests for strip_agegroup_suffix function."""

    def test_with_total_suffix(self):
        """Test stripping _total suffix from names (default)."""
        assert strip_agegroup_suffix("S_total") == "S"
        assert strip_agegroup_suffix("Hosp_total") == "Hosp"

    def test_without_suffix(self):
        """Test names without age group suffix."""
        assert strip_agegroup_suffix("S") == "S"
        assert strip_agegroup_suffix("Hosp") == "Hosp"

    def test_empty_string(self):
        """Test empty string."""
        assert strip_agegroup_suffix("") == ""

    def test_with_numeric_age_group(self):
        """Test stripping numeric age group suffixes like 0-9."""
        assert strip_agegroup_suffix("S_0-9", age_group="0-9") == "S"
        assert strip_agegroup_suffix("I_10-19", age_group="10-19") == "I"
        assert strip_agegroup_suffix("Hosp_65+", age_group="65+") == "Hosp"

    def test_with_custom_age_group(self):
        """Test stripping custom age group suffixes."""
        assert strip_agegroup_suffix("S_child", age_group="child") == "S"
        assert strip_agegroup_suffix("I_adult", age_group="adult") == "I"
        assert strip_agegroup_suffix("R_elderly", age_group="elderly") == "R"


class TestParseTransitionName:
    """Tests for parse_transition_name function."""

    def test_valid_format_total(self):
        """Test parsing valid transition name format with _total."""
        assert parse_transition_name("S_to_I_total") == ("S", "I")
        assert parse_transition_name("Home_sev_to_Hosp_total") == ("Home_sev", "Hosp")

    def test_without_suffix(self):
        """Test parsing transition name without age group suffix."""
        assert parse_transition_name("S_to_I") == ("S", "I")

    def test_with_numeric_age_group(self):
        """Test parsing with numeric age group suffixes."""
        assert parse_transition_name("S_to_I_0-9", age_group="0-9") == ("S", "I")
        assert parse_transition_name("I_to_R_10-19", age_group="10-19") == ("I", "R")
        assert parse_transition_name("Hosp_to_Death_65+", age_group="65+") == ("Hosp", "Death")

    def test_with_custom_age_group(self):
        """Test parsing with custom age group suffixes."""
        assert parse_transition_name("S_to_I_child", age_group="child") == ("S", "I")
        assert parse_transition_name("I_to_R_adult", age_group="adult") == ("I", "R")

    def test_invalid_format_no_to(self):
        """Test invalid format without _to_ separator."""
        with pytest.raises(ValueError, match="Invalid transition name format"):
            parse_transition_name("S_I_total")

    def test_invalid_format_multiple_to(self):
        """Test invalid format with multiple _to_ separators."""
        with pytest.raises(ValueError, match="Invalid transition name format"):
            parse_transition_name("S_to_I_to_R_total")
