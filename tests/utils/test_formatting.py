"""Tests for formatting utilities."""

import pytest

from epymodelingsuite.utils.formatting import format_data_size, format_duration


class TestFormatDuration:
    """Test format_duration function."""

    def test_seconds_only(self):
        """Test formatting durations less than 60 seconds."""
        assert format_duration(3.5) == "3.5s"
        assert format_duration(0.1) == "0.1s"
        assert format_duration(59.9) == "59.9s"

    def test_minutes_and_seconds(self):
        """Test formatting durations in minutes and seconds."""
        assert format_duration(60) == "1m 0s"
        assert format_duration(125) == "2m 5s"
        assert format_duration(3599) == "59m 59s"

    def test_hours_minutes_seconds(self):
        """Test formatting durations with hours."""
        assert format_duration(3600) == "1h 0m 0s"
        assert format_duration(3665) == "1h 1m 5s"
        assert format_duration(7384) == "2h 3m 4s"
        assert format_duration(86400) == "24h 0m 0s"

    def test_zero(self):
        """Test formatting zero duration."""
        assert format_duration(0) == "0.0s"

    def test_large_duration(self):
        """Test formatting very large durations."""
        # 100 hours
        assert format_duration(360000) == "100h 0m 0s"


class TestFormatDataSize:
    """Test format_data_size function."""

    def test_bytes_auto_selection(self):
        """Test auto unit selection for byte inputs."""
        assert format_data_size(0) == "0 B"
        assert format_data_size(512) == "512 B"
        assert format_data_size(1023) == "1023 B"

    def test_kilobytes_auto_selection(self):
        """Test auto unit selection when result is in KB."""
        assert format_data_size(1024) == "1.0 KB"
        assert format_data_size(2048) == "2.0 KB"
        assert format_data_size(1536) == "1.5 KB"

    def test_megabytes_auto_selection(self):
        """Test auto unit selection when result is in MB."""
        assert format_data_size(1024 * 1024) == "1.0 MB"
        assert format_data_size(2 * 1024 * 1024) == "2.0 MB"
        assert format_data_size(int(1.5 * 1024 * 1024)) == "1.5 MB"

    def test_gigabytes_auto_selection(self):
        """Test auto unit selection when result is in GB."""
        assert format_data_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_data_size(2 * 1024 * 1024 * 1024) == "2.0 GB"

    def test_terabytes_auto_selection(self):
        """Test auto unit selection when result is in TB."""
        assert format_data_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"
        assert format_data_size(int(1.5 * 1024 * 1024 * 1024 * 1024)) == "1.5 TB"

    def test_megabyte_input(self):
        """Test with MB as input unit."""
        assert format_data_size(100, "MB") == "100.0 MB"
        assert format_data_size(1024, "MB") == "1.0 GB"
        assert format_data_size(2048, "MB") == "2.0 GB"
        assert format_data_size(1536, "MB") == "1.5 GB"

    def test_gigabyte_input(self):
        """Test with GB as input unit."""
        assert format_data_size(1, "GB") == "1.0 GB"
        assert format_data_size(2.5, "GB") == "2.5 GB"
        assert format_data_size(1024, "GB") == "1.0 TB"

    def test_kilobyte_input(self):
        """Test with KB as input unit."""
        assert format_data_size(1024, "KB") == "1.0 MB"
        assert format_data_size(512, "KB") == "512.0 KB"

    def test_forced_output_unit(self):
        """Test forcing a specific output unit."""
        assert format_data_size(1024, "MB", output_unit="GB") == "1.0 GB"
        assert format_data_size(2048, output_unit="KB") == "2.0 KB"
        assert format_data_size(1536, output_unit="B") == "1536 B"

    def test_precision_control(self):
        """Test controlling decimal precision."""
        assert format_data_size(1536, precision=2) == "1.50 KB"
        assert format_data_size(1536, precision=0) == "2 KB"  # Rounds up
        assert format_data_size(1024 * 1024 * 1.333, precision=3) == "1.333 MB"

    def test_bytes_always_integer(self):
        """Test that bytes are always displayed as integers."""
        assert format_data_size(512.7) == "512 B"
        assert format_data_size(1000, output_unit="B") == "1000 B"

    def test_invalid_input_unit(self):
        """Test that invalid input units raise ValueError."""
        with pytest.raises(ValueError, match="Invalid input_unit"):
            format_data_size(100, "PB")

    def test_invalid_output_unit(self):
        """Test that invalid output units raise ValueError."""
        with pytest.raises(ValueError, match="Invalid output_unit"):
            format_data_size(100, output_unit="PB")

    def test_backward_compatibility_memory(self):
        """Test that MB input produces same output as old format_memory_mb."""
        # Old: format_memory_mb(100) == "100 MB"
        # But with auto-selection, 100 MB in bytes is very small
        # We need to test actual memory values
        assert format_data_size(512, "MB") == "512.0 MB"
        assert format_data_size(1024, "MB") == "1.0 GB"
        assert format_data_size(1536, "MB") == "1.5 GB"

    def test_backward_compatibility_file_size(self):
        """Test that byte input produces same output as old format_file_size."""
        assert format_data_size(512) == "512 B"
        assert format_data_size(1024) == "1.0 KB"
        assert format_data_size(2048) == "2.0 KB"
        assert format_data_size(1024 * 1024) == "1.0 MB"
