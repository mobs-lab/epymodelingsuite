"""Tests for config utility functions."""

from pathlib import Path

import pytest

from epymodelingsuite.utils.config import identify_config_type


class TestIdentifyConfigType:
    """Tests for identify_config_type function."""

    @pytest.fixture
    def temp_yaml_file(self, tmp_path):
        """Create a temporary YAML file for testing."""

        def _create_yaml(content: str, suffix: str = ".yaml") -> Path:
            yaml_file = tmp_path / f"test{suffix}"
            yaml_file.write_text(content)
            return yaml_file

        return _create_yaml

    def test_identify_basemodel_config(self, temp_yaml_file):
        """Test identifying basemodel config with 'model' key."""
        yaml_file = temp_yaml_file("""
model:
  compartments:
    - S
    - I
    - R
""")
        assert identify_config_type(str(yaml_file)) == "basemodel"

    def test_identify_calibration_config(self, temp_yaml_file):
        """Test identifying calibration config with 'modelset.calibration' key."""
        yaml_file = temp_yaml_file("""
modelset:
  calibration:
    strategy: abc
    n_samples: 1000
""")
        assert identify_config_type(str(yaml_file)) == "calibration"

    def test_identify_sampling_config(self, temp_yaml_file):
        """Test identifying sampling config with 'modelset.sampling' key."""
        yaml_file = temp_yaml_file("""
modelset:
  sampling:
    parameters:
      beta:
        distribution: uniform
        min: 0.1
        max: 0.5
""")
        assert identify_config_type(str(yaml_file)) == "sampling"

    def test_identify_output_config(self, temp_yaml_file):
        """Test identifying output config with 'output' key."""
        yaml_file = temp_yaml_file("""
output:
  quantiles:
    - 0.025
    - 0.5
    - 0.975
  trajectories:
    format: csv
""")
        assert identify_config_type(str(yaml_file)) == "output"

    def test_identify_output_config_minimal(self, temp_yaml_file):
        """Test identifying output config with minimal 'output' key."""
        yaml_file = temp_yaml_file("""
output:
  quantiles:
    - 0.5
""")
        assert identify_config_type(str(yaml_file)) == "output"

    def test_unrecognized_config_returns_none(self, temp_yaml_file):
        """Test that unrecognized config structure returns None."""
        yaml_file = temp_yaml_file("""
some_other_key:
  data: value
""")
        assert identify_config_type(str(yaml_file)) is None

    def test_empty_file_returns_none(self, temp_yaml_file):
        """Test that empty YAML file returns None."""
        yaml_file = temp_yaml_file("")
        assert identify_config_type(str(yaml_file)) is None

    def test_non_dict_yaml_returns_none(self, temp_yaml_file):
        """Test that non-dict YAML content returns None."""
        yaml_file = temp_yaml_file("- item1\n- item2\n")
        assert identify_config_type(str(yaml_file)) is None

    def test_file_not_found_raises_error(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            identify_config_type(str(nonexistent))

    def test_non_yaml_file_raises_error(self, tmp_path):
        """Test that non-YAML file raises ValueError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some content")
        with pytest.raises(ValueError, match="must be a YAML file"):
            identify_config_type(str(txt_file))

    def test_accepts_yml_extension(self, temp_yaml_file):
        """Test that .yml extension is accepted."""
        yaml_file = temp_yaml_file(
            """
model:
  compartments:
    - S
""",
            suffix=".yml",
        )
        assert identify_config_type(str(yaml_file)) == "basemodel"

    def test_modelset_without_calibration_or_sampling_returns_none(self, temp_yaml_file):
        """Test that modelset without calibration or sampling returns None."""
        yaml_file = temp_yaml_file("""
modelset:
  other_key: value
""")
        assert identify_config_type(str(yaml_file)) is None
