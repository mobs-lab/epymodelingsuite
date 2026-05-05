"""Integration tests for telemetry with real dispatcher workflows."""

from textwrap import dedent

import pandas as pd
import pytest

from epymodelingsuite.config_loader import (
    load_basemodel_config_from_file,
    load_calibration_config_from_file,
)
from epymodelingsuite.dispatcher.builder import dispatch_builder
from epymodelingsuite.dispatcher.runner import dispatch_runner
from epymodelingsuite.telemetry import ExecutionTelemetry, create_workflow_telemetry

# Simplified flu basemodel config (minimal compartments for fast testing)
BASEMODEL_YAML = dedent("""
model:
  meta:
    description: "Minimal flu model for telemetry testing"
    author: "test"
    version: "1.0"

  timespan:
    start_date: calibrated
    end_date: "2024-12-31"
    delta_t: 1.0

  simulation:
    resample_frequency: W-SAT

  random_seed: 42

  population:
    age_groups:
      - "0-17"
      - "18-49"
      - "50-64"
      - "65+"

  compartments:
    - id: S
      label: "Susceptible"
      init: default
    - id: I
      label: "Infectious"
      init: calibrated
    - id: R
      label: "Recovered"

  transitions:
    - type: mediated
      source: S
      target: I
      mediator: I
      rate: beta
    - type: spontaneous
      source: I
      target: R
      rate: gamma

  parameters:
    beta:
      type: calibrated
    gamma:
      type: scalar
      value: 0.2
""")
# Simplified calibration config
CALIBRATION_YAML = dedent("""
modelset:
  meta:
    description: "Minimal calibration for telemetry testing"

  population_names:
    - US-MA

  calibration:
    strategy:
      name: smc
      options:
        num_particles: 1
        num_generations: 1

    distance_function: rmse
    observed_data_path: PLACEHOLDER

    comparison:
      - observed_date_column: date
        observed_value_column: cases
        simulation:
          - S_to_I_total

    fitting_window:
      start_date: "2024-01-01"
      end_date: "2024-03-31"

    compartments:
      I:
        prior:
          type: scipy
          name: uniform
          args: [0.001, 0.01]

    start_date:
      reference_date: 2024-01-01
      prior:
        type: scipy
        name: randint
        args: [0, 10]

    parameters:
      beta:
        prior:
          type: scipy
          name: uniform
          args: [0.1, 0.5]
""")


@pytest.fixture
def flu_configs(tmp_path):
    """Create flu calibration configs with minimal settings and mock data."""
    # Create mock observed data
    observed_data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", "2024-03-31", freq="W-SAT"),
            "geo_value": "US-MA",
            "cases": [100, 150, 200, 180, 160, 140, 120, 110, 100, 90, 85, 80, 75],
        }
    )
    data_path = tmp_path / "observed_data.csv"
    observed_data.to_csv(data_path, index=False)

    # Write config files
    basemodel_path = tmp_path / "basemodel.yml"
    calibration_path = tmp_path / "calibration.yml"

    basemodel_path.write_text(BASEMODEL_YAML)
    calibration_yaml_with_path = CALIBRATION_YAML.replace("PLACEHOLDER", str(data_path))
    calibration_path.write_text(calibration_yaml_with_path)

    # Load configs
    basemodel_config = load_basemodel_config_from_file(str(basemodel_path))
    calibration_config = load_calibration_config_from_file(str(calibration_path))

    return basemodel_config, calibration_config


def test_flu_calibration_telemetry_builder(flu_configs):
    """Test telemetry in builder stage with flu calibration config."""
    basemodel_config, calibration_config = flu_configs

    with ExecutionTelemetry() as telemetry:
        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

    # Check that we got outputs
    assert isinstance(builder_outputs, list)
    assert len(builder_outputs) == 1

    # Check telemetry text output
    text = telemetry.to_text()

    # Verify age groups display
    assert "Age groups:" in text
    assert "0-17, 18-49, 50-64, 65+" in text

    # Verify calibration subsection
    assert "Calibration:" in text
    assert "Fitting window:" in text
    assert "2024-01-01 to 2024-03-31" in text
    assert "Distance function: rmse" in text

    # Verify population
    assert "US-MA" in text or "United_States_Massachusetts" in text


def test_flu_calibration_telemetry_runner(flu_configs):
    """Test telemetry in runner stage with actual calibration."""
    basemodel_config, calibration_config = flu_configs

    # Build
    with ExecutionTelemetry() as builder_telemetry:
        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

    builder_output = builder_outputs[0]

    # Run calibration
    with ExecutionTelemetry() as runner_telemetry:
        runner_output = dispatch_runner(builder_output)

    # Check telemetry text output
    text = runner_telemetry.to_text()

    # Verify runner stage info
    assert "RUNNER STAGE" in text
    assert "Calibration:" in text
    assert "Strategy: smc" in text
    assert "1 particles, 1 generations" in text or "1 particle, 1 generation" in text

    # Should NOT have calibration subsection in runner (only in builder/config)
    lines = text.split("\n")
    config_section = False
    for line in lines:
        if "CONFIGURATION" in line:
            config_section = True
        if "RUNNER STAGE" in line:
            config_section = False
        # Distance function should only appear in config section, not runner
        if "Distance function:" in line:
            assert config_section, "Distance function should only be in configuration section"


def test_flu_calibration_telemetry_workflow(flu_configs):
    """Test complete workflow telemetry aggregation."""
    basemodel_config, calibration_config = flu_configs

    # Build
    with ExecutionTelemetry() as builder_telemetry:
        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

    # Run
    with ExecutionTelemetry() as runner_telemetry:
        runner_output = dispatch_runner(builder_outputs[0])

    # Create workflow telemetry
    workflow_telemetry = create_workflow_telemetry(
        builder_telemetry=builder_telemetry,
        runner_telemetries=[runner_telemetry],
        output_telemetry=None,
    )

    text = workflow_telemetry.to_text()

    # Verify workflow type
    assert "Workflow: Calibration" in text

    # Verify all three major sections present
    assert "CONFIGURATION" in text
    assert "BUILDER STAGE" in text
    assert "RUNNER STAGE" in text

    # Verify age groups in configuration
    assert "Age groups: 4 (0-17, 18-49, 50-64, 65+)" in text

    # Verify calibration subsection in configuration
    assert "Calibration:" in text
    assert "Fitting window: 2024-01-01 to 2024-03-31" in text
    assert "Distance function: rmse" in text

    # Verify calibration strategy in runner section
    assert "Strategy: smc" in text
    assert "1 particle" in text or "1 particles" in text

    # Verify timing information
    assert "Duration:" in text
    assert "Peak memory:" in text


def test_simulation_no_calibration_subsection():
    """Test that simulation workflow has no calibration subsection."""
    config = load_basemodel_config_from_file("tutorials/data/basic_basemodel_standalone.yml")
    config.model.vaccination = None  # Disable vaccination to avoid data file dependency

    with ExecutionTelemetry() as telemetry:
        builder_output = dispatch_builder(basemodel_config=config)

    text = telemetry.to_text()

    # Should have age groups
    assert "Age groups:" in text

    # Should NOT have calibration subsection
    assert "Calibration:" not in text
    assert "Distance function:" not in text
    assert "Fitting window:" not in text
