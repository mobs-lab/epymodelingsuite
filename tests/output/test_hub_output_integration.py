"""Integration tests for hub output generation, storage, and validation."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from epydemix.calibration import CalibrationResults

from epymodelingsuite.dispatcher.output import generate_calibration_outputs
from epymodelingsuite.output.hub_files import read_hub_table
from epymodelingsuite.output.hub_validation import load_hub_tasks, validate_model_output
from epymodelingsuite.schema.dispatcher import CalibrationOutput
from epymodelingsuite.schema.output import OutputConfig, OutputConfiguration, TabularOutputTypeEnum


def test_generated_hub_parquet_passes_validation(tmp_path):
    """Validate dispatcher-generated hospitalization and ED forecasts after Parquet storage."""
    # Use real result objects so quantiles and samples come from the same projections.
    # Extra weeks exercise horizon filtering; 150 distinct paths require selecting 100.
    dates = pd.date_range("2026-09-26", periods=7, freq="7D").to_list()
    calibrations = []
    for primary_id, population in enumerate(["United_States", "United_States_Alabama"]):
        hosp = np.arange(150 * len(dates), dtype=float).reshape(150, len(dates)) + primary_id + 0.5
        ed = 0.01 + hosp / 100_000
        calibrations.append(
            CalibrationOutput(
                primary_id=primary_id,
                population=population,
                seed=42,
                delta_t=1.0,
                results=CalibrationResults(
                    projections={
                        "baseline": [
                            {"date": dates, "hospitalizations": h, "ed_prop": e}
                            for h, e in zip(hosp, ed, strict=True)
                        ]
                    }
                ),
            )
        )
    config = OutputConfig(
        output=OutputConfiguration(
            tabular_output_types=[TabularOutputTypeEnum.Parquet],
            flusight_format={
                "reference_date": date(2026, 10, 10),
                "prop_ed": {"strategy": "transition", "transition_name": "ed_prop"},
                "samples": {"n_samples": 100},
            },
        )
    )

    outputs = generate_calibration_outputs(calibrations=calibrations, output_config=config)
    (parquet_output,) = outputs["output_hub_formatted"]
    assert parquet_output.name == "output_hub_formatted.parquet"
    path = tmp_path / parquet_output.name
    path.write_bytes(parquet_output.data)
    submission = read_hub_table(path)

    # Validation can accept omitted optional targets/types, so explicitly require
    # both targets and both output types at each location, including leading-zero IDs.
    targets = ["wk inc flu hosp", "wk inc flu prop ed visits"]
    counts = {"quantile": len(config.output.flusight_format.quantiles) * 5, "sample": 100 * 5}
    assert submission.groupby(["location", "target", "output_type"]).size().to_dict() == {
        (location, target, output_type): count
        for location in ["US", "01"]
        for target in targets
        for output_type, count in counts.items()
    }
    assert set(submission.horizon) == {-1, 0, 1, 2, 3}
    # Validate only generated rows against the unchanged official tasks fixture;
    # no official example rows or mocked formatting/serialization fill in the output.
    tasks = load_hub_tasks(Path(__file__).parent.parent / "fixtures" / "flusight" / "tasks.json")
    errors = validate_model_output(submission, tasks)
    assert errors == [], "\n".join(errors)
