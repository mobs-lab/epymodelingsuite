import logging
from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

from .common import Meta

# from ..workflow_dispatcher import get_flusight_quantiles

logger = logging.getLogger(__name__)


def get_flusight_quantiles() -> list[float]:
    """
    Return an array containing the quantiles needed for FluSight submissions.
    The set of quantiles is defined at https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#quantile-output
    """
    import numpy as np

    return np.append(np.append([0.01, 0.025], np.arange(0.05, 0.95 + 0.05, 0.05)), [0.975, 0.99]).astype(float).tolist()


class FluScenariosOutput(BaseModel):
    """Specifications for quantile outputs in Flu Scenario Modeling Hub format."""


class Covid19ForecastOutput(BaseModel):
    """Specifications for quantile outputs in Covid 19 Forecast Hub format."""


class FlusightForecastOutput(BaseModel):
    """Specifications for quantile outputs in flusight forecast hub format."""

    reference_date: date = Field(
        description="'YYYY-MM-DD' date to treat as reference date when creating horizons and target dates for submission file."
    )
    target_transitions: list[str] = Field(
        description="Identifiers of transitions to count for target data, as encoded by epydemix e.g. ['Home_sev_to_Hosp_total', 'Home_sev_vax_to_Hosp_vax_total']"
    )
    rate_trends: bool = Field(
        False,
        description="Add rate-trend forecasts to submission file.",
    )


class DefaultQuantilesOutput(BaseModel):
    """Specifications for quantile outputs in default format."""

    compartments: list[str] | bool = Field(
        False,
        description="Return quantiles for compartments. Set `True` to get all compartments, or provide a list of identifiers (e.g. 'I_total') to select compartments.",
    )
    transitions: list[str] | bool = Field(
        False,
        description="Return quantiles for transitions. Set `True` to get all transitions, or provide a list of identifiers (e.g. 'I_to_R_total') to select transitions.",
    )


class QuantilesOutput(BaseModel):
    """Specifications for quantile outputs."""

    selections: list[float] | None = Field(
        default_factory=get_flusight_quantiles,
        description="Desired quantiles expressed as floats.",
    )
    default_format: DefaultQuantilesOutput | None = Field(
        None, description="Specifications for quantile outputs in default format."
    )
    flusight_format: FlusightForecastOutput | None = Field(
        None, description="Specifications for quantile outputs in FluSight Forecast Hub format."
    )
    covid19_format: Covid19ForecastOutput | None = Field(
        None, description="Specifications for quantile outputs in Covid 19 Forecast Hub format."
    )
    flusmh_format: FluScenariosOutput | None = Field(
        None, description="Specifications for quantile outputs in Flu Scenario Modeling Hub format."
    )

    @model_validator(mode="after")
    def check_formats(self):
        """Ensure output format selections are compatible."""
        hub_formats = [self.flusight_format, self.covid19_format, self.flusmh_format]

        # if self.simulation_default and self.calibration_default:
        #    raise ValueError("Received specifications for both simulation and calibration quantile outputs.")
        if len([1 for _ in hub_formats if bool(_)]) > 1:
            raise ValueError("Received specifications for more than one hub format.")
        # if self.simulation_default and (self.flusight_format or self.covid19_format):
        #    raise ValueError("Simulation results are incompatible with forecast hub formats.")

        return self

    @field_validator("selections")
    @classmethod
    def check_selections(cls, v):
        """Ensure quantiles are in (0, 1)."""
        if not all([0.0 < q < 1.0 for q in v]):
            raise ValueError("Received quantile not in (0, 1).")
        return v


class TrajectoriesOutput(BaseModel):
    """Specifications for trajectory outputs."""

    compartments: list[str] | bool = Field(
        False,
        description="Return trajectories for compartments. Set `True` to get all compartments, or provide a list of identifiers (e.g. 'I_total') to select compartments.",
    )
    transitions: list[str] | bool = Field(
        False,
        description="Return trajectories for transitions. Set `True` to get all transitions, or provide a list of identifiers (e.g. 'I_to_R_total') to select transitions.",
    )


class PosteriorsOutput(BaseModel):
    """Specifications for posterior outputs."""

    generations: list[int] | None = Field(None, description="Generations of SMC to get posteriors for")


class ModelMetaOutput(BaseModel):
    """Specifications for parameter tracking / run metadata outputs."""

    projection_parameters: bool = Field(
        False,
        description="Whether to record projection parameters (calibration parameters always recorded in calibration workflow).",
    )


class OutputConfiguration(BaseModel):
    """Output configuration."""

    meta: Meta | None = Field(None, description="General metadata.")
    quantiles: QuantilesOutput | None = Field(None, description="Specifications for quantile outputs.")
    trajectories: TrajectoriesOutput | None = Field(None, description="Specifications for trajectory outputs.")
    posteriors: PosteriorsOutput | None = Field(None, description="Specifications for posterior outputs.")
    model_meta: ModelMetaOutput = Field(
        default_factory=ModelMetaOutput, description="Specifications for parameter tracking / model metadata outputs."
    )


class OutputConfig(BaseModel):
    """Root configuration model."""

    outputs: OutputConfiguration = Field(description="Modelset configuration")


def validate_output(config: dict) -> OutputConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    OutputConfig
        The validated configuration.
    """
    try:
        root = OutputConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
