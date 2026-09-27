import logging

from pydantic import BaseModel, Field

from .common import Meta

logger = logging.getLogger(__name__)

# ----------------------------------------
# Schema models
# ----------------------------------------


class SurveillanceConfig(BaseModel):
    """
    Specification for surveillance files. Expected to be either on GitHub or local.
    """

    directory: str = Field(description="")
    fit_fname: str = Field(description="")
    recent_fname: str | None = Field(None, description="")
    target_column: str | None = Field(
        "hospitalizations", description="Name of column in trajectory file with target values."
    )
    date_column: str | None = Field("target_end_date", description="Name of column in trajectory file with date.")
    location_column: str | None = Field(
        "location_iso", description="Name of column in trajectory file with location/population."
    )


class AggregatedTrajectoriesConfig(BaseModel):
    """
    Specifications for aggregated trajectories.
    """

    target_column: str = Field(description="Name of column in trajectory file with target values.")
    date_column: str = Field("date", description="Name of column in trajectory file with target date.")
    location_column: str = Field("location", description="Name of column in trajectory file with location/population.")
    week_column: str = Field("epiweek", description="Name of column in trajectory file with target epiweek.")


class SubmissionConfiguration(BaseModel):
    """Configuration for building a hub submission file from aggregated trajectories."""

    meta: Meta | None = Field(None, description="General metadata.")
    submission_week: str | int = Field(description="Epiweek of submission in CDC format, i.e. 'YYYYww'")
    model_name: str = Field(description="'<team>-<model>' part of the submission filename.")
    surveillance: SurveillanceConfig = Field(
        description="Specification for surveillance file. Expected to be either on GitHub or local."
    )
    aggregated: AggregatedTrajectoriesConfig = Field(description="Specifications for aggregated trajectories.")


class SubmissionConfig(BaseModel):
    """Root schema for submission configuration YAML files."""

    submission: SubmissionConfiguration


def validate_submission(config: dict) -> SubmissionConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    SubmissionConfig
        The validated configuration.
    """
    try:
        root = SubmissionConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
