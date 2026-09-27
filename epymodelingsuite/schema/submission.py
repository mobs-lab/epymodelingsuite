import logging
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .common import Meta

logger = logging.getLogger(__name__)

# Keys of epymodelingsuite.multistrain.formatter.SUBMISSION_PROFILES
SubmissionProfile = Literal["flusight_hosp", "flusight_ed", "metrocast", "bphc_ed"]

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
    location_column: str = Field(
        "location",
        description="Column with hub location ids (FIPS for FluSight, e.g. 'location_code' in hosp files; metrocast id).",
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
    profile: SubmissionProfile = Field(description="Hub format of the submission file.")
    model_name: str = Field(description="'<team>-<model>' part of the submission filename.")
    surveillance: SurveillanceConfig | None = Field(
        None, description="Surveillance file for the rate-trend baseline. Only needed by 'flusight_hosp'."
    )
    aggregated: AggregatedTrajectoriesConfig = Field(description="Specifications for aggregated trajectories.")

    @model_validator(mode="after")
    def check_surveillance(self) -> "SubmissionConfiguration":
        """Rate-trend targets need surveillance for the baseline."""
        if self.profile == "flusight_hosp" and self.surveillance is None:
            raise ValueError("Profile 'flusight_hosp' requires 'surveillance' (baseline for rate-trend).")
        return self


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
