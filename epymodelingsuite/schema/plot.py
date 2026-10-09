import logging

from epiweeks import Week
from pydantic import BaseModel, Field, field_validator, model_validator

from .common import Meta
from .submission import AggregatedTrajectoriesConfig, SubmissionProfile, SurveillanceConfig

logger = logging.getLogger(__name__)

# ----------------------------------------
# Schema models
# ----------------------------------------


class SingleStrainConfig(BaseModel):
    """
    Source experiment specification for single-strain comparison.

    Experiment identifiers for integration with an external pipeline, e.g. epycloud.
    Used to locate single-strain trajectories for comparison with multistrain.
    """

    bucket: str = Field(description="Path to bucket on cloud e.g. 'gs://gs_mobs_jessica/pipeline/flu'")
    experiment: str = Field(description="Name of experiment in cloud bucket.")
    run_id: str = Field(
        "latest",
        description="Run ID, set to a value from the google cloud bucket, or 'any' (fails if more than one matching trajectory file exists), or 'latest' (default).",
    )
    trajectory_file: str = Field(description="Name of trajectory file from experiment.")
    target_column: str = Field("hospitalizations", description="Name of column in trajectory file with target values.")
    date_column: str = Field("date", description="Name of column in trajectory file with date.")
    location_column: str = Field(
        "population", description="Name of column in trajectory file with location/population."
    )


class PlotTrajectoriesConfig(AggregatedTrajectoriesConfig):
    """
    Specifications for one set of aggregated trajectories to plot.
    """

    trajectory_file: str = Field("", description="Path relative to the `--aggregated` script argument.")


class FitStartConfig(BaseModel):
    """
    Configuration for marking the beginning of a strain's fitting window.
    """

    label: str = Field(description="Label for strain(s).")
    week: str | int = Field(description="Epiweek in CDC format, i.e. 'YYYYww'")

    @field_validator("week")
    @classmethod
    def validate_week(cls, v: str | int) -> str | int:
        """Ensure valid epiweek"""
        try:
            Week.fromstring(str(v), system="cdc", validate=True)
        except Exception as e:
            raise ValueError(f"Failed to parse epiweek: {e}")

        return v


class PlotConfiguration(BaseModel):
    """Configuration for multistrain comparison plots."""

    meta: Meta | None = Field(None, description="General metadata.")
    submission_week: str | int = Field(description="Epiweek of submission in CDC format, i.e. 'YYYYww'")
    profile: SubmissionProfile = Field(description="Submission profile whose location ids label the panels.")
    surveillance: SurveillanceConfig | None = Field(
        None, description="Specification for surveillance file. Expected to be either on GitHub or local."
    )
    aggregated: dict[str, PlotTrajectoriesConfig] = Field(
        min_length=1, description="Aggregated trajectories by label. All are plotted together."
    )
    single_strain: SingleStrainConfig | None = Field(
        None, description="Source experiment specification for single-strain comparison."
    )
    subplots_per_row: int = Field(4, description="Number of subplots to place in each row.")
    season_start_week: str | int = Field(description="Epiweek in CDC format, i.e. 'YYYYww'")
    season_end_week: str | int = Field(description="Epiweek in CDC format, i.e. 'YYYYww'")
    focus_start_week: str | int = Field(description="Epiweek in CDC format, i.e. 'YYYYww'")
    focus_end_week: str | int = Field(description="Epiweek in CDC format, i.e. 'YYYYww'")
    strain_fit_starts: list[FitStartConfig] = Field(
        default_factory=list, description="Specifications for fitting window markers."
    )

    @model_validator(mode="after")
    def validate_field_combinations(self: "PlotConfiguration") -> "PlotConfiguration":
        """Ensure fields are specified consistently"""
        try:
            s1 = Week.fromstring(str(self.season_start_week), system="cdc", validate=True)
            s2 = Week.fromstring(str(self.season_end_week), system="cdc", validate=True)
            f1 = Week.fromstring(str(self.focus_start_week), system="cdc", validate=True)
            f2 = Week.fromstring(str(self.focus_end_week), system="cdc", validate=True)
            # warning for fit starts
            for fit in self.strain_fit_starts:
                w = Week.fromstring(str(fit.week), system="cdc", validate=True)
                if (w < s1) or (w > s2):
                    logger.warning(f"'{fit.label}' fit start week {fit.week} is outside of plotting windows.")
            # success condition
            if s1 <= f1 < f2 <= s2:
                return self
        except Exception as e:
            raise ValueError(f"Failed to parse and compare plotting window epiweeks: {e}")
        # input epiweeks are valid but do not meet success condition
        raise ValueError(
            "Plot window starts must be before plot window ends, and focus window must be within or equal to season window."
        )


class PlotConfig(BaseModel):
    """Root schema for plot configuration YAML files."""

    plot: PlotConfiguration


def validate_plot(config: dict) -> PlotConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    PlotConfig
        The validated configuration.
    """
    try:
        root = PlotConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
