"""Schema definitions for workflow dispatcher."""

from datetime import date

from epydemix.calibration import ABCSampler, CalibrationResults
from epydemix.model import EpiModel
from epydemix.model.simulation_results import SimulationResults
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calibration import CalibrationStrategy


class SimulationArguments(BaseModel):
    """
    Arguments for a single call to epydemix.EpiModel.run_simulations
    Follows https://epydemix.readthedocs.io/en/stable/epydemix.model.html#epydemix.model.epimodel.EpiModel.run_simulations
    """

    start_date: date = Field(description="Start date of the simulation.")
    end_date: date = Field(description="End date of the simulation.")
    initial_conditions_dict: dict | None = Field(None, description="Initial conditions dictionary.")
    Nsim: int | None = Field(10, description="Number of simulation runs to perform for a single EpiModel.")
    dt: float | None = Field(1.0, description="Timestep for simulation, defaults to 1.0 = 1 day.")
    resample_frequency: str | None = Field(
        None,
        description="The frequency at which to resample the simulation results. Follows https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases",
    )
    # NOTE: percentage_in_agents, resample_aggregation_compartments, resample_aggregation_transitions, and fill_method not used


class ProjectionArguments(BaseModel):
    """Projection arguments."""

    end_date: date = Field(description="End date of the projection.")
    n_trajectories: int = Field(description="Number of trajectories to simulate from posterior after calibration.")
    generation_number: int | None = Field(
        default=None, description="SMC generation number from which to draw parameter sets for projection."
    )


class BuilderOutput(BaseModel):
    """
    A bundle containing a single EpiModel or ABCSampler object paired with instructions for simulation/calibration/projection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_id: int = Field(
        description="Primary identifier of an EpiModel or ABCSampler object paired with instructions for simulation/calibration/projection."
    )
    seed: int | None = Field(None, description="Random seed.")
    model: EpiModel | None = Field(None, description="EpiModel object for simulation.")
    calibrator: ABCSampler | None = Field(None, description="ABCSampler object for calibration (contains an EpiModel).")
    simulation: SimulationArguments | None = Field(
        None, description="Arguments for a single call to EpiModel.run_simulations"
    )
    calibration: CalibrationStrategy | None = Field(
        None, description="Arguments for a single call to ABCSampler.calibrate"
    )
    projection: ProjectionArguments | None = Field(None, description="Arguments for a")

    @model_validator(mode="after")
    def check_fields(self: "BuilderOutput") -> "BuilderOutput":
        """
        Ensure combination of fields is valid.
        """
        assert self.model or self.calibrator, "BuilderOutput must contain an EpiModel or ABCSampler."

        if self.simulation:
            assert not self.calibration and not self.projection, (
                "Simulation workflow cannot be combined with calibration/projection."
            )
            assert self.model, "Simulation workflow requires EpiModel but received only ABCSampler."

        elif self.calibration or self.projection:
            assert self.calibrator, "Calibration/projection workflow requires ABCSampler but received only EpiModel."

        return self


class SimulationOutput(BaseModel):
    """Results of a call to EpiModel.run_simulations() with tracking information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_id: int = Field(
        description="Primary identifier of an EpiModel object paired with instructions for simulation."
    )
    seed: int | None = Field(None, description="Random seed.")
    results: SimulationResults = Field(description="Results of a call to EpiModel.run_simulations()")


class CalibrationOutput(BaseModel):
    """Results of a call to ABCSampler.calibrate() or ABCSampler.run_projections() with tracking information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_id: int = Field(
        description="Primary identifier of an ABCSampler object paired with instructions for calibration/projection."
    )
    seed: int | None = Field(None, description="Random seed.")
    results: CalibrationResults = Field(
        description="Results of a call to ABCSampler.calibrate() or ABCSampler.run_projections()"
    )
