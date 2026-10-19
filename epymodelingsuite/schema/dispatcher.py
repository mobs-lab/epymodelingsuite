"""Schema definitions for workflow dispatcher."""

from datetime import date

from epydemix.calibration import ABCSampler, CalibrationResults
from epydemix.model import EpiModel
from epydemix.model.simulation_results import SimulationResults
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from ..utils.resolved_location import ResolvedLocation
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


class _LocatedOutput(BaseModel):
    """Resolve old builtin inputs once at the construction or unpickle boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    location: ResolvedLocation

    @model_validator(mode="before")
    @classmethod
    def resolve_legacy_location(cls, value):
        if not isinstance(value, dict):
            return value
        data = value.copy()
        population = data.pop("population", None)
        location = data.get("location")
        if location is None:
            reference = population
            if reference is None:
                model = data.get("model")
                if model is None:
                    calibrator = data.get("calibrator")
                    parameters = getattr(calibrator, "parameters", None)
                    model = parameters.get("epimodel") if isinstance(parameters, dict) else None
                reference = getattr(getattr(model, "population", None), "name", None)
            if isinstance(reference, str):
                if reference.startswith("custom_location_"):
                    raise ValueError("Custom artifacts require an explicit resolved location")
                from ..utils.location import resolve_location

                location = resolve_location(reference)
        if location is not None:
            if not isinstance(location, ResolvedLocation):
                location = TypeAdapter(ResolvedLocation).validate_python(location)
            if population is not None and not location.matches(population):
                raise ValueError("Population must match the resolved location")
            data["location"] = location
        return data

    @model_validator(mode="after")
    def check_location_matches_models(self):
        model = getattr(self, "model", None)
        calibrator = getattr(self, "calibrator", None)
        parameters = getattr(calibrator, "parameters", None)
        calibrated_model = parameters.get("epimodel") if isinstance(parameters, dict) else None
        for candidate in (model, calibrated_model):
            name = getattr(getattr(candidate, "population", None), "name", None)
            if name is not None and name not in (self.location.population_name, *self.location.aliases):
                raise ValueError("Resolved location must match the model and calibrator populations")
        return self

    def __setstate__(self, state):
        state = state.copy()
        state["__dict__"] = self.resolve_legacy_location(state["__dict__"])
        if "location" not in state["__dict__"]:
            raise ValueError("Saved artifacts require a resolved location or a builtin population")
        fields = set(state.get("__pydantic_fields_set__", ()))
        state["__pydantic_fields_set__"] = (fields - {"population"}) | {"location"}
        super().__setstate__(state)
        self.check_location_matches_models()


class BuilderOutput(_LocatedOutput):
    """
    A bundle containing a single EpiModel or ABCSampler object paired with instructions for simulation/calibration/projection.
    """

    primary_id: int = Field(
        description="Primary identifier of an EpiModel or ABCSampler object paired with instructions for simulation/calibration/projection."
    )
    seed: int | None = Field(None, description="Random seed.")
    delta_t: float | None = Field(None, description="Timestep.")
    model: EpiModel | None = Field(None, description="EpiModel object for simulation.")
    calibrator: ABCSampler | None = Field(None, description="ABCSampler object for calibration (contains an EpiModel).")
    simulation: SimulationArguments | None = Field(
        None, description="Arguments for a single call to EpiModel.run_simulations"
    )
    calibration: CalibrationStrategy | None = Field(
        None, description="Arguments for a single call to ABCSampler.calibrate"
    )
    projection: ProjectionArguments | None = Field(
        None, description="Arguments for a single call to ABCSampler.run_projections"
    )
    start_date_reference: date | None = Field(
        None, description="Reference date for start_date parameter in calibration (used for posterior visualization)"
    )

    @model_validator(mode="after")
    def check_fields(self: "BuilderOutput") -> "BuilderOutput":
        """
        Ensure combination of fields is valid.
        """
        if not (self.model or self.calibrator):
            msg = "BuilderOutput must contain an EpiModel or ABCSampler."
            raise ValueError(msg)

        if self.simulation:
            if self.calibration or self.projection:
                msg = "Simulation workflow cannot be combined with calibration/projection."
                raise ValueError(msg)
            if not self.model:
                msg = "Simulation workflow requires EpiModel but received only ABCSampler."
                raise ValueError(msg)

        elif self.calibration or self.projection:
            if not self.calibrator:
                msg = "Calibration/projection workflow requires ABCSampler but received only EpiModel."
                raise ValueError(msg)

        return self


class SimulationOutput(_LocatedOutput):
    """Results of a call to EpiModel.run_simulations() with tracking information."""

    primary_id: int = Field(
        description="Primary identifier of an EpiModel object paired with instructions for simulation."
    )
    seed: int | None = Field(None, description="Random seed.")
    delta_t: float | None = Field(None, description="Timestep.")
    results: SimulationResults | None = Field(description="Results of a call to EpiModel.run_simulations()")

    @property
    def population(self) -> str:
        return self.location.population_name


class CalibrationOutput(_LocatedOutput):
    """Results of a call to ABCSampler.calibrate() or ABCSampler.run_projections() with tracking information."""

    primary_id: int = Field(
        description="Primary identifier of an ABCSampler object paired with instructions for calibration/projection."
    )
    seed: int | None = Field(None, description="Random seed.")
    delta_t: float | None = Field(None, description="Timestep.")
    start_date_reference: date | None = Field(
        None,
        description="Reference date for start_date parameter offsets (used for converting posterior distributions to actual dates in visualizations).",
    )
    results: CalibrationResults | None = Field(
        description="Results of a call to ABCSampler.calibrate() or ABCSampler.run_projections()"
    )
    calibration_strategy: CalibrationStrategy | None = Field(None, description="Calibration strategy used.")

    @property
    def population(self) -> str:
        return self.location.population_name
