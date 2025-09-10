import logging

logger = logging.getLogger(__name__)

from datetime import date, today
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .utils import validate_iso3166

# ----------------------------------------
# Schema models
# ----------------------------------------


class Timespan(BaseModel):
    """Date range and timestep for the model."""

    class StartDateTypeEnum(str, Enum):
        """Types of simulation start date."""

        sampled = "sampled"
        calibrated = "calibrated"

    start_date: date | StartDateTypeEnum = Field(..., description="Start date of the simulation.")
    end_date: date = Field(..., description="End date of the simulation.")
    delta_t: float | int = Field(1.0, description="Time step (dt) for the simulation in epydemix.")

    @field_validator("end_date")
    def check_end_date(cls, v: date, info: Any) -> date:
        """Ensure end_date is not before start_date."""
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be after start_date")
        return v


class Simulation(BaseModel):
    """Simulation settings to pass to epydemix."""

    n_sims: int | None = Field(None, description="Number of simulations run for a single model.")
    resample_frequency: str | None = Field(
        None,
        description="The frequency at which to resample the simulation results. Follows https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases",
    )


class Population(BaseModel):
    """Population configuration."""

    name: str | None = Field(
        None,
        description="Location code in ISO 3166. Use ISO 3166-2 for states (e.g., 'US-NY') and ISO 3166-1 alpha 2 for countries (e.g., 'US')",
    )
    age_groups: list[str] = Field(
        ..., description="List of age groups in the population (e.g., ['0-4', '5-17', '18-49', '50-64', '65+'])"
    )

    @field_validator("name")
    def validate_name(cls, v):
        return validate_iso3166(v)


class Compartment(BaseModel):
    """Data model for the compartments (e.g., S, I, R)"""

    class InitCompartmentEnum(str, Enum):
        """Types of compartment initialization."""

        default = "default"  # default compartment
        sampled = "sampled"
        calibrated = "calibrated"

    id: str = Field(..., description="Unique identifier for the compartment.")
    label: str | None = Field(None, description="Human-readable label for the compartment.")
    init: InitCompartmentEnum | float | int | None = Field(None, description="Initial conditions for compartment.")


class Transition(BaseModel):
    """Data model for single transition between compartments."""

    class TransitionTypeEnum(str, Enum):
        """Types of transitions between compartments."""

        spontaneous = "spontaneous"
        mediated = "mediated"
        multi_mediated = "multi_mediated"
        vaccination = "vaccination"

    class Mediator(BaseModel):
        """Required fields for multiple mediated transitions."""

        rate: float | str = Field(..., description="Rate of transition.")
        source: list[str] = Field(..., description="Infectious compartment ids for multiple mediated transition.")

    type: TransitionTypeEnum = Field(..., description="Type of transition.")
    source: str = Field(..., description="Source compartment id.")
    target: str = Field(..., description="Target compartment id.")
    rate: float | str | None = Field(None, description="Rate of transition.")
    mediator: str | None = Field(None, description="Infectious compartment id for single mediated transition.")
    mediators: list[Mediator] | None = Field(
        None, description="Rates and infectious compartments for multiple mediated transition."
    )

    @model_validator(mode="after")
    def check_fields_for_type(cls, m: "Transition") -> "Transition":
        """Enforce required fields for each transition type."""
        if m.type == "spontaneous":
            assert m.rate is not None, "Spontaneous transition must have 'rate'."
            assert m.mediator is None, "Spontaneous transition cannot have 'mediator'."
            assert m.mediators is None, "Spontaneous transition cannot have 'mediators'."
        elif m.type == "mediated":
            assert m.rate is not None, "Mediated (single) transition must have 'rate'."
            assert m.mediator is not None, "Mediated (single) transition must have 'mediator'."
            assert m.mediators is None, "Mediated (single) transition cannot have multiple 'mediators'."
        elif m.type == "multi_mediated":
            assert m.mediators is not None, "Multiple mediated transition must specify 'mediators'."
            assert m.rate is None, (
                "Multiple mediated transition must specify rates within 'mediators', but 'rate' field provided."
            )
            assert m.mediator is None, (
                "Multiple mediated transition must use 'mediators', but 'mediator' field provided."
            )
        elif m.type == "vaccination":
            assert m.rate is None, "Vaccination transition cannot have 'rate'."
            assert m.mediator is None, "Vaccination transition cannot have 'mediator'."
            assert m.mediators is None, "Vaccination transition cannot have 'mediators'."
        return m


class Parameter(BaseModel):
    """Data model for parameters. Parameters can be a constant, expression, age-varying array, calculated, sampled, or calibrated."""

    class ParameterTypeEnum(str, Enum):
        """Types of parameter values."""

        scalar = "scalar"
        age_varying = "age_varying"
        sampled = "sampled"
        calibrated = "calibrated"
        calculated = "calculated"

    type: ParameterTypeEnum = Field(..., description="Type of parameter.")
    value: float | str | None = Field(None, description="Value for scalar or calculated parameters.")
    values: list[float | str] | None = Field(None, description="List of values for age-varying parameters.")

    @model_validator(mode="after")
    def check_param_fields(cls, m: "Parameter") -> "Parameter":
        """Ensure required fields exist for each parameter type."""
        if m.type == "scalar":
            assert m.value is not None, "Constant or expression (scalar) parameter requires 'value'."
            assert m.values is None, "Scalar parameter requires a single 'value', but provided 'values' array."
        elif m.type == "age_varying":
            assert m.values is not None, "Age varying parameter requires 'values'."
            assert m.value is None, "Age varying parameter requires 'values' array, but provided single 'value'."
        elif m.type in ["sampled", "calibrated"]:
            assert m.value is None, (
                "Sampled or calibrated parameter provided extraneous 'value' field, sampling/calibration specs should be provided in modelset."
            )
            assert m.values is None, (
                "Sampled or calibrated parameter provided extraneous 'values' field, sampling/calibration specs should be provided in modelset."
            )
        elif m.type == "calculated":
            assert m.value is not None, "Calculated parameter requires expression in 'value' field."
            assert m.values is None, (
                "Calculated parameter requires expression in 'value' field, but 'values' provided instead."
            )
        return m


class Vaccination(BaseModel):
    """Vaccination configuration, such as data paths."""

    scenario_data_path: str | None = Field(None, description="Path to SMH vaccination scenario data file.")
    preprocessed_vaccination_data_path: str | None = Field(
        None, description="Path to preprocessed vaccination coverage data file."
    )
    origin_compartment: str = Field(..., description="Origin compartment for vaccination.")
    eligible_compartments: list[str] = Field(..., description="Eligible compartments for vaccination.")

    @model_validator(mode="after")
    def check_vax_fields(cls, m: "Vaccination") -> "Vaccination":
        """Ensure vaccination configuration is consistent."""
        assert m.origin_compartment in m.eligible_compartments, "Origin compartment must be in eligible compartments."
        if scenario_data_path is None:
            assert preprocessed_vaccination_data_path is not None, (
                "Must provide one of scenario_data_path or preprocessed_vaccination_data_path."
            )
        else:
            assert preprocessed_vaccination_data_path is None, (
                "Cannot use both scenario_data_path and preprocessed_vaccination_data_path."
            )
        if preprocessed_vaccination_data_path is None:
            assert scenario_data_path is not None, (
                "Must provide one of scenario_data_path or preprocessed_vaccination_data_path."
            )
        else:
            assert scenario_data_path is None, (
                "Cannot use both scenario_data_path and preprocessed_vaccination_data_path."
            )
        return m


class Seasonality(BaseModel):
    """Data model for seasonality effects."""

    class SeasonalityMethodEnum(str, Enum):
        """Methods for defining a seasonally varying function."""

        balcan = "balcan"

    target_parameter: str = Field(..., description="Name of parameter to apply seasonality to")
    method: SeasonalityMethodEnum = Field(..., description="Method for defining a seasonally varying function")
    seasonality_max_date: date = Field(..., description="Date of seasonality peak (max transmissibility)")
    seasonality_min_date: date | None = Field(..., description="Date of seasonality trough (min transmissibility)")
    max_value: float = Field(..., description="Maximum value that the parameter can take after scaling.")
    min_value: float = Field(..., description="Minimum value that the parameter can take after scaling.")

    @field_validator("seasonality_min_date")
    def check_seasonality_dates(cls, v: date, info: Any) -> date:
        max_date = info.data.get("seasonality_max_date")
        if max_date and v < max_date:
            raise ValueError("seasonality_min_date must be after seasonality_max_date")
        return v


class Intervention(BaseModel):
    """Data model for interventions, such as school closures or reductions in parameter values."""

    class InterventionTypeEnum(str, Enum):
        """Types of interventions."""

        school_closure = "school_closure"
        parameter = "parameter"
        contact_matrix = "contact_matrix"

    type: InterventionTypeEnum
    reduction_factor: float | None = Field(None, description="Reduction factor for interventions.")
    target_parameter: str | None = Field(
        None, description="Target parameter for interventions. Only required for 'parameter' interventions."
    )
    start_date: date | None = Field(
        None, description="Start date of intervention. Only required for 'parameter' interventions."
    )
    end_date: date | None = Field(
        None, description="End date of intervention. Only required for 'parameter' interventions."
    )


class BaseEpiModel(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Name of the model")
    version: str | None = Field(None, description="Version of the model")
    date: date | None = Field(default_factory=today, description="Date of work")
    description: str | None = Field(None, description="Human-readable description of the model")
    random_seed: int | None = Field(None, description="Random seed for reproducibility")

    timespan: Timespan = Field(..., description="Date range and timestep for modeling")
    simulation: Simulation | None = Field(None, description="Simulation settings")
    population: Population = Field(..., description="Population configuration")

    compartments: list[Compartment] = Field(..., description="Compartments in the model")
    transitions: list[Transition] = Field(..., description="Transitions between compartments")
    parameters: dict[str, Parameter] = Field(..., description="Model parameters")

    vaccination: Vaccination | None = Field(None, description="Vaccination configuration")
    seasonality: Seasonality | None = Field(None, description="Seasonality configuration")
    interventions: list[Intervention] | None = Field(
        None,
        description="Interventions (school closure, transmissibility reduction, etc.) to apply during the simulation",
    )

    @model_validator(mode="after")
    def check_transition_refs(cls, m: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that each transition.source/target refers to an existing compartment.id."""
        # Collect all defined compartment IDs
        compartment_ids = {c.id for c in m.compartments}

        # Validate each transition
        for t in m.transitions:
            assert t.source in compartment_ids, f"Transition.source='{t.source}' is not a valid Compartment.id"
            assert t.target in compartment_ids, f"Transition.target='{t.target}' is not a valid Compartment.id"
            if t.type == "mediated":
                assert t.mediators.source in compartment_ids, (
                    f"Transition.mediators.source='{t.mediators.source}' is not a valid Compartment.id"
                )
        return m

    @model_validator(mode="after")
    def check_vaccination_refs(cls, m: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that vaccination specs are provided if a vaccination transition is declared, and vice versa."""
        for t in m.transitions:
            if t.type == "vaccination":
                assert m.vaccination is not None, (
                    "Vaccination transition declared but missing vaccination configuration."
                )
                assert t.source == m.vaccination.origin_compartment, (
                    "Vaccination transition source must be vaccination origin compartment."
                )
        if m.vaccination is not None:
            assert "vaccination" in [t.type for t in m.transitions], (
                "Vaccination configuration supplied but no vaccination transition declared."
            )
        return m

    @model_validator(mode="after")
    def check_age_structure_refs(cls, m: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that age-varying parameters match population age structure."""
        n_age_groups = len(m.population.age_groups)
        for p in m.parameters.values():
            if p.type == "age_varying":
                assert len(p.values) == n_age_groups, "Age varying parameters must match population age structure."
        return m


class BasemodelConfig(BaseModel):
    model: BaseEpiModel
    metadata: dict[str, Any] | None = None


def validate_basemodel(config: dict) -> BasemodelConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    BasemodelConfig
        The validated Model.
    """
    try:
        root = BasemodelConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
