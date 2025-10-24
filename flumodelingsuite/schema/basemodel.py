import logging
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..utils import validate_iso3166
from .common import Meta

logger = logging.getLogger(__name__)

# ----------------------------------------
# Schema models
# ----------------------------------------


class Timespan(BaseModel):
    """Date range and timestep for the model."""

    class StartDateTypeEnum(str, Enum):
        """Types of simulation start date."""

        sampled = "sampled"
        calibrated = "calibrated"

    start_date: date | StartDateTypeEnum = Field(description="Start date of the simulation.")
    end_date: date = Field(description="End date of the simulation.")
    delta_t: float | int = Field(1.0, description="Time step (dt) for the simulation in epydemix.")

    @field_validator("end_date")
    @classmethod
    def check_end_date(cls, v: date, info: Any) -> date:
        """Ensure end_date is not before start_date."""
        start = info.data.get("start_date")
        if isinstance(start, date) and v < start:
            raise ValueError("end_date must be after start_date")
        return v

    @field_validator("delta_t")
    @classmethod
    def check_delta_t(cls, v: float) -> float:
        """Ensure delta_t > 0 and return as float"""
        assert v > 0, f"Provided delta_t={v} must be greater than 0."
        return float(v)


class Simulation(BaseModel):
    """Simulation settings to pass to epydemix."""

    n_sims: int | None = Field(10, description="Number of simulations run for a single model.")
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
        description="List of age groups in the population (e.g., ['0-4', '5-17', '18-49', '50-64', '65+'])"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str):
        return validate_iso3166(v)


class Compartment(BaseModel):
    """Data model for the compartments (e.g., S, I, R)"""

    class InitCompartmentEnum(str, Enum):
        """Types of compartment initialization."""

        default = "default"  # default compartment
        sampled = "sampled"
        calibrated = "calibrated"

    id: str = Field(description="Unique identifier for the compartment.")
    label: str | None = Field(None, description="Human-readable label for the compartment.")
    init: InitCompartmentEnum | float | int | list[float | int] | None = Field(
        None,
        description=(
            "Initial conditions for compartment. Can be a scalar (count or proportion), "
            "age-varying list, or special value (default/sampled/calibrated)."
        ),
    )

    @field_validator("init")
    @classmethod
    def enforce_nonnegative_init(cls, v: float | list[float | int], info: Any) -> float | int | list[float | int]:
        """Enforce that compartment initialization is non-negative"""
        if isinstance(v, (float, int)):
            assert v >= 0, f"Negative compartment initialization {v} received for compartment {info.data.get('id')}"
        elif isinstance(v, list):
            for i, val in enumerate(v):
                assert val >= 0, (
                    f"Negative compartment initialization {val} at age group {i} for compartment {info.data.get('id')}"
                )
        return v


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

        rate: float | str = Field(description="Rate of transition.")
        source: list[str] = Field(description="Infectious compartment ids for multiple mediated transition.")

    type: TransitionTypeEnum = Field(description="Type of transition.")
    source: str = Field(description="Source compartment id.")
    target: str = Field(description="Target compartment id.")
    rate: float | str | None = Field(None, description="Rate of transition.")
    mediator: str | None = Field(None, description="Infectious compartment id for single mediated transition.")
    mediators: list[Mediator] | None = Field(
        None, description="Rates and infectious compartments for multiple mediated transition."
    )

    @model_validator(mode="after")
    def check_fields_for_type(self: "Transition") -> "Transition":
        """Enforce required fields for each transition type."""
        if self.type == "spontaneous":
            assert self.rate is not None, "Spontaneous transition must have 'rate'."
            assert self.mediator is None, "Spontaneous transition cannot have 'mediator'."
            assert self.mediators is None, "Spontaneous transition cannot have 'mediators'."
        elif self.type == "mediated":
            assert self.rate is not None, "Mediated (single) transition must have 'rate'."
            assert self.mediator is not None, "Mediated (single) transition must have 'mediator'."
            assert self.mediators is None, "Mediated (single) transition cannot have multiple 'mediators'."
        elif self.type == "multi_mediated":
            assert self.mediators is not None, "Multiple mediated transition must specify 'mediators'."
            assert self.rate is None, (
                "Multiple mediated transition must specify rates within 'mediators', but 'rate' field provided."
            )
            assert self.mediator is None, (
                "Multiple mediated transition must use 'mediators', but 'mediator' field provided."
            )
        elif self.type == "vaccination":
            assert self.rate is None, "Vaccination transition cannot have 'rate'."
            assert self.mediator is None, "Vaccination transition cannot have 'mediator'."
            assert self.mediators is None, "Vaccination transition cannot have 'mediators'."
        return self


class Parameter(BaseModel):
    """Data model for parameters. Parameters can be a constant, expression, age-varying array, calculated, sampled, or calibrated."""

    class ParameterTypeEnum(str, Enum):
        """Types of parameter values."""

        scalar = "scalar"
        age_varying = "age_varying"
        sampled = "sampled"
        calibrated = "calibrated"
        calculated = "calculated"

    type: ParameterTypeEnum = Field(description="Type of parameter.")
    value: float | str | None = Field(None, description="Value for scalar or calculated parameters.")
    values: list[float | str] | None = Field(None, description="List of values for age-varying parameters.")

    @model_validator(mode="after")
    def check_param_fields(self: "Parameter") -> "Parameter":
        """Ensure required fields exist for each parameter type."""
        if self.type == "scalar":
            assert self.value is not None, "Constant or expression (scalar) parameter requires 'value'."
            assert self.values is None, "Scalar parameter requires a single 'value', but provided 'values' array."
        elif self.type == "age_varying":
            assert self.values is not None, "Age varying parameter requires 'values'."
            assert self.value is None, "Age varying parameter requires 'values' array, but provided single 'value'."
        elif self.type in ["sampled", "calibrated"]:
            assert self.value is None, (
                "Sampled or calibrated parameter provided extraneous 'value' field, sampling/calibration specs should be provided in modelset."
            )
            assert self.values is None, (
                "Sampled or calibrated parameter provided extraneous 'values' field, sampling/calibration specs should be provided in modelset."
            )
        elif self.type == "calculated":
            assert self.value is not None, "Calculated parameter requires expression in 'value' field."
            assert self.values is None, (
                "Calculated parameter requires expression in 'value' field, but 'values' provided instead."
            )
        return self


class Vaccination(BaseModel):
    """Vaccination configuration, such as data paths."""

    scenario_data_path: str | None = Field(None, description="Path to SMH vaccination scenario data file.")
    preprocessed_vaccination_data_path: str | None = Field(
        None, description="Path to preprocessed vaccination coverage data file."
    )
    origin_compartment: str = Field(description="Origin compartment for vaccination.")
    eligible_compartments: list[str] = Field(description="Eligible compartments for vaccination.")

    @model_validator(mode="after")
    def check_vax_fields(self: "Vaccination") -> "Vaccination":
        """Ensure vaccination configuration is consistent."""
        assert self.origin_compartment in self.eligible_compartments, (
            "Origin compartment must be in eligible compartments."
        )
        if self.scenario_data_path is None:
            assert self.preprocessed_vaccination_data_path is not None, (
                "Must provide one of scenario_data_path or preprocessed_vaccination_data_path."
            )
        else:
            assert self.preprocessed_vaccination_data_path is None, (
                "Cannot use both scenario_data_path and preprocessed_vaccination_data_path."
            )
        if self.preprocessed_vaccination_data_path is None:
            assert self.scenario_data_path is not None, (
                "Must provide one of scenario_data_path or preprocessed_vaccination_data_path."
            )
        else:
            assert self.scenario_data_path is None, (
                "Cannot use both scenario_data_path and preprocessed_vaccination_data_path."
            )
        return self


class Seasonality(BaseModel):
    """Data model for seasonality effects."""

    class SeasonalityMethodEnum(str, Enum):
        """Methods for defining a seasonally varying function."""

        balcan = "balcan"

    target_parameter: str = Field(description="Name of parameter to apply seasonality to")
    method: SeasonalityMethodEnum = Field(description="Method for defining a seasonally varying function")
    seasonality_max_date: date = Field(description="Date of seasonality peak (max transmissibility)")
    seasonality_min_date: date | None = Field(description="Date of seasonality trough (min transmissibility)")
    max_value: float = Field(description="Maximum value that the parameter can take after scaling.")
    min_value: float = Field(description="Minimum value that the parameter can take after scaling.")

    @field_validator("seasonality_min_date")
    @classmethod
    def check_seasonality_dates(cls, v: date, info: Any) -> date:
        """Ensure date of seasonality trough is after date of seasonality peak."""
        max_date = info.data.get("seasonality_max_date")
        if max_date and v < max_date:
            raise ValueError("seasonality_min_date must be after seasonality_max_date")
        return v

    @field_validator("min_value")
    @classmethod
    def check_scaling_minimum(cls, v: float, info: Any) -> float:
        """Ensure minimum post-scaling seasonal parameter value is lesser than maximum value."""
        max_val = info.data.get("max_value")
        if max_val and v > max_val:
            raise ValueError("Seasonality min_value must be lesser than max_value")
        return v


class Intervention(BaseModel):
    """Data model for interventions, such as school closures or reductions in parameter values."""

    class InterventionTypeEnum(str, Enum):
        """Types of interventions."""

        school_closure = "school_closure"
        parameter = "parameter"
        contact_matrix = "contact_matrix"

    type: InterventionTypeEnum
    scaling_factor: float | None = Field(None, description="Scaling factor for interventions.")
    override_value: float | None = Field(
        None, description="Override value for 'parameter' intervention, alternative to scaling_factor."
    )
    contact_matrix_layer: str | None = Field(
        None,
        description="Name of layer of contact matrix to apply intervention to, e.g. 'school', 'work', 'home', or 'community'.",
    )
    target_parameter: str | None = Field(
        None, description="Target parameter for interventions. Only required for 'parameter' interventions."
    )
    start_date: date | None = Field(None, description="Start date of 'parameter' or 'contact_matrix' intervention.")
    end_date: date | None = Field(None, description="End date of 'parameter' or 'contact_matrix' intervention.")

    @field_validator("scaling_factor")
    @classmethod
    def check_scaling_factor(cls, v: float) -> float:
        """Ensure scaling_factor >= 0."""
        assert v >= 0, f"Provided scaling_factor={v} must be >= 0."
        return v

    @model_validator(mode="after")
    def check_intervention_fields(self: "Intervention") -> "Intervention":
        """Ensure intervention configuration is consistent."""
        # Apply only to parameter interventions, or apply to all except parameter interventions
        if self.type == "parameter":
            assert self.target_parameter, "Parameter intervention is missing 'target_parameter'."
            assert self.start_date and self.end_date, "Parameter intervention must have 'start_date' and 'end_date'."
            assert bool(self.scaling_factor) ^ bool(self.override_value), (
                "Parameter intervention must have exactly one of 'scaling_factor' or 'override_value'."
            )
        else:
            assert not self.override_value, f"{self.type} intervention cannot use 'override_value'"
        # Apply only to contact matrix interventions, or apply to all except contact matrix interventions
        if self.type == "contact_matrix":
            assert self.contact_matrix_layer, (
                "Contact matrix intervention must specify 'contact_matrix_layer' to apply intervention to."
            )
        else:
            assert not self.contact_matrix_layer, f"'{self.type}' intervention cannot use 'contact_matrix_layer'."
        # Apply only to school closure intervention, or apply to all except school closure intervention
        if self.type == "school_closure":
            assert not self.start_date and not self.end_date, (
                "'school_closure' intervention cannot use 'start_date' or 'end_date'."
            )
        else:
            assert self.start_date and self.end_date, (
                f"'{self.type}' intervention must have 'start_date' and 'end_date'."
            )
            assert self.start_date <= self.end_date, (
                f"Start date for {self.type} intervention (given {self.start_date}) cannot be later than end date (given {self.end_date})."
            )
        return self


class BaseEpiModel(BaseModel):
    """Model configuration."""

    meta: Meta | None = Field(None, description="General metadata.")
    name: str | None = Field(None, description="Name of the model")
    random_seed: int | None = Field(None, description="Random seed for reproducibility")

    timespan: Timespan = Field(description="Date range and timestep for modeling")
    simulation: Simulation | None = Field(None, description="Simulation settings")
    population: Population = Field(description="Population configuration")

    compartments: list[Compartment] = Field(description="Compartments in the model")
    transitions: list[Transition] = Field(description="Transitions between compartments")
    parameters: dict[str, Parameter] = Field(description="Model parameters")

    vaccination: Vaccination | None = Field(None, description="Vaccination configuration")
    seasonality: Seasonality | None = Field(None, description="Seasonality configuration")
    interventions: list[Intervention] | None = Field(
        None,
        description="Interventions (school closure, transmissibility reduction, etc.) to apply during the simulation",
    )

    @model_validator(mode="after")
    def check_transition_refs(self: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that each transition.source/target refers to an existing compartment.id."""
        # Collect all defined compartment IDs
        compartment_ids = {c.id for c in self.compartments}

        # Validate each transition
        for t in self.transitions:
            assert t.source in compartment_ids, f"Transition.source='{t.source}' is not a valid Compartment.id"
            assert t.target in compartment_ids, f"Transition.target='{t.target}' is not a valid Compartment.id"
            if t.type == "mediated":
                assert t.mediator in compartment_ids, (
                    f"Transition.mediator='{t.mediator}' is not a valid Compartment.id"
                )
        return self

    @model_validator(mode="after")
    def check_vaccination_refs(self: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that vaccination specs are provided if a vaccination transition is declared, and vice versa."""
        for t in self.transitions:
            if t.type == "vaccination":
                assert self.vaccination is not None, (
                    "Vaccination transition declared but missing vaccination configuration."
                )
                assert t.source == self.vaccination.origin_compartment, (
                    "Vaccination transition source must be vaccination origin compartment."
                )
        if self.vaccination is not None:
            assert "vaccination" in [t.type for t in self.transitions], (
                "Vaccination configuration supplied but no vaccination transition declared."
            )
        return self

    @model_validator(mode="after")
    def check_seasonality_refs(self: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that seasonality target parameter exists"""
        if self.seasonality:
            assert self.seasonality.target_parameter in self.parameters.keys(), (
                f"Seasonality target {self.seasonality.target_parameter} missing from model parameters."
            )
        return self

    @model_validator(mode="after")
    def check_intervention_refs(self: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that intervention target parameters exist"""
        if self.interventions:
            targets = [i.target_parameter for i in self.interventions if i.target_parameter]
            for target in targets:
                assert target in self.parameters.keys(), f"Intervention target {target} missing from model parameters."
        return self

    @model_validator(mode="after")
    def check_age_structure_refs(self: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that age-varying parameters and compartment inits match population age structure."""
        n_age_groups = len(self.population.age_groups)
        for p in self.parameters.values():
            if p.type == "age_varying":
                assert len(p.values) == n_age_groups, "Age varying parameters must match population age structure."
        for c in self.compartments:
            if isinstance(c.init, list):
                assert len(c.init) == n_age_groups, (
                    f"Age varying initialization for compartment '{c.id}' has {len(c.init)} values "
                    f"but population has {n_age_groups} age groups."
                )
        return self

    @model_validator(mode="after")
    def require_default_init(self: "BaseEpiModel") -> "BaseEpiModel":
        """If compartment initialization is provided, require at least one default compartment."""
        inits = [c.init for c in self.compartments if c.init]
        if inits:
            assert inits.count("default") > 0, "Compartment initialization requires at least one default compartment."
        return self

    @model_validator(mode="after")
    def check_init_proportions_sum(self: "BaseEpiModel") -> "BaseEpiModel":
        """Ensure that sum of initialization proportions doesn't exceed 1.0 in any age group."""
        n_age_groups = len(self.population.age_groups)

        # Track proportion sums per age group (excluding default compartments)
        proportion_sums = [0.0] * n_age_groups

        for c in self.compartments:
            if c.init == "default" or c.init is None or c.init in ["sampled", "calibrated"]:
                continue

            if isinstance(c.init, list):
                # Age-varying initialization
                for age_idx, val in enumerate(c.init):
                    if 0 < val < 1:  # Only count proportions, not counts
                        proportion_sums[age_idx] += val
            elif isinstance(c.init, (float, int)) and 0 < c.init < 1:
                # Scalar proportion applied to all age groups
                for age_idx in range(n_age_groups):
                    proportion_sums[age_idx] += c.init

        # Check if any age group exceeds 1.0
        for age_idx, total in enumerate(proportion_sums):
            if total > 1.0:
                age_group_label = self.population.age_groups[age_idx]
                raise ValueError(
                    f"Sum of initialization proportions for age group '{age_group_label}' "
                    f"(index {age_idx}) exceeds 1.0: {total:.3f}"
                )

        return self

    @model_validator(mode="after")
    def disallow_mixing_sampling_calibration(self: "BaseEpiModel") -> "BaseEpiModel":
        """Disallow mixing sampling and calibration workflows"""
        sampled_vars = []
        calibrated_vars = []
        if self.timespan.start_date == "sampled":
            sampled_vars.append("start_date")
        elif self.timespan.start_date == "calibrated":
            calibrated_vars.append("start_date")
        for compartment in self.compartments:
            if compartment.init == "sampled":
                sampled_vars.append(compartment.id)
            elif compartment.init == "calibrated":
                calibrated_vars.append(compartment.id)
        for name, param in self.parameters.items():
            if param.type == "sampled":
                sampled_vars.append(name)
            elif param.type == "calibrated":
                calibrated_vars.append(name)
        if sampled_vars and calibrated_vars:
            msg = f"Cannot mix sampling and calibration workflows.\nDeclared sampled variables: {sampled_vars}\nDeclared calibrated variables: {calibrated_vars}"
            raise ValueError(msg)
        return self

    @field_validator("interventions")
    @classmethod
    def enforce_single_school_closure(cls, v: list[Intervention]) -> list[Intervention]:
        """Enforce that school closure intervention is applied only once."""
        if [i.type for i in v].count("school_closure") > 1:
            raise ValueError("More than one school_closure intervention was provided, maximum is 1")
        return v


class BasemodelConfig(BaseModel):
    """Root model for basemodel."""

    model: BaseEpiModel


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
