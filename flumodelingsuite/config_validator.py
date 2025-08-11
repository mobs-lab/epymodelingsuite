import logging
logger = logging.getLogger(__name__)

from datetime import date
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# ----------------------------------------
# Schema models
# ----------------------------------------
class SimulationConfig(BaseModel):
    """Simulation settings including locations and date range."""
    start_date: date = Field(..., description="Start date of the simulation.")
    end_date: date = Field(..., description="End date of the simulation.")

    @field_validator('end_date')
    def check_end_date(cls, v: date, info: Any) -> date:
        """Ensure end_date is not before start_date."""
        start = info.data.get('start_date')
        if start and v < start:
            raise ValueError("end_date must be after start_date")
        return v

def validate_iso3166(value: str) -> str:
    """
    Validate that `value` follows ISO 3166.

    Parameters
    ----------
		value (str): Location code in ISO 3166. Countries use ISO 3166-1 alpha-2 country code (e.g., "US") and states/regions use ISO 3166-2 subdivision (e.g., "US-NY").
    
    Returns
    -------
        str: The validated ISO 3166 code. Raises error when the code is invalid.

    """
    countries = ['AF', 'AX', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 'BH', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR', 'IO', 'BN', 'BG', 'BF', 'BI', 'CV', 'KH', 'CM', 'CA', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 'KM', 'CG', 'CD', 'CK', 'CR', 'CI', 'HR', 'CU', 'CW', 'CY', 'CZ', 'DK', 'DJ', 'DM', 'DO', 'EC', 'EG', 'SV', 'GQ', 'ER', 'EE', 'SZ', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 'HM', 'VA', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL', 'IT', 'JM', 'JP', 'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MO', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'NC', 'NZ', 'NI', 'NE', 'NG', 'NU', 'NF', 'MK', 'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 'PT', 'PR', 'QA', 'RE', 'RO', 'RU', 'RW', 'BL', 'SH', 'KN', 'LC', 'MF', 'PM', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS', 'ES', 'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO', 'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'US', 'UM', 'UY', 'UZ', 'VU', 'VE', 'VN', 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW']
    subdivisions = {
        'US': ['US-AL', 'US-AK', 'US-AZ', 'US-AR', 'US-CA', 'US-CO', 'US-CT', 'US-DE', 'US-FL', 'US-GA', 'US-HI', 'US-ID', 'US-IL', 'US-IN', 'US-IA', 'US-KS', 'US-KY', 'US-LA', 'US-ME', 'US-MD', 'US-MA', 'US-MI', 'US-MN', 'US-MS', 'US-MO', 'US-MT', 'US-NE', 'US-NV', 'US-NH', 'US-NJ', 'US-NM', 'US-NY', 'US-NC', 'US-ND', 'US-OH', 'US-OK', 'US-OR', 'US-PA', 'US-RI', 'US-SC', 'US-SD', 'US-TN', 'US-TX', 'US-UT', 'US-VT', 'US-VA', 'US-WA', 'US-WV', 'US-WI', 'US-WY', 'US-DC', 'US-AS', 'US-GU', 'US-MP', 'US-PR', 'US-UM', 'US-VI']
    }
    # Check ISO 3166-1 (country)
    if value in countries:
        return value

    # Check ISO 3166-2 (subdivision)
    if value.startswith('US-'):
        if value in subdivisions['US']:
            return value
    else:
        sub = value.split('-')[1]
        if len(sub) == 2:
            return value

    raise ValueError(f'Invalid ISO 3166 code: {value}')

class Population(BaseModel):
    """Population configuration."""

    name: Optional[str] = Field(None, description="Location code in ISO 3166. Use ISO 3166-2 for states (e.g., 'US-NY') and ISO 3166-1 alpha 2 for countries (e.g., 'US')")
    age_groups: Optional[List[str]] = Field(None, description="List of age groups in the population (e.g., ['0-4', '5-17', '18-49', '50-64', '65+'])")

    @field_validator('name')
    def validate_name(cls, v):
        return validate_iso3166(v)

class Compartment(BaseModel):
    """Data model for the compartments (e.g., S, I, R)"""
    id: str = Field(..., description="Unique identifier for the compartment.")
    label: Optional[str] = Field(None, description="Human-readable label for the compartment.")
    is_clinical: Optional[bool] = Field(False, description="Indicates if the compartment is clinical.")
    is_traveller: Optional[bool] = Field(False, description="Indicates if the compartment is for travelers.")

    # class Config:
    #     title = "Compartment"
    #     json_schema_extra = {
    #         "description": "Data model for the compartments (e.g., S, I, R)",
    #     }

class Transition(BaseModel):
    """Data model for single transition between compartments."""

    class TransitionTypeEnum(str, Enum):
        """Types of transitions between compartments."""
        spontaneous = 'spontaneous'
        mediated = 'mediated'
        vaccination = 'vaccination'

    class Mediators(BaseModel):
        """Required fields for mediated transitions."""
        rate: Union[float, str]
        source: str = Field(..., description="Source compartment id")


    type: TransitionTypeEnum
    source: str = Field(..., description="Source compartment id")
    target: str = Field(..., description="Target compartment id")
    rate: Optional[Union[float, str]] = None
    mediators: Optional[Mediators] = Field(None, description="Mediators (infectors) for mediated transitions")

    @model_validator(mode='after')
    def check_fields_for_type(cls, m: "Transition") -> "Transition":
        """Enforce required fields for each transition type."""
        if m.type == 'spontaneous' and m.rate is None:
            raise ValueError("Spontaneous transition requires field 'rate'")
        if m.type == 'mediated' and m.mediators is None:
            raise ValueError("Mediated transition requires field 'mediators'")
        return m

class Intervention(BaseModel):
    """Data model for interventions, such as school closures or reductions in parameter values."""

    class InterventionTypeEnum(str, Enum):
        """Types of interventions."""
        school_closure = 'school_closure'
        parameter = 'parameter'
        contact_matrix = 'contact_matrix'

    type: InterventionTypeEnum
    reduction_factor: Optional[float] = Field(None, description="Reduction factor for interventions.")
    target_parameter: Optional[str] = Field(None, description="Target parameter for interventions. Only required for 'parameter' interventions.")
    start_date: Optional[date] = Field(None, description="Start date of intervention. Only required for 'parameter' interventions.")
    end_date: Optional[date] = Field(None, description="End date of intervention. Only required for 'parameter' interventions.")

class Seasonality(BaseModel):
    """Data model for seasonality effects."""
    model: str
    seasonality_max_date: date = Field(..., description="Date of seasonality peak (max transmissibility)")
    seasonality_min_date: date = Field(..., description="Date of seasonality trough (min transmissibility)")
    transmissibility_max: float = Field(..., description="Maximum transmissibility value")
    transmissibility_min: float = Field(..., description="Minimum transmissibility value")

    @field_validator('seasonality_min_date')
    def check_seasonality_dates(cls, v: date, info: Any) -> date:
        max_date = info.data.get('seasonality_max_date')
        if max_date and v < max_date:
            raise ValueError("seasonality_min_date must be after seasonality_max_date")
        return v

class Distribution(BaseModel):
    """Data model to define a probability distribution."""
    name: str = Field(..., description="Name of the probability distribution. Has to match scipy class (e.g., 'norm', 'uniform', etc.)")
    args: List[float] = Field(default_factory=list, description="Positional arguments for the scipy distribution initializer.  (e.g., [0, 1] for uniform distribution from 0 to 1)")
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Keyword arguments for the scipy distribution initializer.")

class ValueTypeEnum(str, Enum):
    """Types of parameter values."""
    constant = 'constant'
    expression = 'expression'
    array = 'array'
    distribution = 'distribution'

class Parameter(BaseModel):
    """Data model for parameters. Parameters can be a constant, expression, array, or distribution."""
    type: ValueTypeEnum
    value: Optional[Union[float, str, List[float]]] = Field(None, description="Value for constant parameters.")
    values: Optional[List[float]] = Field(None, description="List of values for array parameters.")
    distribution: Optional[Distribution] = Field(None, description="Distribution specification for distribution parameters.")

    @model_validator(mode='after')
    def check_param_fields(cls, m: "Parameter") -> "Parameter":
        """Ensure required fields exist for each parameter type."""
        if m.type == 'constant' and m.value is None:
            raise ValueError("Constant parameter requires 'value'")
        if m.type == 'array' and m.values is None:
            raise ValueError("Array parameter requires 'values'")
        if m.type == 'distribution' and m.distribution is None:
            raise ValueError("Distribution parameter requires 'distribution'")
        return m

class Prior(BaseModel):
    """Prior distribution for calibration."""
    distribution: Distribution

class Vaccination(BaseModel):
    """Vaccination configuration, such as data paths."""
    processed_coverage_data_path: Optional[str] = Field(None, description="Path to processed vaccination coverage data file.")
    smh_vaccination_data_path: Optional[str] = Field(None, description="Path to SMH vaccination data file.")
    scenario: Optional[str] = Field(None, description="Vaccination scenario.")
    origin_compartment: Optional[str] = Field(None, description="Origin compartment for vaccination.")
    eligible_compartments: Optional[List[str]] = Field(None, description="Eligible compartments for vaccination.")

class Model(BaseModel):
    """Model configuration."""
    name: str = Field(..., description="Name of the model")
    version: Optional[str] = Field(None, description="Version of the model")
    date: Optional[date]
    description: Optional[str] = Field(None, description="Human-readable description of the model")

    simulation: SimulationConfig
    population: Optional[Population] = Field(None, description="Population configuration")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    compartments: List[Compartment] = Field(..., description="Compartments in the model")
    transitions: List[Transition] = Field(..., description="Transitions between compartments")
    interventions: Optional[List[Intervention]] = Field(default_factory=list, description="Interventions (school closure, transmissibility reduction, etc.) to apply during the simulation")
    seasonality: Optional[Seasonality] = Field(None, description="Seasonality configuration")
    parameters: Dict[str, Parameter] = Field(..., description="Model parameters")
    priors: Optional[Dict[str, Prior]] = Field(None, description="Priors of calibrating parameters")
    vaccination: Optional[Vaccination] = Field(None, description="Vaccination configuration")

    @model_validator(mode='after')
    def check_transition_refs(cls, m: "Model") -> "Model":
        """Ensure that each transition.source/target refers to an existing compartment.id."""
        # Collect all defined compartment IDs
        compartment_ids = {c.id for c in m.compartments}

        # Validate each transition
        for t in m.transitions:
            if t.source not in compartment_ids:
                raise ValueError(
                    f"Transition.source='{t.source}' is not a valid Compartment.id"
                )
            if t.target not in compartment_ids:
                raise ValueError(
                    f"Transition.target='{t.target}' is not a valid Compartment.id"
                )
        return m

class RootConfig(BaseModel):
    model: Model
    metadata: Optional[Dict[str, Any]] = None

def validate_config(config) -> RootConfig:
    """
    Validate the given configuration against the schema.
    Returns the validated Model.
    """
    try:
        root = RootConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
