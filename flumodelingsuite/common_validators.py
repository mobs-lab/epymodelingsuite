"""Shared validator classes for sampling and calibration configurations."""

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Distribution(BaseModel):
    """Data model to define a probability distribution."""

    class DistributionTypeEnum(str, Enum):
        """Types of distribution."""

        scipy = "scipy"
        custom = "custom"

    type: str | DistributionTypeEnum = Field(..., description="Type of distribution ('scipy' or 'custom')")
    name: str = Field(
        ...,
        description="Name of the probability distribution. Has to match scipy class (e.g., 'norm', 'uniform', etc.) or custom distribution name",
    )
    args: list[float] = Field(
        default_factory=list,
        description="Positional arguments for the distribution initializer (e.g., [0, 1] for uniform distribution from 0 to 1)",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the distribution initializer"
    )


class DateParameter(BaseModel):
    """Date parameter with reference date and distribution."""

    reference_date: date = Field(..., description="Reference date for sampling")
    distribution: Distribution | None = Field(
        None, description="Distribution for date offset sampling"
    )  # Used for sampling
    prior: Distribution | None = Field(None, description="Prior for date offset sampling")  # Used for calibration


class Meta(BaseModel):
    """Metadata section."""

    description: str | None = Field(None, description="Description of the modelset")
    author: str | None = Field(None, description="Author of the modelset")
    version: str | float | None = Field(None, description="Version of the modelset")
