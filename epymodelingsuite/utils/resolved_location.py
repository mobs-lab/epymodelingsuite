"""Immutable regional identity and prepared inputs carried across workflow stages."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ResolvedLocation:
    """One canonical region, with its supported external representations resolved.

    ``source_iso`` identifies inherited contact, vaccination and school data.
    It is separate from the region's own identifiers. Reporting denominators
    may differ from each other and from the model's age-stratified population.
    """

    kind: Literal["iso", "metrocast_location", "custom"]
    location_id: str
    population_name: str
    location_name: str
    location_name_short: str
    source_iso: str
    identifiers: tuple[tuple[str, str], ...]
    aliases: tuple[str, ...] = ()
    total_population: float | None = None
    flusight_population: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.location_id

    @property
    def hub_id(self) -> str:
        return self.format("location_id")

    def format(self, output_format: str) -> str:
        formats = dict(self.identifiers)
        formats.update(
            name=self.location_name,
            name_short=self.location_name_short,
            epydemix_population=self.population_name,
        )
        try:
            return formats[output_format]
        except KeyError:
            raise ValueError(
                f"Unsupported output format {output_format!r} for {self.kind} location {self.location_id}"
            ) from None

    def matches(self, value: str) -> bool:
        aliases = (self.location_id, self.population_name, *self.aliases)
        if self.kind == "iso":
            aliases += (self.location_name, self.location_name_short, *dict(self.identifiers).values())
        return value in aliases


@dataclass(frozen=True)
class VaccinationContext:
    """Age-specific allocation prepared once, independent of simulation start date."""

    source_iso: str
    scaling: tuple[tuple[str, float], ...]
