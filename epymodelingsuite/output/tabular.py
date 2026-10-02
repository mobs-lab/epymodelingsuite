"""Serialize tabular data and wrap it in output objects."""

import io
import logging

import pandas as pd

from ..schema.output import OutputObject, TabularOutputTypeEnum
from .hub_files import cast_hub_dtypes

logger = logging.getLogger(__name__)


def format_tabular_object(df: pd.DataFrame, name: str, output_type: TabularOutputTypeEnum) -> OutputObject:
    """Wrap a formatted table in the requested output representation.

    Parameters
    ----------
    df : pd.DataFrame
        Table whose columns, values and dtypes have already been prepared.
        The input is not modified.
    name : str
        Output name without a filename extension.
    output_type : TabularOutputTypeEnum
        CSVBytes for gzip-compressed CSV, Parquet for Parquet bytes, or
        DataFrame to return the original table.

    Returns
    -------
    OutputObject
        Table or serialized bytes with the requested output type and filename.
        Serialized outputs have a ``.csv.gz`` or ``.parquet`` extension.
    """
    match output_type:
        case TabularOutputTypeEnum.CSVBytes:
            return OutputObject(
                output_type=output_type,
                name=f"{name}.csv.gz",
                data=dataframe_to_gzipped_csv(df, header=True, index=False),
            )
        case TabularOutputTypeEnum.DataFrame:
            return OutputObject(output_type=output_type, name=name, data=df)
        case TabularOutputTypeEnum.Parquet:
            return OutputObject(output_type=output_type, name=f"{name}.parquet", data=df.to_parquet(index=False))
        case _:
            msg = f"Requested undefined tabular object format {output_type}."
            logger.warning(msg)


def format_hub_objects(
    df: pd.DataFrame, name: str, output_types: list[TabularOutputTypeEnum]
) -> list[OutputObject]:
    """Prepare hub tables per format and pass them to the common tabular writer.

    Parameters
    ----------
    df : pd.DataFrame
        Hub forecast table. Parquet output requires all ``HUB_COLUMNS``.
        The input is not modified.
    name : str
        Output name without a filename extension.
    output_types : list[TabularOutputTypeEnum]
        Requested output formats, in the order they should be returned.

    Returns
    -------
    list[OutputObject]
        One object per requested format. Parquet uses a copy with the FluSight
        column order and Arrow dtypes. CSV and DataFrame use the original table.
        An empty format list produces an empty result.
    """
    objects = []
    for output_type in output_types:
        hub_table = df
        if output_type == TabularOutputTypeEnum.Parquet:
            hub_table = cast_hub_dtypes(hub_table)
        objects.append(format_tabular_object(hub_table, name, output_type))
    return objects


def dataframe_to_gzipped_csv(df: pd.DataFrame, **csv_kwargs) -> bytes:
    """Serialize a DataFrame as gzip-compressed CSV bytes.

    Parameters
    ----------
    df : pd.DataFrame
        Table to serialize without modifying it.
    **csv_kwargs : dict
        Additional arguments passed to ``DataFrame.to_csv``, such as ``header``
        and ``index``. Compression and date format are set by this function.

    Returns
    -------
    bytes
        Gzip-compressed CSV with dates formatted as ``YYYY-MM-DD``.
    """
    buffer = io.BytesIO()
    df.to_csv(buffer, date_format="%Y-%m-%d", compression="gzip", **csv_kwargs)
    return buffer.getvalue()
