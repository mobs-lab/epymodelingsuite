"""Utility functions for fetching external data sources."""

import epiweeks
import pandas as pd
from sodapy import Socrata

from epymodelingsuite.utils.location import get_flusight_locations


def fetch_hhs_hospitalizations(
    query_start_date: str | None = None,
    save_path: str | None = None,
    data_type: str = "default",
) -> pd.DataFrame:
    """
    Fetch HHS flu hospitalization data from CDC's Socrata API.

    Parameters
    ----------
    query_start_date : str, optional
        Start date for filtering data (format: 'YYYY-MM-DD').
        If provided, only data with weekendingdate >= query_start_date will be fetched.
        If None, all available data will be fetched.
    save_path : str, optional
        Path to save the resulting CSV file.
        If provided, the DataFrame will be saved to this path before returning.
    data_type : str, optional
        Type of data to fetch. Options are:
        - "default": Non-preliminary data (dataset ID: ua7e-t2fy)
        - "preliminary": Preliminary data (dataset ID: mpgq-jmmr)
        Default is "default".

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - location_iso : str
            ISO 3166 location code in format "US-XX" for states (e.g., "US-CA", "US-TX")
            or "US" for national data
        - location_code : str
            FIPS location code (e.g., "06", "48") or "US" for national data
        - target_end_date : datetime
            End date of the epidemiological week
        - epiweek : int
            MMWR epidemiological week in CDC format (YYYYWW)
        - hospitalizations : float
            Total confirmed flu new admissions

    Examples
    --------
    >>> # Fetch all available data
    >>> df = fetch_hhs_hospitalizations()
    >>> # Fetch data from 2024-10-01 onwards
    >>> df = fetch_hhs_hospitalizations(query_start_date="2024-10-01")
    >>> # Fetch and save to file
    >>> df = fetch_hhs_hospitalizations(
    ...     query_start_date="2024-10-01",
    ...     save_path="data/hhs_flu_2024.csv"
    ... )

    Example Output
    --------------
    >>> df.head(3)
      location_iso location_code target_end_date  epiweek  hospitalizations
    0        US-AL            01      2024-10-19   202442             123.0
    1        US-AL            01      2024-10-26   202443             145.0
    2        US-AK            02      2024-10-19   202442              23.0

    Notes
    -----
    Data Source
        Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction from CDC's
        National Healthcare Safety Network (NHSN). Two datasets are available:

        - Default (non-preliminary): ua7e-t2fy
          https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy
        - Preliminary: mpgq-jmmr
          https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr

    Data Field
        Uses `totalconfflunewadm`: total number of new hospital admissions of patients
        with confirmed influenza captured during the reporting week (Sunday-Saturday).

    Coverage
        Data available from August 2020 onwards. Includes US states, DC, and national
        aggregates. Territories (AS, GU, MP, PR, VI) and HHS regions are excluded from the output.

    """
    # Map data_type to dataset IDs
    dataset_ids = {
        "default": "ua7e-t2fy",  # Non-preliminary data
        "preliminary": "mpgq-jmmr",  # Preliminary data
    }

    if data_type not in dataset_ids:
        msg = f"Invalid data_type: {data_type}. Must be 'default' or 'preliminary'"
        raise ValueError(msg)

    dataset_id = dataset_ids[data_type]

    # Initialize Socrata client
    client = Socrata("data.cdc.gov", None)

    # Build query
    if query_start_date:
        where_clause = f"weekendingdate >= '{query_start_date}'"
        results = client.get(dataset_id, where=where_clause, limit=100000)
    else:
        results = client.get(dataset_id, limit=100000)

    # Convert to pandas DataFrame
    data = pd.DataFrame.from_records(results)
    data = data[["weekendingdate", "jurisdiction", "totalconfflunewadm"]]

    # Process dates and epiweeks
    data["target_end_date"] = pd.to_datetime(data["weekendingdate"])
    data["epiweek"] = data["target_end_date"].apply(lambda key: int(epiweeks.Week.fromdate(key).cdcformat()))

    # Rename totalconfflunewadm and process it
    data["hospitalizations"] = data["totalconfflunewadm"].fillna(0).astype(float)

    # Load location mapping
    locations = get_flusight_locations()

    # Handle USA (national level) separately
    usa_mask = data["jurisdiction"] == "USA"
    data.loc[usa_mask, "location_iso"] = "US"
    data.loc[usa_mask, "location_code"] = "US"

    # Merge with locations to get FIPS codes and ISO codes for states
    data = data.merge(
        locations[["abbreviation", "location"]],
        left_on="jurisdiction",
        right_on="abbreviation",
        how="left",
    )

    # Create ISO location codes for states (only where not already set)
    state_mask = data["location_iso"].isna()
    data.loc[state_mask, "location_iso"] = "US-" + data.loc[state_mask, "abbreviation"]
    data.loc[state_mask, "location_code"] = data.loc[state_mask, "location"]

    # Filter out territories and HHS regions (keep only states, DC, and USA)
    territories = ["AS", "GU", "MP", "PR", "VI"]
    data = data[data["location_iso"].notna()]
    data = data[~data["jurisdiction"].isin(territories)]

    # Select final columns
    data = data[["location_iso", "location_code", "target_end_date", "epiweek", "hospitalizations"]]

    # Sort by location code and date
    data = data.sort_values(["location_code", "target_end_date"]).reset_index(drop=True)

    # Save to file if path provided
    if save_path:
        data.to_csv(save_path, index=False)

    return data


def fetch_nssp_edvisits(
    query_start_date: str | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch NSSP ED visits data from CDC's Socrata API.

    Parameters
    ----------
    query_start_date : str, optional
        Start date for filtering data (format: 'YYYY-MM-DD').
        If provided, only data with weekendingdate >= query_start_date will be fetched.
        If None, all available data will be fetched.
    save_path : str, optional
        Path to save the resulting CSV file.
        If provided, the DataFrame will be saved to this path before returning.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - location_iso : str
            ISO 3166 location code in format "US-XX" for states (e.g., "US-CA", "US-TX")
            or "US" for national data
        - location_code : str
            FIPS location code (e.g., "06", "48") or "US" for national data
        - target_end_date : datetime
            End date of the epidemiological week
        - epiweek : int
            MMWR epidemiological week in CDC format (YYYYWW)
        - prop_ed_visits : float
            Proportion of ED visits attributed to flu

    Examples
    --------
    >>> # Fetch all available data
    >>> df = fetch_nssp_edvisits()
    >>> # Fetch data from 2024-10-01 onwards
    >>> df = fetch_nssp_edvisits(query_start_date="2024-10-01")
    >>> # Fetch and save to file
    >>> df = fetch_nssp_edvisits(
    ...     query_start_date="2024-10-01",
    ...     save_path="data/nssp_flu_2024.csv"
    ... )

    Example Output
    --------------
    >>> df.head(3)
      location_iso location_code target_end_date  epiweek  hospitalizations
    0        US-AL            01      2024-10-19   202442             123.0
    1        US-AL            01      2024-10-26   202443             145.0
    2        US-AK            02      2024-10-19   202442              23.0

    Notes
    -----
    Data Source
        Each row provides a percentage of emergency department patient visits for the specified pathogen
        of all ED patient visits for the specified geography that were observed for the given week from
        data submitted to the National Syndromic Surveillance Program (NSSP).

    Data Field
        Uses `percent_visits_influenza`

    Coverage
        Data available from August 2020 onwards. Includes US states, DC, and national
        aggregates. Territories (AS, GU, MP, PR, VI) and HHS regions are excluded from the output.

    """
    # There is only one version of the dataset
    dataset_id = "rdmq-nq56"

    # Initialize Socrata client
    client = Socrata("data.cdc.gov", None)

    # Build query
    if query_start_date:
        where_clause = f"week_end >= '{query_start_date}'"
        results = client.get(dataset_id, where=where_clause, limit=100000)
    else:
        results = client.get(dataset_id, limit=100000)

    # Convert to pandas DataFrame
    data = pd.DataFrame.from_records(results)
    data = data[data.county == "All"]
    data = data[["week_end", "geography", "percent_visits_influenza"]]

    # Process dates and epiweeks
    data["target_end_date"] = pd.to_datetime(data["week_end"])
    data["epiweek"] = data["target_end_date"].apply(lambda key: int(epiweeks.Week.fromdate(key).cdcformat()))

    # Rename percent_visits_influenza and process it
    data["prop_ed_visits"] = data["percent_visits_influenza"].fillna(0).astype(float)

    # Load location mapping
    locations = get_flusight_locations()

    # Handle USA (national level) separately
    usa_mask = data["geography"] == "United States"
    data.loc[usa_mask, "location_iso"] = "US"
    data.loc[usa_mask, "location_code"] = "US"

    # Merge with locations to get FIPS codes and ISO codes for states
    data = data.merge(
        locations[["abbreviation", "location", "location_name"]],
        left_on="geography",
        right_on="location_name",
        how="left",
    )

    # Create ISO location codes for states (only where not already set)
    state_mask = data["location_iso"].isna()
    data.loc[state_mask, "location_iso"] = "US-" + data.loc[state_mask, "abbreviation"]

    data.loc[state_mask, "location_code"] = data.loc[state_mask, "location"]

    # Filter out territories and HHS regions (keep only states, DC, and USA)
    territories = ["AS", "GU", "MP", "PR", "VI"]
    data = data[data["location_iso"].notna()]
    data = data[~data["location"].isin(territories)]

    # Select final columns
    data = data[["location_iso", "location_code", "target_end_date", "epiweek", "prop_ed_visits"]]

    # Sort by location code and date
    data = data.sort_values(["location_code", "target_end_date"]).reset_index(drop=True)

    # Save to file if path provided
    if save_path:
        data.to_csv(save_path, index=False)

    return data
