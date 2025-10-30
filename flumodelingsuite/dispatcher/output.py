"""Output generation functions for formatting and saving results."""

import copy
import io
import logging
from datetime import date

import numpy as np
import pandas as pd

from ..schema.dispatcher import CalibrationOutput, SimulationOutput
from ..schema.output import OutputConfig, get_flusight_quantiles
from ..utils.location import get_flusight_population

logger = logging.getLogger(__name__)


# ===== Output Generator Helper Functions =====


def dataframe_to_gzipped_csv(df: pd.DataFrame, **csv_kwargs) -> bytes:
    """
    Convert a DataFrame to gzip-compressed CSV bytes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert
    **csv_kwargs
        Additional keyword arguments to pass to DataFrame.to_csv()

    Returns
    -------
    bytes
        Gzip-compressed CSV data as bytes
    """
    buffer = io.BytesIO()
    df.to_csv(buffer, compression="gzip", **csv_kwargs)
    return buffer.getvalue()


def format_quantiles_flusightforecast(quantiles_df: pd.DataFrame, reference_date: date) -> pd.DataFrame:
    """
    Create FluSight forecast formatted quantile outputs for a single model. Rate-trends are handled separately.

    Parameters
    ----------
    quantiles_df: pd.DataFrame
    """
    formatted = copy.deepcopy(quantiles_df)

    # Horizons required for quantile outputs
    flusight_horizons = range(-1, 4)

    # Create horizon column and filter for appropriate horizons
    formatted.insert(
        0,
        "horizon",
        (formatted.date - pd.to_datetime(reference_date)).apply(lambda x: x / np.timedelta64(1, "W")).astype(int),
    )
    formatted = formatted[formatted.horizon.isin(flusight_horizons)]

    # Name and format remaining fields
    # FRAGILE: the name 'hospitalizations' is user-supplied in the modelset as the column to look for in the surveillance data.
    formatted.hospitalizations = formatted.hospitalizations.round().astype(int)
    formatted.rename(
        columns={"date": "target_end_date", "hospitalizations": "value", "quantile": "output_type_id"}, inplace=True
    )
    formatted.insert(2, "output_type", "quantile")
    formatted.insert(2, "target", "wk inc flu hosp")

    return formatted


def format_quantiles_flusmh(quantiles_df: pd.DataFrame) -> pd.DataFrame:
    """"""

    formatted = copy.deepcopy(quantiles_df)
    # TODO
    return pd.DataFrame()


def format_quantiles_covid19forecast(quantiles_df: pd.DataFrame) -> pd.DataFrame:
    """"""

    formatted = copy.deepcopy(quantiles_df)
    # TODO
    return pd.DataFrame()


def compare_thresholds_flusightforecast(
    stable_thres: float, change_thres: float, rate_change: float, count_change: float
) -> str:
    """
    Compare the simulated rate-change and count-change against the provided thresholds.
    Comparisons against thresholds are defined in FluSight documentation.
    https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#rate-trend-forecast-specifications

    Parameters
    ----------
    stable_thres: A simulated rate-change with magnitude less than this threshold is stable (unless count_change < 10).
    change_thres: This threshold defines whether non-stable rate-changes are a large increase/decrease or not.
    rate_change: The difference between the last observed rate (/100k population) and the simulated rate (diff = simulated - observed).
    count_change: The difference between the last observed count and the simulated count (diff = simulated - observed).

    Returns
    -------
    A string representing the category of the rate-change.
    """
    if abs(rate_change) < stable_thres or abs(count_change) < 10:
        return "stable"
    if 0 < rate_change < change_thres:
        return "increase"
    if change_thres <= rate_change:
        return "large_increase"
    if -change_thres < rate_change < 0:
        return "decrease"
    if rate_change <= -change_thres:
        return "large_decrease"


def categorize_rate_change_flusightforecast(rate_change: float, count_change: float, horizon: int) -> str:
    """
    Categorize the simulated rate-change using the appropriate thresholds for the horizon.
    Thresholds for different horizons are defined in FluSight documentation.
    https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#rate-trend-forecast-specifications

    Parameters
    ----------
    rate_change: The difference between the last observed rate (/100k population) and the simulated rate (diff = simulated - observed).
    count_change: The difference between the last observed count and the simulated count (diff = simulated - observed).
    horizon: The horizon on which the simulated changes are calculated.
    """
    assert -1 <= rate_change <= 1, "Received invalid rate-change."

    if horizon == 0:
        stable_thres = 0.3 / 100000
        change_thres = 1.7 / 100000
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    if horizon == 1:
        stable_thres = 0.5 / 100000
        change_thres = 3 / 100000
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    if horizon == 2:
        stable_thres = 0.7 / 100000
        change_thres = 4 / 100000
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    if horizon == 3:
        stable_thres = 1 / 100000
        change_thres = 5 / 100000
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    msg = f"Received invalid horizon {horizon}."
    raise AssertionError(msg)


def make_rate_trends_flusightforecast(formatted_quantiles: pd.DataFrame) -> pd.DataFrame:
    """Create FluSight rate-trend forecasts."""
    # Horizons required for rate-trend outputs
    flusight_horizons = range(4)

    # Population needed for converting to /100k rates
    loc_pop = get_flusight_population("25")

    # Set up DataFrame
    rate_trends = quantiles.copy()
    rate_trends.output_type = "pmf"
    rate_trends.target = "wk flu hosp rate change"
    rate_trends = rate_trends[rate_trends.horizon.isin(flusight_horizons)]

    # Get forecasted rates
    rate_trends.forecasted_rate = rate_trends.value * 100000 / loc_pop

    # TODO: get the previous week's observed rate
    last_observed_rate = None

    # Calculate the rate change
    rate_trends.rate_change = np.abs(rate_trends.forecasted_rate - last_observed_rate)

    # TODO: obtain probabilities for rate-change categories

    return pd.DataFrame()


# ===== Output Generator Registry and Functions =====


OUTPUT_GENERATOR_REGISTRY = {}


def register_output_generator(kind_set):
    """Decorator for output generation dispatch."""

    def deco(fn):
        OUTPUT_GENERATOR_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_output_generator({"simulations", "output_config"})
def generate_simulation_outputs(*, simulations: list[SimulationOutput], output_config: OutputConfig, **_) -> dict:
    """
    Create a dictionary of outputs specified in an OutputConfig for a simulation workflow.

    Parameters
    ----------
        simulations: a list of SimulationOutputs containing SimulationResults.
        output_config: an OutputConfig instance with output specifications.

    Returns
    -------
        A dictionary where keys are intended filenames for writing data, and values are gzip-compressed CSV strings.
    """
    logger.info("OUTPUT GENERATOR: dispatched for simulation")
    output = output_config.output
    warnings = set()

    quantiles_compartments = pd.DataFrame()
    quantiles_transitions = pd.DataFrame()
    trajectories_compartments = pd.DataFrame()
    trajectories_transitions = pd.DataFrame()
    hub_format_output = pd.DataFrame()
    model_meta = pd.DataFrame()

    ### Quantiles
    if output.quantiles:
        for simulation in simulations:
            # Compartments
            if output.quantiles.compartments:
                quan_df = simulation.results.get_quantiles_compartments(quantiles=output.quantiles.selections)
                if hasattr(output.quantiles.default_format.compartments, "__len__"):
                    try:
                        quan_df = quan_df[["date", "quantile"] + output.quantiles.default_format.compartments]
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all compartments: {e}"
                        )
                quan_df.insert(0, "primary_id", simulation.primary_id)
                quan_df.insert(1, "seed", simulation.seed)
                quan_df.insert(2, "population", simulation.population)
                quantiles_compartments = pd.concat([quantiles_compartments, quan_df])

            # Transitions
            if output.quantiles.transitions:
                quan_df = simulation.results.get_quantiles_transitions(quantiles=output.quantiles.selections)
                if hasattr(output.quantiles.default_format.transitions, "__len__"):
                    try:
                        quan_df = quan_df[["date", "quantile"] + output.quantiles.default_format.transitions]
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured selecting transition quantiles, returning all transitions: {e}"
                        )
                quan_df.insert(0, "primary_id", simulation.primary_id)
                quan_df.insert(1, "seed", simulation.seed)
                quan_df.insert(2, "population", simulation.population)
                quantiles_transitions = pd.concat([quantiles_transitions, quan_df])

    ### Trajectories
    if output.trajectories:
        for simulation in simulations:
            for i, traj in enumerate(simulation.results.trajectories):
                # Compartments
                if output.trajectories.compartments:
                    traj_df = pd.DataFrame(traj.compartments)
                    if hasattr(output.trajectories.compartments, "__len__"):
                        try:
                            traj_df = traj_df[output.trajectories.compartments]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment trajectories, returning all compartments: {e}"
                            )
                    traj_df.insert(0, "primary_id", simulation.primary_id)
                    traj_df.insert(1, "sim_id", i)
                    traj_df.insert(2, "seed", simulation.seed)
                    traj_df.insert(3, "population", simulation.population)
                    trajectories_compartments = pd.concat([trajectories_compartments, traj_df])

                # Transitions
                if output.trajectories.transitions:
                    traj_df = pd.DataFrame(traj.transitions)
                    if hasattr(output.trajectories.transitions, "__len__"):
                        try:
                            traj_df = traj_df[output.trajectories.transitions]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting transition trajectories, returning all transitions: {e}"
                            )
                    traj_df.insert(0, "primary_id", simulation.primary_id)
                    traj_df.insert(1, "sim_id", i)
                    traj_df.insert(2, "seed", simulation.seed)
                    traj_df.insert(3, "population", simulation.population)
                    trajectories_transitions = pd.concat([trajectories_transitions, traj_df])

    ### Hub Formats

    # Unsupported formats
    if output.flusight_format or output.covid19_format:
        warnings.add("OUTPUT_GENERATOR: Requested forecast hub quantile format for simulation data, ignoring.")

    # Flu SMH
    if output.flusmh_format:
        pass

    ### Model Metadata
    if output.model_meta:
        if output.model_meta.projection_parameters:
            warnings.add("OUTPUT_GENERATOR: Requested projection parameter metadata in simulation workflow, ignoring.")

        meta_dict = {}
        for simulation in simulations:
            meta_dict.set_default("primary_id", [])
            meta_dict["primary_id"].append(simulation.primary_id)

            meta_dict.set_default("seed", [])
            meta_dict["seed"].append(simulation.seed)

            meta_dict.set_default("delta_t", [])
            meta_dict["delta_t"].append(simulation.delta_t)

            meta_dict.set_default("population", [])
            meta_dict["population"].append(simulation.population)

            meta_dict.set_default("n_sims", [])
            meta_dict["n_sims"].append(simulation.results.Nsim)

            meta_dict.set_default("start_date", [])
            meta_dict["start_date"].append(str(sorted(simulation.results.dates)[0]))

            meta_dict.set_default("end_date", [])
            meta_dict["end_date"].append(str(sorted(simulation.results.dates)[-1]))

            # Parameters
            for p, v in simulation.results.parameters.items():
                meta_dict.set_default(p, [])
                meta_dict[p].append(str(v))

            # Initial conditions
            inits = {k: [int(v[0]) for v in vs] for k, vs in simulation.results.get_stacked_compartments().items()}
            for c, i in inits:
                colname = f"init_{c}"
                meta_dict.set_default(colname, [])
                meta_dict[colname].append(str(i))

        model_meta = pd.DataFrame(meta_dict)

    ### Cleanup and return
    for warning in warnings:
        logger.warning(warning)

    out_dict = {}
    if not quantiles_compartments.empty:
        out_dict["quantiles_compartments.csv.gz"] = dataframe_to_gzipped_csv(
            quantiles_compartments, header=True, index=False
        )
    if not quantiles_transitions.empty:
        out_dict["quantiles_transitions.csv.gz"] = dataframe_to_gzipped_csv(
            quantiles_transitions, header=True, index=False
        )
    if not trajectories_compartments.empty:
        out_dict["trajectories_compartments.csv.gz"] = dataframe_to_gzipped_csv(
            trajectories_compartments, header=True, index=False
        )
    if not trajectories_transitions.empty:
        out_dict["trajectories_transitions.csv.gz"] = dataframe_to_gzipped_csv(
            trajectories_transitions, header=True, index=False
        )
    if not hub_format_output.empty:
        # will want to build filename to be something better, like to fit hub standards
        out_dict["output_hub_formatted.csv.gz"] = dataframe_to_gzipped_csv(hub_format_output, header=True, index=False)
    if not model_meta.empty:
        out_dict["model_metadata.csv.gz"] = dataframe_to_gzipped_csv(model_meta, header=True, index=False)

    logger.info("OUTPUT GENERATOR: completed for simulation")

    return out_dict


@register_output_generator({"calibrations", "output_config"})
def generate_calibration_outputs(*, calibrations: list[CalibrationOutput], output_config: OutputConfig, **_) -> dict:
    """
    Create a dictionary of outputs specified in an OutputConfig for a calibration workflow.

    Parameters
    ----------
        calibrations: a list of CalibrationOutputs containing CalibrationResults.
        output_config: an OutputConfig instance with output specifications.

    Returns
    -------
        A dictionary where keys are intended filenames for writing data, and values are gzip-compressed CSV strings.
    """
    logger.info("OUTPUT GENERATOR: dispatched for calibration")
    output = output_config.output
    warnings = set()

    quantiles_compartments = pd.DataFrame()
    quantiles_transitions = pd.DataFrame()
    trajectories_compartments = pd.DataFrame()
    trajectories_transitions = pd.DataFrame()
    posteriors = pd.DataFrame()
    hub_format_output = pd.DataFrame()
    model_meta = pd.DataFrame()

    ### Quantiles
    if output.quantiles:
        for calibration in calibrations:
            # Obtain quantiles
            try:
                quan_df = calibration.results.get_projection_quantiles(quantiles=output.quantiles.selections)
            except ValueError:
                warnings.add(
                    f"OUTPUT GENERATOR: failed to obtain projection quantiles for model with primary_id={calibration.primary_id}, continuing to next model."
                )
                continue

            transition_columns = [c for c in quan_df.columns if "_to_" in c]

            # Compartments
            if output.quantiles.compartments:
                if hasattr(output.quantiles.compartments, "__len__"):
                    # Filter for explicitly requested compartments
                    try:
                        quanc_df = copy.deepcopy(quan_df[["date", "quantile"] + output.quantiles.compartments])
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all compartments: {e}"
                        )
                        # Use all compartments, filter out transitions
                        quanc_df.drop(columns=transition_columns, inplace=True)
                else:
                    # Use all compartments, filter out transitions
                    quanc_df.drop(columns=transition_columns, inplace=True)
                quanc_df.insert(0, "primary_id", calibration.primary_id)
                quanc_df.insert(1, "seed", calibration.seed)
                quanc_df.insert(2, "population", calibration.population)
                quantiles_compartments = pd.concat([quantiles_compartments, quanc_df])

            # Transitions
            if output.quantiles.transitions:
                if hasattr(output.quantiles.transitions, "__len__"):
                    # Filter for explicitly requested transitions
                    try:
                        quant_df = copy.deepcopy(
                            quan_df[["date", "quantile"] + output.quantiles.default_format.transitions]
                        )
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all transitions: {e}"
                        )
                        # Use all transitions, filter out compartments
                        # TODO: add target prediction data column name below
                        quant_df = quant_df[["date", "quantile"] + transition_columns]
                else:
                    # Use all transitions, filter out compartments
                    # TODO: add target prediction data column name below
                    quant_df = quant_df[["date", "quantile"] + transition_columns]
                quant_df.insert(0, "primary_id", calibration.primary_id)
                quant_df.insert(1, "seed", calibration.seed)
                quant_df.insert(2, "population", calibration.population)
                quantiles_transitions = pd.concat([quantiles_transitions, quant_df])

    ### Trajectories
    if output.trajectories:
        for calibration in calibrations:
            # Collect all trajectories
            try:
                traj = calibration.results.get_projection_trajectories()
            except Exception:
                warnings.add(
                    f"OUTPUT GENERATOR: failed to obtain projection trajectories for model with primary_id={calibration.primary_id}, continuing to next model."
                )
                continue

            trajectories = pd.DataFrame()
            for i in range(len(traj["date"])):
                columns = []
                for name, values in traj.items():
                    columns.append(pd.Series(values[i], name=name))
                traj_df = pd.concat(columns, axis=1)
                traj_df.insert(0, "sim_id", i)
                trajectories = pd.concat([trajectories, traj_df])

            transition_columns = [c for c in trajectories.columns if "_to_" in c]

            # Compartments
            if output.trajectories.compartments:
                traj_c = copy.deepcopy(trajectories)
                if hasattr(output.trajectories.compartments, "__len__"):
                    # Filter for explicitly requested compartments
                    try:
                        traj_c = traj_c[["sim_id", "date"] + output.trajectories.compartments]
                    except Exception:
                        warnings.add(
                            "OUTPUT GENERATOR: failed to filter trajectories for selected compartments, returning all compartments."
                        )
                        # Use all compartments, filter out transitions
                        traj_c.drop(columns=transition_columns, inplace=True)
                else:
                    # Use all compartments, filter out transitions
                    traj_c.drop(columns=transition_columns, inplace=True)
                traj_c.insert(0, "primary_id", calibration.primary_id)
                traj_c.insert(2, "seed", calibration.seed)
                traj_c.insert(3, "population", calibration.population)
                trajectories_compartments = pd.concat([trajectories_compartments, traj_c])

            # Transitions
            if output.trajectories.transitions:
                traj_t = copy.deepcopy(trajectories)
                if hasattr(output.trajectories.transitions, "__len__"):
                    # filter for explicitly requested transitions
                    try:
                        traj_t = traj_t[["sim_id", "date"] + output.trajectories.transitions]
                    except Exception:
                        warnings.add(
                            "OUTPUT GENERATOR: failed to filter trajectories for selected transitions, returning all transitions."
                        )
                        # Use all transitions, filter out compartments
                        traj_t = traj_t[["sim_id", "date"] + transition_columns]
                else:
                    # Use all transitions, filter out compartments
                    traj_t = traj_t[["sim_id", "date"] + transition_columns]
                traj_t.insert(0, "primary_id", calibration.primary_id)
                traj_t.insert(2, "seed", calibration.seed)
                traj_t.insert(3, "population", calibration.population)
                trajectories_transitions = pd.concat([trajectories_transitions, traj_t])

    ### Posteriors
    if output.posteriors:
        for calibration in calibrations:
            if output.posteriors.generations:
                post_df = pd.DataFrame()
                for g in output.posteriors.generations:
                    try:
                        post = calibration.results.get_posterior_distribution(generation=g)
                        post.insert(0, "generation", g)
                        post_df = pd.concat([post_df, post])
                    except Exception:
                        warnings.add(
                            f"OUTPUT GENERATOR: failed to obtain posterior for generation {g} from model with primary_id={calibration.primary_id}, continuing."
                        )
                post_df.insert(0, "primary_id", calibration.primary_id)
                post_df.insert(2, "seed", calibration.seed)
                post_df.insert(3, "population", calibration.population)
                posteriors = pd.concat([posteriors, post_df])
            else:
                post_df = calibration.results.get_posterior_distribution()
                post_df.insert(0, "primary_id", calibration.primary_id)
                post_df.insert(1, "seed", calibration.seed)
                post_df.insert(2, "population", calibration.population)
                posteriors = pd.concat([posteriors, post_df])

    ### Hub Formats

    # FluSight Forecast Hub
    if output.flusight_format:
        # Quantile forecasts
        for calibration in calibrations:
            try:
                # FRAGILE: the name 'hospitalizations' is user-supplied in the modelset as the column to look for in the surveillance data.
                quanf_df = calibration.results.get_projection_quantiles(
                    quantiles=get_flusight_quantiles(), variables=["date", "quantile", "hospitalizations"]
                )
            except ValueError:
                warnings.add(
                    f"OUTPUT GENERATOR: failed to obtain projection quantiles for model with primary_id={calibration.primary_id}, continuing to next model."
                )
                continue
            quanf_df = format_quantiles_flusightforecast(quanf_df, outputs.quantiles.flusight_format.reference_date)
            quanf_df.insert(0, "reference_date", outputs.quantiles.flusight_format.reference_date)
            quanf_df.insert(0, "location", convert_location_name_format(calibration.population, "FIPS"))
            hub_format_output = pd.concat([hub_format_output, quanf_df])

        # Rate-trend forecasts
        if output.flusight_format.rate_trends:
            trends_df = make_rate_trends_flusightforecast(hub_format_output)
            hub_format_output = pd.concat([hub_format_output, trends_df])

    # Covid19 Forecast Hub
    elif output.quantiles.covid19_format:
        pass

    ### Model Metadata
    if output.model_meta:
        # TODO
        pass

    ### Cleanup and return
    for warning in warnings:
        logger.warning(warning)

    out_dict = {}
    if not quantiles_compartments.empty:
        out_dict["quantiles_compartments.csv.gz"] = dataframe_to_gzipped_csv(
            quantiles_compartments, header=True, index=False
        )
    if not quantiles_transitions.empty:
        out_dict["quantiles_transitions.csv.gz"] = dataframe_to_gzipped_csv(
            quantiles_transitions, header=True, index=False
        )
    if not trajectories_compartments.empty:
        out_dict["trajectories_compartments.csv.gz"] = dataframe_to_gzipped_csv(
            trajectories_compartments, header=True, index=False
        )
    if not trajectories_transitions.empty:
        out_dict["trajectories_transitions.csv.gz"] = dataframe_to_gzipped_csv(
            trajectories_transitions, header=True, index=False
        )
    if not posteriors.empty:
        out_dict["posteriors.csv.gz"] = dataframe_to_gzipped_csv(posteriors, header=True, index=False)
    if not hub_format_output.empty:
        # will want to build filename to be something better, like to fit hub standards
        out_dict["output_hub_formatted.csv.gz"] = dataframe_to_gzipped_csv(hub_format_output, header=True, index=False)
    if not model_meta.empty:
        out_dict["model_metadata.csv.gz"] = dataframe_to_gzipped_csv(model_meta, header=True, index=False)

    return out_dict


def dispatch_output_generator(**configs) -> dict:
    """Dispatch output generator functions. Returns dictionary of filenames and gzip-compressed CSV strings."""
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return OUTPUT_GENERATOR_REGISTRY[kinds](**configs)
