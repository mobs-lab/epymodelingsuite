"""Output generation functions for formatting and saving results."""

import io
import logging

import pandas as pd

from ..schema.dispatcher import CalibrationOutput, SimulationOutput
from ..schema.output import OutputConfig, get_flusight_quantiles

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


def format_quantiles_flusightforecast(quantiles_df: pd.DataFrame) -> pd.DataFrame:
    """"""

    formatted = copy.deepcopy(quantiles_df)
    # TODO
    return pd.DataFrame()


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


def make_rate_trends_flusightforecast(formatted_quantiles: pd.DataFrame) -> pd.DataFrame:
    """"""
    # TODO
    return pd.DataFrame()


# ===== Output Generator Registry and Functions =====


OUTPUT_GENERATOR_REGISTRY = {}


def register_output_generator(kind_set):
    """Decorator for output generation dispatch."""

    def deco(fn):
        OUTPUT_GENERATOR_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_output_generator({"simulations", "outputs"})
def generate_simulation_outputs(*, simulations: list[SimulationOutput], outputs: OutputConfig, **_) -> dict:
    """
    Create a dictionary of outputs specified in an OutputConfig for a simulation workflow.

    Parameters
    ----------
        simulations: a list of SimulationOutputs containing SimulationResults.
        outputs: an OutputConfig instance with output specifications.

    Returns
    -------
        A dictionary where keys are intended filenames for writing data, and values are gzip-compressed CSV strings.
    """
    logger.info("OUTPUT GENERATOR: dispatched for simulation")
    warnings = set()

    quantiles_compartments = pd.DataFrame()
    quantiles_transitions = pd.DataFrame()
    quantiles_formatted = pd.DataFrame()
    trajectories_compartments = pd.DataFrame()
    trajectories_transitions = pd.DataFrame()
    model_meta = pd.DataFrame()

    # Quantiles
    if outputs.quantiles:
        # Unsupported formats
        if outputs.quantiles.flusight_format or outputs.quantiles.covid19_format:
            warnings.add("OUTPUT_GENERATOR: Requested forecast hub quantile format for simulation data, ignoring.")

        for simulation in simulations:
            # Default format
            if outputs.quantiles.default_format:
                # Compartments
                if outputs.quantiles.default_format.compartments:
                    quan_df = simulation.results.get_quantiles_compartments(quantiles=outputs.quantiles.selections)
                    if hasattr(outputs.quantiles.default_format.compartments, "__len__"):
                        try:
                            quan_df = quan_df[["date", "quantile"] + outputs.quantiles.default_format.compartments]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all compartments: {e}"
                            )
                    quan_df.insert(0, "primary_id", simulation.primary_id)
                    quan_df.insert(1, "seed", simulation.seed)
                    quan_df.insert(2, "population", simulation.population)
                    quantiles_compartments = pd.concat([quantiles_compartments, quan_df])

                # Transitions
                if outputs.quantiles.default_format.transitions:
                    quan_df = simulation.results.get_quantiles_transitions(quantiles=outputs.quantiles.selections)
                    if hasattr(outputs.quantiles.default_format.transitions, "__len__"):
                        try:
                            quan_df = quan_df[["date", "quantile"] + outputs.quantiles.default_format.transitions]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting transition quantiles, returning all transitions: {e}"
                            )
                    quan_df.insert(0, "primary_id", simulation.primary_id)
                    quan_df.insert(1, "seed", simulation.seed)
                    quan_df.insert(2, "population", simulation.population)
                    quantiles_transitions = pd.concat([quantiles_transitions, quan_df])

            # Hub format
            if outputs.quantiles.flusmh_format:
                quan_df = pd.DataFrame()
                quanf_df = format_quantiles_flusmh(quan_df)
                quantiles_formatted = pd.concat([quantiles_formatted, quanf_df])

            # Unsupported formats
            if outputs.quantiles.flusight_format or outputs.quantiles.covid19_format:
                warnings.add("OUTPUT_GENERATOR: Requested forecast hub format for simulation data, ignoring.")

    # Trajectories
    if outputs.trajectories:
        for simulation in simulations:
            for i, traj in enumerate(simulation.results.trajectories):
                # Compartments
                if outputs.trajectories.compartments:
                    traj_df = pd.DataFrame(traj.compartments)
                    if hasattr(outputs.trajectories.compartments, "__len__"):
                        try:
                            traj_df = traj_df[outputs.trajectories.compartments]
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
                if outputs.trajectories.transitions:
                    traj_df = pd.DataFrame(traj.transitions)
                    if hasattr(outputs.trajectories.transitions, "__len__"):
                        try:
                            traj_df = traj_df[outputs.trajectories.transitions]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting transition trajectories, returning all transitions: {e}"
                            )
                    traj_df.insert(0, "primary_id", simulation.primary_id)
                    traj_df.insert(1, "sim_id", i)
                    traj_df.insert(2, "seed", simulation.seed)
                    traj_df.insert(3, "population", simulation.population)
                    trajectories_transitions = pd.concat([trajectories_transitions, traj_df])

    # Model Metadata
    if outputs.model_meta:
        if outputs.model_meta.projection_parameters:
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

    # Cleanup and return
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
    if not quantiles_formatted.empty:
        # will want to build filename to be something better, like to fit hub standards
        out_dict["quantiles_hub_formatted.csv.gz"] = dataframe_to_gzipped_csv(
            quantiles_formatted, header=True, index=False
        )
    if not trajectories_compartments.empty:
        out_dict["trajectories_compartments.csv.gz"] = dataframe_to_gzipped_csv(
            trajectories_compartments, header=True, index=False
        )
    if not trajectories_transitions.empty:
        out_dict["trajectories_transitions.csv.gz"] = dataframe_to_gzipped_csv(
            trajectories_transitions, header=True, index=False
        )
    if not model_meta.empty:
        out_dict["model_metadata.csv.gz"] = dataframe_to_gzipped_csv(model_meta, header=True, index=False)

    logger.info("OUTPUT GENERATOR: completed for simulation")

    return out_dict


@register_output_generator({"calibrations", "outputs"})
def generate_calibration_outputs(*, calibrations: list[CalibrationOutput], outputs: OutputConfig, **_) -> dict:
    """Generate calibration outputs from CalibrationOutput objects."""
    logger.info("OUTPUT GENERATOR: dispatched for calibration")
    warnings = set()

    quantiles_compartments = pd.DataFrame()
    quantiles_transitions = pd.DataFrame()
    quantiles_formatted = pd.DataFrame()
    trajectories_compartments = pd.DataFrame()
    trajectories_transitions = pd.DataFrame()
    posteriors = pd.DataFrame()
    model_meta = pd.DataFrame()

    # Quantiles
    if outputs.quantiles:
        for calibration in calibrations:
            # Default format
            if outputs.quantiles.default_format:
                # Compartments
                if outputs.quantiles.default_format.compartments:
                    try:
                        quan_df = calibration.results.get_projection_quantiles(quantiles=outputs.quantiles.selections)
                    except ValueError:
                        warnings.add(
                            f"OUTPUT GENERATOR: failed to obtail projection quantiles for model with primary_id={calibration.primary_id}, continuing to next model."
                        )
                        continue
                    if hasattr(outputs.quantiles.default_format.compartments, "__len__"):
                        # Filter for explicitly requested compartments
                        try:
                            quan_df = quan_df[["date", "quantile"] + outputs.quantiles.default_format.compartments]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all compartments: {e}"
                            )
                            # Use all compartments, filter out transitions
                            transition_columns = [
                                c for c in calibration.results.get_projection_quantiles().columns if "_to_" in c
                            ]
                            quan_df.drop(columns=transition_columns, inplace=True)
                    else:
                        # Use all compartments, filter out transitions
                        transition_columns = [
                            c for c in calibration.results.get_projection_quantiles().columns if "_to_" in c
                        ]
                        quan_df.drop(columns=transition_columns, inplace=True)
                    quan_df.insert(0, "primary_id", calibration.primary_id)
                    quan_df.insert(1, "seed", calibration.seed)
                    quan_df.insert(2, "population", calibration.population)
                    quantiles_compartments = pd.concat([quantiles_compartments, quan_df])

                # Transitions
                if outputs.quantiles.default_format.transitions:
                    try:
                        quan_df = calibration.results.get_projection_quantiles(quantiles=outputs.quantiles.selections)
                    except ValueError:
                        warnings.add(
                            f"OUTPUT GENERATOR: failed to obtain projection quantiles for model with primary_id={calibration.primary_id}, continuing to next model."
                        )
                        continue
                    if hasattr(outputs.quantiles.default_format.transitions, "__len__"):
                        # Filter for explicitly requested transitions
                        try:
                            quan_df = quan_df[["date", "quantile"] + outputs.quantiles.default_format.transitions]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all transitions: {e}"
                            )
                            # Use all transitions, filter out compartments
                            transition_columns = [
                                c for c in calibration.results.get_projection_quantiles().columns if "_to_" in c
                            ]
                            # TODO: add target prediction data column name below
                            quan_df = quan_df[["date", "quantile"] + transition_columns]
                    else:
                        # Use all transitions, filter out compartments
                        transition_columns = [
                            c for c in calibration.results.get_projection_quantiles().columns if "_to_" in c
                        ]
                        # TODO: add target prediction data column name below
                        quan_df = quan_df[["date", "quantile"] + transition_columns]
                    quan_df.insert(0, "primary_id", calibration.primary_id)
                    quan_df.insert(1, "seed", calibration.seed)
                    quan_df.insert(2, "population", calibration.population)
                    quantiles_transitions = pd.concat([quantiles_transitions, quan_df])

            if outputs.quantiles.flusight_format:
                try:
                    quanf_df = calibration.results.get_projection_quantiles(quantiles=get_flusight_quantiles())
                except ValueError:
                    warnings.add(
                        f"OUTPUT GENERATOR: failed to obtain projection quantiles for model with primary_id={calibration.primary_id}, continuing to next model."
                    )
                    continue
                quanf_df.insert(0, "population", calibration.population)
                quanf_df = format_quantiles_flusightforecast(quanf_df)
                quantiles_formatted = pd.concat([quantiles_formatted, quanf_df])

                if outputs.quantiles.rate_trends:
                    trends_df = make_rate_trends_flusightforecast(quanf_df)
                    quantiles_formatted = pd.concat([quantiles_formatted, trends_df])

            elif outputs.quantiles.covid19_format:
                quanf_df = format_quantiles_covid19forecast(quan_df)
                quantiles_formatted = pd.concat([quantiles_formatted, quanf_df])

    # Trajectories
    if outputs.trajectories:
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

            # Compartments
            if outputs.trajectories.compartments:
                traj_c = trajectories.copy()
                if hasattr(outputs.trajectories.compartments, "__len__"):
                    # Filter for explicitly requested compartments
                    try:
                        traj_c = traj_c[["sim_id", "date"] + outputs.trajectories.compartments]
                    except Exception:
                        warnings.add(
                            "OUTPUT GENERATOR: failed to filter trajectories for selected compartments, returning all compartments."
                        )
                        # Use all compartments, filter out transitions
                        transition_columns = [c for c in traj_c.columns if "_to_" in c]
                        traj_c.drop(columns=transition_columns, inplace=True)
                else:
                    # Use all compartments, filter out transitions
                    transition_columns = [c for c in traj_c.columns if "_to_" in c]
                    traj_c.drop(columns=transition_columns, inplace=True)
                traj_c.insert(0, "primary_id", calibration.primary_id)
                traj_c.insert(2, "seed", calibration.seed)
                traj_c.insert(3, "population", calibration.population)
                trajectories_compartments = pd.concat([trajectories_compartments, traj_c])

            # Transitions
            if outputs.trajectories.transitions:
                traj_t = trajectories.copy()
                if hasattr(outputs.trajectories.transitions, "__len__"):
                    # filter for explicitly requested transitions
                    try:
                        traj_t = traj_t[["sim_id", "date"] + outputs.trajectories.transitions]
                    except Exception:
                        warnings.add(
                            "OUTPUT GENERATOR: failed to filter trajectories for selected transitions, returning all transitions."
                        )
                        # Use all transitions, filter out compartments
                        transition_columns = [c for c in traj_t.columns if "_to_" in c]
                        traj_t = traj_t[["sim_id", "date"] + transition_columns]
                else:
                    # Use all transitions, filter out compartments
                    transition_columns = [c for c in traj_t.columns if "_to_" in c]
                    traj_t = traj_t[["sim_id", "date"] + transition_columns]
                traj_t.insert(0, "primary_id", calibration.primary_id)
                traj_t.insert(2, "seed", calibration.seed)
                traj_t.insert(3, "population", calibration.population)
                trajectories_transitions = pd.concat([trajectories_transitions, traj_t])

    # Posteriors
    if outputs.posteriors:
        for calibration in calibrations:
            if outputs.posteriors.generations:
                post_df = pd.DataFrame()
                for g in outputs.posteriors.generations:
                    try:
                        post = calibration.results.get_posterior_distribution(generation=g)
                        post.insert(0, "generation", g)
                        post_df = pd.concat([post_df, post])
                    except Exception:
                        warnings.add(f"OUTPUT GENERATOR: failed to obtain posterior for generation {g}, continuing.")
                post_df.insert(0, "primary_id", calibration.primary_id)
                post_df.insert(2, "seed", calibration.seed)
                post_df.insert(3, "population", calibration.population)
                posteriors = post_df
            else:
                post_df = calibration.results.get_posterior_distribution()
                post_df.insert(0, "primary_id", calibration.primary_id)
                post_df.insert(1, "seed", calibration.seed)
                post_df.insert(2, "population", calibration.population)
                posteriors = post_df

    # Model Metadata
    if outputs.model_meta:
        # TODO
        pass

    # Cleanup and return
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
    if not quantiles_formatted.empty:
        # will want to build filename to be something better, like to fit hub standards
        out_dict["quantiles_hub_formatted.csv.gz"] = dataframe_to_gzipped_csv(
            quantiles_formatted, header=True, index=False
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
    if not model_meta.empty:
        out_dict["model_metadata.csv.gz"] = dataframe_to_gzipped_csv(model_meta, header=True, index=False)

    return out_dict


def dispatch_output_generator(**configs) -> dict:
    """Dispatch output generator functions. Returns dictionary of filenames and gzip-compressed CSV strings."""
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return OUTPUT_GENERATOR_REGISTRY[kinds](**configs)
