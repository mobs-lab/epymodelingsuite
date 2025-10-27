"""Output generation functions for formatting and saving results."""

import logging

from ..schema.dispatcher import CalibrationOutput, SimulationOutput
from ..schema.output import OutputConfig, get_flusight_quantiles

logger = logging.getLogger(__name__)


# ===== Output Generator Helper Functions =====


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


@register_output_generator({"simulation", "outputs"})
def generate_simulation_outputs(*, simulation: list[SimulationOutput], outputs: OutputConfig, **_) -> dict:
    """
    Create a dictionary of outputs specified in an OutputConfig for a simulation workflow.

    Parameters
    ----------
        simulation: a list of SimulationOutputs containing SimulationResults.
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
        for model in simulation:
            # Default format
            if outputs.quantiles.simulation_default:
                # Compartments
                if outputs.quantiles.simulation_default.compartments:
                    quan_df = model.results.get_quantiles_compartments(quantiles=outputs.quantiles.selections)
                    if hasattr(outputs.quantiles.simulation_default.compartments, "__len__"):
                        try:
                            quan_df = quan_df[["date", "quantile"] + outputs.quantiles.simulation_default.compartments]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all compartments: {e}"
                            )
                    quan_df.insert(0, "primary_id", model.primary_id)
                    quan_df.insert(1, "seed", model.seed)
                    quan_df.insert(2, "population", model.population)
                    quantiles_compartments = pd.concat([quantiles_compartments, quan_df])

                # Transitions
                if outputs.quantiles.simulation_default.transitions:
                    quan_df = model.results.get_quantiles_transitions(quantiles=outputs.quantiles.selections)
                    if hasattr(outputs.quantiles.simulation_default.transitions, "__len__"):
                        try:
                            quan_df = quan_df[["date", "quantile"] + outputs.quantiles.simulation_default.transitions]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting transition quantiles, returning all transitions: {e}"
                            )
                    quan_df.insert(0, "primary_id", model.primary_id)
                    quan_df.insert(1, "seed", model.seed)
                    quan_df.insert(2, "population", model.population)
                    quantiles_transitions = pd.concat([quantiles_transitions, quan_df])

            # Hub format
            if outputs.quantiles.flusmh_format:
                quan_df = pd.DataFrame()
                quanf_df = format_quantiles_flusmh(quan_df)
                quantiles_formatted = pd.concat([quantiles_formatted, quanf_df])

    # Trajectories
    if outputs.trajectories:
        for model in simulation:
            for i, traj in enumerate(model.results.trajectories):
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
                    traj_df.insert(0, "primary_id", model.primary_id)
                    traj_df.insert(1, "sim_id", i)
                    traj_df.insert(2, "seed", model.seed)
                    traj_df.insert(3, "population", model.population)
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
                    traj_df.insert(0, "primary_id", model.primary_id)
                    traj_df.insert(1, "sim_id", i)
                    traj_df.insert(2, "seed", model.seed)
                    traj_df.insert(3, "population", model.population)
                    trajectories_transitions = pd.concat([trajectories_transitions, traj_df])

    # Model Metadata
    if outputs.model_meta:
        if outputs.model_meta.projection_parameters:
            logger.warning(
                "OUTPUT_GENERATOR: Requested projection parameter metadata in simulation workflow, ignoring."
            )

        meta_dict = {}
        for model in simulation:
            meta_dict.set_default("primary_id", [])
            meta_dict["primary_id"].append(model.primary_id)

            meta_dict.set_default("seed", [])
            meta_dict["seed"].append(model.seed)

            meta_dict.set_default("delta_t", [])
            meta_dict["delta_t"].append(model.delta_t)

            meta_dict.set_default("population", [])
            meta_dict["population"].append(model.population)

            meta_dict.set_default("n_sims", [])
            meta_dict["n_sims"].append(model.results.Nsim)

            meta_dict.set_default("start_date", [])
            meta_dict["start_date"].append(str(sorted(model.results.dates)[0]))

            meta_dict.set_default("end_date", [])
            meta_dict["end_date"].append(str(sorted(model.results.dates)[-1]))

            # Parameters
            for p, v in model.results.parameters.items():
                meta_dict.set_default(p, [])
                meta_dict[p].append(str(v))

            # Initial conditions
            inits = {k: [int(v[0]) for v in vs] for k, vs in model.results.get_stacked_compartments().items()}
            for c, i in inits:
                colname = f"init_{c}"
                meta_dict.set_default(colname, [])
                meta_dict[colname].append(str(i))

        model_meta = pd.DataFrame(meta_dict)

    for warning in warnings:
        logger.warning(warning)

    out_dict = {}
    if not quantiles_compartments.empty:
        out_dict["quantiles_compartments.csv.gz"] = quantiles_compartments.to_csv(
            path_or_buf=None, header=True, index=False, compression="gzip"
        )
    if not quantiles_transitions.empty:
        out_dict["quantiles_transitions.csv.gz"] = quantiles_transitions.to_csv(
            path_or_buf=None, header=True, index=False, compression="gzip"
        )
    if not quantiles_formatted.empty:
        # will want to build filename to be something better, like to fit hub standards
        out_dict["quantiles_hub_formatted.csv.gz"] = quantiles_formatted.to_csv(
            path_or_buf=None, header=True, index=False, compression="gzip"
        )
    if not trajectories_compartments.empty:
        out_dict["trajectories_compartments.csv.gz"] = trajectories_compartments.to_csv(
            path_or_buf=None, header=True, index=False, compression="gzip"
        )
    if not trajectories_transitions.empty:
        out_dict["trajectories_transitions.csv.gz"] = trajectories_transitions.to_csv(
            path_or_buf=None, header=True, index=False, compression="gzip"
        )
    if not model_meta.empty:
        out_dict["model_metadata.csv.gz"] = model_meta.to_csv(
            path_or_buf=None, header=True, index=False, compression="gzip"
        )

    logger.info("OUTPUT GENERATOR: completed for simulation")

    return out_dict


@register_output_generator({"calibration", "outputs"})
def generate_calibration_outputs(*, calibration: list[CalibrationOutput], outputs: OutputConfig, **_) -> dict:
    """"""
    logger.info("OUTPUT GENERATOR: dispatched for calibration")
    warnings = set()

    quantiles = pd.DataFrame()
    quantiles_formatted = pd.DataFrame()
    trajectories = pd.DataFrame()
    posteriors = pd.DataFrame()
    model_meta = pd.DataFrame()

    # Quantiles
    if outputs.quantiles:
        for model in calibration:
            # Default format
            if outputs.quantiles.calibration_default:
                #
                pass



                
            try:
                quan_df = model.results.get_projection_quantiles(outputs.quantiles.selections)
            except ValueError:
                quan_df = model.results.get_calibration_quantiles(outputs.quantiles.selections)
            quan_df.insert(0, "primary_id", model.primary_id)
            quan_df.insert(1, "seed", model.seed)
            quan_df.insert(2, "population", model.population)
            quantiles = pd.concat([quantiles, quan_df])

            if outputs.quantiles.data_format == "flusightforecast":
                try:
                    quanf_df = model.results.get_projection_quantiles(get_flusight_quantiles())
                except ValueError:
                    warnings.add("OUTPUT GENERATOR: ")
                quanf_df.insert(0, "population", model.population)
                quanf_df = format_quantiles_flusightforecast(quanf_df)
                quantiles_formatted = pd.concat([quantiles_formatted, quanf_df])

                if outputs.quantiles.rate_trends:
                    trends_df = make_rate_trends_flusightforecast(quanf_df)
                    quantiles_formatted = pd.concat([quantiles_formatted, trends_df])

            elif outputs.quantiles.data_format == "covid19forecast":
                quanf_df = format_quantiles_covid19forecast(quan_df)
                quantiles_formatted = pd.concat([quantiles_formatted, quanf_df])

    # Trajectories
    if outputs.trajectories:
        for model in calibration:
            for i, traj in enumerate(model.results.get_projection_trajectories()):
                if outputs.trajectories.resample_freq:
                    try:
                        traj.resample(freq=outputs.trajectories.resample_freq)
                    except Exception as e:
                        logger.warning(
                            f"OUTPUT GENERATOR: Exception occured resampling trajectories, continuing without resampling: {e}"
                        )
                traj_df = pd.DataFrame(traj.compartments)
                if outputs.trajectories.compartments:
                    try:
                        traj_df = traj_df[outputs.trajectories.compartments]
                    except Exception as e:
                        logger.warning(
                            f"OUTPUT GENERATOR: Exception occured selecting compartments, returning all compartments: {e}"
                        )
                traj_df.insert(0, "primary_id", model.primary_id)
                traj_df.insert(1, "sim_id", i)
                traj_df.insert(2, "seed", model.seed)
                traj_df.insert(3, "population", model.population)
                trajectories = pd.concat([trajectories, traj_df])

    out_dict = {}
    if not quantiles.empty:
        out_dict["quantiles.csv.gz"] = quantiles.to_csv(path_or_buf=None, header=True, index=False, compression="gzip")
    if not quantiles_formatted.empty:
        filename = f"quantiles_{outputs.quantiles.data_format}.csv.gz"
        out_dict[filename] = quantiles_formatted.to_csv(path_or_buf=None, header=True, index=False, compression="gzip")
    if not trajectories.empty:
        out_dict["trajectories.csv.gz"] = trajectories.to_csv(
            path_or_buf=None, header=True, index=False, compression="gzip"
        )

    return out_dict


def dispatch_output_generator(**configs) -> dict:
    """Dispatch output generator functions. Returns dictionary of filenames and gzip-compressed CSV strings."""
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return OUTPUT_GENERATOR_REGISTRY[kinds](**configs)
