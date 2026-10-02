"""Requests shared by tables, hub forecasts and quantile plots."""

from ..schema.output import get_metrocast_quantiles
from ..utils.quantiles import PreparedQuantiles


def prepare_quantiles(calibrations, output):
    from ..visualization.generators import get_locations_to_plot

    # Import here: the existing selection policy belongs to the dispatcher.
    from .output import _projection_quantile_variables

    prepared = PreparedQuantiles()
    single_locations = get_locations_to_plot(calibrations, output.plots.quantiles.single) if output.plots else set()
    for item in calibrations:
        results = item.results
        if results is None:
            continue

        def add(kind, levels, variables, generation=None):
            try:
                trajectories = (
                    results.get_selected_trajectories(generation)
                    if kind == "calibration"
                    else results.projections.get("baseline", [])
                )
                dates = trajectories[0].get("date") if trajectories else None
            except (ValueError, AttributeError, TypeError, IndexError):
                return  # The consumer retains its existing warning/skip policy.
            kwargs = dict(quantiles=levels, variables=variables, dates=dates, ignore_nan=True)
            if kind == "calibration":
                kwargs["generation"] = generation
            prepared.add(results, kind, **kwargs)

        quantiles = output.quantiles
        if quantiles:
            if quantiles.calibration:
                generations = quantiles.calibration if isinstance(quantiles.calibration, list) else [None]
                for generation in generations:
                    add("calibration", quantiles.selections, ["data"], generation)
            if quantiles.compartments or quantiles.transitions:
                projections = results.projections.get("baseline", [])
                add("projection", quantiles.selections, _projection_quantile_variables(projections, quantiles))
        hub = output.flusight_format
        if hub:
            levels = get_metrocast_quantiles() if hub.metrocast else hub.quantiles
            if hub.hospitalizations:
                add("projection", levels, ["hospitalizations"])
            if hub.prop_ed:
                if hub.prop_ed.strategy == "calibration_window":
                    add("calibration", levels, ["data"])
                elif hub.prop_ed.strategy == "transition":
                    add("projection", levels, [hub.prop_ed.transition_name])
        if output.plots:
            plots = output.plots.quantiles
            grid = plots.grid is not False and plots.grid.enabled
            single = item.population in single_locations
            if grid or single:
                if any(o.show_calibration for o in plots.outputs):
                    add("calibration", plots.quantiles, ["data"])
                if any(o.show_projection for o in plots.outputs):
                    add("projection", plots.quantiles, [plots.value_column])
    return prepared
