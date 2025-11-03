import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from matplotlib import cm

def get_n_colors(n: int) -> List[str]:
    """Return a list of `n` visually distinct, plot-friendly hex colors.

    Characteristics:
    - Avoids pure yellow and obvious fluorescent greens.
    - Returns hex color strings (e.g. "#2a9d8f").
    - If n is larger than the curated palette, colors are repeated by
      cycling through the palette (keeps consistency and visibility).

    Args:
        n: number of colors requested (n >= 0).

    Returns:
        list of hex color strings of length n.
    """

    if n <= 0:
        return []

    # Curated palette (no yellow or neon/fluo greens). These are
    # chosen to be distinct and visible on light/dark backgrounds.
    palette = [
        "#3850D8",  # deep cyan/teal
        "#b53bda",  # purple
        "#f37805",  # orange (not yellow)
        "#22baec",  # sky blue
        "#b5179e",  # magenta
        "#2a9d8f",  # teal (non-fluorescent)
        "#EEAAC4",  # very dark blue
        "#6e4b96",  # maroon
        "#0077b6",  # medium blue
        "#264653",  # slate
        "#8ecae6",  # light blue
        "#6a4c93",  # violet
        "#fb8500",  # darker orange
        "#2b2d42",  # near-black blue
        "#5a189a",  # deep purple
        "#ff6b6b",  # coral
        "#e63946",  # strong red
        "#d62828",  # deep red
        "#ef476f",  # pink
        "#4cc9f0",  # bright sky blue
    ]

    # If we need more colors than in the palette, cycle through it.
    colors = [palette[i % len(palette)] for i in range(n)]
    return colors

def plot_hydrograph(
    qobs,
    qsim,
    results: dict = None,
    basin: str = None,
    freq: str = None,
    plot_peaks: bool = False,
    top_percent: int = 5,
    width: int = 1000,
    height: int = 500,
):
    """
    Plot an interactive hydrograph with Plotly.

    Parameters
    ----------
    qobs : xarray.DataArray or pandas.Series
        Observed streamflow data.
    qsim : xarray.DataArray or pandas.Series
        Simulated streamflow data.
    results : dict, optional
        Dictionary containing performance metrics (e.g., NSE, MAPE, RMSE).
        Can follow NeuralHydrology's output format (e.g., results['massa']['1h']).
    basin : str, optional
        Basin name (used for labeling when results is provided).
    freq : str, optional
        Temporal resolution label (e.g., '1h', '1d') for display.
    plot_peaks : bool, default=False
        If True, highlight the top `top_percent` of observed flows as flood peaks.
    top_percent : int, default=5
        Percentile threshold for defining high-flow events.
    width : int, default=1000
        Plot width in pixels.
    height : int, default=500
        Plot height in pixels.
    """

    # --- Extract time axis ---
    if hasattr(qobs, "coords"):
        if "date" in qobs.coords:
            dates = qobs["date"].values
        elif "time" in qobs.coords:
            dates = qobs["time"].values
        else:
            raise KeyError("No 'date' or 'time' coordinate found in qobs.")
    elif hasattr(qobs, "index"):
        dates = qobs.index.values
    else:
        raise TypeError("qobs must be an xarray DataArray or pandas Series with a date/time index.")

    # --- Convert to numpy and flatten ---
    qobs_vals = np.ravel(np.asarray(qobs.values, dtype=float))
    qsim_vals = np.ravel(np.asarray(qsim.values, dtype=float))

    # --- Trim to same length ---
    min_len = min(len(dates), len(qobs_vals), len(qsim_vals))
    dates, qobs_vals, qsim_vals = dates[:min_len], qobs_vals[:min_len], qsim_vals[:min_len]

    # --- Create Plotly figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=qobs_vals,
        mode='lines',
        name='Observed flow',
        line=dict(color='#da2843', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=qsim_vals,
        mode='lines',
        name='Simulated flow',
        line=dict(color='royalblue', width=1.5)
    ))

    # --- Optionally highlight high-flow peaks ---
    if plot_peaks:
        threshold = np.nanpercentile(qobs_vals, 100 - top_percent)
        high_flow_mask = qobs_vals >= threshold
        fig.add_trace(go.Scatter(
            x=dates[high_flow_mask],
            y=qobs_vals[high_flow_mask],
            mode='markers',
            name=f'High-flow events (top {top_percent}%)',
            marker=dict(color='gold', size=6, line=dict(color='black', width=0.5))
        ))

    # --- Build title ---
    if results and basin and freq:
        metrics = results.get(basin, {}).get(freq, {})
        mape = metrics.get("MAPE", np.nan)
        nse = metrics.get("NSE", np.nan)
        rmse = metrics.get("RMSE", np.nan)
        title = f"{basin} – {freq} | MAPE {mape:.3f}, NSE {nse:.3f}, RMSE {rmse:.3f}"
    else:
        title = "Hydrograph Comparison"

    # --- Layout ---
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Discharge (mm/d)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        width=width,
        height=height,
        margin=dict(l=50, r=30, t=80, b=50)
    )

    # --- Zoom range slider ---
    fig.update_xaxes(rangeslider_visible=True)

    # --- Show ---
    fig.show()


# ----------------

def plot_training_curves(train_df, val_df):
    """Plot loss and validation metrics."""
    fig, ax1 = plt.subplots(figsize=(6, 3))

    c1 = "#1e3a7a"
    c2 = "#dc6b2b"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color=c1)
    ax1.plot(train_df["epoch"], train_df["avg_total_loss"], label="Training Loss", lw=2, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel("Validation Loss", color=c2)
    ax2.plot(val_df["epoch"], val_df["avg_val_loss"], label="Validation Loss", lw=2, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    

    plt.title("Training vs Validation Loss")
    plt.box()
    plt.grid(alpha=0.5)
    fig.tight_layout()
    plt.show()



def plot_validation_metrics(val_df):
    """Plot validation metrics like NSE, RMSE, etc."""
    c = "#50c878"
    metrics = ["MAPE", "NSE", "MSE", "RMSE"]
    fig, axs = plt.subplots(2, 2, figsize=(8, 5))
    axs = axs.ravel()

    for i, m in enumerate(metrics):
        if m in val_df.columns:
            axs[i].plot(val_df["epoch"], val_df[m], lw=2, color=c)
            axs[i].set_title(m)
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel(m)
            axs[i].grid(alpha=0.5)

    fig.suptitle("Validation Metrics over Epochs", fontsize=14)
    plt.tight_layout()
    plt.show()

# --------------------



def plot_forecast_24h(
    xr_ds=None,
    mode="list",             # "mean", "quantiles", or "spaghetti"
    quantiles=(0.1, 0.9),    # only used in quantiles mode
    var_obs="streamflow_obs",
    var_sim="streamflow_sim",
    title="24h Rolling Forecast vs Observed",
    horizons=[24],
    width: int = 1000,
    height: int = 500,
    measured = None,
    predicted=None
):
    """
    Plot 24h horizon forecasts issued hourly from an AR-LSTM model.

    Parameters
    ----------
    xr_ds : xarray.Dataset
        Dataset containing forecast results with dimensions (date, time_step).
    mode : str
        'mean' → average of overlapping forecasts;
        'quantiles' → shaded P10–P90 fan;
        'spaghetti' → plot every individual forecast line.
    quantiles : tuple(float, float)
        Quantiles for the shaded fan if mode='quantiles'.
    var_obs : str
        Observation variable name in xr_ds.
    var_sim : str
        Simulation variable name in xr_ds.
    title : str
        Plot title.
    """

    # Convert to pandas
    if (measured is not None) and (predicted is not None):
        obs_df = measured.to_pandas()
        sim_df = predicted.to_pandas()
    else:
        obs_df = xr_ds[var_obs].to_pandas()
        sim_df = xr_ds[var_sim].to_pandas()

    horizon_len = len(obs_df.columns)
    # Combine the two observed segments into a single trace so only one legend entry appears.
    # First segment: full-series shifted left by the horizon length (start of each forecast)
    x1 = obs_df.index - pd.Timedelta(hours=horizon_len-1)
    y1 = obs_df.iloc[:, 0].values

    # Second segment: the last `horizon_len` timestamps using the last column (end of each horizon)
    x2 = obs_df[-horizon_len:].index
    y2 = obs_df.iloc[-horizon_len:, -1].values

    # Build a single Series, concatenate and sort by time to produce a continuous trace
    s1 = pd.Series(y1, index=pd.to_datetime(x1))
    s2 = pd.Series(y2, index=pd.to_datetime(x2))
    obs_combined = pd.concat([s1, s2]).sort_index()
    obs_combined = obs_combined[~obs_combined.index.duplicated(keep="first")]

    # Build Plotly figure
    fig = go.Figure()

    # store trace data (x,y arrays) for dynamic y-scaling callback
    trace_data = []

    fig.add_trace(
        go.Scatter(
            x=obs_combined.index,
            y=obs_combined.values,
            mode="lines",
            name="Observed Flow",
            line=dict(color="crimson", width=2),
        )
    )
    trace_data.append({"x": np.asarray(obs_combined.index).astype("datetime64[ns]"), "y": np.asarray(obs_combined.values)})

    if mode == "spaghetti":
        # Plot every forecast trajectory
        print("not yet implemented")
    elif mode == "mean":
        print("not yet implemented")
    elif mode == "quantiles":
        print("not yet implemented")
    elif mode == "one":
        print("not yet implemented")
    elif mode == "list":
        # print(sim_df.iloc[:, 25-1].values)
        colors = get_n_colors(len(horizons))
        # note: index 0 means t+24h, -23 means t+1h
        for color, hor in zip(colors,horizons):
            xs = sim_df.index - pd.Timedelta(hours=(horizon_len-hor))
            ys = sim_df.iloc[:, hor-1].values
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=f"Simulated Flow | +{hor}h",
                    line=dict(color=color, width=1.5)
                )
            )
            trace_data.append({"x": np.asarray(xs).astype("datetime64[ns]"), "y": np.asarray(ys)})
    elif mode == "last":
        # display the last prediction of the horizon (t+24h)
        fig.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=sim_df[0],
                mode="lines",
                name=f"Simulated Flow",
                line=dict(color="royalblue", width=1.5)
            )
        )
        trace_data.append({"x": np.asarray(sim_df.index).astype("datetime64[ns]"), "y": np.asarray(sim_df[0].values)})
    else:
        raise ValueError("unknown mode detected")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Streamflow",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=width,
        height=height,
        margin=dict(l=50, r=30, t=80, b=50),
        dragmode="zoom"   # enable box zoom by default
    )

    # --- Zoom range slider ---
    fig.update_xaxes(rangeslider_visible=True)

    # Convert to FigureWidget for interactive callbacks in Jupyter
    try:
        figw = go.FigureWidget(fig)
    except Exception:
        # fallback to non-widget figure if not in Jupyter environment
        figw = fig

    # dynamic y-axis scaling callback (works when using FigureWidget in a Jupyter notebook)
    if isinstance(figw, go.FigureWidget):
        def _update_yaxis(relayout_data):
            # handle autorange/reset
            if relayout_data is None:
                return
            # when reset axes or autorange requested, use autorange
            if relayout_data.get("xaxis.autorange") or relayout_data.get("autosize"):
                figw.layout.yaxis.autorange = True
                return

            # typical relayout keys from slider/zoom are 'xaxis.range[0]' and 'xaxis.range[1]'
            x0 = relayout_data.get("xaxis.range[0]") or relayout_data.get("xaxis.range")
            x1 = relayout_data.get("xaxis.range[1]") if "xaxis.range[1]" in relayout_data else None

            # handle tuple form or single key
            if isinstance(x0, (list, tuple)) and len(x0) == 2:
                x0, x1 = x0[0], x0[1]
            if x0 is None or x1 is None:
                return

            # convert to pandas Timestamp for comparisons
            try:
                x0_ts = pd.to_datetime(x0)
                x1_ts = pd.to_datetime(x1)
            except Exception:
                return

            # compute min/max across visible portion of each stored trace
            ys_min = []
            ys_max = []
            for td in trace_data:
                xs = pd.to_datetime(td["x"])
                mask = (xs >= x0_ts) & (xs <= x1_ts)
                if mask.any():
                    ys = np.asarray(td["y"])[mask]
                    # filter nan
                    ys = ys[~np.isnan(ys)]
                    if ys.size > 0:
                        ys_min.append(ys.min())
                        ys_max.append(ys.max())

            if len(ys_min) == 0:
                return

            new_min = float(np.min(ys_min))
            new_max = float(np.max(ys_max))
            # add small padding
            pad = (new_max - new_min) * 0.06 if new_max > new_min else max(abs(new_min), 1.0) * 0.06
            figw.update_yaxes(range=[new_min - pad, new_max + pad], autorange=False)

        # attach callback
        figw.on_relayout(_update_yaxis)

    # Show figure
    figw.show()
    # fig.show()

# ==========================================

import pandas as pd
import plotly.graph_objects as go


def plot_forecast_horizon_24h(
    issue_date,
    xr_ds=None,
    var_obs="streamflow_obs",
    var_sim="streamflow_sim",
    title=None,
    measured = None,
    predicted=None
):
    """
    Plot a single 24-hour forecast horizon (spaghetti view) for a given issue date.

    Parameters
    ----------
    xr_ds : xarray.Dataset
        Dataset with dimensions (date, time_step) and variables for obs/sim.
    issue_date : str or pd.Timestamp
        Forecast issue datetime (must exist in xr_ds['date']).
        Example: "2025-03-01 00:00:00"
    var_obs : str
        Observation variable name.
    var_sim : str
        Simulation variable name.
    title : str or None
        Optional custom title.
    """

    # Ensure proper timestamp
    issue_date = pd.Timestamp(issue_date)

    # Find the closest forecast issue (exact or nearest)
    all_dates = pd.to_datetime(xr_ds["date"].values)
    closest_date = all_dates[np.argmin(np.abs(all_dates - issue_date))]

    # Extract corresponding forecast row
    obs_row = xr_ds[var_obs].sel(date=closest_date).to_pandas()
    sim_row = xr_ds[var_sim].sel(date=closest_date).to_pandas()

    # Forecast horizon hours (relative to issue)
    horizons = xr_ds["time_step"].values.astype(int)
    abs_times = closest_date + pd.to_timedelta(horizons, unit="h")

    # Build figure
    fig = go.Figure()

    # Simulated line
    fig.add_trace(
        go.Scatter(
            x=abs_times,
            y=sim_row.values,
            mode="lines+markers",
            name="Predicted",
            line=dict(color="royalblue", width=3),
            marker=dict(size=6),
        )
    )

    # Observed line
    fig.add_trace(
        go.Scatter(
            x=abs_times,
            y=obs_row.values,
            mode="lines+markers",
            name="Observed",
            line=dict(color="crimson", width=3),
            marker=dict(size=5),
        )
    )

    # Title & layout
    if title is None:
        title = f"24h Forecast Horizon — Issued at {closest_date.strftime('%Y-%m-%d %H:%M')}"

    fig.update_layout(
        title=title,
        xaxis_title="Forecast Time (absolute)",
        yaxis_title="Streamflow",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
    )

    return fig


#====================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib


def plot_forecast_horizon_series(
    xr_ds,
    issue_date,
    n_forecasts=1,
    var_obs="streamflow_obs",
    var_sim="streamflow_sim",
    title=None,
):
    """
    Plot N consecutive forecast horizons (each 24 h window) starting from a given issue date.
    Shows observed values once (only within horizon coverage) and forecasts in a gradient of colors.

    Parameters
    ----------
    xr_ds : xarray.Dataset
        Dataset with dimensions (date, time_step) and variables for obs/sim.
    issue_date : str or pd.Timestamp
        Forecast issue datetime to start from (must exist in xr_ds['date']).
    n_forecasts : int
        Number of consecutive forecasts to plot.
    var_obs : str
        Observation variable name.
    var_sim : str
        Simulation variable name.
    title : str or None
        Optional title.
    """

    issue_date = pd.Timestamp(issue_date)
    all_dates = pd.to_datetime(xr_ds["date"].values)

    # Find the starting index
    idx_start = np.argmin(np.abs(all_dates - issue_date))
    dates_to_plot = all_dates[idx_start : idx_start + n_forecasts]

    if len(dates_to_plot) == 0:
        raise ValueError("No forecasts found for the given date range.")

    time_steps = xr_ds["time_step"].values.astype(int)

    # Compute the full coverage window of all horizons
    earliest_time = dates_to_plot[0] + pd.to_timedelta(time_steps.min(), unit="h")
    latest_time = dates_to_plot[-1] + pd.to_timedelta(time_steps.max(), unit="h")

    # --- Extract observed data only within coverage range
    obs_df = xr_ds[var_obs].to_pandas()
    obs_df = obs_df[(obs_df.index >= earliest_time) & (obs_df.index <= latest_time)]

    obs_series = obs_df.stack().reset_index(level=1, drop=True)
    obs_series.index = pd.to_datetime(obs_series.index)
    obs_series = obs_series[(obs_series.index >= earliest_time) & (obs_series.index <= latest_time)]

    # --- Build the colormap (matplotlib 3.7+ compatible)
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [
        f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.8)"
        for r, g, b, _ in cmap(np.linspace(0.1, 1, n_forecasts))
    ]

    # --- Prepare Plotly figure
    fig = go.Figure()

    # Observed line (only once, restricted in time)
    fig.add_trace(
        go.Scatter(
            x=obs_df.index,
            y=obs_df[0],
            mode="lines",
            name="Observed",
            line=dict(color="crimson", width=3),
            opacity=0.95,
        )
    )

    # --- Add each forecast horizon
    for i, forecast_date in enumerate(dates_to_plot):
        sim_row = xr_ds[var_sim].sel(date=forecast_date).to_pandas()
        abs_times = forecast_date + pd.to_timedelta(time_steps, unit="h")

        fig.add_trace(
            go.Scatter(
                x=abs_times,
                y=sim_row.values,
                mode="lines+markers",
                name=f"Forecast {forecast_date.strftime('%Y-%m-%d %H:%M')}",
                line=dict(color=colors[i], width=2),
                marker=dict(size=5, color=colors[i]),
                opacity=0.9,
            )
        )

    # --- Layout
    if title is None:
        title = (
            f"{n_forecasts} consecutive forecasts from {dates_to_plot[0].strftime('%Y-%m-%d %H:%M')} "
            f"to {dates_to_plot[-1].strftime('%Y-%m-%d %H:%M')}"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date (absolute forecast time)",
        yaxis_title="Streamflow",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=10)
        ),
        height=600,
    )

    return fig





def plot_mape_horizon(mapes, horizons=None, color="#2a9d8f", figsize=(10, 5),
                      marker="o", linestyle="-", title=None, xlabel="Forecast Horizon (hours)",
                      ylabel="MAPE (%)", annotate=True, show=True):
    """
    Nicely formatted plot of MAPE vs forecast horizon.
    Tries to use a nicer matplotlib style but falls back if not available.

    Parameters
    - mapes: array-like of MAPE values (one per horizon)
    - horizons: array-like of horizon indices. If None, will be 1..len(mapes)
    - color: main color for line and fill
    - figsize: figure size tuple
    - marker, linestyle: marker and line style for the series
    - title: plot title (if None, a generic title is used)
    - xlabel, ylabel: axis labels
    - annotate: annotate mean value on the plot
    - show: whether to call plt.show()
    Returns: (fig, ax)
    """
    # try a nicer style but fall back to a safe default
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        plt.style.use("ggplot")

    mapes_arr = np.asarray(mapes)
    if horizons is None:
        horizons = np.arange(1, len(mapes_arr) + 1)
    else:
        horizons = np.asarray(horizons)

    fig, ax = plt.subplots(figsize=figsize)

    # Line + markers
    ax.plot(horizons, mapes_arr, color=color, marker=marker, linestyle=linestyle,
            linewidth=2, markersize=6, label="MAPE")

    # Soft filled area under the curve down to zero
    ax.fill_between(horizons, mapes_arr, 0, color=color, alpha=0.10)

    # Mean line (dashed)
    mean_val = mapes_arr.mean()
    ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.2f}%")

    # Tidy axis, ticks and limits
    ax.set_xticks(horizons)
    ymin = mapes_arr.min() * 0.9
    ymax = mapes_arr.max() * 1.12
    ax.set_ylim(ymin, ymax)

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = "MAPE vs Forecast Horizon"
    ax.set_title(title, fontsize=12, pad=10)

    # Grid and legend
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, edgecolor="0.85")

    # Annotate last point and mean
    if annotate and len(horizons) > 0:
        ax.scatter(horizons[-1], mapes_arr[-1], s=70, facecolors="white", edgecolors=color, zorder=5)
        ax.text(horizons[-1] + 0.3, mapes_arr[-1], f"{mapes_arr[-1]:.2f}%", va="center", ha="left", fontsize=9)
        # place mean label near right side
        ax.text(horizons[-1] + 0.3, mean_val, f"Mean: {mean_val:.2f}%", va="center", ha="left", fontsize=9, color="gray")

    plt.tight_layout()
    if show:
        plt.show()
    # return fig, ax
#===================