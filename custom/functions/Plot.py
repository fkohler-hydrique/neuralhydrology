import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(10, 5))
    plt.semilogy(train_df["epoch"], train_df["avg_total_loss"], label="Training Loss", lw=2)
    plt.semilogy(val_df["epoch"], val_df["avg_val_loss"], 'o--', label="Validation Loss", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

# -------------------------

def plot_validation_metrics(val_df):
    """Plot validation metrics like NSE, RMSE, etc."""
    metrics = ["MAPE", "NSE", "MSE", "RMSE"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    for i, m in enumerate(metrics):
        if m in val_df.columns:
            axs[i].plot(val_df["epoch"], val_df[m], marker='o', lw=2)
            axs[i].set_title(m)
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel(m)
            axs[i].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Validation Metrics over Epochs", fontsize=14)
    plt.tight_layout()
    plt.show()

# --------------------


import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_forecast_24h(
    xr_ds,
    mode="last",             # "mean", "quantiles", or "spaghetti"
    quantiles=(0.1, 0.9),    # only used in quantiles mode
    var_obs="streamflow_obs",
    var_sim="streamflow_sim",
    title="24h Rolling Forecast vs Observed",
    int_horizon=0
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
    obs_df = xr_ds[var_obs].to_pandas()
    sim_df = xr_ds[var_sim].to_pandas()

    # Build forecast issue/horizon timestamps
    all_forecasts = []
    for issue_time, row in sim_df.iterrows():
        for h, val in enumerate(row.values):
            horizon = int(xr_ds["time_step"].values[h])
            # convert to absolute time
            abs_time = pd.Timestamp(issue_time) + pd.Timedelta(hours=horizon)
            all_forecasts.append((abs_time, issue_time, val))
    df_forecasts = pd.DataFrame(all_forecasts, columns=["abs_time", "issue_time", "forecast"])
    df_forecasts.sort_values("abs_time", inplace=True)

    # Flatten observed data
    obs_series = obs_df.stack().reset_index(level=1, drop=True)
    obs_series.index = pd.to_datetime(obs_series.index)

    # Build Plotly figure
    fig = go.Figure()

    # Observed line
    fig.add_trace(
        go.Scatter(
            x=obs_df.index,
            y=obs_df[0],
            mode="lines",
            name="Observed Flow",
            line=dict(color="crimson", width=3),
        )
    )

    if mode == "spaghetti":
        # Plot every forecast trajectory
        for issue_time, group in df_forecasts.groupby("issue_time"):
            fig.add_trace(
                go.Scatter(
                    x=group["abs_time"],
                    y=group["forecast"],
                    mode="lines",
                    line=dict(color="royalblue", width=0.7),
                    opacity=0.3,
                    name="Forecast" if issue_time == df_forecasts["issue_time"].iloc[0] else None,
                    showlegend=(issue_time == df_forecasts["issue_time"].iloc[0]),
                )
            )

    elif mode == "mean":
        df_mean = df_forecasts.groupby("abs_time")["forecast"].mean()
        fig.add_trace(
            go.Scatter(
                x=df_mean.index,
                y=df_mean.values,
                mode="lines",
                name="Forecast mean",
                line=dict(color="royalblue", width=2),
            )
        )

    elif mode == "quantiles":
        q_low, q_high = quantiles
        df_q = (
            df_forecasts.groupby("abs_time")["forecast"]
            .quantile([q_low, 0.5, q_high])
            .unstack(level=1)
            .rename(columns={q_low: "low", 0.5: "median", q_high: "high"})
        )

        # Build the filled fan (convert index to Series for concat)
        x_vals = pd.concat([pd.Series(df_q.index), pd.Series(df_q.index[::-1])])
        y_vals = pd.concat([df_q["high"], df_q["low"][::-1]])

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                fill="toself",
                fillcolor="rgba(65,105,225,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"Forecast {int(q_low*100)}–{int(q_high*100)}%",
            )
        )
        # Median line
        fig.add_trace(
            go.Scatter(
                x=df_q.index,
                y=df_q["median"],
                mode="lines",
                name="Forecast median",
                line=dict(color="royalblue", width=2),
            )
        )
    elif mode == "one":
        # display the last prediction of the horizon (t+24h)
        fig.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=sim_df[int_horizon],
                mode="lines",
                name=f"Simulated Flow",
                line=dict(color="royalblue", width=3),
                marker=dict(size=6),
            )
        )
    elif mode == "last":
        # display the last prediction of the horizon (t+24h)
        fig.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=sim_df[0],
                mode="lines",
                name=f"Simulated Flow",
                line=dict(color="royalblue", width=3),
                marker=dict(size=6),
            )
        )
    else:
        raise ValueError("mode must be one of: 'mean', 'quantiles', or 'spaghetti'")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Streamflow",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig

# ==========================================

import pandas as pd
import plotly.graph_objects as go


def plot_forecast_horizon_24h(
    xr_ds,
    issue_date,
    var_obs="streamflow_obs",
    var_sim="streamflow_sim",
    title=None,
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


#===================