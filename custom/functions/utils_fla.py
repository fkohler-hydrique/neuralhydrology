import re
import pandas as pd
from pathlib import Path
import numpy as np

def parse_neuralhydrology_log(log_path: str | Path, verbose: bool = True):
    """Parse output.log from NeuralHydrology training run.
    improvement to implement: takes the list of metrics given in the config file"""
    log_path = Path(log_path)
    text = log_path.read_text(encoding="utf-8")

    # Regex patterns
    train_pattern = r"Epoch (\d+) average loss: avg_loss: ([\d\.]+), avg_total_loss: ([\d\.]+)"
    val_pattern = (
        r"Epoch (\d+) average validation loss: ([\d\.]+).*?MAPE: ([\d\.]+), NSE: ([\d\.]+), "
        r"MSE: ([\d\.]+), RMSE: ([\d\.]+)"
    )

    # Extract
    train_data = re.findall(train_pattern, text)
    val_data = re.findall(val_pattern, text)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data, columns=["epoch", "avg_loss", "avg_total_loss"]).astype(float)
    val_df = pd.DataFrame(val_data, columns=["epoch", "avg_val_loss", "MAPE", "NSE", "MSE", "RMSE"]).astype(float)

    if verbose: print(f"✅ Parsed {len(train_df)} training epochs and {len(val_df)} validation evaluations.")
    return train_df, val_df


def mape_array(observations: np.ndarray, predictions: np.ndarray, verbose: bool = False) -> list[float]:
    """Mean Absolute Percentage Error (MAPE) computed per forecast horizon.

    Args:
        observations: np.ndarray, ground truth observations (shape: [T, H] or [T])
        predictions:  np.ndarray, model predictions (shape: [T, H] or [T])
        verbose: bool, if True prints a transposed table of MAPE per horizon

    Returns:
        list[float]: MAPE values per horizon (in percent). NaN if no valid observations for a horizon.
    """
    def _mape_np(obs: np.ndarray, pred: np.ndarray) -> float:
        """Compute MAPE for 1D numpy arrays (returns percentage)."""
        return float(np.mean(np.abs((obs - pred) / np.clip(np.abs(obs), 1e-6, None))) * 100.0)

    # Ensure numpy arrays and 2D shape [T, H]
    obs = np.asarray(observations)
    pred = np.asarray(predictions)

    if obs.ndim == 1:
        obs = obs[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]

    if obs.shape[0] != pred.shape[0]:
        raise ValueError("observations and predictions must have the same number of time steps (axis 0).")

    n_horizons = pred.shape[1]
    mapes: list[float] = []

    for h in range(n_horizons):
        obs_h = obs[:, h]
        pred_h = pred[:, h]
        valid_mask = ~np.isnan(obs_h)
        obs_valid = obs_h[valid_mask]
        pred_valid = pred_h[valid_mask]

        if obs_valid.size == 0:
            mape_h = float("nan")
        else:
            mape_h = _mape_np(obs_valid, pred_valid)

        mapes.append(mape_h)

    if verbose:
        # Prepare string representations
        vals = []
        for m in mapes:
            vals.append("NaN" if np.isnan(m) else f"{m:.2f}")

        mean_mape = float(np.nan) if all(np.isnan(m) for m in mapes) else float(np.nanmean(mapes))
        mean_str = "NaN" if np.isnan(mean_mape) else f"{mean_mape:.2f}"

        # Column headers: Metric | H1 | H2 | ... | Mean
        col_headers = ["Metric"] + [f"H{h}" for h in range(1, n_horizons + 1)] + ["Mean"]
        row_values = ["MAPE (%)"] + vals + [mean_str]

        # Compute column widths
        col_widths = [max(len(col_headers[i]), len(row_values[i])) for i in range(len(col_headers))]
        # Build and print header row
        header = " | ".join(col_headers[i].rjust(col_widths[i]) for i in range(len(col_headers)))
        sep = "-+-".join("-" * col_widths[i] for i in range(len(col_headers)))
        row = " | ".join(row_values[i].rjust(col_widths[i]) for i in range(len(col_headers)))

        print(header)
        print(sep)
        print(row)

    return mapes

def info_results(results_):
    print("----- Dataset information -----")
    print(results_)
    print("-------------------------------")
    # print first date of emissions and last date of emissions - both synthaxes are ok -
    print("First date of prediction: \t", results_.streamflow_obs.date.values[0])
    print("Last date of prediction: \t", results_['streamflow_obs'].date.values[-1])
    print("Number of time steps: \t\t", results_.sizes['date'])
    print("Shape of data: \t\t\t", results_.sizes) 
    print("-------------------------------")


def dict_without_xr_key(old_dict:dict):
    new_dict = old_dict.copy()
    new_dict.pop('xr', None)
    return new_dict



from typing import Iterable, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

def hydrograph_flows_distribution(
    flows: Iterable[float],
    bins: Optional[Iterable[float]] = None,
    plot: bool = False,
    plot_kind: str = "both",  # "both", "hydrograph", or "hist"
    ax: Optional[Tuple[plt.Axes, ...]] = None,
) -> Dict[str, Any]:
    """
    Analyze a hydrograph (time series of flow rates) by classifying flows into classes.

    Parameters
    ----------
    flows : Iterable[float]
        1D sequence of flow values (e.g. discharge in m³/s) over time.
    bins : Iterable[float], optional
        Class boundaries for flow ranges. Must be monotonically increasing.
        - If None, defaults to [0, 10, 40, 80, np.inf] corresponding to:
          [0–10), [10–40), [40–80), [80–inf).
        - You can pass your own like [0, 5, 20, 100, np.inf].
    plot : bool, default False
        Whether to produce a plot.
    plot_kind : {"both", "hydrograph", "hist"}, default "both"
        Type of plot to generate if `plot=True`:
        - "hydrograph": time series only
        - "hist": bar plot of class percentages
        - "both": hydrograph + histogram
    ax : matplotlib Axes or tuple of Axes, optional
        Axes to plot on. If None, new figure(s) will be created.
        - For plot_kind="both", pass a tuple (ax1, ax2).
        - For "hydrograph" or "hist", pass a single Axes.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "bins": np.ndarray of bin edges
        - "counts": np.ndarray of counts per class
        - "percentages": np.ndarray of percentages per class (sum to 100)
        - "labels": list of str, human-readable class labels
        - "flows": np.ndarray of input flows (cleaned 1D array)
    """
    # ---- 1. Prepare data ----
    flows = np.asarray(flows, dtype=float).ravel()
    if flows.size == 0:
        raise ValueError("`flows` is empty. Provide a non-empty sequence of flow values.")

    if bins is None:
        bins = [0.0, 10.0, 40.0, 80.0, np.inf]
    bins = np.asarray(bins, dtype=float)

    if not np.all(np.diff(bins) > 0):
        raise ValueError("`bins` must be strictly monotonically increasing.")

    # ---- 2. Compute histogram (class counts) ----
    counts, edges = np.histogram(flows, bins=bins)
    total = counts.sum()
    if total == 0:
        percentages = np.zeros_like(counts, dtype=float)
    else:
        percentages = counts / total * 100.0

    # ---- 3. Make human-readable class labels ----
    def _format_edge(val: float) -> str:
        if np.isinf(val):
            return "∞"
        # Trim trailing .0 for nicer labels
        s = f"{val:g}"
        return s

    labels = []
    for i in range(len(edges) - 1):
        left = _format_edge(edges[i])
        right = _format_edge(edges[i + 1])
        if np.isinf(edges[i + 1]):
            label = f"{left}–∞"
        else:
            # Convention: [left, right)
            label = f"{left}–{right}"
        labels.append(label)

    result = {
        "bins": edges,
        "counts": counts,
        "percentages": percentages,
        "labels": labels,
        "flows": flows,
    }

    # ---- 4. Optional plotting ----
    if plot:
        plot_kind = plot_kind.lower()
        if plot_kind not in {"both", "hydrograph", "hist"}:
            raise ValueError("plot_kind must be one of {'both', 'hydrograph', 'hist'}")

        # Figure + axes handling
        if plot_kind == "both":
            if ax is None:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(8, 6), sharex=False,
                    gridspec_kw={"height_ratios": [2, 1]}
                )
            else:
                ax1, ax2 = ax
        else:
            if ax is None:
                fig, ax_single = plt.subplots(figsize=(8, 4))
            else:
                ax_single = ax

        # Hydrograph plot
        if plot_kind in {"both", "hydrograph"}:
            if plot_kind == "both":
                ax_h = ax1
            else:
                ax_h = ax_single

            t = np.arange(flows.size)
            ax_h.plot(t, flows, lw=1.5)
            ax_h.set_xlabel("Time step")
            ax_h.set_ylabel("Flow")
            ax_h.set_title("Hydrograph")

        # Histogram / class distribution plot
        if plot_kind in {"both", "hist"}:
            if plot_kind == "both":
                ax_bar = ax2
            else:
                ax_bar = ax_single

            x = np.arange(len(labels))
            ax_bar.bar(x, percentages)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(labels, rotation=45, ha="right")
            ax_bar.set_ylabel("Percentage of time (%)")
            ax_bar.set_title("Flow class distribution")
            ax_bar.grid(True, axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()

    return result


def print_hydrograph_distribution(res: dict):
    labels = res["labels"]
    counts = res["counts"]
    percentages = res["percentages"]

    # 1) Optional: overall summary
    print(f"Total points: {counts.sum()}\n")

    # 2) Column headers
    print(f"{'Class':<10} {'Count':>10} {'Percentage':>12}")
    print("-" * 34)

    # 3) Rows
    for label, c, p in zip(labels, counts, percentages):
        print(f"{label:<10} {c:>10d} {p:>11.2f}%")



import numpy as np
from typing import Iterable, Dict, Any, Optional

def mape_by_flow_class(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    bins: Optional[Iterable[float]] = None,
    ignore_zero: bool = True,
) -> Dict[str, Any]:
    """
    Compute MAPE per flow class, where classes are defined on the *observed* flows.

    Parameters
    ----------
    y_true : array-like
        Observed/measured flows.
    y_pred : array-like
        Simulated/predicted flows. Must be broadcastable to y_true.
    bins : array-like, optional
        Class boundaries for flow ranges, monotonically increasing.
        If None, defaults to [0, 10, 40, 80, np.inf].
    ignore_zero : bool, default True
        If True, exclude points where y_true == 0 from MAPE calculation
        (to avoid division by zero). If False, those MAPEs will be set to np.nan.

    Returns
    -------
    result : dict
        {
            "bins": np.ndarray of bin edges,
            "labels": list of str for classes,
            "counts": np.ndarray of number of points in each class (all points),
            "valid_counts": np.ndarray of points actually used for MAPE
                            (depends on ignore_zero),
            "mape_per_class": np.ndarray of MAPE values (percent) per class,
            "global_mape": float, overall MAPE over all valid points,
        }
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape after ravel().")
    if y_true.size == 0:
        raise ValueError("Empty input arrays.")

    if bins is None:
        bins = [0.0, 10.0, 40.0, 80.0, np.inf]
    bins = np.asarray(bins, dtype=float)
    if not np.all(np.diff(bins) > 0):
        raise ValueError("`bins` must be strictly monotonically increasing.")

    # Helper: pretty class labels
    def _fmt(v: float) -> str:
        if np.isinf(v):
            return "∞"
        return f"{v:g}"

    labels = []
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        if np.isinf(right):
            labels.append(f"{_fmt(left)}–∞")
        else:
            labels.append(f"{_fmt(left)}–{_fmt(right)}")

    counts = np.zeros(len(labels), dtype=int)
    valid_counts = np.zeros(len(labels), dtype=int)
    mape_per_class = np.full(len(labels), np.nan, dtype=float)

    # Global mask for valid MAPE points
    if ignore_zero:
        global_valid = y_true != 0
    else:
        # Avoid division by zero by marking zeros invalid;
        # they will just never enter valid sets, so MAPE may be nan.
        global_valid = y_true != 0

    abs_pct_errors_all = np.abs((y_pred[global_valid] - y_true[global_valid]) / y_true[global_valid]) * 100

    # Class-specific MAPE
    for i in range(len(labels)):
        lower, upper = bins[i], bins[i + 1]
        class_mask = (y_true >= lower) & (y_true < upper)
        counts[i] = np.count_nonzero(class_mask)

        valid_mask = class_mask & global_valid
        valid_counts[i] = np.count_nonzero(valid_mask)

        if valid_counts[i] > 0:
            mape_per_class[i] = np.mean(
                np.abs((y_pred[valid_mask] - y_true[valid_mask]) / y_true[valid_mask])
            ) * 100.0
        else:
            mape_per_class[i] = np.nan

    # Global MAPE (over all valid points)
    global_mape = np.nan
    if abs_pct_errors_all.size > 0:
        global_mape = abs_pct_errors_all.mean()

    return {
        "bins": bins,
        "labels": labels,
        "counts": counts,
        "valid_counts": valid_counts,
        "mape_per_class": mape_per_class,
        "global_mape": global_mape,
    }
