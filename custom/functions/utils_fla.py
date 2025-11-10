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