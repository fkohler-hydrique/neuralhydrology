# Import
import logging
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from typing import Dict, List, Union

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class SwissHourly(BaseDataset):
    """Custom dataset for Swiss hourly streamflow and meteorological data.
    Expects:

    - Reads basin CSVs: expects a datetime-like column and feature columns.
    - Ensures a DatetimeIndex (hourly if possible), no MultiIndex.
    - Computes rolling features:
        prec_*_3h, prec_*_24h, temp_binn_6h_mean, degree_day_24h, is_snow_bruchji
    - Raises if required dynamic_inputs (non-rolling) are missing.
    - Does NOT create autoregressive shifted columns (BaseDataset does that).
    - Attribute TXT files under `<data_dir>/swiss_attributes_v1.0/`
      with ';'-separated values, containing 'gauge_id' as the index.
    """
    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basin: str = None,
        additional_features: List[Dict[str, pd.DataFrame]] = None,
        id_to_int: Dict[str, int] = None,
        scaler: Dict[str, Union[pd.Series, xr.DataArray]] = None,
    ):
        super().__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            additional_features=additional_features or [],
            id_to_int=id_to_int or {},
            scaler=scaler or {},
        )

    def _find_datetime_col(self, df: pd.DataFrame) -> str:
        # prefer common names
        if "date" in df.columns:
            return "date"
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "datetime", "timestamp"))]
        if candidates:
            return candidates[0]
        # fallback: test each column parseability
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                continue
            if parsed.notna().mean() > 0.95:
                return c
        raise ValueError("No datetime-like column detected in CSV.")

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        # flatten MultiIndex if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        time_col = self._find_datetime_col(df)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if df[time_col].isna().any():
            LOGGER.warning(f"Some timestamps failed to parse in column '{time_col}'. Dropping NaT rows.")
            df = df[~df[time_col].isna()]
        df = df.sort_values(time_col)
        df = df.set_index(time_col)
        return df

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load hourly time series data for the given basin.

        Parameters
        ----------
        basin : str
            Basin (gauge) ID.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with feature columns.
        """
        csv_path = Path(self.cfg.data_dir) / f"{basin}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Time series file not found for basin '{basin}' at {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["date"])
        if "date" not in df.columns:
            raise ValueError(f"CSV file {csv_path} must contain a 'date' column.")

        df = self._ensure_datetime_index(df)
        # try to set hourly frequency
        try:
            df = df.asfreq("h")
        except Exception:
            try:
                inferred = pd.infer_freq(df.index)
                if inferred is not None:
                    df = df.asfreq(inferred)
            except Exception:
                LOGGER.warning(f"Could not enforce frequency for basin {basin}, proceeding.")
        # drop duplicate timestamps
        if df.index.duplicated().any():
            raise ValueError(f"{basin}: duplicate timestamps found for basin {basin} — keeping first occurrence.")
        
        # print(self.cfg.SH_addRollingFeatures)
        if self.cfg.SH_addRollingFeatures:
            # compute rolling features
            df = add_rolling_features(df)
        # print("After rolling features added, columns are: ", df.columns)

        # Build list of required dynamic inputs (non-rolling ones)
        # dynamic_inputs in config is a list of basic features (not necessarily rolling ones)
        required_dyn = []
        if isinstance(self.cfg.dynamic_inputs, list):
            required_dyn = list(self.cfg.dynamic_inputs)
        else:
            required_dyn = [i for inputs in self.cfg.dynamic_inputs.values() for i in inputs]

        # Identify rolling-derived names we add; we won't require them to be present in CSV
        rolling_derived = []
        for base in ["prec_bruchji", "prec_fieschertal", "prec_binn", "prec_visp"]:
            rolling_derived += [f"{base}_3h", f"{base}_24h"]
        rolling_derived += ["temp_binn_6h_mean", "degree_day_24h", "is_snow_bruchji"]

        # If any required dynamic input (from config) is missing in df and is NOT a rolling-derived feature -> raise
        missing = [c for c in required_dyn if c not in df.columns and c not in rolling_derived]
        if missing:
            raise KeyError(f"Missing required dynamic input(s) for basin {basin}: {missing}")
        # print("No missing")  
        # Now compute the keep_cols as BaseDataset will expect (mirrors BaseDataset logic)
        if isinstance(self.cfg.dynamic_inputs, list):
            # print("ouais alors Jacqueline...")
            dynamic_cols_to_keep = getattr(self.cfg, "dynamic_inputs_flattened", self.cfg.dynamic_inputs)
        else:
            dynamic_cols_to_keep = [i for inputs in self.cfg.dynamic_inputs.values() for i in inputs]
        # print(dynamic_cols_to_keep)    
        # print("target variables: ", self.cfg.target_variables)
        keep_cols = (
            list(self.cfg.target_variables)
            + list(getattr(self.cfg, "evolving_attributes", []))
            + list(getattr(self.cfg, "mass_inputs", []))
            # + list(getattr(self.cfg, "autoregressive_inputs", []))
            + list(dynamic_cols_to_keep)
            + list(getattr(self.cfg, "dynamic_conceptual_inputs", []))
        )
        keep_cols = list(sorted(set(keep_cols)))
        # print("columns: ", keep_cols)
        
        # Restrict to keep_cols in stable order
        df = df[keep_cols]
        # print("After restricting to keep_cols, columns are: ", df.columns)
                

        # Final guard: ensure DatetimeIndex
        if not isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError(f"Final DataFrame for basin {basin} does not have a datetime index")
        # LOGGER.info(f"\n [SwissHourly] {basin} final columns: {list(df.columns)}")
        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load basin attributes from Swiss attribute files.

        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame containing static attributes.
        """
        attr_dir = Path(self.cfg.data_dir) / "swiss_attributes_v1.0"
        if not attr_dir.exists():
            raise RuntimeError(f"Attribute folder not found at {attr_dir}")

        txt_files = list(attr_dir.glob("swiss_*.txt"))
        if not txt_files:
            raise RuntimeError(f"No attribute files found in {attr_dir}")

        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=";", header=0, dtype={"gauge_id": str})
            if "gauge_id" not in df_temp.columns:
                raise ValueError(f"Attribute file {txt_file} must contain a 'gauge_id' column.")
            df_temp = df_temp.set_index("gauge_id")
            dfs.append(df_temp)

        # Merge on index to avoid duplicate columns instead of blind concat
        df = pd.concat(dfs, axis=1)

        # Filter to basins in use
        if self.basins:
            missing = [b for b in self.basins if b not in df.index]
            if missing:
                raise ValueError(f"Missing attributes for basins: {missing}")
            df = df.loc[self.basins]

        # Optional: clean column names (strip whitespace)
        df.columns = df.columns.str.strip()

# Function specific to Swisshourly dataset (not modularized yet!)
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # precipitation aggregates
    for prec_col in ["prec_bruchji", "prec_fieschertal", "prec_binn", "prec_visp"]:
        if prec_col in df.columns:
            df[f"{prec_col}_3h"] = df[prec_col].rolling(window=3, min_periods=1).sum()
            df[f"{prec_col}_24h"] = df[prec_col].rolling(window=24, min_periods=1).sum()
    # temperature derived
    if "temp_binn" in df.columns:
        df["temp_binn_6h_mean"] = df["temp_binn"].rolling(window=6, min_periods=1).mean()
        df["degree_day_24h"] = df["temp_binn"].clip(lower=0).rolling(window=24, min_periods=1).sum()
        df["is_snow_bruchji"] = (df["temp_binn"] <= 0).astype(int)
    return df
