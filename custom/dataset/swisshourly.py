import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import xarray as xr

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class SwissHourly(BaseDataset):
    """Custom dataset for Swiss hourly streamflow and meteorological data.

    Responsibilities
    ----------------
    - Load basin CSVs (one per basin) and ensure:
      * A proper DatetimeIndex (ideally hourly), no MultiIndex.
      * No duplicate timestamps.
    - Optionally compute rolling features:
        * prec_*_3h, prec_*_24h
        * temp_binn_6h_mean
        * degree_day_24h
        * is_snow_bruchji
    - Validate that required dynamic inputs (non-rolling ones) are present.
    - Load static attributes from TXT files under:
        <data_dir>/swiss_attributes_v1.0/
      Files are ';'-separated and must contain a 'gauge_id' column.

    Note
    ----
    - This dataset does NOT create autoregressive shifted columns; this is
      handled by BaseDataset.
    """

    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basin: str | None = None,
        additional_features: List[Dict[str, pd.DataFrame]] | None = None,
        id_to_int: Dict[str, int] | None = None,
        scaler: Dict[str, Union[pd.Series, xr.DataArray]] | None = None,
    ) -> None:
        """Initialize the SwissHourly dataset."""
        super().__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            additional_features=additional_features or [],
            id_to_int=id_to_int or {},
            scaler=scaler or {},
        )

    # -------------------------------------------------------------------------
    # Helper methods for time handling
    # -------------------------------------------------------------------------
    def _find_datetime_col(self, df: pd.DataFrame) -> str:
        """Infer which column in `df` should be used as the datetime column.

        Strategy:
        - Prefer a column literally named "date".
        - Otherwise, pick the first column whose name contains one of:
          "date", "time", "datetime", "timestamp" (case-insensitive).
        - As a fallback, attempt to parse each column as datetime and select
          the one where >95% of values parse successfully.

        Raises
        ------
        ValueError
            If no suitable datetime-like column can be identified.
        """
        # Prefer a column literally named "date"
        if "date" in df.columns:
            return "date"

        # Next, try by column name heuristics
        candidates = [
            c
            for c in df.columns
            if any(key in c.lower() for key in ("date", "time", "datetime", "timestamp"))
        ]
        if candidates:
            return candidates[0]

        # Fallback: test parseability for each column
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                continue
            if parsed.notna().mean() > 0.95:
                return col

        raise ValueError("No datetime-like column detected in CSV.")

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` with a DatetimeIndex.

        - Flattens a MultiIndex (if present) by resetting it.
        - Detects the datetime column using `_find_datetime_col`.
        - Parses timestamps and drops rows with invalid timestamps.
        - Sorts by time and sets the time column as index.
        """
        df = df.copy()

        # Flatten any MultiIndex on the rows
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        time_col = self._find_datetime_col(df)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        if df[time_col].isna().any():
            LOGGER.warning(
                "Some timestamps failed to parse in column '%s'. Dropping NaT rows.",
                time_col,
            )
            df = df[df[time_col].notna()]

        df = df.sort_values(time_col)
        df = df.set_index(time_col)

        return df

    # -------------------------------------------------------------------------
    # Loading time series
    # -------------------------------------------------------------------------
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load hourly time series data for the given basin.

        Parameters
        ----------
        basin : str
            Basin (gauge) ID.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with feature columns and hourly frequency.
        """
        csv_path = Path(self.cfg.data_dir) / f"{basin}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Time series file not found for basin '{basin}' at {csv_path}"
            )

        # Let _ensure_datetime_index handle datetime parsing and index
        df = pd.read_csv(csv_path)
        df = self._ensure_datetime_index(df)

        # Try to enforce an hourly frequency whenever possible
        try:
            df = df.asfreq("h")
        except Exception:
            # Fall back to inferred frequency if available
            try:
                inferred = pd.infer_freq(df.index)
                if inferred is not None:
                    df = df.asfreq(inferred)
            except Exception:
                LOGGER.warning(
                    "Could not enforce a regular frequency for basin %s. Proceeding with "
                    "original index.",
                    basin,
                )

        # Drop duplicate timestamps, keeping the first occurrence
        if df.index.duplicated().any():
            LOGGER.warning(
                "%s: duplicate timestamps found. Dropping all but the first occurrence.",
                basin,
            )
            df = df[~df.index.duplicated(keep="first")]

        # Optionally add rolling features
        if getattr(self.cfg, "SH_addRollingFeatures", False):
            df = add_rolling_features(df)

        # ---------------------------------------------------------------------
        # Determine which dynamic inputs we logically require
        # ---------------------------------------------------------------------
        # dynamic_inputs in config can be:
        # - a flat list, or
        # - a dictionary of lists.
        if isinstance(self.cfg.dynamic_inputs, list):
            required_dyn = list(self.cfg.dynamic_inputs)
        else:
            required_dyn = [
                item
                for inputs in self.cfg.dynamic_inputs.values()
                for item in inputs
            ]

        # Names of rolling-derived features that we create in add_rolling_features()
        rolling_derived = [
            f"{base}_{window}"
            for base in ("prec_bruchji", "prec_fieschertal", "prec_binn", "prec_visp")
            for window in ("3h", "24h")
        ]
        rolling_derived += ["temp_binn_6h_mean", "degree_day_24h", "is_snow_bruchji"]

        # Missing dynamic inputs (excluding rolling-derived ones) should raise
        missing = [
            col
            for col in required_dyn
            if col not in df.columns and col not in rolling_derived
        ]
        if missing:
            raise KeyError(f"Missing required dynamic input(s) for basin {basin}: {missing}")

        # ---------------------------------------------------------------------
        # Collect all columns we want to keep
        # ---------------------------------------------------------------------
        if isinstance(self.cfg.dynamic_inputs, list):
            dynamic_cols_to_keep = getattr(
                self.cfg, "dynamic_inputs_flattened", self.cfg.dynamic_inputs
            )
        else:
            dynamic_cols_to_keep = [
                item
                for inputs in self.cfg.dynamic_inputs.values()
                for item in inputs
            ]

        keep_cols = (
            list(self.cfg.target_variables)
            + list(getattr(self.cfg, "evolving_attributes", []))
            + list(getattr(self.cfg, "mass_inputs", []))
            # autoregressive inputs are handled by BaseDataset
            + list(dynamic_cols_to_keep)
            + list(getattr(self.cfg, "dynamic_conceptual_inputs", []))
        )
        # Ensure uniqueness and stable ordering
        keep_cols = sorted(set(keep_cols))

        # Restrict DataFrame to columns used by the model
        df = df[keep_cols]

        # Final guard: ensure index really is time-like
        if not isinstance(
            df.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)
        ):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                raise ValueError(
                    f"Final DataFrame for basin {basin} does not have a datetime index"
                ) from exc

        # LOGGER.debug("[SwissHourly] %s final columns: %s", basin, list(df.columns))
        return df

    # -------------------------------------------------------------------------
    # Loading static attributes
    # -------------------------------------------------------------------------
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

        dfs: list[pd.DataFrame] = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(
                txt_file,
                sep=";",
                header=0,
                dtype={"gauge_id": str},
            )
            if "gauge_id" not in df_temp.columns:
                raise ValueError(
                    f"Attribute file {txt_file} must contain a 'gauge_id' column."
                )

            df_temp = df_temp.set_index("gauge_id")
            dfs.append(df_temp)

        # Merge on index; this aligns basins and stacks attributes by columns
        attr_df = pd.concat(dfs, axis=1)

        # Filter to basins in use (if `self.basins` is defined)
        if self.basins:
            missing = [b for b in self.basins if b not in attr_df.index]
            if missing:
                raise ValueError(f"Missing attributes for basins: {missing}")
            attr_df = attr_df.loc[self.basins]

        # Clean column names (e.g. trailing spaces)
        attr_df.columns = attr_df.columns.str.strip()

        return attr_df


# -------------------------------------------------------------------------
# Rolling feature computation (SwissHourly specific utility)
# -------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling/aggregated features specific to the SwissHourly dataset.

    Features added (if base columns exist):
    - For each precipitation column in:
        prec_bruchji, prec_fieschertal, prec_binn, prec_visp
      * <prec>_3h  : 3-hour rolling sum
      * <prec>_24h : 24-hour rolling sum

    - For `temp_binn` (if present):
      * temp_binn_6h_mean : 6-hour rolling mean
      * degree_day_24h    : 24-hour rolling sum of positive temperatures
      * is_snow_bruchji   : indicator (1 if temp <= 0, else 0)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a time index.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling/aggregated columns.
    """
    df = df.copy()

    # Precipitation aggregates
    for prec_col in ("prec_bruchji", "prec_fieschertal", "prec_binn", "prec_visp"):
        if prec_col in df.columns:
            df[f"{prec_col}_3h"] = df[prec_col].rolling(window=3, min_periods=1).sum()
            df[f"{prec_col}_24h"] = df[prec_col].rolling(window=24, min_periods=1).sum()

    # Temperature-derived features
    if "temp_binn" in df.columns:
        temp = df["temp_binn"]
        df["temp_binn_6h_mean"] = temp.rolling(window=6, min_periods=1).mean()
        df["degree_day_24h"] = temp.clip(lower=0).rolling(window=24, min_periods=1).sum()
        df["is_snow_bruchji"] = (temp <= 0).astype("int8")

    return df
