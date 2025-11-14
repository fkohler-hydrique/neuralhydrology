import logging
import pickle
import re
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from numba import NumbaPendingDeprecationWarning, njit, prange
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from ruamel.yaml import YAML
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError, NoTrainDataError
from neuralhydrology.utils import samplingutils

LOGGER = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset class to load and preprocess data.

    Subclasses implement dataset-specific logic for loading basin time series and
    static attributes (e.g., CamelsUS, CamelsGB, SwissHourly).

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        If True, dataset is used for training:
        - Means/stds are computed and stored.
        - One-hot encoding mapping is created (if enabled).
        If False, `scaler` (and `id_to_int` if basin ID encoding is used) must be provided.
    period : {'train', 'validation', 'test'}
        Which period to load.
    basin : str, optional
        If provided, only this basin is loaded. Otherwise, basins are read from
        the corresponding basin file for `period`.
    additional_features : list[dict[str, pd.DataFrame]], optional
        List of dicts mapping basin ID → DataFrame with additional dynamic columns.
        All columns become available as dynamic inputs / evolving attributes / targets.
    id_to_int : dict[str, int], optional
        Mapping basin ID → integer index for one-hot encoding.
        Required for validation/test if `use_basin_id_encoding` is True.
    scaler : dict[str, Union[pd.Series, xarray.Dataset]], optional
        Centering and scaling information for features. Required for validation/test.
    """

    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basin: str | None = None,
        additional_features: List[Dict[str, pd.DataFrame]] | None = None,
        id_to_int: Dict[str, int] | None = None,
        scaler: Dict[str, Union[pd.Series, xarray.Dataset]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train

        if period not in ["train", "validation", "test"]:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test'")
        self.period = period

        # For validation / test we require scaler (and id_to_int if encoding is used)
        if period in ["validation", "test"]:
            if not scaler:
                raise ValueError(
                    "During evaluation of validation or test period, a scaler dictionary must be passed."
                )
            if cfg.use_basin_id_encoding and not id_to_int:
                raise ValueError(
                    "For basin ID embedding, the id_to_int dictionary must be passed for validation/test."
                )

        # Optional timestep counters for hindcast/forecast sequences
        if self.cfg.timestep_counter:
            if not self.cfg.forecast_inputs_flattened:
                raise ValueError("Timestep counter only works for forecast data.")
            if cfg.forecast_overlap:
                overlap_zeros = torch.zeros((cfg.forecast_overlap, 1))
                forecast_counter = torch.arange(
                    1, cfg.forecast_seq_length - cfg.forecast_overlap + 1
                ).unsqueeze(-1)
                self.forecast_counter = torch.concatenate(
                    [overlap_zeros, forecast_counter], dim=0
                )
                self.hindcast_counter = torch.zeros(
                    (cfg.seq_length - cfg.forecast_seq_length + cfg.forecast_overlap, 1)
                )
            else:
                self.forecast_counter = torch.arange(
                    1, cfg.forecast_seq_length + 1
                ).unsqueeze(-1)
                self.hindcast_counter = torch.zeros(
                    (cfg.seq_length - cfg.forecast_seq_length, 1)
                )

        # Determine basins to load
        if basin is None:
            self.basins = utils.load_basin_file(getattr(cfg, f"{period}_basin_file"))
        else:
            self.basins = [basin]

        # Avoid mutable default arguments
        self.additional_features: list[dict[str, pd.DataFrame]] = additional_features or []
        self.id_to_int: dict[str, int] = id_to_int or {}
        self.scaler: dict[str, Union[pd.Series, xarray.Dataset]] = scaler or {}

        # Don't compute scale when fine-tuning with a pre-existing scaler
        self._compute_scaler = bool(is_train and not scaler)

        # Frequency-related configuration
        self.frequencies: list[str] = []
        self.seq_len: list[int] | None = None
        self._predict_last_n: list[int] | None = None
        self._initialize_frequency_configuration()

        # During training we log preprocessing with progress bars; not during validation/testing
        self._disable_pbar = cfg.verbose == 0 or not self.is_train

        # Initialize containers for loaded data
        self._x_d: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        self._x_s: dict[str, dict[str, torch.Tensor]] = {}
        self._attributes: dict[str, torch.Tensor] = {}
        self._y: dict[str, dict[str, torch.Tensor]] = {}
        self._per_basin_target_stds: dict[str, torch.Tensor] = {}
        self._dates: dict[str, dict[str, np.ndarray]] = {}
        self.start_and_end_dates: dict[str, dict[str, list[pd.Timestamp]]] = {}
        self.num_samples: int = 0
        self.period_starts: dict[str, pd.Timestamp] = {}

        # Get start/end date(s) per basin
        self._get_start_and_end_dates()

        # If additional feature files are specified in config, load them
        if not self.additional_features and cfg.additional_feature_files:
            self._load_additional_features()

        # Basin-ID encoding
        if cfg.use_basin_id_encoding and self.is_train:
            self._create_id_to_int()

        # Load/preprocess all data
        self._load_data()

        # Persist scaler to disk after training
        if self.is_train:
            self._dump_scaler()

    # --------------------------------------------------------------------- #
    # PyTorch Dataset interface
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, item: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | np.ndarray]:
        """Return a single training/evaluation sample for index `item`."""
        basin, indices = self.lookup_table[item]
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor] | np.ndarray] = {}

        for freq, seq_len, idx in zip(self.frequencies, self.seq_len, indices):
            # If there's just one frequency, don't use suffixes.
            freq_suffix = "" if len(self.frequencies) == 1 else f"_{freq}"

            # Slicing bounds (note: slice end is exclusive)
            hindcast_start_idx = idx + 1 - seq_len
            global_end_idx = idx + 1

            if self.cfg.forecast_seq_length:
                hindcast_end_idx = idx + 1 - self.cfg.forecast_seq_length
                forecast_start_idx = idx + 1 - self.cfg.forecast_seq_length
                if self.cfg.forecast_overlap and self.cfg.forecast_overlap > 0:
                    hindcast_end_idx += self.cfg.forecast_overlap
            else:
                hindcast_end_idx = None
                forecast_start_idx = None

            x_d_key = f"x_d{freq_suffix}"
            x_d_hindcast_key = f"{x_d_key}_hindcast"
            x_d_forecast_key = f"{x_d_key}_forecast"

            sample[x_d_key] = {}
            sample[x_d_hindcast_key] = {}
            sample[x_d_forecast_key] = {}

            # Split dynamic inputs into hindcast/forecast parts if configured
            for feature_name, values in self._x_d[basin][freq].items():
                if feature_name in self.cfg.hindcast_inputs_flattened:
                    sample[x_d_hindcast_key][feature_name] = values[
                        hindcast_start_idx:hindcast_end_idx
                    ]
                if feature_name in self.cfg.forecast_inputs_flattened:
                    sample[x_d_forecast_key][feature_name] = values[
                        forecast_start_idx:global_end_idx
                    ]
                if not self.cfg.hindcast_inputs_flattened:
                    # No split: the whole sequence is hindcast
                    sample[x_d_key][feature_name] = values[
                        hindcast_start_idx:global_end_idx
                    ]

            # Sample NaN streaks (data dropout) only during training
            if self.is_train and (
                self.cfg.nan_step_probability or self.cfg.nan_sequence_probability
            ):
                if self.cfg.hindcast_inputs_flattened:
                    sample[x_d_hindcast_key] = self._add_nan_streaks(
                        sample[x_d_hindcast_key], groups=self.cfg.hindcast_inputs
                    )
                    sample[x_d_forecast_key] = self._add_nan_streaks(
                        sample[x_d_forecast_key], groups=self.cfg.forecast_inputs
                    )
                else:
                    sample[x_d_key] = self._add_nan_streaks(
                        sample[x_d_key], groups=self.cfg.dynamic_inputs
                    )

            # Targets and dates for this frequency
            y_key = f"y{freq_suffix}"
            date_key = f"date{freq_suffix}"
            sample[y_key] = self._y[basin][freq][
                hindcast_start_idx:global_end_idx
            ]
            sample[date_key] = self._dates[basin][freq][
                hindcast_start_idx:global_end_idx
            ]

            # Static inputs: dataset-specific static attrs + evolving attrs (x_s)
            static_inputs: list[torch.Tensor] = []
            if self._attributes:
                static_inputs.append(self._attributes[basin])
            if self._x_s:
                static_inputs.append(self._x_s[basin][freq][idx])
            if static_inputs:
                sample[f"x_s{freq_suffix}"] = torch.cat(static_inputs, dim=-1)

            # Optional timestep counters
            if self.cfg.timestep_counter:
                sample[x_d_hindcast_key]["hindcast_counter"] = self.hindcast_counter
                sample[x_d_forecast_key]["forecast_counter"] = self.forecast_counter

        # Per-basin target standard deviations for NSE / weighted NSE losses
        if self._per_basin_target_stds:
            sample["per_basin_target_stds"] = self._per_basin_target_stds[basin]

        # Optional basin one-hot encoding
        if self.id_to_int:
            sample["x_one_hot"] = torch.nn.functional.one_hot(
                torch.tensor(self.id_to_int[basin]),
                num_classes=len(self.id_to_int),
            ).to(torch.float32)

        return sample

    # --------------------------------------------------------------------- #
    # Utility methods used in __getitem__
    # --------------------------------------------------------------------- #
    def _add_nan_streaks(
        self, x_d: dict[str, torch.Tensor], groups: list[list[str]]
    ) -> dict[str, torch.Tensor]:
        """Randomly mask streaks of dynamic inputs with NaNs for robustness.

        Parameters
        ----------
        x_d : dict[str, torch.Tensor]
            Dynamic input features (seq_len, 1) per feature.
        groups : list[list[str]]
            Groups of feature names; dropout is applied group-wise.

        Returns
        -------
        dict[str, torch.Tensor]
            Updated `x_d` with NaNs inserted.
        """
        if not groups or not isinstance(groups[0], list):
            raise ValueError("For dropout streaks, dynamic_inputs must be a list of lists.")

        seq_length = x_d[groups[0][0]].shape[0]
        drop_masks = np.zeros((len(groups), seq_length, 1), dtype=bool)

        # Decide which groups have full-sequence dropout
        drop_sequences = np.random.choice(
            [True, False],
            p=[self.cfg.nan_sequence_probability, 1 - self.cfg.nan_sequence_probability],
            size=len(groups),
        )
        if drop_sequences.all():
            # Don't allow all sequences to be completely dropped
            drop_sequences[np.random.choice(len(groups))] = False

        # Per-step dropout within each group
        for i in range(len(groups)):
            drop_steps = np.random.choice(
                [True, False],
                p=[self.cfg.nan_step_probability, 1 - self.cfg.nan_step_probability],
                size=(seq_length, 1),
            )
            drop_masks[i] = drop_sequences[i] | drop_steps

        drop_masks_t = torch.from_numpy(drop_masks)
        for i, group in enumerate(groups):
            for feature in group:
                x_d[feature] = torch.where(drop_masks_t[i], torch.nan, x_d[feature])

        return x_d

    # --------------------------------------------------------------------- #
    # Abstract hooks for subclasses
    # --------------------------------------------------------------------- #
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Return time-indexed DataFrame with all required columns for a basin."""
        raise NotImplementedError

    def _load_attributes(self) -> pd.DataFrame:
        """Return basin-indexed DataFrame with static attributes."""
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # ID encoding & scaler dumping
    # --------------------------------------------------------------------- #
    def _create_id_to_int(self) -> None:
        """Create random basin → integer mapping for one-hot encoding and dump to disk."""
        self.id_to_int = {
            str(basin_id): idx
            for idx, basin_id in enumerate(np.random.permutation(self.basins))
        }

        file_path = self.cfg.train_dir / "id_to_int.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(self.id_to_int, fp)

    def _dump_scaler(self) -> None:
        """Store scaler dictionary to disk for inference/validation."""
        scaler_dict: dict[str, dict] = defaultdict(dict)
        for key, value in self.scaler.items():
            if isinstance(value, pd.Series) or isinstance(value, xarray.Dataset):
                scaler_dict[key] = value.to_dict()
            else:
                raise RuntimeError(
                    f"Unknown datatype for scaler '{key}'. "
                    "Supported types are pandas.Series and xarray.Dataset."
                )

        file_path = self.cfg.train_dir / "train_data_scaler.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(dict(scaler_dict), fp)

    # --------------------------------------------------------------------- #
    # Period handling
    # --------------------------------------------------------------------- #
    def _get_start_and_end_dates(self) -> None:
        """Populate `self.start_and_end_dates` for all basins."""
        # Single periods from config (same for all basins)
        if getattr(self.cfg, f"per_basin_{self.period}_periods_file") is None:
            # Ensure we always have lists for iteration
            start_cfg = getattr(self.cfg, f"{self.period}_start_date")
            end_cfg = getattr(self.cfg, f"{self.period}_end_date")

            start_dates = start_cfg if isinstance(start_cfg, list) else [start_cfg]
            end_dates = end_cfg if isinstance(end_cfg, list) else [end_cfg]

            if self.period != "train" and len(start_dates) > 1:
                raise ValueError("Evaluation on split periods is currently not supported.")

            self.start_and_end_dates = {
                b: {"start_dates": start_dates, "end_dates": end_dates} for b in self.basins
            }
        # Per-basin periods from file
        else:
            with open(
                getattr(self.cfg, f"per_basin_{self.period}_periods_file"), "rb"
            ) as fp:
                self.start_and_end_dates = pickle.load(fp)

    def _load_additional_features(self) -> None:
        """Load additional feature pickles specified in the config."""
        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    # --------------------------------------------------------------------- #
    # DataFrame preprocessing helpers
    # --------------------------------------------------------------------- #
    def _duplicate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Duplicate selected features as additional columns."""
        for feature, n_duplicates in self.cfg.duplicate_features.items():
            for n in range(1, n_duplicates + 1):
                df[f"{feature}_copy{n}"] = df[feature]
        return df

    def _add_missing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure target columns exist (filled with NaN if missing)."""
        for var in self.cfg.target_variables:
            if var not in df.columns:
                df[var] = np.nan
        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create shifted copies of features as specified in cfg.lagged_features."""
        # Check that all autoregressive inputs are contained in the list of shifted variables
        self._check_autoregressive_inputs()

        for feature, shift in self.cfg.lagged_features.items():
            if isinstance(shift, list):
                # Only consider unique shift values to avoid duplicate column names
                for s in set(shift):
                    df[f"{feature}_shift{s}"] = df[feature].shift(periods=s, freq="infer")
            elif isinstance(shift, int):
                df[f"{feature}_shift{shift}"] = df[feature].shift(periods=shift, freq="infer")
            else:
                raise ValueError(
                    "The value of 'lagged_features' must be an int or a list of ints."
                )
        return df

    def _check_autoregressive_inputs(self) -> None:
        """Ensure that autoregressive inputs correspond to valid lagged features."""
        for input_name in self.cfg.autoregressive_inputs:
            capture = re.compile(r"^(.*)_shift(\d+)$").search(input_name)
            if not capture:
                raise ValueError(
                    "Autoregressive inputs must be shifted variables with form "
                    "<variable>_shift<lag> where <lag> is an integer. "
                    f"Instead got: {input_name}."
                )
            base_var, lag_str = capture.groups()
            lag_int = int(lag_str)
            if base_var not in self.cfg.lagged_features or lag_int not in self.cfg.lagged_features[base_var]:
                raise ValueError(
                    "Autoregressive inputs must be present in 'lagged_features'. "
                    f"Missing ({base_var}, shift={lag_int})."
                )

    # --------------------------------------------------------------------- #
    # Xarray dataset creation / loading
    # --------------------------------------------------------------------- #
    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        """Create or load xarray.Dataset of all basins and time slices.

        Returns
        -------
        xarray.Dataset
            Dataset with coords: date, basin, and all dynamic/static variables.
        """
        # If no netCDF/pickled dataset is passed, or if not in training mode, create from raw basin files.
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list: list[xarray.Dataset] = []

            # List of columns to keep; everything else is dropped to reduce memory.
            keep_cols: list[str] = (
                self.cfg.target_variables
                + self.cfg.evolving_attributes
                + self.cfg.mass_inputs
                + self.cfg.autoregressive_inputs
            )

            # Dynamic inputs (single or multi-frequency)
            if isinstance(self.cfg.dynamic_inputs, list):
                keep_cols += self.cfg.dynamic_inputs_flattened
            else:
                keep_cols += [
                    col for inputs in self.cfg.dynamic_inputs.values() for col in inputs
                ]

            # Dynamic conceptual inputs
            keep_cols += self.cfg.dynamic_conceptual_inputs

            # Remove duplicates and sort for stable order
            keep_cols = sorted(set(keep_cols))

            if not self._disable_pbar:
                LOGGER.info("Loading basin data into xarray dataset.")

            for basin in tqdm(
                self.basins, file=sys.stdout, disable=self._disable_pbar, leave=False
            ):
                df = self._load_basin_data(basin)

                # Optional target normalization (currently assumes 'streamflow' config)
                if self.cfg.target_normalization is not None:
                    for target in self.cfg.target_variables:
                        specs = self.cfg.target_normalization.get("streamflow", {})
                        if specs.get("transform") == "log":
                            log_offset = specs.get("log_offset", 0.0)
                            df[target] = np.log(df[target] + log_offset)

                # Add additional dynamic features from external DataFrames
                if self.additional_features:
                    df = pd.concat(
                        [df, *[feat_dict[basin] for feat_dict in self.additional_features]],
                        axis=1,
                    )

                # During evaluation, ensure missing targets exist (NaN)
                if not self.is_train:
                    df = self._add_missing_targets(df)

                # Duplicate and lagged features as configured
                df = self._duplicate_features(df)
                df = self._add_lagged_features(df)

                # Keep only columns needed by the model
                try:
                    df = df[keep_cols]
                except KeyError:
                    missing = [c for c in keep_cols if c not in df.columns]
                    msg = [
                        f"The following features are not available in the data: {missing}. ",
                        f"Available features: {df.columns.tolist()}",
                    ]
                    raise KeyError("".join(msg))

                # Randomly hold out parts of specified dynamic features
                for holdout_variable, holdout_dict in self.cfg.random_holdout_from_dynamic_features.items():
                    df[holdout_variable] = samplingutils.bernoulli_subseries_sampler(
                        data=df[holdout_variable].values,
                        missing_fraction=holdout_dict["missing_fraction"],
                        mean_missing_length=holdout_dict["mean_missing_length"],
                    )

                # Enlarge end dates to last second of day to include all hours (not just 00:00)
                start_dates = self.start_and_end_dates[basin]["start_dates"]
                end_dates = [
                    date + pd.Timedelta(days=1, seconds=-1)
                    for date in self.start_and_end_dates[basin]["end_dates"]
                ]

                native_frequency = utils.infer_frequency(df.index)
                if not self.frequencies:
                    # Use df's native resolution by default
                    self.frequencies = [native_frequency]

                # Ensure used frequencies are not higher than native frequency
                try:
                    freq_vs_native = [
                        utils.compare_frequencies(freq, native_frequency)
                        for freq in self.frequencies
                    ]
                except ValueError:
                    LOGGER.warning(
                        "Cannot compare provided frequencies with native frequency. "
                        "Make sure frequencies are not higher than the native frequency."
                    )
                    freq_vs_native = []

                if any(comparison > 1 for comparison in freq_vs_native):
                    raise ValueError(
                        f"Frequency is higher than native data frequency {native_frequency}."
                    )

                # Warmup offsets (per frequency), computed in terms of timedelta
                offsets = [
                    (self.seq_len[i] - self._predict_last_n[i]) * to_offset(freq)
                    for i, freq in enumerate(self.frequencies)
                ]

                basin_data_list: list[pd.DataFrame] = []
                # Create per-period slices with warmup
                for start_date, end_date in zip(start_dates, end_dates):
                    # Start date must align with all frequencies
                    misaligned = [
                        freq
                        for freq in self.frequencies
                        if not to_offset(freq).is_on_offset(start_date)
                    ]
                    if misaligned:
                        raise ValueError(
                            f"Start date {start_date} is not aligned with frequencies {misaligned}."
                        )

                    # Warmup starts at earliest (largest) offset across frequencies
                    warmup_start_date = min(start_date - offset for offset in offsets)
                    df_sub = df[warmup_start_date:end_date]

                    # Ensure df_sub covers full date range; fill gaps with NaNs
                    full_range = pd.date_range(
                        start=warmup_start_date, end=end_date, freq=native_frequency
                    )
                    df_sub = df_sub.reindex(
                        pd.DatetimeIndex(full_range, name=df_sub.index.name)
                    )

                    # Set targets before the period start to NaN
                    df_sub.loc[df_sub.index < start_date, self.cfg.target_variables] = np.nan

                    basin_data_list.append(df_sub)

                if not basin_data_list:
                    # Skip basin if no period defined
                    continue

                # Stack all time slices into a single time series
                df = pd.concat(basin_data_list, axis=0)

                # Handle duplicate timestamps due to overlapping slices:
                # - Keep non-duplicated entries as is.
                # - For duplicates, prefer rows with non-NaN targets.
                df_non_duplicated = df[~df.index.duplicated(keep=False)]
                df_duplicated = df[df.index.duplicated(keep=False)]

                filtered_duplicates: list[pd.DataFrame] = []
                for _, grp in df_duplicated.groupby("date"):
                    mask = ~grp[self.cfg.target_variables].isna().any(axis=1)
                    if not mask.any():
                        # All duplicates have NaN targets; keep the first row
                        filtered_duplicates.append(grp.head(1))
                    else:
                        # Keep the first row with non-NaN targets
                        filtered_duplicates.append(grp[mask].head(1))

                if filtered_duplicates:
                    df_filtered_duplicates = pd.concat(filtered_duplicates, axis=0)
                    df = pd.concat([df_non_duplicated, df_filtered_duplicates], axis=0)
                else:
                    df = df_non_duplicated

                # Sort and reindex to full continuous date range
                df = df.sort_index(axis=0, ascending=True)
                df = df.reindex(
                    pd.DatetimeIndex(
                        data=pd.date_range(df.index[0], df.index[-1], freq=native_frequency),
                        name=df.index.name,
                    )
                )

                # Convert to xarray Dataset and attach basin coordinate
                ds_basin = xarray.Dataset.from_dataframe(df.astype(np.float32))
                ds_basin = ds_basin.assign_coords({"basin": basin})
                data_list.append(ds_basin)

            if not data_list:
                # No valid time slices for any basin
                if self.is_train:
                    raise NoTrainDataError
                raise NoEvaluationDataError

            # Concatenate all basins into one dataset along basin dimension
            ds = xarray.concat(data_list, dim="basin")

            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(ds)

        else:
            # Load precomputed train dataset from disk
            with self.cfg.train_data_file.open("rb") as fp:
                d = pickle.load(fp)
            ds = xarray.Dataset.from_dict(d)
            if not self.frequencies:
                native_frequency = utils.infer_frequency(ds["date"].values)
                self.frequencies = [native_frequency]

        return ds

    def _save_xarray_dataset(self, ds: xarray.Dataset) -> None:
        """Store newly created train dataset to disk (pickled dict)."""
        file_path = self.cfg.train_dir / "train_data.p"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # netCDF has issues with "/" in var names; store as dict + pickle instead
        with file_path.open("wb") as fp:
            pickle.dump(ds.to_dict(), fp)

    # --------------------------------------------------------------------- #
    # Per-basin statistics & sample lookup table
    # --------------------------------------------------------------------- #
    def _calculate_per_basin_std(self, ds: xarray.Dataset) -> None:
        """Compute per-basin standard deviation of target variables."""
        if not self._disable_pbar:
            LOGGER.info("Calculating target variable stds per basin")

        nan_basins: list[str] = []
        for basin in tqdm(self.basins, file=sys.stdout, disable=self._disable_pbar, leave=False):
            obs = ds.sel(basin=basin)[self.cfg.target_variables].to_array().values
            if np.sum(~np.isnan(obs)) > 1:
                per_basin_target_stds = torch.tensor(
                    np.expand_dims(np.nanstd(obs, axis=1), 0), dtype=torch.float32
                )
            else:
                nan_basins.append(basin)
                per_basin_target_stds = torch.full(
                    (1, obs.shape[0]), np.nan, dtype=torch.float32
                )
            self._per_basin_target_stds[basin] = per_basin_target_stds

        if nan_basins:
            LOGGER.warning(
                "The following basins had not enough valid target values to calculate a standard deviation: "
                f"{', '.join(nan_basins)}. NSE loss values for these basins will be NaN."
            )

    def _create_lookup_table(self, ds: xarray.Dataset) -> None:
        """Create a lookup table mapping dataset index → (basin, indices_per_frequency)."""
        lookup: list[tuple[str, list[int]]] = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to PyTorch tensor")

        basins_without_samples: list[str] = []
        basin_coordinates = ds["basin"].values.tolist()

        for basin in tqdm(
            basin_coordinates, file=sys.stdout, disable=self._disable_pbar, leave=False
        ):
            # x_d: per frequency, dict[feature_name] -> np.ndarray (time, 1)
            x_d: dict[str, dict[str, np.ndarray]] = {}
            x_s: dict[str, np.ndarray] = {}
            y: dict[str, np.ndarray] = {}
            dates: dict[str, np.ndarray] = {}

            # Keys: frequencies; values: mapping lowest-frequency sample index → index in this frequency
            frequency_maps: dict[str, np.ndarray] = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]

            # Converting from xarray to pandas is faster for resampling
            df_native = ds.sel(basin=basin).to_dataframe()

            for freq in self.frequencies:
                # Dynamic columns for this frequency; mass inputs first
                if isinstance(self.cfg.dynamic_inputs, list):
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs_flattened
                else:
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs[freq]

                dynamic_cols += self.cfg.dynamic_conceptual_inputs

                df_resampled = df_native[
                    dynamic_cols
                    + self.cfg.target_variables
                    + self.cfg.evolving_attributes
                    + self.cfg.autoregressive_inputs
                ].resample(freq).mean()

                # Dynamic inputs (per feature)
                x_d[freq] = {col: df_resampled[[col]].values for col in dynamic_cols}
                # Targets
                y[freq] = df_resampled[self.cfg.target_variables].values
                # Evolving static-like inputs
                if self.cfg.evolving_attributes:
                    x_s[freq] = df_resampled[self.cfg.evolving_attributes].values

                # Dates
                dates[freq] = df_resampled.index.to_numpy()

                # Number of frequency steps in one lowest-frequency step
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                if len(df_resampled) % frequency_factor != 0:
                    raise ValueError(
                        f"The length of the dataframe at frequency {freq} is {len(df_resampled)} "
                        f"(including warmup), which is not a multiple of {frequency_factor} "
                        f"(factor between lowest frequency {lowest_freq} and {freq}). "
                        f"Adjust the {self.period} start/end dates so that the period "
                        f"(including warmup) length is divisible by {frequency_factor}."
                    )
                frequency_maps[freq] = (
                    np.arange(len(df_resampled) // frequency_factor) * frequency_factor
                    + (frequency_factor - 1)
                )

            # Store first date for period to be able to restore dates during inference
            if not self.is_train:
                self.period_starts[basin] = pd.to_datetime(
                    ds.sel(basin=basin)["date"].values[0]
                )

            # Validate samples using numba-accelerated function
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

                if self.is_train:
                    # Concatenate features along last dimension for validation
                    x_d_validate = [
                        np.concatenate(list(x_d[freq].values()), axis=-1)
                        for freq in self.frequencies
                    ]
                    x_s_validate = (
                        [x_s[freq] for freq in self.frequencies] if x_s else None
                    )
                    y_validate = [y[freq] for freq in self.frequencies]
                else:
                    # During inference, we accept samples with NaNs in inputs
                    x_d_validate = None
                    x_s_validate = None
                    y_validate = None

                flag = _validate_samples(
                    x_d=x_d_validate,
                    x_s=x_s_validate,
                    y=y_validate,
                    frequency_maps=[frequency_maps[freq] for freq in self.frequencies],
                    seq_length=self.seq_len,
                    predict_last_n=self._predict_last_n,
                )

            # Concatenate autoregressive columns to dynamic inputs *after* validation,
            # so as not to remove samples with missing AR inputs.
            if self.cfg.autoregressive_inputs:
                if len(self.frequencies) > 1:
                    raise ValueError(
                        "Autoregressive inputs are not supported for datasets with multiple frequencies."
                    )
                freq = self.frequencies[0]
                x_d[freq].update(
                    {col: df_resampled[[col]].values for col in self.cfg.autoregressive_inputs}
                )

            valid_samples = np.argwhere(flag == 1)

            for idx in valid_samples:
                # Store pointer to basin and this sample's index for each frequency
                lookup.append(
                    (
                        basin,
                        [frequency_maps[freq][int(idx)] for freq in self.frequencies],
                    )
                )

            # Only store basin data if there is at least one valid sample
            if valid_samples.size > 0:
                if self.cfg.forecast_inputs_flattened and not self.cfg.hindcast_inputs_flattened:
                    raise ValueError(
                        "Hindcast inputs must be provided if forecast inputs are provided."
                    )

                self._x_d[basin] = {
                    freq: {
                        feature_name: torch.from_numpy(values.astype(np.float32))
                        for feature_name, values in freq_dict.items()
                    }
                    for freq, freq_dict in x_d.items()
                }
                self._y[basin] = {
                    freq: torch.from_numpy(vals.astype(np.float32))
                    for freq, vals in y.items()
                }
                if x_s:
                    self._x_s[basin] = {
                        freq: torch.from_numpy(vals.astype(np.float32))
                        for freq, vals in x_s.items()
                    }
                self._dates[basin] = dates
            else:
                basins_without_samples.append(basin)

        if basins_without_samples:
            LOGGER.info(
                "These basins do not have a single valid sample in the %s period: %s",
                self.period,
                basins_without_samples,
            )

        # Map integer index → (basin, indices_per_frequency)
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError

    # --------------------------------------------------------------------- #
    # Static attributes
    # --------------------------------------------------------------------- #
    def _load_hydroatlas_attributes(self) -> pd.DataFrame:
        """Load HydroATLAS attributes for all basins."""
        df = utils.load_hydroatlas_attributes(self.cfg.data_dir, basins=self.basins)

        # Remove attributes not specified in the config
        drop_cols = [c for c in df.columns if c not in self.cfg.hydroatlas_attributes]
        df = df.drop(drop_cols, axis=1)

        if self.is_train:
            utils.attributes_sanity_check(df=df)

        return df

    def _load_combined_attributes(self) -> None:
        """Load dataset-specific and HydroATLAS attributes and combine them."""
        dfs: list[pd.DataFrame] = []

        # Dataset-specific attributes
        if self.cfg.static_attributes:
            df = self._load_attributes()

            missing_attrs = [
                attr for attr in self.cfg.static_attributes if attr not in df.columns
            ]
            if missing_attrs:
                raise ValueError(f"Static attributes {missing_attrs} are missing.")

            df = df[self.cfg.static_attributes]

            if self._compute_scaler:
                utils.attributes_sanity_check(df=df)

            dfs.append(df)

        # HydroATLAS attributes
        if self.cfg.hydroatlas_attributes:
            dfs.append(self._load_hydroatlas_attributes())

        if not dfs:
            return

        df_combined = pd.concat(dfs, axis=1)

        # Ensure all configured attributes are available
        combined_attr_names = self.cfg.static_attributes + self.cfg.hydroatlas_attributes
        missing_columns = [
            attr for attr in combined_attr_names if attr not in df_combined.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following attributes are not available in the dataset: {missing_columns}"
            )

        # Sort columns alphabetically
        df_combined = df_combined.sort_index(axis=1)

        # Compute / apply normalization
        if self._compute_scaler:
            self.scaler["attribute_means"] = df_combined.mean()
            self.scaler["attribute_stds"] = df_combined.std()

        if any(key.startswith("camels_attr") for key in self.scaler.keys()):
            LOGGER.warning(
                "Deprecation warning: using old scaler files ('camels_attr_*') "
                "won't be supported in a future release."
            )
            df_combined = (
                df_combined - self.scaler["camels_attr_means"]
            ) / self.scaler["camels_attr_stds"]
        else:
            df_combined = (
                df_combined - self.scaler["attribute_means"]
            ) / self.scaler["attribute_stds"]

        # Store per-basin attributes as float32 tensors
        for basin in self.basins:
            attrs = df_combined.loc[basin].values.astype(np.float32)
            self._attributes[basin] = torch.from_numpy(attrs)

    # --------------------------------------------------------------------- #
    # High-level load + normalization
    # --------------------------------------------------------------------- #
    def _load_data(self) -> None:
        """Load everything: attributes, xarray dataset, normalization, lookup table."""
        # Load attributes first to sanity-check them
        self._load_combined_attributes()

        ds = self._load_or_create_xarray_dataset()

        # Compute per-basin std used in NSE/weightedNSE/custom losses
        self._calculate_per_basin_std(ds)

        # Compute feature-wise center/scale if needed
        if self._compute_scaler:
            self._setup_normalization(ds)

        # Normalize all features
        ds = (ds - self.scaler["xarray_feature_center"]) / (
            self.scaler["xarray_feature_scale"] + 1e-9
        )

        # Build lookup_table and per-basin tensors
        self._create_lookup_table(ds)

    def _setup_normalization(self, ds: xarray.Dataset) -> None:
        """Set up feature-wise centering and scaling for normalization."""
        # Default: mean/std
        self.scaler["xarray_feature_scale"] = ds.std(skipna=True)
        self.scaler["xarray_feature_center"] = ds.mean(skipna=True)

        # Feature-wise custom normalization
        for feature, feature_specs in self.cfg.custom_normalization.items():
            for key, val in feature_specs.items():
                # Centering
                if key == "centering":
                    if val is None or str(val).lower() == "none":
                        self.scaler["xarray_feature_center"][feature] = np.float32(0.0)
                    elif str(val).lower() == "median":
                        self.scaler["xarray_feature_center"][feature] = ds[
                            feature
                        ].median(skipna=True)
                    elif str(val).lower() == "min":
                        self.scaler["xarray_feature_center"][feature] = ds[
                            feature
                        ].min(skipna=True)
                    elif str(val).lower() == "mean":
                        # Default; nothing to do
                        pass
                    else:
                        raise ValueError(f"Unknown centering method {val} for feature {feature}")
                # Scaling
                elif key == "scaling":
                    if val is None or str(val).lower() == "none":
                        self.scaler["xarray_feature_scale"][feature] = np.float32(1.0)
                    elif val == "minmax":
                        self.scaler["xarray_feature_scale"][feature] = (
                            ds[feature].max(skipna=True)
                            - (ds[feature].min(skipna=True) + 1e-9)
                        )
                    elif val == "std":
                        # Default; nothing to do
                        pass
                    else:
                        raise ValueError(f"Unknown scaling method {val} for feature {feature}")
                else:
                    raise ValueError(
                        "Unknown dict key in custom_normalization. "
                        "Use 'centering' and/or 'scaling' for each feature."
                    )

    # --------------------------------------------------------------------- #
    # Misc API helpers
    # --------------------------------------------------------------------- #
    def get_period_start(self, basin: str) -> pd.Timestamp:
        """Return the first date in the period for a given basin."""
        return self.period_starts[basin]

    def _initialize_frequency_configuration(self) -> None:
        """Check and extract configuration for frequencies, seq_length, predict_last_n."""
        self.frequencies = self.cfg.use_frequencies
        self.seq_len = self.cfg.seq_length
        self._predict_last_n = self.cfg.predict_last_n

        if not self.frequencies:
            # No multi-frequency: seq_length and predict_last_n must be ints
            if not isinstance(self.seq_len, int) or not isinstance(self._predict_last_n, int):
                raise ValueError(
                    "seq_length and predict_last_n must be integers if use_frequencies is not provided."
                )
            self.seq_len = [self.seq_len]
            self._predict_last_n = [self._predict_last_n]
        else:
            # Multi-frequency: both configs must be dicts keyed by frequency
            if (
                not isinstance(self.seq_len, dict)
                or not isinstance(self._predict_last_n, dict)
                or any(freq not in self.seq_len for freq in self.frequencies)
                or any(freq not in self._predict_last_n for freq in self.frequencies)
            ):
                raise ValueError(
                    "seq_length and predict_last_n must be dictionaries with one key per frequency."
                )
            self.seq_len = [self.seq_len[freq] for freq in self.frequencies]
            self._predict_last_n = [self._predict_last_n[freq] for freq in self.frequencies]

    @staticmethod
    def collate_fn(
        samples: List[
            Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]
        ]
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]:
        """Custom collate_fn for PyTorch DataLoader.

        - Dates are stacked as numpy arrays.
        - Dynamic inputs ('x_d*') are dicts stacked per feature.
        - Everything else is stacked as torch tensors.
        """
        batch: dict[str, Union[torch.Tensor, np.ndarray, dict[str, torch.Tensor]]] = {}

        if not samples:
            return batch

        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith("date"):
                batch[feature] = np.stack([sample[feature] for sample in samples], axis=0)
            elif feature.startswith("x_d"):
                batch[feature] = {
                    k: torch.stack([sample[feature][k] for sample in samples], dim=0)
                    for k in samples[0][feature]
                }
            else:
                batch[feature] = torch.stack([sample[feature] for sample in samples], dim=0)

        return batch


# ------------------------------------------------------------------------- #
# Numba-accelerated sample validation
# ------------------------------------------------------------------------- #
@njit()
def _validate_samples(
    x_d: List[np.ndarray],
    x_s: List[np.ndarray],
    y: List[np.ndarray],
    seq_length: List[int],
    predict_last_n: List[int],
    frequency_maps: List[np.ndarray],
) -> np.ndarray:
    """Check for invalid samples due to NaN or insufficient sequence length.

    Parameters
    ----------
    x_d : List[np.ndarray]
        List of dynamic input arrays; one entry per frequency (time, features).
    x_s : List[np.ndarray]
        List of static/slowly varying input arrays; one entry per frequency.
    y : List[np.ndarray]
        List of target arrays; one entry per frequency.
    seq_length : List[int]
        Sequence lengths per frequency.
    predict_last_n : List[int]
        Number of last steps used for prediction per frequency.
    frequency_maps : List[np.ndarray]
        For each frequency, maps lowest-frequency sample index → index in that frequency.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) with 1 for valid and 0 for invalid samples.
    """
    # Number of samples is number of lowest-frequency samples
    n_samples = len(frequency_maps[0])

    # 1 = valid sample, 0 = invalid sample
    flag = np.ones(n_samples)

    for i in range(len(frequency_maps)):  # frequencies
        for j in prange(n_samples):  # lowest-frequency samples
            last_sample_of_freq = frequency_maps[i][j]

            # Too early in the series: insufficient history for this frequency
            if last_sample_of_freq < seq_length[i] - 1:
                flag[j] = 0
                continue

            # Any NaN in dynamic inputs makes sample invalid
            if x_d is not None:
                _x_d = x_d[i][
                    last_sample_of_freq - seq_length[i] + 1 : last_sample_of_freq + 1
                ]
                if np.any(np.isnan(_x_d)):
                    flag[j] = 0
                    continue

            # All-NaN targets in prediction window make sample invalid
            if y is not None:
                _y = y[i][
                    last_sample_of_freq - predict_last_n[i] + 1 : last_sample_of_freq + 1
                ]
                if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                    flag[j] = 0
                    continue

            # Any NaN in static features makes sample invalid
            if x_s is not None:
                _x_s = x_s[i][last_sample_of_freq]
                if np.any(np.isnan(_x_s)):
                    flag[j] = 0

    return flag
