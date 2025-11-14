import logging
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.utils import (
    get_frequency_factor,
    load_basin_file,
    load_scaler,
    sort_frequencies,
)
from neuralhydrology.evaluation import plots
from neuralhydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralhydrology.evaluation.utils import load_basin_id_encoding, metrics_to_dataframe
from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.training import get_loss_obj, get_regularization_obj
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import AllNaNError, NoEvaluationDataError

LOGGER = logging.getLogger(__name__)


class BaseTester:
    """Base class to run inference on a model.

    Use subclasses of this class to evaluate a trained model on its train, test, or validation period.
    For regression settings, :class:`RegressionTester` is used; for uncertainty prediction,
    :class:`UncertaintyTester` is used.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}, optional
        The period to evaluate, by default 'test'.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in
        ``run_dir``.
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = "test",
        init_model: bool = True,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.init_model = init_model

        if period not in ["train", "validation", "test"]:
            raise ValueError(
                f'Invalid period {period}. Must be one of ["train", "validation", "test"]'
            )
        self.period = period

        # determine device
        self._set_device()

        if self.init_model:
            self.model = get_model(cfg).to(self.device)

        # disable progress bars if verbose == 0
        self._disable_pbar = cfg.verbose == 0

        # will be initialized in _load_run_data
        self.basins: List[str] | None = None
        self.scaler: Optional[dict] = None
        self.id_to_int: Dict[str, int] = {}
        self.additional_features: List[Dict[str, pd.DataFrame]] = []

        # cached validation datasets (per basin)
        self.cached_datasets: Dict[str, BaseDataset] = {}

        # initialize loss object to compute the loss of the evaluation data
        self.loss_obj = get_loss_obj(cfg)
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

        self._load_run_data()

    # --------------------------------------------------------------------- #
    # Device & run data loading
    # --------------------------------------------------------------------- #
    def _set_device(self) -> None:
        """Determine computation device based on config and availability."""
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                # Allow 'cuda' or 'cuda:N'
                try:
                    gpu_id = int(self.cfg.device.split(":")[-1])
                except ValueError:
                    gpu_id = 0

                if gpu_id >= torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id}.")
                self.device = torch.device(f"cuda:{gpu_id}")
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS device is not available.")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def _load_run_data(self) -> None:
        """Load run-specific data from the run directory.

        - Basin list
        - Feature scaler
        - Basin ID encoding (if used)
        - Additional feature pickles
        """
        # get list of basins
        self.basins = load_basin_file(getattr(self.cfg, f"{self.period}_basin_file"))

        # load feature scaler
        self.scaler = load_scaler(self.run_dir)

        # check for old scaler files; rename keys if needed
        if "xarray_means" in self.scaler:
            self.scaler["xarray_feature_center"] = self.scaler.pop("xarray_means")
        if "xarray_stds" in self.scaler:
            self.scaler["xarray_feature_scale"] = self.scaler.pop("xarray_stds")

        # load basin_id to integer dictionary for one-hot-encoding
        if self.cfg.use_basin_id_encoding:
            self.id_to_int = load_basin_id_encoding(self.run_dir)

        # load additional features if specified
        for file in getattr(self.cfg, "additional_feature_files", []):
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    # --------------------------------------------------------------------- #
    # Weight file handling
    # --------------------------------------------------------------------- #
    def _get_weight_file(self, epoch: Optional[int]) -> Path:
        """Return file path to a weight file.

        If `epoch` is None, returns the last available epoch file.
        """
        if epoch is None:
            candidates = sorted(self.run_dir.glob("model_epoch*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No model_epoch*.pt files found in run directory {self.run_dir}"
                )
            weight_file = candidates[-1]
        else:
            weight_file = self.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

        return weight_file

    def _get_weight_file_from_best(self) -> Path:
        """Return the path to the 'best_model.pt' weights file."""
        weight_file = self.run_dir / "best_model.pt"
        if not weight_file.exists():
            raise FileNotFoundError(f"Best model file not found at {weight_file}")
        return weight_file

    def _load_weights(
        self, epoch: Optional[int] = None, from_best: bool = False
    ) -> None:
        """Load weights of a certain (or the last) epoch into the model."""
        if from_best:
            weight_file = self._get_weight_file_from_best()
        else:
            weight_file = self._get_weight_file(epoch)

        LOGGER.info("Using the model weights from %s", weight_file)
        self.model.load_state_dict(torch.load(weight_file, map_location=self.device))

    # --------------------------------------------------------------------- #
    # Dataset helper
    # --------------------------------------------------------------------- #
    def _get_dataset(self, basin: str) -> BaseDataset:
        """Get dataset for a single basin."""
        ds = get_dataset(
            cfg=self.cfg,
            is_train=False,
            period=self.period,
            basin=basin,
            additional_features=self.additional_features,
            id_to_int=self.id_to_int,
            scaler=self.scaler,
        )
        return ds

    # --------------------------------------------------------------------- #
    # Main evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        epoch: Optional[int] = None,
        save_results: bool = True,
        save_all_output: bool = False,
        metrics: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        model: Optional[torch.nn.Module] = None,
        experiment_logger: Optional[Logger] = None,
        from_best: bool = False,
    ) -> dict:
        """Evaluate the model on the configured period.

        Parameters
        ----------
        epoch : int, optional
            If given, evaluate weights from this epoch. Otherwise, use the last checkpoint.
        save_results : bool, optional
            If True, store evaluation metrics/results in the run directory.
        save_all_output : bool, optional
            If True, store all model outputs in the run directory.
        metrics : list or dict, optional
            Metrics to compute during evaluation. May be:
            - list of metric names, applied to all target variables, or
            - dict mapping target variable → list of metrics.
        model : torch.nn.Module, optional
            If provided, this model is used for evaluation; otherwise the internal model is used.
        experiment_logger : Logger, optional
            Logger instance to log step-wise metrics (e.g. during training).
        from_best : bool, optional
            If True, use weights from ``best_model.pt`` instead of epoch-specific weights.

        Returns
        -------
        dict
            Dictionary containing one xarray per basin with evaluation results.
        """
        if model is None:
            if not self.init_model:
                raise RuntimeError("No model was initialized for the evaluation.")
            if from_best:
                self._load_weights(from_best=True)
            else:
                self._load_weights(epoch=epoch)
            model = self.model

        # during validation, optionally evaluate only on a random subset of basins
        basins = list(self.basins)  # copy to avoid in-place shuffling side-effects
        if self.period == "validation":
            if len(basins) > self.cfg.validate_n_random_basins:
                random.shuffle(basins)
                basins = basins[: self.cfg.validate_n_random_basins]

        # force model to train-mode when doing mc-dropout evaluation
        if self.cfg.mc_dropout:
            model.train()
        else:
            model.eval()

        results: dict = defaultdict(dict)
        all_output: dict = {basin: None for basin in basins}

        pbar = tqdm(
            basins, file=sys.stdout, disable=self._disable_pbar, leave=False
        )
        pbar.set_description(
            "# Validation" if self.period == "validation" else "# Evaluation"
        )

        for basin in pbar:
            if self.cfg.cache_validation_data and basin in self.cached_datasets:
                ds = self.cached_datasets[basin]
            else:
                try:
                    ds = self._get_dataset(basin)
                except NoEvaluationDataError:
                    # no evaluation data for this basin → skip
                    continue

                if self.cfg.cache_validation_data and self.period == "validation":
                    self.cached_datasets[basin] = ds

            loader = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                num_workers=0,
                collate_fn=ds.collate_fn,
            )

            y_hat, y, dates, all_losses, all_output[basin] = self._evaluate(
                model, loader, ds.frequencies, save_all_output
            )

            # log loss of this basin plus number of batches/samples in the logger to compute epoch aggregates later
            if experiment_logger is not None:
                experiment_logger.log_step(
                    **{k: (v, len(loader)) for k, v in all_losses.items()}
                )

            predict_last_n = self.cfg.predict_last_n
            seq_length = self.cfg.seq_length

            # if predict_last_n / seq_length are int, there's only one frequency
            if isinstance(predict_last_n, int):
                predict_last_n = {ds.frequencies[0]: predict_last_n}
            if isinstance(seq_length, int):
                seq_length = {ds.frequencies[0]: seq_length}

            lowest_freq = sort_frequencies(ds.frequencies)[0]

            for freq in ds.frequencies:
                if predict_last_n[freq] == 0:
                    continue  # this frequency is not being predicted

                results[basin][freq] = {}

                # inverse feature scaling of observations
                feature_scaler = (
                    self.scaler["xarray_feature_scale"][self.cfg.target_variables]
                    .to_array()
                    .values
                )
                feature_center = (
                    self.scaler["xarray_feature_center"][self.cfg.target_variables]
                    .to_array()
                    .values
                )
                y_freq = y[freq] * feature_scaler + feature_center

                # inverse feature scaling of predictions
                if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                elif y_hat[freq].ndim == 4:
                    # if y_hat has 4 dim and we have multiple features we expand the dimensions for scaling
                    feature_scaler_exp = np.expand_dims(feature_scaler, (0, 1, 3))
                    feature_center_exp = np.expand_dims(feature_center, (0, 1, 3))
                    y_hat_freq = y_hat[freq] * feature_scaler_exp + feature_center_exp
                else:
                    raise RuntimeError(
                        f"Simulations have {y_hat[freq].ndim} dimensions. Only 3 and 4 are supported."
                    )

                # --- Inverse normalization for log-transformed targets variables ---
                if self.cfg.target_normalization is not None:
                    for var in self.cfg.target_variables:
                        norm_params = self.cfg.target_normalization.get(var, {})
                        transform = norm_params.get("transform", "center").lower()
                        log_offset = float(norm_params.get("log_offset", 0.0))

                        # Only handle 'log' here — other options are left unchanged
                        if transform == "log":
                            var_idx = self.cfg.target_variables.index(var)

                            # y_freq and y_hat_freq are log(Q + offset) here → invert with exp() - offset
                            y_freq[..., var_idx] = np.exp(y_freq[..., var_idx]) - log_offset
                            y_hat_freq[..., var_idx] = (
                                np.exp(y_hat_freq[..., var_idx]) - log_offset
                            )

                # Create data_vars dictionary for the xarray.Dataset
                data_vars = self._create_xarray_data_vars(y_hat_freq, y_freq)

                # freq_range are the steps of the current frequency at each lowest-frequency step
                frequency_factor = int(get_frequency_factor(lowest_freq, freq))

                # coordinates for the xarray.Dataset:
                # - 'date': last timestep of each sequence on the lowest frequency
                # - 'time_step': position relative to the last timestep, measured in units of `freq`
                coords = {
                    "date": dates[lowest_freq][:, -1],
                    "time_step": (
                        (dates[freq][0, :] - dates[freq][0, -1]) / pd.Timedelta(freq)
                    ).astype(np.int64)
                    + frequency_factor
                    - 1,
                }
                xr = xarray.Dataset(data_vars=data_vars, coords=coords)
                xr = xr.reindex(
                    {
                        "date": pd.DatetimeIndex(
                            pd.date_range(
                                xr["date"].values[0],
                                xr["date"].values[-1],
                                freq=lowest_freq,
                            ),
                            name="date",
                        )
                    }
                )
                results[basin][freq]["xr"] = xr

                # create datetime range at the current frequency
                freq_date_range = pd.date_range(
                    start=dates[lowest_freq][0, -1],
                    end=dates[freq][-1, -1],
                    freq=freq,
                )
                # remove datetime steps that are not being predicted from the datetime range
                mask = np.ones(frequency_factor, dtype=bool)
                mask[:-predict_last_n[freq]] = False
                freq_date_range = freq_date_range[np.tile(mask, len(xr["date"]))]

                # calculate metrics, if requested
                if metrics:
                    for target_variable in self.cfg.target_variables:
                        # stack dates and time_steps so we don't just evaluate every 24h
                        # when use_frequencies=[1D, 1h]
                        obs = (
                            xr.isel(time_step=slice(-frequency_factor, None))
                            .stack(datetime=["date", "time_step"])
                            .drop_vars({"datetime", "date", "time_step"})[
                                f"{target_variable}_obs"
                            ]
                        )
                        obs["datetime"] = freq_date_range

                        if not all(obs.isnull()):
                            sim = (
                                xr.isel(time_step=slice(-frequency_factor, None))
                                .stack(datetime=["date", "time_step"])
                                .drop_vars({"datetime", "date", "time_step"})[
                                    f"{target_variable}_sim"
                                ]
                            )
                            sim["datetime"] = freq_date_range

                            # clip negative predictions to zero, if variable is listed in config 'clip_target_to_zero'
                            if target_variable in self.cfg.clip_targets_to_zero:
                                sim = xarray.where(sim < 0, 1e-5, sim)

                            if "samples" in sim.dims:
                                # average across samples dimension for metrics
                                sim = sim.mean(dim="samples")

                            var_metrics = metrics if isinstance(metrics, list) else metrics[target_variable]
                            if "all" in var_metrics:
                                var_metrics = get_available_metrics()

                            try:
                                values = calculate_metrics(obs, sim, metrics=var_metrics, resolution=freq)
                            except AllNaNError as err:
                                msg = (
                                    f"Basin {basin} "
                                    + (f"{target_variable} " if len(self.cfg.target_variables) > 1 else "")
                                    + (f"{freq} " if len(ds.frequencies) > 1 else "")
                                    + str(err)
                                )
                                LOGGER.warning(msg)
                                values = {metric: np.nan for metric in var_metrics}

                            # add variable identifier to metrics if needed
                            if len(self.cfg.target_variables) > 1:
                                values = {f"{target_variable}_{key}": val for key, val in values.items()}
                            # add frequency identifier to metrics if needed
                            if len(ds.frequencies) > 1:
                                values = {f"{key}_{freq}": val for key, val in values.items()}

                            if experiment_logger is not None:
                                experiment_logger.log_step(**values)
                            for k, v in values.items():
                                results[basin][freq][k] = v


        # convert default dict back to plain dict
        results = dict(results)

        if (
            self.period == "validation"
            and self.cfg.log_n_figures > 0
            and experiment_logger is not None
            and results
        ):
            self._create_and_log_figures(results, experiment_logger, epoch)

        # save model output to file, if requested
        results_to_save = results if save_results else None
        states_to_save = all_output if save_all_output else None
        if save_results or save_all_output:
            self._save_results(
                results=results_to_save,
                states=states_to_save,
                epoch=epoch,
                from_best=from_best,
            )

        return results

    # --------------------------------------------------------------------- #
    # Visualization & persistence helpers
    # --------------------------------------------------------------------- #
    def _create_and_log_figures(
        self, results: dict, experiment_logger: Logger, epoch: Optional[int]
    ) -> None:
        """Create and log regression/uncertainty figures for a subset of basins."""
        basins = list(results.keys())
        random.shuffle(basins)

        for target_var in self.cfg.target_variables:
            max_figures = min(
                self.cfg.validate_n_random_basins,
                self.cfg.log_n_figures,
                len(basins),
            )
            for freq in results[basins[0]].keys():
                figures = []
                for i in range(max_figures):
                    xr = results[basins[i]][freq]["xr"]
                    obs = xr[f"{target_var}_obs"].values
                    sim = xr[f"{target_var}_sim"].values
                    # clip negative predictions to zero, if variable is listed in config 'clip_target_to_zero'
                    if target_var in self.cfg.clip_targets_to_zero:
                        sim = xarray.where(sim < 0, 1e-5, sim)
                    figures.append(
                        self._get_plots(
                            obs,
                            sim,
                            title=f"{target_var} - Basin {basins[i]} - Epoch {epoch} - Frequency {freq}",
                        )[0]
                    )
                # sanitize preamble so that it is a valid file name
                preamble = re.sub(r"[^A-Za-z0-9\._\-]+", "", target_var)
                experiment_logger.log_figures(figures, freq, preamble=preamble)

    def _save_results(
        self,
        results: Optional[dict],
        states: Optional[dict] = None,
        epoch: Optional[int] = None,
        from_best: bool = False,
    ) -> None:
        """Store results in various formats to disk.

        Notes
        -----
        We cannot store the time series data (the xarray objects) as netCDF file but have to use pickle as a
        wrapper. netCDF has constraints on variable names, which do not always hold in this project.
        Metrics, if calculated, are also stored separately in a CSV file.
        """
        # use name of weight file as part of the result folder name
        if from_best:
            weight_file = self._get_weight_file_from_best()
            parent_directory = self.run_dir / self.period / "model_from_best"
        else:
            weight_file = self._get_weight_file(epoch)
            parent_directory = self.run_dir / self.period / weight_file.stem

        parent_directory.mkdir(parents=True, exist_ok=True)

        # save metrics any time this function is called, as long as they exist
        df_metrics = None
        if self.cfg.metrics and results is not None:
            metrics_list = self.cfg.metrics
            if isinstance(metrics_list, dict):
                metrics_list = list(set(m for v in metrics_list.values() for m in v))
            if "all" in metrics_list:
                metrics_list = get_available_metrics()
            df_metrics = metrics_to_dataframe(
                results, metrics_list, self.cfg.target_variables
            )
            metrics_file = parent_directory / f"{self.period}_metrics.csv"
            df_metrics.to_csv(metrics_file)

        # store all results packed as pickle file
        if results is not None:
            result_file = parent_directory / f"{self.period}_results.p"
            with result_file.open("wb") as fp:
                pickle.dump(results, fp)

        # store all model outputs packed as pickle file
        if states is not None:
            states_file = parent_directory / f"{self.period}_all_output.p"
            with states_file.open("wb") as fp:
                pickle.dump(states, fp)
            LOGGER.info("Stored states at %s", states_file)

    # --------------------------------------------------------------------- #
    # Core evaluation loop
    # --------------------------------------------------------------------- #
    def _evaluate(
        self,
        model: BaseModel,
        loader: DataLoader,
        frequencies: List[str],
        save_all_output: bool = False,
    ) -> Tuple[dict, dict, dict, Dict[str, float], dict]:
        """Evaluate the model for one basin and return predictions, observations and losses."""
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            # if predict_last_n is int, there's only one frequency
            predict_last_n = {frequencies[0]: predict_last_n}

        preds: dict = {}
        obs: dict = {}
        dates: dict = {}
        all_output: dict = {}
        losses: List[Dict[str, float]] = []

        with torch.no_grad():
            for data in loader:
                # move everything (except dates) to device
                for key in data:
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                # optional head-specific pre-hook
                if self.cfg.head.lower() == "umal":
                    data = model.pre_model_hook(data, is_train=False)

                predictions, loss_dict = self._get_predictions_and_loss(model, data)

                # collect raw model outputs if requested
                if all_output:
                    for key, value in predictions.items():
                        if value is not None and not isinstance(value, dict):
                            all_output[key].append(value.detach().cpu().numpy())
                elif save_all_output:
                    all_output = {
                        key: [value.detach().cpu().numpy()]
                        for key, value in predictions.items()
                        if value is not None and not isinstance(value, dict)
                    }

                # per-frequency slicing of predictions/targets/dates
                for freq in frequencies:
                    if predict_last_n[freq] == 0:
                        continue  # no predictions for this frequency

                    freq_key = "" if len(frequencies) == 1 else f"_{freq}"
                    y_hat_sub, y_sub = self._subset_targets(
                        model,
                        data,
                        predictions,
                        predict_last_n[freq],
                        freq_key,
                    )

                    # Date subsetting is universal across all models and happens here.
                    date_sub = data[f"date{freq_key}"][:, -predict_last_n[freq] :]

                    if freq not in preds:
                        preds[freq] = y_hat_sub.detach().cpu()
                        obs[freq] = y_sub.cpu()
                        dates[freq] = date_sub
                    else:
                        preds[freq] = torch.cat(
                            (preds[freq], y_hat_sub.detach().cpu()), dim=0
                        )
                        obs[freq] = torch.cat((obs[freq], y_sub.cpu()), dim=0)
                        dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

                losses.append(loss_dict)

            # convert predictions / obs to numpy
            for freq in preds.keys():
                preds[freq] = preds[freq].numpy()
                obs[freq] = obs[freq].numpy()

        # concatenate all output variables (currently a dict-of-lists) into a single-level dict
        for key, list_of_data in all_output.items():
            all_output[key] = np.concatenate(list_of_data, axis=0)

        # aggregate losses across batches
        mean_losses: Dict[str, float] = {}
        if not losses:
            mean_losses["loss"] = np.nan
        else:
            for loss_name in losses[0].keys():
                loss_values = [loss[loss_name] for loss in losses]
                if np.all(np.isnan(loss_values)):
                    mean_losses[loss_name] = np.nan
                else:
                    mean_losses[loss_name] = float(np.nanmean(loss_values))

        return preds, obs, dates, mean_losses, all_output

    # --------------------------------------------------------------------- #
    # Abstract hooks for subclasses
    # --------------------------------------------------------------------- #
    def _get_predictions_and_loss(
        self, model: BaseModel, data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Run model forward pass and compute loss.

        Returns
        -------
        predictions : dict
            Raw model outputs (e.g. containing 'y_hat*' keys).
        losses : dict
            Loss components as scalars (float).
        """
        predictions = model(data)
        loss, all_losses = self.loss_obj(predictions, data)
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self,
        model: BaseModel,
        data: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        predict_last_n: int,
        freq: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return subset of predictions/targets for last `predict_last_n` steps (per frequency)."""
        raise NotImplementedError

    def _create_xarray_data_vars(
        self, y_hat: np.ndarray, y: np.ndarray
    ) -> Dict[str, Tuple[Tuple[str, str], np.ndarray]]:
        """Create `data_vars` dictionary for xarray.Dataset from predictions and observations."""
        raise NotImplementedError

    def _get_plots(
        self, qobs: np.ndarray, qsim: np.ndarray, title: str
    ) -> List[object]:
        """Return list of matplotlib Figure objects for plotting."""
        raise NotImplementedError


# ------------------------------------------------------------------------- #
# Regression tester
# ------------------------------------------------------------------------- #
class RegressionTester(BaseTester):
    """Tester class to run inference on a regression model."""

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = "test",
        init_model: bool = True,
    ) -> None:
        super().__init__(cfg, run_dir, period, init_model)

    def _subset_targets(
        self,
        model: BaseModel,
        data: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        predict_last_n: int,
        freq: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat_sub = predictions[f"y_hat{freq}"][:, -predict_last_n:, :]
        y_sub = data[f"y{freq}"][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(
        self, y_hat: np.ndarray, y: np.ndarray
    ) -> Dict[str, Tuple[Tuple[str, str], np.ndarray]]:
        data: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (("date", "time_step"), y[:, :, i])
            data[f"{var}_sim"] = (("date", "time_step"), y_hat[:, :, i])
        return data

    def _get_plots(
        self, qobs: np.ndarray, qsim: np.ndarray, title: str
    ) -> List[object]:
        return plots.regression_plot(qobs, qsim, title)


# ------------------------------------------------------------------------- #
# Uncertainty tester
# ------------------------------------------------------------------------- #
class UncertaintyTester(BaseTester):
    """Tester class to run inference on an uncertainty model."""

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = "test",
        init_model: bool = True,
    ) -> None:
        super().__init__(cfg, run_dir, period, init_model)

    def _get_predictions_and_loss(
        self, model: BaseModel, data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Override to draw multiple samples while still computing loss on mean prediction."""
        outputs = model(data)
        _, all_losses = self.loss_obj(outputs, data)
        predictions = model.sample(data, self.cfg.n_samples)
        model.eval()  # switch back to eval mode for downstream usage
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self,
        model: BaseModel,
        data: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        predict_last_n: int,
        freq: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat_sub = predictions[f"y_hat{freq}"][:, -predict_last_n:, :]
        y_sub = data[f"y{freq}"][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(
        self, y_hat: np.ndarray, y: np.ndarray
    ) -> Dict[str, Tuple[Tuple[str, str, str], np.ndarray]]:
        data: Dict[str, Tuple[Tuple[str, str, str], np.ndarray]] = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (("date", "time_step"), y[:, :, i])
            data[f"{var}_sim"] = (("date", "time_step", "samples"), y_hat[:, :, i, :])
        return data

    def _get_plots(
        self, qobs: np.ndarray, qsim: np.ndarray, title: str
    ) -> List[object]:
        return plots.uncertainty_plot(qobs, qsim, title)
