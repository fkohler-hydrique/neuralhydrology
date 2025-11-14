from __future__ import annotations

import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralhydrology.training.loss as loss
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from neuralhydrology.training.earlystopper import EarlyStopper
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper: autoregressive teacher forcing (currently unused, but kept intact)
# --------------------------------------------------------------------------- #
def build_ar_teacher_forcing(
    y: torch.Tensor,
    seq_h: int,
    seq_f: int,
    lags: int = 1,
) -> torch.Tensor:
    """Build autoregressive tensor for teacher forcing, based on previous targets.

    Parameters
    ----------
    y : torch.Tensor
        Full target sequence, shape (batch, seq_total, out_dim).
    seq_h : int
        Hindcast length.
    seq_f : int
        Forecast length.
    lags : int, optional
        Number of lags to include.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, seq_f, lags), where last dim has [lag1, lag2, ...].
    """
    batch, seq_total, out_dim = y.shape
    ar = torch.zeros((batch, seq_f, lags), device=y.device, dtype=y.dtype)

    for t in range(seq_f):
        for lag in range(1, lags + 1):
            idx = seq_h + t - lag
            if idx >= 0:
                ar[:, t, lag - 1] = y[:, idx, 0]  # assumes scalar target in dim 0
            else:
                ar[:, t, lag - 1] = y[:, 0, 0]  # fallback for missing history

    return ar


class BaseTrainer:
    """Default class to train a model.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Will be initialized in initialize_training()
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_obj: Optional[loss.BaseLoss] = None
        self.experiment_logger: Optional[Logger] = None
        self.loader: Optional[DataLoader] = None
        self.validator: Optional[BaseTester] = None
        self.noise_sampler_y: Optional[torch.distributions.Normal] = None

        self._target_mean: Optional[torch.Tensor] = None
        self._target_std: Optional[torch.Tensor] = None
        self._scaler: Dict = {}

        self._allow_subsequent_nan_losses: int = cfg.allow_subsequent_nan_losses
        self._disable_pbar: bool = cfg.verbose == 0
        self._max_updates_per_epoch: Optional[int] = cfg.max_updates_per_epoch

        # Early stopping
        self._early_stopping: bool = cfg.early_stopping
        self._patience_early_stopping: int = cfg.patience_early_stopping
        self._min_delta_early_stopping: float = cfg.min_delta_early_stopping
        self._minimum_epochs_before_early_stopping: int = cfg.minimum_epochs_before_early_stopping

        # Dynamic learning rate
        self._dynamic_learning_rate: bool = cfg.dynamic_learning_rate
        self._patience_dynamic_learning_rate: int = cfg.patience_dynamic_learning_rate
        self._factor_dynamic_learning_rate: float = cfg.factor_dynamic_learning_rate

        # Save-best logic
        self._save_best_enabled: bool = cfg.save_best_enabled
        self._save_best_criterion: str = cfg.save_best_criterion
        self._best_score: float = float("inf")
        self._best_epoch: Optional[int] = None

        # Load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # Check at which epoch training starts
        self._epoch: int = self._get_start_epoch_number()

        # Folder structure and logging
        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info("### Folder structure created at %s", self.cfg.run_dir)

        if self.cfg.is_continue_training:
            LOGGER.info("### Continue training of run stored in %s", self.cfg.base_run_dir)

        if self.cfg.is_finetuning:
            LOGGER.info("### Start finetuning with pretrained model stored in %s", self.cfg.base_run_dir)

        LOGGER.info("### Run configurations for %s", self.cfg.experiment_name)
        for key, val in self.cfg.as_dict().items():
            LOGGER.info("%s: %s", key, val)

        self._set_random_seeds()
        self._set_device()

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #
    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self) -> None:
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

    def _get_tester(self) -> BaseTester:
        return get_tester(
            cfg=self.cfg,
            run_dir=self.cfg.run_dir,
            period="validation",
            init_model=False,
        )

    def _get_data_loader(self, ds: BaseDataset) -> DataLoader:
        """Create DataLoader for training set."""
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            collate_fn=ds.collate_fn,
        )

    # ------------------------------------------------------------------ #
    # Fine-tuning helpers
    # ------------------------------------------------------------------ #
    def _freeze_model_parts(self) -> None:
        """Freeze all parameters and unfreeze only those specified in cfg.finetune_modules."""
        # Freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        # Unfreeze parameters specified in config as tunable parameters
        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
            # Dictionary form: {module_group: module_parts}
            for module_group, module_parts in self.cfg.finetune_modules.items():
                if module_group in self.model.module_parts:
                    if isinstance(module_parts, str):
                        module_parts = [module_parts]
                    for module_part in module_parts:
                        module = getattr(self.model, module_group)[module_part]
                        for param in module.parameters():
                            param.requires_grad = True
                else:
                    unresolved_modules.append(module_group)

        if unresolved_modules:
            LOGGER.warning(
                "Could not resolve the following module parts for finetuning: %s",
                unresolved_modules,
            )

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize_training(self) -> None:
        """Initialize model, loss, optimizer, dataset, dataloader, logging, and validator.

        If called in a ``continue_training`` context, this also restores the model
        and optimizer state.
        """
        if self.cfg.is_finetuning:
            # Load scaler from pre-trained model.
            self._scaler = load_scaler(self.cfg.base_run_dir)

        # Initialize dataset before the model is loaded
        ds = self._get_dataset()
        self.ds = ds  # keep reference to save scaler for best model

        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")

        self.loader = self._get_data_loader(ds=ds)
        self.model = self._get_model().to(self.device)

        # Load checkpoint if requested
        if self.cfg.checkpoint_path is not None:
            LOGGER.info("Starting training from checkpoint %s", self.cfg.checkpoint_path)
            self.model.load_state_dict(
                torch.load(str(self.cfg.checkpoint_path), map_location=self.device)
            )
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            # Default for finetuning: last model state from base_run_dir
            checkpoint_path = sorted(self.cfg.base_run_dir.glob("model_epoch*.pt"))[-1]
            LOGGER.info("Starting finetuning from checkpoint %s", checkpoint_path)
            self.model.load_state_dict(
                torch.load(str(checkpoint_path), map_location=self.device)
            )

        # Freeze model parts from pre-trained model
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)

        # Add regularization terms to loss function
        self._set_regularization()

        # Restore optimizer and model state if training is continued
        if self.cfg.is_continue_training:
            self._restore_training_state()

        self.experiment_logger = Logger(cfg=self.cfg)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            # Set epoch and iteration step counter to continue from the selected checkpoint
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        # Validator
        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                LOGGER.warning(
                    "Validation set to validate every %s epoch(s), but "
                    "'validate_n_random_basins' not set or set to zero. "
                    "Will validate on the entire validation set.",
                    self.cfg.validate_every,
                )
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        # Target noise for data augmentation
        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(
                loc=0.0, scale=self.cfg.target_noise_std
            )
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables]
                .to_array()
                .values
            ).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables]
                .to_array()
                .values
            ).to(self.device)

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #
    def train_and_validate(self) -> None:
        """Train and validate the model for the configured number of epochs."""
        # Early stopping
        if self._early_stopping:
            if self.cfg.is_continue_training:
                LOGGER.warning("Early stopping state is reset.")
            early_stopper = EarlyStopper(
                patience=self._patience_early_stopping,
                min_delta=self._min_delta_early_stopping,
            )
        else:
            early_stopper = None

        # Dynamic learning rate scheduler
        if self._dynamic_learning_rate:
            if self.cfg.is_continue_training:
                LOGGER.warning("Scheduler state is reset.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self._factor_dynamic_learning_rate,
                patience=self._patience_dynamic_learning_rate,
                threshold=0.05,
                threshold_mode="rel",
                cooldown=3,
                min_lr=1e-7,
            )
        else:
            scheduler = None

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            # Static learning rate schedule (if dynamic LR is disabled)
            if not self._dynamic_learning_rate:
                if epoch in self.cfg.learning_rate.keys():
                    new_lr = self.cfg.learning_rate[epoch]
                    LOGGER.info("Setting learning rate to %s", new_lr)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr

            # Train one epoch
            self._train_epoch(epoch=epoch)
            avg_losses = self.experiment_logger.summarise()
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info("Epoch %d average loss: %s", epoch, loss_str)

            # Save weights and optimizer periodically
            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)

            # Validation
            if (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                self.validator.evaluate(
                    epoch=epoch,
                    save_results=self.cfg.save_validation_results,
                    save_all_output=self.cfg.save_all_output,
                    metrics=self.cfg.metrics,
                    model=self.model,
                    experiment_logger=self.experiment_logger.valid(),
                )

                valid_metrics = self.experiment_logger.summarise()
                if "avg_total_loss" in valid_metrics:
                    print_msg = (
                        f"Epoch {epoch} average validation loss: "
                        f"{valid_metrics['avg_total_loss']:.5f}"
                    )
                else:
                    print_msg = f"Epoch {epoch} validation done (no avg_total_loss key)."

                if self.cfg.metrics:
                    print_msg += " -- Median validation metrics: "
                    print_msg += ", ".join(
                        f"{k}: {v:.5f}"
                        for k, v in valid_metrics.items()
                        if k != "avg_total_loss"
                    )
                LOGGER.info(print_msg)

                # --------------------------- #
                # Save-best logic
                # --------------------------- #
                if self._save_best_enabled:
                    # choose criterion
                    if self._save_best_criterion == "val_loss":
                        score = valid_metrics.get("avg_total_loss", float("inf"))
                    elif self._save_best_criterion == "sum_metrics":
                        # sum = MSE + MAPE + (1 - NSE)  (lower is better)
                        mse = valid_metrics.get("MSE", 0.0)
                        mape = valid_metrics.get("MAPE", 0.0)
                        nse = valid_metrics.get("NSE", 0.0)
                        score = float(mse) + float(mape) + (1.0 - float(nse))
                    elif self._save_best_criterion in ["NSE", "MSE", "MAPE"]:
                        score = valid_metrics.get(self._save_best_criterion, 10.0)
                    else:
                        LOGGER.warning(
                            "Unknown save_best_criterion '%s', skipping saveBest.",
                            self.cfg.save_best_criterion,
                        )
                        score = float("inf")

                    if score < self._best_score:
                        LOGGER.info(
                            "[saveBest] New best model found at epoch %d: %.6f (prev %.6f)",
                            epoch,
                            score,
                            self._best_score,
                        )
                        self._best_score = score
                        self._best_epoch = epoch

                        # Save model and optimizer
                        best_model_path = self.cfg.run_dir / "best_model.pt"
                        torch.save(self.model.state_dict(), str(best_model_path))

                        best_opt_path = self.cfg.run_dir / "best_optimizer_state.pt"
                        if self.optimizer is not None:
                            torch.save(self.optimizer.state_dict(), str(best_opt_path))

                        # Save scaler if available on dataset
                        try:
                            if hasattr(self, "ds") and getattr(self.ds, "scaler", None) is not None:
                                best_scaler_path = self.cfg.run_dir / "best_model_scaler.pth"
                                torch.save(self.ds.scaler, str(best_scaler_path))
                        except Exception as e:  # noqa: BLE001
                            LOGGER.warning("[saveBest] Could not save scaler: %s", e)

                        # Save validation metrics snapshot
                        try:
                            metrics_path = self.cfg.run_dir / "best_model_metrics.csv"
                            metrics_path.parent.mkdir(parents=True, exist_ok=True)
                            df_metrics = pd.DataFrame([valid_metrics])
                            df_metrics.to_csv(metrics_path, index=False)
                        except Exception as e:  # noqa: BLE001
                            LOGGER.warning("[saveBest] Could not save metrics: %s", e)
                # --------------------------- #

                # Early stopping
                val_loss_for_es = valid_metrics.get("avg_total_loss", None)
                if (
                    self._early_stopping
                    and early_stopper is not None
                    and val_loss_for_es is not None
                    and epoch > self._minimum_epochs_before_early_stopping
                ):
                    if early_stopper.check_early_stopping(val_loss_for_es):
                        LOGGER.info(
                            "Early stopping triggered at epoch %d with validation loss %.5f. Training stopped.",
                            epoch,
                            val_loss_for_es,
                        )
                        break

                # Dynamic LR scheduling
                if self._dynamic_learning_rate and scheduler is not None and val_loss_for_es is not None:
                    old_lr = scheduler.get_last_lr()[-1]
                    scheduler.step(val_loss_for_es)
                    new_lr = scheduler.get_last_lr()[-1]

                    if old_lr != new_lr:
                        LOGGER.info(
                            "[Scheduler] Learning rate changed from %.1e to %.1e",
                            old_lr,
                            new_lr,
                        )

        # Make sure to close tensorboard to avoid losing the last epoch
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    # ------------------------------------------------------------------ #
    # Epoch training loop
    # ------------------------------------------------------------------ #
    def _train_epoch(self, epoch: int) -> None:
        """Train model for a single epoch."""
        self.model.train()
        self.experiment_logger.train()

        if self._max_updates_per_epoch is not None:
            n_iter = min(self._max_updates_per_epoch, len(self.loader))
        else:
            n_iter = None

        pbar = tqdm(
            self.loader,
            file=sys.stdout,
            disable=self._disable_pbar,
            total=n_iter,
            leave=False,
        )
        pbar.set_description(f"# Epoch {epoch}")

        nan_count = 0

        # Iterate over training set
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            # Move tensors to device
            for key in data.keys():
                if key.startswith("x_d"):
                    data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                elif not key.startswith("date"):
                    data[key] = data[key].to(self.device)

            # Apply possible pre-processing to the batch before the forward pass
            if self.cfg.head.lower() == "umal":
                data = self.model.pre_model_hook(data, is_train=True)

            # ---- Teacher forcing hook (commented out, kept for reference) ----
            # seq_h = self.cfg.hindcast_length
            # seq_f = self.cfg.predict_last_n
            # lags = getattr(self.cfg, "autoregressive_lags", 1)
            # ar_tensor = build_ar_teacher_forcing(data["y"], seq_h, seq_f, lags=lags)
            # data["x_d_forecast"][self.cfg.autoregressive_inputs[0]] = ar_tensor
            # -----------------------------------------------------------------

            # Forward pass
            predictions = self.model(data)

            # Make model available to regularization (e.g. L2)
            predictions_with_model = dict(predictions)
            predictions_with_model["model"] = self.model

            # Add noise to targets (data augmentation)
            if self.noise_sampler_y is not None:
                for key in filter(lambda k: "y" in k, data.keys()):
                    noise = self.noise_sampler_y.sample(data[key].shape)
                    # make sure we add near-zero noise to originally near-zero targets
                    data[key] += (data[key] + self._target_mean / self._target_std) * noise.to(
                        self.device
                    )

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            batch_loss, all_losses = self.loss_obj(predictions_with_model, data)

            # Handle NaN loss (skip or stop)
            if torch.isnan(batch_loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(
                        f"Loss was NaN for {nan_count} times in a row. Stopped training."
                    )
                LOGGER.warning(
                    "Loss is NaN; ignoring step. (#{}/{}).",
                    nan_count,
                    self._allow_subsequent_nan_losses,
                )
            else:
                nan_count = 0

                # Backprop and optimizer step
                batch_loss.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.clip_gradient_norm,
                    )

                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {batch_loss.item():.4f}")
            self.experiment_logger.log_step(
                **{k: v.item() for k, v in all_losses.items()}
            )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _get_start_epoch_number(self) -> int:
        """Determine starting epoch index (for continue_training or fresh run)."""
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = sorted(self.cfg.run_dir.glob("model_epoch*.pt"))[-1]
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self) -> None:
        """Restore model and optimizer state for continue_training runs."""
        if self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = sorted(self.cfg.base_run_dir.glob("model_epoch*.pt"))[-1]
            epoch = weight_path.name[-6:-3]

        optimizer_path = self.cfg.base_run_dir / f"optimizer_state_epoch{epoch}.pt"

        LOGGER.info("Continue training from epoch %d", int(epoch))
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.optimizer.load_state_dict(
            torch.load(str(optimizer_path), map_location=self.device)
        )

    def _save_weights_and_optimizer(self, epoch: int) -> None:
        """Persist model and optimizer state for a given epoch."""
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def _set_random_seeds(self) -> None:
        """Fix random seeds for reproducibility."""
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)

    def _set_device(self) -> None:
        """Choose training device from configuration and availability."""
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
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

        LOGGER.info("### Device %s will be used for training", self.device)

    def _create_folder_structure(self) -> None:
        """Create run directory and subfolders (train_data, img_log, ...)."""
        # Continue-training: create subdirectory within run directory of base run
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"

            # Store dir of base run for easier access in weight loading
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name

        # New folder structure
        else:
            now = datetime.now()
            day = f"{now.day:02d}"
            month = f"{now.month:02d}"
            hour = f"{now.hour:02d}"
            minute = f"{now.minute:02d}"
            second = f"{now.second:02d}"
            run_name = f"{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}"

            # If no base directory for runs is specified, create 'runs' folder in CWD
            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        # Create folder + necessary subfolder
        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")

        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
