from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from neuralhydrology.training.regularization import BaseRegularization
from neuralhydrology.utils.config import Config

ONE_OVER_2PI_SQUARED = 1.0 / np.sqrt(2.0 * np.pi)


class BaseLoss(torch.nn.Module):
    """Base loss class.

    All concrete losses extend this class by implementing `_get_loss`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    prediction_keys : list[str]
        Keys expected in the `prediction` dict (without any frequency suffix).
    ground_truth_keys : list[str]
        Keys expected in the `data` dict (without any frequency suffix).
    additional_data : list[str], optional
        Keys from `data` to be passed as additional kwargs to `_get_loss`.
    output_size_per_target : int, optional
        Number of model outputs per target variable (e.g. 1 for regression,
        `n_distributions` for mixture heads).
    """

    def __init__(
        self,
        cfg: Config,
        prediction_keys: List[str],
        ground_truth_keys: List[str],
        additional_data: List[str] | None = None,
        output_size_per_target: int = 1,
    ) -> None:
        super().__init__()

        self._predict_last_n = _get_predict_last_n(cfg)
        self._frequencies = [
            f for f in self._predict_last_n.keys() if f not in cfg.no_loss_frequencies
        ]
        self._output_size_per_target = output_size_per_target

        self._regularization_terms: list[BaseRegularization] = []

        # Names of keys to subset in time / per-target
        self._prediction_keys = prediction_keys
        self._ground_truth_keys = ground_truth_keys

        # Additional loss-specific inputs (e.g., per_basin_target_stds for NSE)
        self._additional_data = additional_data or []

        # Per-target weights (for multi-target output)
        if cfg.target_loss_weights is None:
            n_targets = len(cfg.target_variables)
            if n_targets <= 0:
                raise ValueError("cfg.target_variables must contain at least one target.")
            weights = [1.0 / n_targets] * n_targets
        else:
            if len(cfg.target_loss_weights) != len(cfg.target_variables):
                raise ValueError(
                    "Number of target_loss_weights must equal number of target variables."
                )
            weights = list(cfg.target_loss_weights)

        # store as simple Python floats to avoid device mismatches
        self._target_weights = [float(w) for w in weights]

    # ------------------------------------------------------------------ #
    # Forward: orchestrates per-frequency, per-target loss + regularization
    # ------------------------------------------------------------------ #
    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate the loss.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Dictionary of predictions for each frequency. If more than one frequency is predicted,
            keys must have suffixes ``_{frequency}``; without suffix otherwise.
        data : dict[str, torch.Tensor]
            Dictionary of ground truth data for each frequency, same suffixing rules as prediction.

        Returns
        -------
        total_loss : torch.Tensor
            Overall loss value (core loss + regularization).
        all_losses : dict[str, torch.Tensor]
            Individual components, including:
            - 'loss'        : main data-fit loss (sum over targets & frequencies)
            - '<reg_name>'  : each regularization term
            - 'total_loss'  : final combined loss
        """
        # unpack loss-specific additional arguments from data (no subsetting here)
        kwargs = {key: data[key] for key in self._additional_data}

        losses: list[torch.Tensor] = []
        prediction_sub: dict[str, torch.Tensor] = {}
        ground_truth_sub: dict[str, torch.Tensor] = {}

        for freq in self._frequencies:
            if self._predict_last_n[freq] == 0:
                continue  # no predictions for this frequency

            freq_suffix = "" if freq == "" else f"_{freq}"

            # apply predict_last_n for this frequency at once
            freq_pred, freq_gt = self._subset_in_time(
                {key: prediction[f"{key}{freq_suffix}"] for key in self._prediction_keys},
                {key: data[f"{key}{freq_suffix}"] for key in self._ground_truth_keys},
                self._predict_last_n[freq],
            )

            # remember subsets for multi-frequency regularization
            prediction_sub.update(
                {f"{key}{freq_suffix}": freq_pred[key] for key in freq_pred.keys()}
            )
            ground_truth_sub.update(
                {f"{key}{freq_suffix}": freq_gt[key] for key in freq_gt.keys()}
            )

            # per-target loop
            for n_target, weight in enumerate(self._target_weights):
                # subset to this target's slice
                target_pred, target_gt = self._subset_target(freq_pred, freq_gt, n_target)

                # subset any additional data
                kwargs_sub = self._subset_additional_data(kwargs, n_target)

                # compute loss component for this target
                loss_val = self._get_loss(target_pred, target_gt, **kwargs_sub)
                losses.append(loss_val * weight)

        if not losses:
            # fallback: no loss was computed (e.g., all predict_last_n = 0)
            example_tensor = next(iter(prediction.values()))
            loss = torch.zeros((), device=example_tensor.device, dtype=example_tensor.dtype)
        else:
            loss = torch.stack(losses).sum()

        total_loss = loss.clone()
        all_losses = defaultdict(lambda: torch.tensor(0.0, device=total_loss.device))
        all_losses["loss"] = loss

        # add regularization terms (may use prediction_sub / ground_truth_sub / other_model_data)
        other_model_data = {
            k: v for k, v in prediction.items() if k not in self._prediction_keys
        }
        for reg_module in self._regularization_terms:
            reg_out = reg_module(prediction_sub, ground_truth_sub, other_model_data)
            total_loss = total_loss + reg_module.weight * reg_out
            # one name may appear multiple times; sum them up for logging
            all_losses[reg_module.name] = all_losses[reg_module.name] + reg_out

        all_losses["total_loss"] = total_loss
        return total_loss, all_losses

    # ------------------------------------------------------------------ #
    # Helpers for subsetting in time and per-target
    # ------------------------------------------------------------------ #
    def _subset_in_time(
        self,
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        predict_last_n: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Subset ground truth and prediction to the last `predict_last_n` timesteps.

        Supports:
        - Prediction shape (batch, seq, out) → standard case.
        - Prediction shape (batch, out_flat):
          * if out_flat == predict_last_n * output_size_per_target:
              reshape to (batch, predict_last_n, output_size_per_target)
          * if out_flat == output_size_per_target:
              treat as seq_len = 1 and unsqueeze time axis.
        """
        # subset ground truth (expects [batch, seq, out])
        ground_truth_sub = {key: gt[:, -predict_last_n:, :] for key, gt in ground_truth.items()}

        prediction_sub: dict[str, torch.Tensor] = {}
        for key, pred in prediction.items():
            if pred.dim() == 3:
                prediction_sub[key] = pred[:, -predict_last_n:, :]
            elif pred.dim() == 2:
                bs, out_flat = pred.shape
                expected_out = predict_last_n * self._output_size_per_target

                if out_flat == expected_out:
                    if self._output_size_per_target == 1:
                        prediction_sub[key] = pred[:, -predict_last_n:].unsqueeze(-1)
                    else:
                        prediction_sub[key] = pred.reshape(
                            bs, predict_last_n, self._output_size_per_target
                        )
                elif out_flat == self._output_size_per_target:
                    # single-timestep output: (batch, out) -> (batch, 1, out)
                    prediction_sub[key] = pred.unsqueeze(1)
                else:
                    # fallback: interpret as single timestep and warn user if needed
                    prediction_sub[key] = pred.unsqueeze(1)
            else:
                raise ValueError(
                    f"Prediction tensor for key '{key}' must be 2D or 3D, got {pred.dim()}D."
                )
        return prediction_sub, ground_truth_sub

    def _subset_target(
        self,
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        n_target: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Subset (batch, seq, out) tensors to a single target index."""
        start = n_target * self._output_size_per_target
        end = (n_target + 1) * self._output_size_per_target
        prediction_sub = {key: pred[:, :, start:end] for key, pred in prediction.items()}
        ground_truth_sub = {
            key: gt[:, :, n_target : n_target + 1] for key, gt in ground_truth.items()
        }
        return prediction_sub, ground_truth_sub

    @staticmethod
    def _subset_additional_data(
        additional_data: Dict[str, torch.Tensor],
        n_target: int,
    ) -> Dict[str, torch.Tensor]:
        # by default, nothing happens; subclasses can override
        return additional_data

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def set_regularization_terms(
        self, regularization_modules: List[BaseRegularization]
    ) -> None:
        """Register the passed regularization terms to be added to the loss function."""
        self._regularization_terms = regularization_modules


# --------------------------------------------------------------------------- #
# Concrete Losses
# --------------------------------------------------------------------------- #
class MaskedMSELoss(BaseLoss):
    """Mean squared error loss (masked on NaNs in ground truth)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, prediction_keys=["y_hat"], ground_truth_keys=["y"])

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        mask = ~torch.isnan(ground_truth["y"])
        return 0.5 * torch.mean((prediction["y_hat"][mask] - ground_truth["y"][mask]) ** 2)


class MaskedMAPELoss(BaseLoss):
    """Mean Absolute Percentage Error (MAPE) loss (masked on NaNs)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, prediction_keys=["y_hat"], ground_truth_keys=["y"])

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        y_true = ground_truth["y"]
        y_pred = prediction["y_hat"]

        mask = ~torch.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        eps = 1e-9
        return torch.mean(
            torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=eps))
        ) * 100.0


class MaskedSMAPELoss(BaseLoss):
    """Symmetric Mean Absolute Percentage Error (SMAPE) loss.

    Requires 3D tensors (batch, seq, out) for both y_true and y_pred.
    Handles:
    - batch/seq transposition
    - expanded batch dimension (e.g. UMAL samples)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, prediction_keys=["y_hat"], ground_truth_keys=["y"])

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        y_true = ground_truth["y"]
        y_pred = prediction["y_hat"]

        if y_pred.dim() != 3 or y_true.dim() != 3:
            raise ValueError(
                "y_pred and y_true must be 3D tensors (batch, seq, out) for MaskedSMAPELoss."
            )

        # Align shapes
        if y_pred.shape[:2] == y_true.shape[:2]:
            pass  # already aligned
        elif y_pred.shape[0] == y_true.shape[1] and y_pred.shape[1] == y_true.shape[0]:
            y_pred = y_pred.transpose(0, 1)
        else:
            # handle expanded batch (e.g. UMAL with multiple tau samples)
            if y_pred.shape[1:] == y_true.shape[1:] and y_pred.shape[0] % y_true.shape[0] == 0:
                n = y_pred.shape[0] // y_true.shape[0]
                y_true = y_true.repeat(n, 1, 1)
            elif y_true.shape[1:] == y_pred.shape[1:] and y_true.shape[0] % y_pred.shape[0] == 0:
                n = y_true.shape[0] // y_pred.shape[0]
                y_pred = y_pred.repeat(n, 1, 1)
            else:
                raise ValueError(
                    f"Incompatible shapes for SMAPE: y_pred {tuple(y_pred.shape)} vs y_true {tuple(y_true.shape)}"
                )

        mask = ~torch.isnan(y_true)
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]

        eps = 1e-8
        numerator = torch.abs(y_true_valid - y_pred_valid)
        denominator = (torch.abs(y_true_valid) + torch.abs(y_pred_valid)) / 2.0 + eps
        return 100.0 * torch.mean(numerator / denominator)


class CombinedLoss(BaseLoss):
    """Combine multiple BaseLoss instances with specified weights.

    Supported sub-loss names: 'mse', 'nse', 'smape', 'mape'.
    """

    def __init__(
        self,
        cfg: Config,
        losses_list_str: List[str],
        weights: List[float] | None = None,
    ) -> None:
        # Instantiate sub-losses
        losses: list[BaseLoss] = []
        for loss_name in losses_list_str:
            ln = loss_name.lower()
            if ln == "mse":
                losses.append(MaskedMSELoss(cfg))
            elif ln == "nse":
                losses.append(MaskedNSELoss(cfg))
            elif ln == "smape":
                losses.append(MaskedSMAPELoss(cfg))
            elif ln == "mape":
                losses.append(MaskedMAPELoss(cfg))
            else:
                raise ValueError(
                    f"Loss '{loss_name}' is not recognized for CombinedLoss. "
                    "Supported: mse, nse, smape, mape."
                )

        self.losses = losses

        if weights is None:
            self.weights = [1.0 / len(losses)] * len(losses)
        else:
            if len(weights) != len(losses):
                raise ValueError("Number of weights must match number of losses.")
            self.weights = [float(w) for w in weights]

        # gather all keys across all loss objects
        pred_keys: list[str] = []
        gt_keys: list[str] = []
        add_data: list[str] = []

        for loss_obj in losses:
            pred_keys.extend(getattr(loss_obj, "_prediction_keys", []) or [])
            gt_keys.extend(getattr(loss_obj, "_ground_truth_keys", []) or [])
            add_data.extend(getattr(loss_obj, "_additional_data", []) or [])

        def _unique(seq: List[str]) -> List[str]:
            seen = set()
            out: list[str] = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        prediction_key = _unique(pred_keys)
        ground_truth_key = _unique(gt_keys)
        additional_data = _unique(add_data)

        super().__init__(
            cfg,
            prediction_keys=prediction_key,
            ground_truth_keys=ground_truth_key,
            additional_data=additional_data,
        )

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        # Provide dummy per_basin_target_stds if NSE is used but not passed
        if "per_basin_target_stds" not in kwargs:
            example_tensor = next(iter(prediction.values()))
            kwargs["per_basin_target_stds"] = torch.ones_like(example_tensor)

        combined = 0.0
        for loss_obj, w in zip(self.losses, self.weights):
            loss_kwargs = {
                k: v for k, v in kwargs.items() if k in loss_obj._additional_data  # type: ignore[attr-defined]
            }
            combined = combined + w * loss_obj._get_loss(prediction, ground_truth, **loss_kwargs)
        return combined


class MaskedRMSELoss(BaseLoss):
    """Root mean squared error loss (masked on NaNs)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, prediction_keys=["y_hat"], ground_truth_keys=["y"])

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        mask = ~torch.isnan(ground_truth["y"])
        mse = torch.mean((prediction["y_hat"][mask] - ground_truth["y"][mask]) ** 2)
        return torch.sqrt(0.5 * mse)


class MaskedNSELoss(BaseLoss):
    """Basin-averaged Nash–Sutcliffe Efficiency (NSE) loss (Kratzert et al., 2019)."""

    def __init__(self, cfg: Config, eps: float = 0.1) -> None:
        super().__init__(
            cfg,
            prediction_keys=["y_hat"],
            ground_truth_keys=["y"],
            additional_data=["per_basin_target_stds"],
        )
        self.eps = eps

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        mask = ~torch.isnan(ground_truth["y"])
        y_hat = prediction["y_hat"][mask]
        y = ground_truth["y"][mask]
        per_basin_target_stds = kwargs["per_basin_target_stds"]

        # expand per-basin stds to y_hat shape, then mask
        per_basin_target_stds = per_basin_target_stds.expand_as(prediction["y_hat"])[mask]

        squared_error = (y_hat - y) ** 2
        weights = 1.0 / (per_basin_target_stds + self.eps) ** 2
        scaled_loss = weights * squared_error
        return torch.mean(scaled_loss)

    @staticmethod
    def _subset_additional_data(
        additional_data: Dict[str, torch.Tensor],
        n_target: int,
    ) -> Dict[str, torch.Tensor]:
        # keep shape [bs, seq, 1] per target
        return {key: value[:, :, n_target : n_target + 1] for key, value in additional_data.items()}


class MaskedGMMLoss(BaseLoss):
    """Average negative log-likelihood for a Gaussian Mixture Model (GMM) head."""

    def __init__(self, cfg: Config, eps: float = 1e-10) -> None:
        super().__init__(
            cfg,
            prediction_keys=["mu", "sigma", "pi"],
            ground_truth_keys=["y"],
            output_size_per_target=cfg.n_distributions,
        )
        self.eps = eps

    @staticmethod
    def _gaussian_distribution(
        mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # (y - mu) / sigma
        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = -0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * ONE_OVER_2PI_SQUARED

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        mask = ~torch.isnan(ground_truth["y"]).any(1).any(1)
        y = ground_truth["y"][mask]
        m = prediction["mu"][mask]
        s = prediction["sigma"][mask]
        p = prediction["pi"][mask]

        result = self._gaussian_distribution(m, s, y) * p
        result = torch.sum(result, dim=-1)
        result = -torch.log(result + self.eps)
        return torch.mean(result)


class MaskedCMALLoss(BaseLoss):
    """Average negative log-likelihood for the CMAL head."""

    def __init__(self, cfg: Config, eps: float = 1e-8) -> None:
        super().__init__(
            cfg,
            prediction_keys=["mu", "b", "tau", "pi"],
            ground_truth_keys=["y"],
            output_size_per_target=cfg.n_distributions,
        )
        self.eps = eps

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        mask = ~torch.isnan(ground_truth["y"]).any(1).any(1)
        y = ground_truth["y"][mask]
        m = prediction["mu"][mask]
        b = prediction["b"][mask]
        t = prediction["tau"][mask]
        p = prediction["pi"][mask]

        error = y - m
        log_like = (
            torch.log(t)
            + torch.log(1.0 - t)
            - torch.log(b)
            - torch.max(t * error, (t - 1.0) * error) / b
        )
        log_weights = torch.log(p + self.eps)

        result = torch.logsumexp(log_weights + log_like, dim=2)
        result = -torch.mean(torch.sum(result, dim=1))
        return result


class MaskedUMALLoss(BaseLoss):
    """Average negative log-likelihood for the UMAL head."""

    def __init__(self, cfg: Config, eps: float = 1e-5) -> None:
        super().__init__(
            cfg,
            prediction_keys=["mu", "b"],
            ground_truth_keys=["y_extended", "tau"],
            output_size_per_target=2,
        )
        self.eps = eps
        self._n_taus_count = cfg.n_taus
        # store log(n_taus) as float to avoid device issues
        self._n_taus_log = float(np.log(cfg.n_taus).astype("float32"))

    def _get_loss(
        self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        mask = ~torch.isnan(ground_truth["y_extended"]).any(1).any(1)
        y = ground_truth["y_extended"][mask]
        t = ground_truth["tau"][mask]
        m = prediction["mu"][mask]
        b = prediction["b"][mask]

        error = y - m
        log_like = (
            torch.log(t)
            + torch.log(1.0 - t)
            - torch.log(b)
            - torch.max(t * error, (t - 1.0) * error) / b
        )

        original_batch_size = int(log_like.shape[0] / self._n_taus_count)
        log_like_split = torch.cat(log_like[:, :, :].split(original_batch_size, 0), 2)

        result = torch.logsumexp(log_like_split, dim=2) - self._n_taus_log
        result = -torch.mean(torch.sum(result, dim=1))
        return result


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _get_predict_last_n(cfg: Config) -> dict:
    predict_last_n = cfg.predict_last_n
    if isinstance(predict_last_n, int):
        predict_last_n = {"": predict_last_n}
    if len(predict_last_n) == 1:
        # if there's only one frequency, omit its identifier
        predict_last_n = {"": list(predict_last_n.values())[0]}
    return predict_last_n
