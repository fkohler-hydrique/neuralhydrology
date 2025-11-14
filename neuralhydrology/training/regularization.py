from typing import Dict

import torch

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.utils.config import Config


class BaseRegularization(torch.nn.Module):
    """Base class for regularization terms.

    Parameters
    ----------
    cfg : Config
        Run configuration.
    name : str
        Name of the regularization term (for logging).
    weight : float, optional
        Base weight of the regularization term. Used by the loss wrapper.
    """

    def __init__(self, cfg: Config, name: str, weight: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.weight = weight

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        other_model_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class TiedFrequencyMSERegularization(BaseRegularization):
    """Penalize inconsistent predictions across multiple frequencies.

    For each pair of adjacent frequencies f (high) and f' (low), it:
    1. Aggregates f's predictions to f' resolution.
    2. Penalizes MSE between aggregated f and f' predictions.

    Requires at least two predicted frequencies.
    """

    def __init__(self, cfg: Config, weight: float = 1.0) -> None:
        super().__init__(cfg, name="tie_frequencies", weight=weight)
        self._frequencies = sort_frequencies(
            [f for f in cfg.use_frequencies if cfg.predict_last_n[f] > 0 and f not in cfg.no_loss_frequencies]
        )
        if len(self._frequencies) < 2:
            raise ValueError("TiedFrequencyMSERegularization needs at least two frequencies.")

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        *args,
    ) -> torch.Tensor:
        loss = 0.0
        for idx, freq in enumerate(self._frequencies):
            if idx == 0:
                continue
            higher = self._frequencies[idx]
            lower = self._frequencies[idx - 1]

            frequency_factor = int(get_frequency_factor(lower, higher))

            freq_pred = prediction[f"y_hat_{higher}"]  # (bs, seq_high, out)
            mean_freq_pred = freq_pred.view(
                freq_pred.shape[0],
                freq_pred.shape[1] // frequency_factor,
                frequency_factor,
                -1,
            ).mean(dim=2)

            lower_freq_pred = prediction[f"y_hat_{lower}"][:, -mean_freq_pred.shape[1] :]
            loss = loss + torch.mean((lower_freq_pred - mean_freq_pred) ** 2)

        return loss


class ForecastOverlapMSERegularization(BaseRegularization):
    """Squared error regularization over overlapping hindcast/forecast windows.

    Expects models to provide in `other_model_output`:
    - other_model_output['y_hindcast_overlap'] : dict[str, Tensor]
    - other_model_output['y_forecast_overlap'] : dict[str, Tensor]
    """

    def __init__(self, cfg: Config, weight: float = 1.0) -> None:
        super().__init__(cfg, name="forecast_overlap", weight=weight)

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        other_model_output: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if "y_hindcast_overlap" not in other_model_output or not other_model_output["y_hindcast_overlap"]:
            raise ValueError("y_hindcast_overlap is not present in the model output.")
        if "y_forecast_overlap" not in other_model_output or not other_model_output["y_forecast_overlap"]:
            raise ValueError("y_forecast_overlap is not present in the model output.")

        loss = 0.0
        for key in other_model_output["y_hindcast_overlap"]:
            hindcast = other_model_output["y_hindcast_overlap"][key]
            forecast = other_model_output["y_forecast_overlap"][key]
            loss = loss + torch.mean((hindcast - forecast) ** 2)

        return loss


class L2Regularization(BaseRegularization):
    """L2 regularization (weight decay) applied to model parameters.

    Skips biases and normalization parameters (common practice).
    """

    def __init__(self, cfg: Config, weight: float = 1.0) -> None:
        super().__init__(cfg, name="l2", weight=weight)

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        other_model_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute unweighted L2 penalty (outer loss will apply `self.weight`)."""
        if "model" not in other_model_data:
            raise ValueError("L2Regularization requires 'model' in other_model_data.")

        model = other_model_data["model"]
        l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Skip biases and normalization parameters
            if name.endswith(".bias"):
                continue
            if "norm" in name.lower():
                continue

            l2_loss = l2_loss + torch.sum(param ** 2)

        return l2_loss
