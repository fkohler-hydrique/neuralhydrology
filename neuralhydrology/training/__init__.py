from __future__ import annotations

import logging
import warnings
from typing import List

import torch

import neuralhydrology.training.loss as loss
from neuralhydrology.training import regularization
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Return optimizer instance based on configuration.

    Currently supported:
    - 'Adam'
    - 'AdamW'

    Parameters
    ----------
    model : torch.nn.Module
        Model to be optimized.
    cfg : Config
        Run configuration containing optimizer settings.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer.
    """
    opt_name = cfg.optimizer.lower()

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate[0])
    else:
        raise NotImplementedError(
            f"{cfg.optimizer} not implemented or not linked in `get_optimizer()`"
        )

    return optimizer


def get_custom_loss(cfg: Config) -> loss.BaseLoss:
    """Build a CombinedLoss object from the 'custom_loss' configuration."""
    return loss.CombinedLoss(
        cfg,
        losses_list_str=cfg.custom_loss["losses"],
        weights=cfg.custom_loss["weights"],
    )


def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Return the loss object specified in the configuration.

    Supported:
    - 'MSE'         → MaskedMSELoss
    - 'NSE'         → MaskedNSELoss
    - 'RMSE'        → MaskedRMSELoss
    - 'GMMLoss'     → MaskedGMMLoss
    - 'CMALLoss'    → MaskedCMALLoss
    - 'UMALLoss'    → MaskedUMALLoss
    - 'custom_loss' → CombinedLoss (from cfg.custom_loss)
    - 'MAPE'        → MaskedMAPELoss
    - 'SMAPE'       → MaskedSMAPELoss
    """
    name = cfg.loss.lower()

    if name == "mse":
        loss_obj = loss.MaskedMSELoss(cfg)
    elif name == "nse":
        loss_obj = loss.MaskedNSELoss(cfg)
    elif name == "weightednse":
        warnings.warn(
            "'WeightedNSE' loss has been removed. Use 'NSE' with 'target_loss_weights' instead.",
            FutureWarning,
        )
        loss_obj = loss.MaskedNSELoss(cfg)
    elif name == "rmse":
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif name == "gmmloss":
        loss_obj = loss.MaskedGMMLoss(cfg)
    elif name == "cmalloss":
        loss_obj = loss.MaskedCMALLoss(cfg)
    elif name == "umalloss":
        loss_obj = loss.MaskedUMALLoss(cfg)
    elif name == "custom_loss":
        loss_obj = get_custom_loss(cfg)
    elif name == "mape":
        loss_obj = loss.MaskedMAPELoss(cfg)
    elif name == "smape":
        loss_obj = loss.MaskedSMAPELoss(cfg)
    else:
        raise NotImplementedError(f"{cfg.loss} not implemented or not linked in `get_loss_obj()`")

    return loss_obj


def get_regularization_obj(cfg: Config) -> List[regularization.BaseRegularization]:
    """Return list of regularization objects specified in the configuration.

    Supported entries in cfg.regularization:
    - 'tie_frequencies'
    - 'forecast_overlap'
    - 'l2' (weight decay wrapper)

    Each entry may be:
    - 'name'                  → uses default weight 1.0
    - ('name', weight: float) → custom weight
    """
    regularization_modules: list[regularization.BaseRegularization] = []

    for reg_item in cfg.regularization:
        if isinstance(reg_item, str):
            reg_name = reg_item
            reg_weight = 1.0
        else:
            reg_name, reg_weight = reg_item

        if reg_name == "tie_frequencies":
            regularization_modules.append(
                regularization.TiedFrequencyMSERegularization(cfg=cfg, weight=reg_weight)
            )
        elif reg_name == "forecast_overlap":
            regularization_modules.append(
                regularization.ForecastOverlapMSERegularization(cfg=cfg, weight=reg_weight)
            )
        elif reg_name == "l2":
            regularization_modules.append(
                regularization.L2Regularization(cfg=cfg, weight=reg_weight)
            )
        else:
            raise NotImplementedError(
                f"{reg_name} not implemented or not linked in `get_regularization_obj()`."
            )

    return regularization_modules
