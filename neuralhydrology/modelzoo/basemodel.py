from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.samplingutils import sample_pointpredictions, umal_extend_batch


class BaseModel(nn.Module):
    """Abstract base model class, don't use this class for model training.

    Use subclasses of this class for training/evaluating different models, e.g. use `CudaLSTM` for a standard LSTM
    model or `EA-LSTM` for an Entity-Aware-LSTM.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts: list[str] = []

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Cache for scaler to avoid reloading from disk on every sampling call
        self._scaler_cache: Dict | None = None

        # Output size of the regression/probabilistic head
        self.output_size = len(cfg.target_variables)

        head = cfg.head.lower()
        if head == "gmm":
            # GMM: mu, sigma, pi → 3 × n_distributions
            self.output_size *= 3 * cfg.n_distributions
        elif head == "cmal":
            # CMAL: mu, b, tau, pi → 4 × n_distributions
            self.output_size *= 4 * cfg.n_distributions
        elif head == "umal":
            # UMAL: mu, b → 2 × targets
            self.output_size *= 2
        # regression and any other deterministic heads keep output_size as number of targets

    def _get_scaler(self) -> Dict:
        """Lazy-load and cache scaler from disk for sampling."""
        if self._scaler_cache is None:
            self._scaler_cache = load_scaler(self.cfg.run_dir)
        return self._scaler_cache

    def sample(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        n_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample point predictions from a probabilistic model.

        Wraps :func:`sample_pointpredictions`, which implements the sampling logic for different
        uncertainty estimation approaches (GMM, CMAL, UMAL, ...).

        Parameters
        ----------
        data : dict
            Dictionary containing model inputs (and optionally labels).
        n_samples : int
            Number of point prediction samples to draw from the model.

        Returns
        -------
        dict
            Sampled point predictions (e.g. containing ``'y_hat'`` or similar keys).
        """
        scaler = self._get_scaler()
        return sample_pointpredictions(self, data, n_samples, scaler)

    def forward(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        This method must be implemented by subclasses.

        Parameters
        ----------
        data : dict
            Dictionary containing input features (and potentially labels).

        Returns
        -------
        dict
            Model outputs and intermediate states/activations.
        """
        raise NotImplementedError

    def pre_model_hook(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        is_train: bool,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Optional hook executed before the forward pass in train/validation/test.

        For UMAL heads, this extends the batch by tau-samples along an extra dimension.

        Parameters
        ----------
        data : dict
            Input dictionary containing features (and labels).
        is_train : bool
            Whether we are in training mode.

        Returns
        -------
        dict
            Possibly modified input data used for the forward pass.
        """
        if self.cfg.head.lower() == "umal":
            data = umal_extend_batch(
                data,
                self.cfg,
                n_taus=self.cfg.n_taus,
                extend_y=True,
            )
        else:
            # hook for additional pre-processing strategies, if needed in the future
            pass

        return data
