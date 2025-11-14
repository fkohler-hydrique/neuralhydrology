from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class SequentialForecastLSTM(BaseModel):
    """Single-LSTM model that rolls through hindcast + forecast in one sequence.

    Compared to a standard CudaLSTM, this model:
    - Uses separate embedding networks for hindcast and forecast.
    - Concatenates hindcast and forecast sequences in time and runs one LSTM over both.
    - Returns only the last `predict_last_n` steps (forecast part) via ``y_hat``.

    Note
    ----
    - Do not use this model with ``forecast_overlap > 0``.
    """

    # Submodules that can be used for fine-tuning
    module_parts = ["hindcast_embedding_net", "forecast_embedding_net", "lstm", "dropout", "head"]

    def __init__(self, cfg: Config) -> None:
        # Handle predict_last_n being int or dict (single-frequency case)
        predict_last_n = cfg.predict_last_n
        if isinstance(predict_last_n, dict):
            # Single-frequency: just take the first value
            predict_last_n = next(iter(predict_last_n.values()))
        self._predict_last_n: int = int(predict_last_n)

        super().__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError(
                "Forecast overlap cannot be set for a sequential forecasting model. "
                "Please set it to 0 or remove it from the config file."
            )

        # Embedding networks for hindcast and forecast dynamic/static inputs
        # Output shape: (seq_len, batch, embedding_dim)
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type="hindcast")
        self.forecast_embedding_net = InputLayer(cfg, embedding_type="forecast")

        if self.forecast_embedding_net.output_size != self.hindcast_embedding_net.output_size:
            raise ValueError(
                "Forecast and hindcast embedding nets must have the same output size when "
                "using a SequentialForecastLSTM."
            )

        self.lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.lstm_dropout,
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # Head expects (batch, seq_len, hidden); see forward() for actual usage
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #
    def _reset_parameters(self) -> None:
        """Special initialization of certain model weights.

        If `initial_forget_bias` is specified, initialize the forget gate bias.
        """
        if self.cfg.initial_forget_bias is None:
            return

        # Forget gate is second quarter of the bias vector for LSTM
        b = float(self.cfg.initial_forget_bias)
        h = self.cfg.hidden_size

        with torch.no_grad():
            if hasattr(self.lstm, "bias_hh_l0"):
                self.lstm.bias_hh_l0[h : 2 * h].fill_(b)
            if hasattr(self.lstm, "bias_ih_l0"):
                self.lstm.bias_ih_l0[h : 2 * h].fill_(b)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the SequentialForecastLSTM model.

        Parameters
        ----------
        data : dict
            Dictionary containing input features as key-value pairs.

        Returns
        -------
        dict
            - ``y_hat``: last `predict_last_n` predictions (forecast part).
        """
        # Embedding of hindcast and forecast inputs; each: (seq_len, batch, emb_dim)
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # Concatenate along time dimension: (seq_total, batch, emb_dim)
        x_combined = torch.cat([x_h, x_f], dim=0)

        # LSTM expects (seq_len, batch, input_size)
        lstm_out, _ = self.lstm(x_combined)
        # -> (batch, seq_len, hidden_size)
        lstm_out = lstm_out.transpose(0, 1)

        head_in = self.dropout(lstm_out)

        # Head must return a dict with 'y_hat'
        pred = self.head(head_in)
        if not isinstance(pred, dict) or "y_hat" not in pred:
            raise ValueError("Head output must be a dictionary containing a 'y_hat' tensor.")

        y_hat_full = pred["y_hat"]  # (batch, seq_total, n_targets)

        # Keep only last predict_last_n steps (forecast part)
        y_hat_predicted = y_hat_full[:, -self._predict_last_n :, :]

        return {"y_hat": y_hat_predicted}
