from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer


class HandoffForecastLSTM(BaseModel):
    """Encoder/decoder LSTM model with state handoff for forecasting.

    Workflow
    --------
    1. Hindcast LSTM runs over the hindcast sequence (and optional overlap).
    2. Its final (h, c) state is passed through a fully-connected handoff network.
    3. The handoff network output initializes the forecast LSTM state.
    4. Forecast LSTM rolls out over the forecast period (optionally with overlap).
    5. A single head is applied to the forecast LSTM sequence to produce `y_hat`.

    Overlap
    -------
    - The overlap between hindcast and forecast is controlled by ``forecast_overlap``.
    - If you use regularization like ``ForecastOverlapMSERegularization``, it will consume
      overlap outputs present in the prediction dict; here we at least return `y_hat`
      over the forecast period (post-overlap).

    Parameters
    ----------
    cfg : Config
        Run configuration.

    Raises
    ------
    ValueError
        If `state_handoff_network` is not specified in the config.
    """

    # Submodules that can be used for fine-tuning
    module_parts = [
        "hindcast_embedding_net",
        "forecast_embedding_net",
        "hindcast_lstm",
        "handoff_net",
        "forecast_lstm",
        "dropout",
        "head",
    ]

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg=cfg)

        # Handle predict_last_n as int/dict (single-frequency)
        predict_last_n = cfg.predict_last_n
        if isinstance(predict_last_n, dict):
            predict_last_n = next(iter(predict_last_n.values()))
        self._predict_last_n: int = int(predict_last_n)

        # Hindcast length = full seq length - forecast seq length
        self.initial_hindcast_seq_length = cfg.seq_length - cfg.forecast_seq_length

        # Embedding networks for hindcast and forecast inputs
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type="hindcast")
        self.forecast_embedding_net = InputLayer(cfg, embedding_type="forecast")

        # Hindcast LSTM
        self._hindcast_hidden_size = cfg.hindcast_hidden_size
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=self._hindcast_hidden_size,
            num_layers=cfg.num_layers,
        )

        # Forecast LSTM
        self._forecast_hidden_size = cfg.forecast_hidden_size
        self.forecast_lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=self._forecast_hidden_size,
            num_layers=cfg.num_layers,
        )

        # State handoff network (required)
        if not cfg.state_handoff_network:
            raise ValueError(
                "The handoff forecast LSTM requires a state_handoff_network "
                "specified in the config file."
            )

        self._hiddens_handoff = cfg.state_handoff_network["hiddens"]
        activation_name = cfg.state_handoff_network["activation"].lower()

        if activation_name == "relu":
            activation_layer = nn.ReLU()
        elif activation_name == "sigmoid":
            activation_layer = nn.Sigmoid()
        elif activation_name == "id":
            activation_layer = nn.Identity()
        elif activation_name == "tanh":
            activation_layer = nn.Tanh()
        elif activation_name == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation '{activation_name}' for state_handoff_network.")

        handoff_layers: list[nn.Module] = [
            nn.Linear(self._hindcast_hidden_size * 2, self._hiddens_handoff[0]),
            activation_layer,
            nn.Dropout(cfg.state_handoff_network["dropout"]),
        ]
        for i, hidden_size in enumerate(self._hiddens_handoff[1:]):
            handoff_layers.append(nn.Linear(self._hiddens_handoff[i], hidden_size))
            handoff_layers.append(activation_layer)
            handoff_layers.append(nn.Dropout(cfg.state_handoff_network["dropout"]))

        # Final layer maps to concatenated (h, c) for forecast LSTM
        handoff_layers.append(
            nn.Linear(self._hiddens_handoff[-1], self._forecast_hidden_size * 2)
        )
        self.handoff_net = nn.Sequential(*handoff_layers)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self._forecast_hidden_size, n_out=self.output_size)

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _reset_parameters(self) -> None:
        """Special initialization of certain model weights (forget gate bias)."""
        if self.cfg.initial_forget_bias is None:
            return

        b = float(self.cfg.initial_forget_bias)
        h_h = self._hindcast_hidden_size
        h_f = self._forecast_hidden_size

        with torch.no_grad():
            # Hindcast LSTM
            if hasattr(self.hindcast_lstm, "bias_hh_l0"):
                self.hindcast_lstm.bias_hh_l0[h_h : 2 * h_h].fill_(b)
            if hasattr(self.hindcast_lstm, "bias_ih_l0"):
                self.hindcast_lstm.bias_ih_l0[h_h : 2 * h_h].fill_(b)

            # Forecast LSTM
            if hasattr(self.forecast_lstm, "bias_hh_l0"):
                self.forecast_lstm.bias_hh_l0[h_f : 2 * h_f].fill_(b)
            if hasattr(self.forecast_lstm, "bias_ih_l0"):
                self.forecast_lstm.bias_ih_l0[h_f : 2 * h_f].fill_(b)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the HandoffForecastLSTM model.

        Parameters
        ----------
        data : dict
            Dictionary containing input features as key-value pairs.

        Returns
        -------
        dict
            - ``y_hat``: forecast predictions for the non-overlap part
              (shape: [batch, forecast_seq_length - forecast_overlap, n_targets]).
        """
        # Embedding: each (seq_len, batch, emb_dim)
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # --------------------------------------
        # 1) Hindcast LSTM (past → issue time)
        # --------------------------------------
        # Main hindcast sequence
        hindcast_main = x_h[: self.initial_hindcast_seq_length, ...]
        lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.hindcast_lstm(hindcast_main)
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)  # (batch, seq_hind_main, h_h)

        # Optional overlap part of hindcast
        if x_h.shape[0] > self.initial_hindcast_seq_length:
            hindcast_overlap = x_h[self.initial_hindcast_seq_length :, ...]
            lstm_output_hindcast_overlap, _ = self.hindcast_lstm(
                hindcast_overlap, (h_n_hindcast, c_n_hindcast)
            )
            lstm_output_hindcast_overlap = lstm_output_hindcast_overlap.transpose(0, 1)
        else:
            lstm_output_hindcast_overlap = None

        # --------------------------------------
        # 2) State handoff → initial state for forecast LSTM
        # --------------------------------------
        initial_state = self.handoff_net(torch.cat([h_n_hindcast, c_n_hindcast], dim=-1))
        h_n_handoff, c_n_handoff = initial_state.chunk(2, dim=-1)
        h_n_handoff = h_n_handoff.contiguous()
        c_n_handoff = c_n_handoff.contiguous()

        # --------------------------------------
        # 3) Forecast LSTM
        # --------------------------------------
        lstm_output_forecast, (h_n_forecast, c_n_forecast) = self.forecast_lstm(
            x_f, (h_n_handoff, c_n_handoff)
        )
        lstm_output_forecast = lstm_output_forecast.transpose(0, 1)  # (batch, seq_forecast_total, h_f)

        # Separate overlap vs. non-overlap part in forecast sequence
        overlap = self.cfg.forecast_overlap or 0
        if overlap > 0:
            lstm_output_forecast_overlap = lstm_output_forecast[:, :overlap, :]
            lstm_output_forecast_main = lstm_output_forecast[:, overlap:, :]
        else:
            lstm_output_forecast_overlap = None
            lstm_output_forecast_main = lstm_output_forecast

        # --------------------------------------
        # 4) Apply head to forecast main sequence
        # --------------------------------------
        y_forecast = self.head(self.dropout(lstm_output_forecast_main))

        if not isinstance(y_forecast, dict) or "y_hat" not in y_forecast:
            raise ValueError("Head output must be a dictionary containing a 'y_hat' tensor.")

        y_hat = y_forecast["y_hat"]  # (batch, seq_forecast_main, n_targets)

        # Note:
        # We only return 'y_hat' here because the training loss and regularization
        # in your current setup expect that key. If in the future you need
        # overlap-based regularization, we can extend this dict with:
        # y_forecast_overlap, y_hindcast_overlap, etc.

        return {"y_hat": y_hat}
