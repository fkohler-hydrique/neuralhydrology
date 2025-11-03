from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class SequentialForecastLSTM(BaseModel):
    """A forecasting model that uses a single LSTM sequence with multiple embedding layers.

    This is a forecasting model that uses a single sequential (LSTM) model that rolls 
    out through both the hindcast and forecast sequences. The difference between this
    and a standard ``CudaLSTM`` is (1) this model uses both hindcast and forecast
    input features, and (2) it uses a separate embedding network for the hindcast
    period and the forecast period. 
    
    Do not use this model with ``forecast_overlap`` > 0.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Raises
    ------
    ValueError if forecast_overlap > 0
    ValueError if forecast and hindcast embedding nets have different output sizes.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'lstm', 'dropout', 'head']
    def __init__(self, cfg: Config):
        super(SequentialForecastLSTM, self).__init__(cfg=cfg)
        self._predict_last_n = cfg.predict_last_n
        if cfg.forecast_overlap:
            raise ValueError('Forecast overlap cannot be set for a sequential forecasting model. '
                             'Please set to None or remove from config file.')

        # output : (L_subseq, batch, hidden_emb)
        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        if self.forecast_embedding_net.output_size != self.hindcast_embedding_net.output_size:
            raise ValueError('Forecast and hindcast embedding nets must have the same output size when using a sequential forecast LSTM.')

        self.lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers
            # dropout=0.02
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias



    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the SequentialForecastLSTM model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary. Key `'y_hat'` contains
            predictions with shape (batch, predict_last_n, n_targets).
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)
        # when no embedding is provided, returns a tensor of shape 
        # (seq_L(hind|fore), batch size, number of features)

        # print("\nShape of input (SequentialLSTM), after 'embedding' ")
        # print(x_h.shape)
        # print(x_f.shape)

        # combine sequences on seq-dim (seq, batch, hidden or num_Feat when not embedding)
        x_combined = torch.cat([x_h, x_f], dim=0)

        # run LSTM -> (seq_in_LSTM, batch, hidden or num_feat)
        lstm_out, (h_n, c_n) = self.lstm(x_combined)
        lstm_out = lstm_out.transpose(0, 1)
        # returns (batch, seq_tot, hidden_lstm)
        # print("LSTM out", lstm_out.shape)

        lstm_forecast_part = lstm_out[:, -self._predict_last_n:, :]
        head_in = self.dropout(lstm_forecast_part)

        # Run head - heads may return a tensor or a dict with 'y_hat'
        head_out = self.head(head_in)

        # extract raw tensor from head_out
        if isinstance(head_out, dict):
            if 'y_hat' in head_out:
                y_raw = head_out['y_hat']
            else:
                # try to find first tensor value
                vals = [v for v in head_out.values() if isinstance(v, torch.Tensor)]
                if not vals:
                    raise RuntimeError("Head returned a dict without tensor outputs")
                y_raw = vals[0]
        elif isinstance(head_out, torch.Tensor):
            y_raw = head_out
        else:
            raise RuntimeError("Head returned unsupported type: %s" % type(head_out))

        # normalize to (batch, predict_last_n, n_targets)
        B = y_raw.shape[0]
        pred_n = self._predict_last_n
        n_targets = self.output_size

        # possible shapes handling
        if y_raw.dim() == 3:
            # (batch, seq_out, out_dim)
            seq_out, out_dim = y_raw.shape[1], y_raw.shape[2]
            if seq_out == pred_n and out_dim == n_targets:
                y_hat = y_raw
            elif seq_out == 1 and out_dim == (pred_n * n_targets):
                y_hat = y_raw.reshape(B, pred_n, n_targets)
            elif seq_out == 1 and out_dim == pred_n and n_targets == 1:
                y_hat = y_raw.squeeze(1).unsqueeze(-1)  # (batch, pred_n, 1)
            elif seq_out == pred_n and out_dim == 1 and n_targets == 1:
                y_hat = y_raw  # already (batch, pred_n, 1)
            else:
                raise ValueError(f"Cannot interpret head output shape {tuple(y_raw.shape)} as (batch,{pred_n},{n_targets})")
        elif y_raw.dim() == 2:
            # (batch, out_flat)
            out_flat = y_raw.shape[1]
            if out_flat == pred_n * n_targets:
                y_hat = y_raw.reshape(B, pred_n, n_targets)
            elif out_flat == pred_n and n_targets == 1:
                y_hat = y_raw.unsqueeze(-1)
            elif out_flat == n_targets:
                # single timestep returned
                y_hat = y_raw.unsqueeze(1)
            else:
                raise ValueError(f"Cannot interpret head output shape {tuple(y_raw.shape)} as (batch,{pred_n},{n_targets})")
        else:
            raise ValueError(f"Unsupported head output dimension: {y_raw.dim()}")

        return {'y_hat': y_hat}
    