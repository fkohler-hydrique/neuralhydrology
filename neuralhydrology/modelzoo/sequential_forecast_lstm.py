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
        self._predict_last_n = cfg.predict_last_n
        super(SequentialForecastLSTM, self).__init__(cfg=cfg)
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
            num_layers=cfg.num_layers,
            dropout=cfg.lstm_dropout
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights.

        If `initial_forget_bias` is specified, we initialize the forget gate bias to this value.
        """
        if self.cfg.initial_forget_bias is not None:
            # Initialize the forget gate bias to a constant value.
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
            A dictionary containing the model predictions.
                - y_hat: last 'predict_last_n' predictions
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

        head_in = self.dropout(lstm_out)

        # Run head - heads may return a tensor or a dict with 'y_hat'
        pred = self.head(head_in)

        # The head might return a dict, and we are interested in 'y_hat'.
        # If it's not a dict, we assume the tensor is the prediction.
        if not isinstance(pred, dict) or 'y_hat' not in pred:
            raise ValueError("Head output must be a dictionary containing a 'y_hat' tensor")
        
        y_hat_full = pred['y_hat']
        
        # y_hat_hindcast = y_hat_full[:, :-self._predict_last_n, :]
        y_hat_predicted = y_hat_full[:, -self._predict_last_n:, :]

        # The training loop expects 'y_hat', so we provide the concatenated version.
        # Specific evaluation can be done on the hindcast/forecast parts.
        return {
            'y_hat': y_hat_predicted
        }
    