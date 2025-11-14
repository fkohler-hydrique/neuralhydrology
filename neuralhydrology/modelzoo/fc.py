from typing import List

import numpy as np
import torch
import torch.nn as nn


class FC(nn.Module):
    """Auxiliary class to build (multi-layer) fully-connected networks.

    This class is used to build fully-connected embedding networks for static and/or dynamic input data.
    Use the config argument ``statics/dynamics_embedding`` to specify the architecture of the embedding network.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_sizes : list[int]
        Sizes of hidden and output layers. The last entry defines the output size.
    activation : {'tanh', 'sigmoid', 'relu', 'linear'}, optional
        Activation function for intermediate layers (default: 'tanh').
    dropout : float, optional
        Dropout rate in intermediate layers (default: 0.0).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(hidden_sizes) == 0:
            raise ValueError(
                "hidden_sizes must contain at least one entry to create a fully-connected net."
            )

        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0.")

        self.output_size = hidden_sizes[-1]
        hidden_only = hidden_sizes[:-1]

        activation_layer = self._get_activation(activation)

        # Build network: [Linear -> Activation -> Dropout]* + Final Linear
        layers: list[nn.Module] = []
        if hidden_only:
            prev_size = input_size
            for hidden_size in hidden_only:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(activation_layer)
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, self.output_size))
        else:
            # single-layer network
            layers.append(nn.Linear(input_size, self.output_size))

        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Return activation module from name."""
        name_l = name.lower()
        if name_l == "tanh":
            return nn.Tanh()
        if name_l == "sigmoid":
            return nn.Sigmoid()
        if name_l == "relu":
            return nn.ReLU()
        if name_l == "linear":
            return nn.Identity()
        raise NotImplementedError(
            f"{name} is currently not supported as activation in this class."
        )

    def _reset_parameters(self) -> None:
        """Special initialization of linear layers (uniform with fan-in based gain)."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                n_in = layer.weight.shape[1]
                gain = np.sqrt(3.0 / n_in)
                nn.init.uniform_(layer.weight, -gain, gain)
                nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass on the FC network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., input_size].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [..., output_size], where ``output_size`` is the last value in ``hidden_sizes``.
        """
        return self.net(x)
