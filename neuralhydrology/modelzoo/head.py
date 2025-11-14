import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_head(cfg: Config, n_in: int, n_out: int) -> nn.Module:
    """Return a head module as specified in the configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.

    Returns
    -------
    nn.Module
        Instantiated model head.
    """
    head_name = (cfg.head or "").lower()

    if head_name == "regression":
        return Regression(n_in=n_in, n_out=n_out, activation=cfg.output_activation)
    if head_name == "gmm":
        return GMM(n_in=n_in, n_out=n_out)
    if head_name == "umal":
        return UMAL(n_in=n_in, n_out=n_out)
    if head_name == "cmal":
        return CMAL(n_in=n_in, n_out=n_out)
    if head_name == "":
        raise ValueError(
            f"No 'head' specified in the config but it is required for model '{cfg.model}'."
        )

    raise NotImplementedError(f"Head '{cfg.head}' not implemented or not linked in `get_head()`.")


class Regression(nn.Module):
    """Single-layer regression head with configurable output activation.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation : {'linear', 'relu', 'softplus'}, optional
        Output activation function.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear") -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(n_in, n_out)]

        act = activation.lower()
        if act != "linear":
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(
                    "Ignored unsupported output activation '%s' and used 'linear' instead.",
                    activation,
                )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of regression head.

        Parameters
        ----------
        x : torch.Tensor
            Latent representation from previous model layers.

        Returns
        -------
        dict
            Dictionary with a single key ``'y_hat'`` containing predictions.
        """
        return {"y_hat": self.net(x)}


class GMM(nn.Module):
    """Gaussian Mixture Density Network head.

    A mixture density network with Gaussian components. Uses a hidden layer, exponential activation
    for variance estimates, and softmax for mixture weights.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons (3 × num_components).
    n_hidden : int, optional
        Hidden layer size (default: 100).
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of GMM head.

        Parameters
        ----------
        x : torch.Tensor
            Latent representation from previous model layers.

        Returns
        -------
        dict
            Dictionary with:
            - 'mu'    : component means
            - 'sigma' : component standard deviations
            - 'pi'    : mixture weights
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        # split output into mu, sigma and weights
        mu, sigma, pi = h.chunk(3, dim=-1)

        return {
            "mu": mu,
            "sigma": torch.exp(sigma) + self._eps,
            "pi": torch.softmax(pi, dim=-1),
        }


class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians (CMAL) head.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons (4 × num_components).
    n_hidden : int, optional
        Hidden layer size (default: 100).
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(beta=2.0)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of CMAL head.

        Parameters
        ----------
        x : torch.Tensor
            Latent representation from previous model layers.

        Returns
        -------
        dict
            Dictionary with:
            - 'mu'  : component means
            - 'b'   : scale parameters (> 0)
            - 'tau' : skewness parameters in (0, 1)
            - 'pi'  : mixture weights (> 0, sum to 1)
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restriction on means
        b = self._softplus(b_latent) + self._eps  # scale > 0
        t = (1.0 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 < tau < 1
        p = (1.0 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps  # pi > 0, sum(pi) ~ 1

        return {"mu": m, "b": b, "tau": t, "pi": p}


class UMAL(nn.Module):
    """Uncountable Mixture of Asymmetric Laplacians (UMAL) head.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons (2 × output_size → mean + scale).
    n_hidden : int, optional
        Hidden layer size (default: 100).
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        # empirically chosen upper bound for scale; may be adapted for unnormalized outputs
        self._upper_bound_scale = 0.5
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of UMAL head.

        Parameters
        ----------
        x : torch.Tensor
            Latent representation from previous model layers.

        Returns
        -------
        dict
            Dictionary with:
            - 'mu' : mean parameter
            - 'b'  : scale parameter (bounded on both sides)
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent = h.chunk(2, dim=-1)

        m = m_latent  # means are unconstrained
        b = self._upper_bound_scale * torch.sigmoid(b_latent) + self._eps

        return {"mu": m, "b": b}
