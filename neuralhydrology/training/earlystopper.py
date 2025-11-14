from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopper:
    """Simple early stopping helper.

    Stops training if the validation loss doesn't improve by at least ``min_delta``
    for ``patience`` consecutive validation checks.

    Parameters
    ----------
    patience : int
        Number of validation checks with no sufficient improvement before stopping.
    min_delta : float
        Minimum loss improvement required to be considered as "better".
    """

    patience: int
    min_delta: float

    def __post_init__(self) -> None:
        self._counter: int = 0
        self._min_validation_loss: float = float("inf")

    def check_early_stopping(self, validation_loss: float) -> bool:
        """Update internal state and decide whether training should stop.

        Parameters
        ----------
        validation_loss : float
            Current validation loss.

        Returns
        -------
        bool
            True if early stopping criterion is met, False otherwise.
        """
        if validation_loss < self._min_validation_loss - self.min_delta:
            self._min_validation_loss = validation_loss
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    def reset(self) -> None:
        """Reset early stopping state (e.g., when restarting training)."""
        self._counter = 0
        self._min_validation_loss = float("inf")
