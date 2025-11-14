from pathlib import Path

from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config


def start_training(cfg: Config) -> Path:
    """Start model training for the given configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    Path
        The run directory where outputs and checkpoints are stored.
    """
    # MC-LSTM is a special case where the head returns an empty string but the model
    # is still trained as a regression model.
    if cfg.head.lower() in ["regression", "gmm", "umal", "cmal", ""]:
        trainer = BaseTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")

    trainer.initialize_training()
    trainer.train_and_validate()
    return cfg.run_dir
