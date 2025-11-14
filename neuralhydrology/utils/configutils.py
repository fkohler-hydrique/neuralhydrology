"""Utility script to generate config files from a base config and a defined set of variations."""

import itertools
from pathlib import Path
from typing import Dict, List, Any

from neuralhydrology.utils.config import Config


def create_config_files(
    base_config_path: Path,
    modify_dict: Dict[str, List[Any]],
    output_dir: Path,
) -> None:
    """Create multiple config files from a base config and a grid of hyperparameters.

    For every combination of values in ``modify_dict``, this function:

    1. Updates the base config with that combination.
    2. Builds a unique experiment name (base name + key/value suffixes).
    3. Writes the resulting config to ``output_dir/config_<i>.yml``.

    Parameters
    ----------
    base_config_path : Path
        Path to a base config file (.yml).
    modify_dict : dict[str, list]
        Mapping from parameter names to lists of possible values.
    output_dir : Path
        Directory where the generated configs will be stored.
    """
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Load base config once; subsequent `update_config` calls just override keys.
    base_config = Config(base_config_path)
    experiment_name = base_config.experiment_name
    option_names = list(modify_dict.keys())

    # Iterate over each possible combination of hyper parameters
    for i, options in enumerate(itertools.product(*modify_dict.values()), start=1):
        # 1) update config with the current choice of hyperparameters
        updates = dict(zip(option_names, options))
        base_config.update_config(updates)

        # 2) create a unique run name
        name = experiment_name
        for key, val in zip(option_names, options):
            name += f"_{key}{val}"
        base_config.update_config({"experiment_name": name})

        # 3) dump to disk
        base_config.dump_config(output_dir, f"config_{i}.yml")

    print(f"Finished. Configs are stored in {output_dir}")
