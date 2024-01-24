# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import hashlib
import importlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type, Union


@dataclass
class Experiment:
    optimizer: Union[dict, Type]
    benchmark: Union[dict, Type]
    n_evaluations: int
    n_studies: int
    compute: str

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Experiment):
            return False

        return hash_experiment_config(self) == hash_experiment_config(__o)


def _parse_simple_type(value: Any) -> Union[str, int, float]:
    """Transform any input to the closest simple type representation.

    String representations of numeric values are casted to int/float respectively.
    For objects without a human friendly string representation, the combination of
    module and class name are used.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            if str(value).startswith("<class"):
                return f"{value.__module__}.{value.__name__}"
            elif "object at" in str(value):
                return f"{value.__class__.__module__}.{value.__class__.__name__}"
            else:
                return str(value)


def parse_experiment_config(config):
    """Transform input into a representation with exclusively simple types.

    Complex types are converted to string representations according to
    `_parse_simple_type`.
    """
    if isinstance(config, dict):
        return {k: parse_experiment_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [parse_experiment_config(v) for v in config]
    else:
        return _parse_simple_type(config)


def hash_experiment_config(config: Experiment, short: bool = False) -> str:
    """SHA-256 hash of the given experiment configuration without the compute attribute.

    Args:
        config: The experiment configuration of which to compute the fingerprint.
        short: If true return the first seven characters of the resulting hash.
    """
    # Without this, compute is popped by reference from the passed data class instance.
    config_dict = deepcopy(config.__dict__)
    config_dict.pop("compute")

    # Exclude any data paths that might be different depending on the compute target
    # from computing the unique configuration fingerprint
    if (
        isinstance(config_dict["benchmark"], dict)
        and "kwargs" in config_dict["benchmark"]
    ):
        # HPOBench
        config_dict["benchmark"]["kwargs"].pop("data_dir", None)
        # FCNet
        if "target_task_file" in config_dict["benchmark"]["kwargs"]:
            config_dict["benchmark"]["kwargs"]["target_task_file"] = Path(
                config_dict["benchmark"]["kwargs"]["target_task_file"]
            ).name
        if "meta_task_files" in config_dict["benchmark"]["kwargs"]:
            config_dict["benchmark"]["kwargs"]["meta_task_files"] = [
                Path(mtf).name
                for mtf in config_dict["benchmark"]["kwargs"]["meta_task_files"]
            ]

    parsed_config = parse_experiment_config(config_dict)

    config_hash = hashlib.sha256(json.dumps(parsed_config).encode()).hexdigest()

    if short:
        config_hash = config_hash[:7]

    return config_hash


def get_experiments_config_from_module(module: str) -> Dict[str, Experiment]:
    """Return value of the global variable `EXPERIMENTS` from a given module, or raise
    an informative `ValueError` in case no such variable is defined.
    """
    _module = importlib.import_module(module)

    if not hasattr(_module, "EXPERIMENTS"):
        raise ValueError(
            f"Module {module} is missing the global variable EXPERIMENTS that contains "
            + "all experiment configurations."
        )

    return _module.EXPERIMENTS  # type: ignore
