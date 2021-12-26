import argparse
import toml
import glob
import logging

logger = logging.getLogger(__name__)


def read():
    """Reads arguments from config file (from a path passed to CLI)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf8") as config_file:
        config = toml.load(config_file)

    logger.info(config)

    return config


def _replace_within_config(config, key_part, parse_function):
    for key in config.keys():
        if isinstance(config[key], dict):
            _replace_within_config(config[key], key_part, parse_function)
        elif key_part in key:
            config[key] = parse_function(config[key])
        else:
            pass
