#!/usr/bin/env python3
"""
Configuration module for MIC analysis.
Provides centralized configuration management via YAML/TOML files and environment variables.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define project paths
try:
    # Assumes config.py is in PROJECT_ROOT/src/data
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive session)
    PROJECT_ROOT = Path(os.getcwd()).resolve()
    logger.warning(f"__file__ not defined. Assuming project root is current working directory: {PROJECT_ROOT}")

CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


def get_default_config() -> Dict[str, Any]:
    """Returns the hardcoded default configuration."""
    return {
        "project": {
            "root": str(PROJECT_ROOT),
        },
        "data": {
            "raw_dir": str(PROJECT_ROOT / "data" / "raw"),
            "interim_dir": str(PROJECT_ROOT / "data" / "interim"),
            "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
            "external_dir": str(PROJECT_ROOT / "data" / "external"),
        },
        "database": {
            "path": str(PROJECT_ROOT / "data" / "processed" / "mic_analysis.duckdb"),
        },
        "loading": {
            "proquest": {
                "enabled": True,
                "source_subdir": "", # Relative to data.raw_dir, empty means raw_dir itself
                "filename_prefix": "ProQuestDocuments",
                "recursive": True,
                "target_table": "raw.articles",
                "excluded_subdirs": [], # Relative to the proquest source dir
            },
            "nyt": {
                "enabled": True,
                "source_subdir": "New York Times/2011-2014", # Relative to data.raw_dir
                "filename_prefix": "sorted",
                "recursive": True,
                "target_table": "raw.parsed_articles",
                "bad_keys_table": "staging.bad_keys",
                "article_separator": "---------------------------------------------------------------",
                "text_start_marker": ">>>>>>>>>>>>>>>>>>>>>>",
                "text_end_marker": "<<<<<<<<<<<<<<<<<<<<<<"
            }
        },
        "system": {
            "parallel_workers": max(1, os.cpu_count() - 1)
        }
    }

def merge_config(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Recursively merge source configuration into target configuration.
    Overwrites existing keys in target with values from source.
    Handles nested dictionaries.

    Args:
        target: Target dictionary to update.
        source: Source dictionary with new values.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            # If both target and source have a dict for this key, recurse
            merge_config(target[key], value)
        else:
            # Otherwise, overwrite the target value with the source value
            target[key] = value


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, starting with hardcoded defaults,
    then merging the default YAML file (if exists), and finally merging
    a custom YAML file (if provided).

    Args:
        config_path: Path to a custom configuration file (optional).

    Returns:
        Dictionary containing the final configuration values.
    """
    # Start with hardcoded defaults
    config = get_default_config()
    logger.debug("Loaded hardcoded default config.")

    # Try to load from default config file (e.g., config/default.yaml)
    if DEFAULT_CONFIG_PATH.exists():
        try:
            with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
                default_yaml_config = yaml.safe_load(f)
                if default_yaml_config and isinstance(default_yaml_config, dict):
                    merge_config(config, default_yaml_config)
                    logger.info(f"Loaded and merged default configuration from {DEFAULT_CONFIG_PATH}")
                elif default_yaml_config:
                     logger.warning(f"Default configuration file {DEFAULT_CONFIG_PATH} does not contain a valid dictionary structure. Skipping merge.")
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing default configuration file {DEFAULT_CONFIG_PATH}: {e}. Using hardcoded defaults.")
        except Exception as e:
            logger.warning(f"Error loading default configuration file {DEFAULT_CONFIG_PATH}: {e}. Using hardcoded defaults.")
    else:
        logger.info(f"Default configuration file not found at {DEFAULT_CONFIG_PATH}. Using hardcoded defaults.")

    # Try to load from specified custom config file if provided
    if config_path:
        custom_config_file = Path(config_path)
        if custom_config_file.exists():
            try:
                with open(custom_config_file, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
                    if custom_config and isinstance(custom_config, dict):
                        merge_config(config, custom_config)
                        logger.info(f"Loaded and merged custom configuration from {config_path}")
                    elif custom_config:
                         logger.warning(f"Custom configuration file {config_path} does not contain a valid dictionary structure. Skipping merge.")

            except yaml.YAMLError as e:
                logger.warning(f"Error parsing custom configuration file {config_path}: {e}. Previous configuration state retained.")
            except Exception as e:
                logger.warning(f"Error loading custom configuration file {config_path}: {e}. Previous configuration state retained.")
        else:
            logger.warning(f"Custom configuration file not found: {config_path}. Using previously loaded configuration.")

    # Handle relative paths by resolving them against PROJECT_ROOT
    resolve_relative_paths(config, PROJECT_ROOT)

    return config


def resolve_relative_paths(config: Dict[str, Any], base_path: Path) -> None:
    """
    Resolve relative paths in configuration against the given base path.
    Modifies the config dictionary in place.

    Args:
        config: Configuration dictionary to process.
        base_path: Base path to resolve relative paths against.
    """
    # Resolve data directory paths
    for key, value in config.get('data', {}).items():
        if key.endswith('_dir') and value and not os.path.isabs(value):
            config['data'][key] = str((base_path / value).resolve())
            logger.debug(f"Resolved relative path '{value}' to '{config['data'][key]}'")

    # Resolve database path
    db_path = config.get('database', {}).get('path')
    if db_path and not os.path.isabs(db_path):
        config['database']['path'] = str((base_path / db_path).resolve())
        logger.debug(f"Resolved relative database path '{db_path}' to '{config['database']['path']}'")


def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update configuration with values from command line arguments.
    Only overrides if the argument was actually provided by the user.

    Args:
        config: Configuration dictionary to update.
        args: Command line arguments namespace.

    Returns:
        Updated configuration dictionary.
    """
    # Database path
    if hasattr(args, 'db_path') and args.db_path:
        path = Path(args.db_path)
        config['database']['path'] = str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
        logger.info(f"Overriding database path from command line: {config['database']['path']}")

    # Raw data directory
    if hasattr(args, 'data_dir') and args.data_dir:
        path = Path(args.data_dir)
        config['data']['raw_dir'] = str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
        logger.info(f"Overriding raw data directory from command line: {config['data']['raw_dir']}")

    # ProQuest specific args
    if hasattr(args, 'proquest_dir') and args.proquest_dir:
        config['loading']['proquest']['source_subdir'] = args.proquest_dir
        logger.info(f"Overriding ProQuest source subdir from command line: {args.proquest_dir}")
    # Check if --recursive or --no-recursive was explicitly used for ProQuest
    if hasattr(args, 'proquest_recursive') and args.proquest_recursive is not None:
        config['loading']['proquest']['recursive'] = args.proquest_recursive
        logger.info(f"Overriding ProQuest recursive setting from command line: {args.proquest_recursive}")

    # NYT specific args
    if hasattr(args, 'nyt_dir') and args.nyt_dir:
        config['loading']['nyt']['source_subdir'] = args.nyt_dir
        logger.info(f"Overriding NYT source subdir from command line: {args.nyt_dir}")
    # Check if --recursive or --no-recursive was explicitly used for NYT
    if hasattr(args, 'nyt_recursive') and args.nyt_recursive is not None:
        config['loading']['nyt']['recursive'] = args.nyt_recursive
        logger.info(f"Overriding NYT recursive setting from command line: {args.nyt_recursive}")

    # Enable/disable loading types
    # If --load-all is used, it implies both are true unless explicitly disabled
    load_all = getattr(args, 'load_all', False)
    # If --load-proquest or --load-nyt is explicitly mentioned, it overrides --load-all for that type
    proquest_explicitly_set = hasattr(args, 'load_proquest') and args.load_proquest is not None
    nyt_explicitly_set = hasattr(args, 'load_nyt') and args.load_nyt is not None

    if proquest_explicitly_set:
         config['loading']['proquest']['enabled'] = args.load_proquest
    elif load_all:
         config['loading']['proquest']['enabled'] = True
    # else: keep config file value

    if nyt_explicitly_set:
         config['loading']['nyt']['enabled'] = args.load_nyt
    elif load_all:
         config['loading']['nyt']['enabled'] = True
    # else: keep config file value

    logger.info(f"ProQuest loading enabled: {config['loading']['proquest']['enabled']}")
    logger.info(f"NYT loading enabled: {config['loading']['nyt']['enabled']}")

    return config


def get_config(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Get configuration combining defaults, config files, and command line arguments.

    Args:
        args: Command line arguments namespace (optional).

    Returns:
        Final configuration dictionary.
    """
    # Determine config file path from args if provided
    config_path = args.config if args and hasattr(args, 'config') and args.config else None

    # Load configuration from files (hardcoded -> default.yaml -> custom.yaml)
    config = load_config(config_path)

    # Update with command line arguments if provided
    if args:
        config = update_config_from_args(config, args)

    # Final validation or adjustments can happen here
    # e.g., ensure directories exist if needed by the calling script

    return config


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add configuration-related arguments to an argument parser.

    Args:
        parser: Argument parser to add arguments to.

    Returns:
        Updated argument parser.
    """
    # General config
    parser.add_argument('--config', type=str, help='Path to custom YAML configuration file.')
    parser.add_argument('--db_path', type=str, help='Override path to DuckDB database file.')
    parser.add_argument('--data_dir', type=str, help='Override path to the main raw data directory.')

    # Loading type selection
    load_group = parser.add_argument_group('Loading Selection (Default: Both enabled via config)')
    load_group.add_argument('--load-proquest', action='store_true', dest='load_proquest',
                           help='Enable loading ProQuest files.')
    load_group.add_argument('--no-load-proquest', action='store_false', dest='load_proquest',
                           help='Disable loading ProQuest files.')
    load_group.add_argument('--load-nyt', action='store_true', dest='load_nyt',
                           help='Enable loading NYT-style files.')
    load_group.add_argument('--no-load-nyt', action='store_false', dest='load_nyt',
                           help='Disable loading NYT-style files.')
    load_group.add_argument('--load-all', action='store_true',
                           help='Convenience flag to ensure both loaders are enabled (can be overridden by --no-load-...).')
    parser.set_defaults(load_proquest=None, load_nyt=None)


    # ProQuest specific overrides
    pq_group = parser.add_argument_group('ProQuest Loading Overrides')
    pq_group.add_argument('--proquest-dir', type=str,
                          help='Override ProQuest source sub-directory (relative to raw data dir).')
    pq_group.add_argument('--proquest-recursive', action='store_true', dest='proquest_recursive', 
                          help='Enable recursive search for ProQuest files.')
    pq_group.add_argument('--no-proquest-recursive', action='store_false', dest='proquest_recursive',
                          help='Disable recursive search for ProQuest files.')
    parser.set_defaults(proquest_recursive=None)

    # NYT specific overrides
    nyt_group = parser.add_argument_group('NYT Loading Overrides')
    nyt_group.add_argument('--nyt-dir', type=str,
                         help='Override NYT source sub-directory (relative to raw data dir).')
    nyt_group.add_argument('--nyt-recursive', action='store_true', dest='nyt_recursive',
                         help='Enable recursive search for NYT files.')
    nyt_group.add_argument('--no-nyt-recursive', action='store_false', dest='nyt_recursive',
                         help='Disable recursive search for NYT files.')
    parser.set_defaults(nyt_recursive=None)

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Config Loading")
    parser = add_config_args(parser)
    # Add a dummy arg to test merging
    parser.add_argument('--dummy', action='store_true', help='Dummy flag')

    # Test cases
    print("\n--- Test Case 1: Hardcoded Defaults Only ---")
    args1 = parser.parse_args([])
    cfg1 = get_config(args1)
    # print(yaml.dump(cfg1, indent=2)) # Requires PyYAML

    print(f"\n--- Test Case 2: With Relative Paths ---")
    dummy_config = {
        "project": {"root": "."},
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed"
        },
        "database": {
            "path": "data/processed/test.duckdb"
        }
    }
    
    # Test relative path resolution
    resolve_relative_paths(dummy_config, PROJECT_ROOT)
    print(f"Resolved raw_dir: {dummy_config['data']['raw_dir']}")
    print(f"Resolved database path: {dummy_config['database']['path']}")

    print("\n--- Test Case 3: Command Line Overrides ---")
    args3 = parser.parse_args([
        '--db_path', 'data/temp/test.db',  # Relative path
        '--data_dir', '/absolute/path/to/data',  # Absolute path
        '--proquest-dir', 'New York Times',
    ])
    cfg3 = get_config(args3)
    print(f"DB Path: {cfg3['database']['path']}")
    print(f"Raw Dir: {cfg3['data']['raw_dir']}")
    print(f"ProQuest Source Dir: {cfg3['loading']['proquest']['source_subdir']}")