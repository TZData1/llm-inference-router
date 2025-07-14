import pandas as pd
import yaml
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_results(df: pd.DataFrame, results_dir: Path, filename_prefix: str, format: str = 'csv'):
    """
    Saves a DataFrame to the results directory with a timestamp.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        results_dir (Path): The directory to save the results file in.
        filename_prefix (str): The prefix for the filename (e.g., 'a1_detailed_results').
        format (str): The format to save in ('csv', 'parquet', 'json'). Defaults to 'csv'.
    """
    if not results_dir.is_dir():
        logger.error(f"Results directory does not exist: {results_dir}. Cannot save results.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.{format}"
    save_path = results_dir / filename

    try:
        if format == 'csv':
            df.to_csv(save_path, index=False)
        elif format == 'parquet':
            df.to_parquet(save_path, index=False)
        elif format == 'json':
            df.to_json(save_path, orient='records', indent=2)
        else:
            logger.error(f"Unsupported save format: {format}. Use 'csv', 'parquet', or 'json'.")
            return
        logger.info(f"Results saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {save_path}: {e}")

def load_results(results_dir: Path, filename_pattern: str, format: str = 'csv'):
    """
    Loads the most recent results file matching a pattern from the results directory.

    Args:
        results_dir (Path): The directory containing results files.
        filename_pattern (str): A glob pattern to match the filename (e.g., 'a1_summary_stats_*.csv').
        format (str): The format of the file ('csv', 'parquet', 'json'). Defaults to 'csv'.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if no file is found or error occurs.
    """
    if not results_dir.is_dir():
        logger.error(f"Results directory does not exist: {results_dir}. Cannot load results.")
        return None

    try:
        files = sorted(list(results_dir.glob(filename_pattern)), reverse=True)
        if not files:
            logger.warning(f"No files found matching pattern '{filename_pattern}' in {results_dir}")
            return None
        
        latest_file = files[0]
        logger.info(f"Loading latest results file: {latest_file}")

        if format == 'csv':
            # Directly read with single header row assumption
            return pd.read_csv(latest_file, header=0)
        elif format == 'parquet':
            return pd.read_parquet(latest_file)
        elif format == 'json':
            return pd.read_json(latest_file, orient='records')
        else:
            logger.error(f"Unsupported load format: {format}. Use 'csv', 'parquet', or 'json'.")
            return None
    except Exception as e:
        logger.error(f"Failed to load results from {results_dir} with pattern '{filename_pattern}': {e}")
        return None

def save_experiment_metadata(metadata: dict, results_dir: Path, filename_prefix: str):
    """
    Saves experiment metadata (e.g., configs used, run ID) to a YAML file.

    Args:
        metadata (dict): Dictionary containing metadata to save.
        results_dir (Path): The directory to save the metadata file in.
        filename_prefix (str): Prefix for the metadata filename.
    """
    if not results_dir.is_dir():
        logger.error(f"Results directory does not exist: {results_dir}. Cannot save metadata.")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_metadata_{timestamp}.yaml"
    save_path = results_dir / filename
    
    try:
        with open(save_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Experiment metadata saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata to {save_path}: {e}") 