import yaml
from pathlib import Path
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricNormalizer:
    """
    Normalize metrics to [0, 1] range using dataset-specific min/max bounds
    loaded from a configuration file. Assumes 1 is best, 0 is worst.
    """
    _DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config' / 'metrics.yaml'

    def __init__(self, config_path=None):
        """
        Initialize the normalizer by loading min/max bounds from a YAML config file.

        Args:
            config_path (str or Path, optional): Path to the metrics bounds YAML file.
                                                 Defaults to config/metrics.yaml in the project root.
        """
        self.config_path = Path(config_path) if config_path else self._DEFAULT_CONFIG_PATH
        self.bounds = self._load_bounds()
        if not self.bounds:
             logger.warning(f"Metric bounds configuration could not be loaded from {self.config_path}. Normalization may use defaults or fail.")

    def _load_bounds(self):
        """Load normalization bounds from the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                bounds_data = yaml.safe_load(f)
                logger.info(f"Successfully loaded metric bounds from {self.config_path}")
                if isinstance(bounds_data, dict):
                    return bounds_data
                else:
                    logger.error(f"Invalid format in metric bounds file {self.config_path}. Expected a dictionary.")
                    return {}
        except FileNotFoundError:
            logger.error(f"Metric bounds configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing metric bounds YAML file {self.config_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred loading metric bounds from {self.config_path}: {e}")
            return {}

    def normalize(self, metric_name, value, dataset):
        """
        Normalize a metric value to the [0, 1] range using dataset-specific bounds.

        Args:
            metric_name (str): The name of the metric (e.g., 'accuracy', 'latency').
            value (float): The raw value of the metric to normalize.
            dataset (str): The name of the dataset the metric value belongs to.

        Returns:
            float: The normalized value clamped between 0.0 and 1.0.
                   Returns 0.5 if bounds are missing or invalid, or if min == max.
        """
        if value is None or np.isnan(value):
             logger.warning(f"Cannot normalize None or NaN value for metric '{metric_name}' on dataset '{dataset}'. Returning 0.5.")
             return 0.5
        dataset_bounds = self.bounds.get(dataset, self.bounds.get('default'))

        if not dataset_bounds:
            logger.warning(f"No bounds found for dataset '{dataset}' (and no default) in {self.config_path}. Cannot normalize '{metric_name}'. Returning 0.5.")
            return 0.5

        metric_bounds = dataset_bounds.get(metric_name)

        if not metric_bounds or 'min' not in metric_bounds or 'max' not in metric_bounds:
            logger.warning(f"Incomplete/missing bounds for metric '{metric_name}' in dataset '{dataset}' config. Cannot normalize. Returning 0.5.")
            return 0.5

        min_val = metric_bounds['min']
        max_val = metric_bounds['max']
        if max_val == min_val:

            logger.debug(f"Min and Max bounds are identical ({min_val}) for metric '{metric_name}' on dataset '{dataset}'. Returning 0.5.")
            return 0.5
        try:
            value_f = float(value)
            min_val_f = float(min_val)
            max_val_f = float(max_val)
            
            normalized = (value_f - min_val_f) / (max_val_f - min_val_f)
        except ZeroDivisionError:
             logger.warning(f"Division by zero during normalization for metric '{metric_name}' on dataset '{dataset}'. Min={min_val}, Max={max_val}. Returning 0.5.")
             return 0.5
        except (ValueError, TypeError) as e:
             logger.warning(f"Type error during normalization calculation for metric '{metric_name}' on dataset '{dataset}'. Value={value}, Min={min_val}, Max={max_val}. Error: {e}. Returning 0.5.")
             return 0.5
        clamped_value = max(0.0, min(1.0, normalized))
        return clamped_value

    def update_bounds(self, metric_name, value, task_type=None):
        """Update bounds if value extends the range."""
        key = f"{metric_name}_{task_type}" if task_type else metric_name
        
        if key not in self.bounds:
            self.bounds[key] = {"min": value, "max": value}
            return
        if value < self.bounds[key]["min"]:
            self.bounds[key]["min"] = value
        elif value > self.bounds[key]["max"]:
            self.bounds[key]["max"] = value
