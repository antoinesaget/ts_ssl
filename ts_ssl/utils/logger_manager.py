import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, ListConfig

try:
    from torch.utils.tensorboard.writer import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

try:
    from neptune.utils import stringify_unsupported

    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False


class _FilterCallback(logging.Filterer):
    """Filter for removing noisy Neptune log messages."""

    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
            )
        )


class LoggerManager:
    """Manages multiple loggers for training metrics and general logging."""

    def __init__(
        self,
        output_dir: Path,
        loggers: Optional[List[str]] = None,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        neptune_run=None,
    ):
        """Initialize logger manager.

        Args:
            output_dir: Output directory for logs
            loggers: List of loggers to use. Options: ["tensorboard", "csv", "neptune"]
                    Note: Python logger is always enabled by default
            log_file: Optional path to log file (defaults to output_dir/train.log if not provided)
            level: Logging level for Python logger
            format: Log message format for Python logger
            neptune_run: Optional pre-initialized Neptune run object
        """
        self.output_dir = output_dir

        self.loggers = list(set(["logger"] + (loggers or [])))
        if neptune_run is not None:
            self.loggers.append("neptune")
        self.writers = {}

        # Set up Python logging
        if log_file is None:
            log_file = output_dir / "train.log"

        # Create handlers
        handlers = [logging.StreamHandler(sys.stdout)]

        # Create directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

        # Configure logging
        logging.basicConfig(
            level=level,
            format=format,
            handlers=handlers,
            force=True,  # Force reconfiguration
        )

        # Reduce logging level for some noisy modules
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
            logging.WARNING
        )

        # Add neptune filter
        logging.getLogger("neptune").addFilter(_FilterCallback())

        # Initialize Python logger with a placeholder - it will be replaced in each logging call
        self.writers["logger"] = None

        # Initialize other loggers
        if TENSORBOARD_AVAILABLE and "tensorboard" in self.loggers:
            tensorboard_dir = self.output_dir / "tensorboard"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writers["tensorboard"] = SummaryWriter(tensorboard_dir)

        if "csv" in self.loggers:
            self.writers["csv"] = open(self.output_dir / "logs.csv", "w", newline="")
            self.csv_writer = csv.writer(self.writers["csv"])
            # Write header
            self.csv_writer.writerow(["step", "metric", "value"])

        if NEPTUNE_AVAILABLE and "neptune" in self.loggers:
            if neptune_run is None:
                raise ValueError(
                    "Neptune logger is enabled but no neptune_run object was provided"
                )
            self.writers["neptune"] = neptune_run

    def _get_caller_logger(self):
        """Get a logger named after the calling module."""
        import inspect

        # Get the calling frame (skipping this function and the logging function)
        frame = inspect.currentframe()
        try:
            # Skip this function and the logging function
            caller_frame = frame.f_back.f_back
            module_name = caller_frame.f_globals["__name__"]
            return logging.getLogger(module_name)
        finally:
            del frame  # Avoid reference cycles

    def info(self, msg: str, logger: Optional[logging.Logger] = None):
        """Log an info message using the Python logger."""
        logger = logger or self._get_caller_logger()
        logger.info(msg)
        if "neptune" in self.loggers and NEPTUNE_AVAILABLE:
            self.writers["neptune"]["logs/info"].append(msg)

    def warning(self, msg: str, logger: Optional[logging.Logger] = None):
        """Log a warning message using the Python logger."""
        logger = logger or self._get_caller_logger()
        logger.warning(msg)
        if "neptune" in self.loggers and NEPTUNE_AVAILABLE:
            self.writers["neptune"]["logs/warning"].append(msg)

    def error(self, msg: str, logger: Optional[logging.Logger] = None):
        """Log an error message using the Python logger."""
        logger = logger or self._get_caller_logger()
        logger.error(msg)
        if "neptune" in self.loggers and NEPTUNE_AVAILABLE:
            self.writers["neptune"]["logs/error"].append(msg)

    def debug(self, msg: str, logger: Optional[logging.Logger] = None):
        """Log a debug message using the Python logger."""
        logger = logger or self._get_caller_logger()
        logger.debug(msg)
        if "neptune" in self.loggers and NEPTUNE_AVAILABLE:
            self.writers["neptune"]["logs/debug"].append(msg)

    def log_metrics(
        self,
        metrics,
        step,
        ignore_loggers=None,
    ):
        """Log metrics to all configured loggers.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
            ignore_loggers: Optional list of logger names to ignore for this call
        """
        ignore_loggers = ignore_loggers or []
        active_loggers = [logger for logger in self.loggers if logger not in ignore_loggers]

        for logger_name in active_loggers:
            if logger_name == "tensorboard":
                for name, value in metrics.items():
                    self.writers["tensorboard"].add_scalar(name, value, step)

            elif logger_name == "csv":
                for name, value in metrics.items():
                    self.csv_writer.writerow([step, name, value])
                self.writers["csv"].flush()

            elif logger_name == "neptune" and NEPTUNE_AVAILABLE:
                run = self.writers["neptune"]
                for name, value in metrics.items():
                    run[name].append(value, step=step)

            elif logger_name == "logger":
                metrics_str = ", ".join(
                    [
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in metrics.items()
                    ]
                )
                self.info(
                    f"Step {step}: {metrics_str}", logger=self._get_caller_logger()
                )

    def log_hyperparameters(self, config):
        """Log hyperparameters to all available loggers.

        Args:
            config: Configuration object or dictionary containing hyperparameters
        """
        # Log to TensorBoard
        if "tensorboard" in self.loggers:
            # Convert config to flat dictionary with TensorBoard-compatible types
            def flatten_dict(d, parent_key="", sep="/"):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        # Convert to TensorBoard-compatible types
                        if isinstance(v, (int, float, str, bool)):
                            items.append((new_key, v))
                        elif isinstance(v, (list, tuple)):
                            items.append((new_key, str(v)))
                        elif hasattr(v, "__name__"):  # Handle function/class references
                            items.append((new_key, v.__name__))
                        elif v is None:
                            items.append((new_key, "None"))
                        else:
                            items.append((new_key, str(v)))
                return dict(items)

            flat_config = flatten_dict(config)
            # TensorBoard requires at least one metric, so we add a dummy one
            self.writers["tensorboard"].add_hparams(flat_config, {"dummy_metric": 0})

        # Log to Neptune
        if "neptune" in self.loggers and NEPTUNE_AVAILABLE:
            run = self.writers["neptune"]
            run["parameters"] = stringify_unsupported(config, expand=True)

        # Log to text file
        def format_value(value, indent_level=0):
            indent = "\t" * indent_level
            if isinstance(value, (dict, DictConfig)):
                result = "\n"
                for k, v in value.items():
                    result += f"{indent}\t{k}: {format_value(v, indent_level + 1)}"
                return result
            elif isinstance(value, (list, ListConfig)):
                # Check if all items are primitive types
                if all(not isinstance(x, (list, ListConfig)) for x in value):
                    return f"{value}\n"
                # Otherwise format nested structure
                result = "\n"
                for item in value:
                    result += f"{indent}\t- {format_value(item, indent_level + 1)}"
                return result
            else:
                return f"{value}\n"

        config_str = "\n=== Configuration ===\n"
        for category, params in config.items():
            config_str += f"\t{category}: {format_value(params, 1)}"
        config_str += "==================\n"
        self._get_caller_logger().info(config_str)

    def close(self):
        """Close all loggers."""
        for logger_name, writer in self.writers.items():
            if logger_name == "tensorboard":
                writer.close()
            elif logger_name == "csv":
                writer.close()
            elif logger_name == "neptune":
                writer.stop()
