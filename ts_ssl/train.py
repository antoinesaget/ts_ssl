import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchinfo import summary

try:
    import neptune

    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from ts_ssl.data.datamodule import (
    SSLGroupedTimeSeriesDataset,
    SupervisedGroupedTimeSeriesDataset,
)
from ts_ssl.evaluate import evaluate_main
from ts_ssl.trainer import Trainer
from ts_ssl.utils.logger_manager import LoggerManager

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Initialize Neptune if enabled
    neptune_run = None
    if config.logging.neptune.enabled and NEPTUNE_AVAILABLE:
        # Convert OmegaConf objects to native Python types
        neptune_config = OmegaConf.to_container(config.logging.neptune, resolve=True)
        neptune_run = neptune.init_run(
            project=neptune_config["project"],
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            capture_hardware_metrics=neptune_config["capture_hardware_metrics"],
            capture_stderr=neptune_config["capture_stderr"],
            capture_stdout=neptune_config["capture_stdout"],
            source_files=neptune_config["source_files"],
        )

    # Check if Hugging Face is required but not available
    if (
        hasattr(config.dataset, "dataset_type")
        and config.dataset.dataset_type == "huggingface"
        and not HUGGINGFACE_AVAILABLE
    ):
        raise ImportError(
            "Hugging Face datasets package is required but not installed. "
            "Please install it with `pip install datasets`"
        )

    # Initialize logger manager
    logger = LoggerManager(
        output_dir=Path(config.output_dir),
        loggers=config.logging.enabled_loggers,
        log_file=Path(config.output_dir) / "train.log",
        neptune_run=neptune_run,
    )

    # Log hyperparameters to all loggers
    logger.log_hyperparameters(config)

    # Initialize model
    logger.info("Instantiating model")
    model = hydra.utils.instantiate(config.model)

    # Log model summary
    model_summary = summary(
        model,
        input_size=(
            config.training.batch_size,
            config.training.n_samples_per_group,
            config.dataset.n_timesteps,
            config.dataset.n_channels,
        ),
        device=config.device,
        verbose=0,
        mode="train",
    )
    logger.info(f"\nModel Summary:\n{model_summary}")
    logger.info(
        "Please note that the model summary size estimates won't be accurate due to AMP."
    )

    model.compile()

    # Initialize datasets and dataloaders
    logger.info("Instantiating SSL training dataset and dataloader")
    train_dataset = SSLGroupedTimeSeriesDataset(
        data=config.dataset.ssl_data,
        n_samples_per_group=config.training.n_samples_per_group,
        percentiles=config.dataset.percentiles,
        config=config.augmentations,
        logger=logger,
        normalize_data=config.dataset.normalize,
        dataset_type=config.dataset.dataset_type,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
    )

    # Conditionally instantiate the supervised validation dataset and dataloader
    val_loader = None
    if getattr(config.dataset, "validation_data", None):
        logger.info("Instantiating supervised validation dataset and dataloader")
        val_dataset = SupervisedGroupedTimeSeriesDataset(
            data=config.dataset.validation_data,
            percentiles=config.dataset.percentiles,
            logger=logger,
            normalize_data=config.dataset.normalize,
            dataset_type=config.dataset.dataset_type,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.validation.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
        )
    else:
        logger.warning(
            "No validation set provided in the config; proceeding without validation."
        )

    # Initialize optimizer
    logger.info("Instantiating SGD optimizer")
    optimizer = SGD(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
    )

    # Initialize trainer (pass val_batch_size only if validation loader exists)
    logger.info("Instantiating trainer")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=config.device,
        output_dir=config.output_dir,
        val_check_interval=config.training.val_check_interval,
        logger=logger,
        val_batch_size=config.validation.batch_size if val_loader is not None else None,
        config=config,
    )

    # Train model
    logger.info("Training model")
    trainer.train(
        max_steps=config.training.max_examples_seen // config.training.batch_size
    )

    # Run post-training evaluation if enabled
    if config.post_training_eval:
        logger.info("Running post-training evaluation")

        eval_config = config.copy()
        eval_config.output_dir = str(Path(config.output_dir) / "eval")

        best_checkpoint_path = trainer.get_best_checkpoint_path()
        if best_checkpoint_path is None:
            raise ValueError("No checkpoints found for evaluation")

        eval_config.eval.checkpoint_path = str(best_checkpoint_path)
        evaluate_main(eval_config, logger)

    logger.close()


if __name__ == "__main__":
    main()
