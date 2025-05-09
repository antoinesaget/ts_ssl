import os
from pathlib import Path

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from ts_ssl.utils.logger_manager import LoggerManager
from ts_ssl.utils.lr_factory import create_logistic_regression


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        output_dir,
        val_check_interval,
        config,
        logger: LoggerManager,
        val_batch_size: int = None,
        n_classes: int = 20,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.val_check_interval = val_check_interval
        self.val_batch_size = val_batch_size
        self.logger = logger
        self.n_classes = n_classes
        self.config = config

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize training state and checkpoint tracking
        self.global_step = 0
        self.best_metrics = {}
        self.best_checkpoint_paths = {}  # Track best checkpoint path for each metric
        self.last_checkpoint_path = None  # Track last checkpoint path

        # If no validation set, monitor training loss instead
        if val_loader is None:
            self.monitored_metrics = [
                {
                    "metric": "training/train_loss",
                    "mode": "min",
                    "filename": "train_loss.pt",
                }
            ]
        else:
            self.monitored_metrics = config.training.checkpoints.monitored_metrics

        for metric_config in self.monitored_metrics:
            self.best_metrics[metric_config["metric"]] = (
                float("inf") if metric_config["mode"] == "min" else float("-inf")
            )
            self.best_checkpoint_paths[metric_config["metric"]] = None

        self.scaler = GradScaler()

        # Initialize validation metrics based on task type
        self.accuracy_metrics = {}
        self.majority_accuracy_metrics = {}
        for n_samples in config.validation.samples_per_class:
            self.accuracy_metrics[n_samples] = Accuracy(
                task="multiclass", num_classes=n_classes
            ).to(device)
            self.majority_accuracy_metrics[n_samples] = Accuracy(
                task="multiclass", num_classes=n_classes
            ).to(device)
            # Compile metrics
            self.accuracy_metrics[n_samples] = torch.compile(
                self.accuracy_metrics[n_samples]
            )
            self.majority_accuracy_metrics[n_samples] = torch.compile(
                self.majority_accuracy_metrics[n_samples]
            )
        # Initialize learning rate scheduler
        if config.optimizer.scheduler.name == "one_cycle":
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=config.optimizer.scheduler.max_lr,
                total_steps=config.training.max_examples_seen
                // config.training.batch_size,
                pct_start=config.optimizer.scheduler.pct_start,
                div_factor=config.optimizer.scheduler.div_factor,
                final_div_factor=config.optimizer.scheduler.final_div_factor,
            )
        elif config.optimizer.scheduler.name == "cosine":
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.max_examples_seen // config.training.batch_size,
                eta_min=config.optimizer.scheduler.eta_min,
            )
        else:  # steady
            self.scheduler = None

    def train(self, max_steps: int):
        """Main training loop"""
        self.model.train()
        epoch = 0
        step_in_epoch = 0

        with tqdm(total=max_steps, desc="Training", ncols=80) as pbar:
            while self.global_step < max_steps:
                self.log_metrics({"training/epoch": epoch})
                epoch += 1
                for batch in self.train_loader:
                    # Preprocess batch
                    x1, x2 , y = batch
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    y = y.to(self.device)
                    # Training step
                    loss_value = self.training_step((x1, x2 ,y))

                    # Log metrics
                    if self.global_step % 100 == 0:
                        self.log_metrics(
                            {"training/train_loss": loss_value},
                            ignore_loggers=[
                                "logger"
                            ],  # Don't print training loss to console
                        )

                    # Validation
                    if (
                        self.val_loader is not None
                        and self.global_step % self.val_check_interval == 0
                    ):
                        self.validate()
                        self.model.train()

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({"train_loss": f"{loss_value:.4f}"})

                    self.global_step += 1
                    step_in_epoch += 1

                    if self.global_step >= max_steps:
                        break

    def training_step(self, batch):
        """Execute one training step"""
        self.optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            loss = self.model.training_step(batch)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)

        old_scaler = self.scaler.get_scale()
        self.scaler.update()
        new_scaler = self.scaler.get_scale()

        loss_value = loss.item()

        # Log and potentially checkpoint training metrics
        if self.global_step % 100 == 0:
            metrics = {"training/train_loss": loss_value}
            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]["lr"]
            metrics["training/learning_rate"] = lr

            self.log_metrics(metrics, ignore_loggers=["logger"])

            # Check for checkpointing if we're using training metrics
            if self.val_loader is None:
                self.check_and_save_checkpoints(metrics)

        # Step the scheduler if it exists
        if self.scheduler is not None and old_scaler <= new_scaler:
            self.scheduler.step()

        return loss_value

    def validate(self):
        """Run validation with linear evaluation using either scikit-learn or cuML"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        features_list = []
        labels_list = []

        # Extract features and labels
        with torch.no_grad():
            for batch in self.val_loader:
                x, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)

                # Get features
                features = self.model.get_features(x, aggregate=False)

                features_list.append(features)
                labels_list.append(labels)

        # Concatenate all features and labels
        features = torch.cat(features_list)  # N x G x F
        labels = torch.cat(labels_list)  # N

        results = {}

        # Evaluate with different numbers of samples
        for n_samples in self.config.validation.samples_per_class:
            accuracy = self.accuracy_metrics[n_samples]
            majority_accuracy = self.majority_accuracy_metrics[n_samples]
            # Create stratified train-test split
            classes = torch.unique(labels)
            train_indices = []

            # Check if we have enough samples for each class
            enough_samples = True
            for c in classes:
                class_indices = torch.where(labels == c)[0]
                if len(class_indices) < n_samples:
                    print(
                        f"Not enough samples for class {c}. Required {n_samples}, found {len(class_indices)}"
                    )
                    enough_samples = False
                    break
                # Random selection on GPU
                perm = torch.randperm(len(class_indices), device=self.device)
                selected_indices = class_indices[perm[:n_samples]]
                train_indices.append(selected_indices)

            if not enough_samples:
                continue

            # Reset metrics
            accuracy.reset()
            majority_accuracy.reset()

            # Combine all selected indices
            train_indices = torch.cat(train_indices)
            # Create test indices mask
            test_mask = torch.ones(len(labels), dtype=torch.bool, device=self.device)
            test_mask[train_indices] = False
            test_indices = torch.where(test_mask)[0]

            # Split data (already on GPU)
            X_train = features[train_indices]  # M x G x F
            y_train = labels[train_indices]  # M
            X_test = features[test_indices]  # K x G x F
            y_test = labels[test_indices]  # K

            # Train and evaluate with each evaluator in config
            for (
                evaluator_name,
                evaluator_cfg,
            ) in self.config.validation.downstream_models.items():
                # Create evaluator using factory function
                evaluator = create_logistic_regression(
                    **evaluator_cfg, logger=self.logger
                )

                if len(X_train.shape) == 3:
                    # Convert data to numpy for scikit-learn/cuml
                    X_train_flat = (
                        X_train.view(-1, self.model.embedding_dim).cpu().numpy()
                    )
                    y_train_flat = (
                        y_train.repeat_interleave(self.config.dataset.group_size)
                        .cpu()
                        .numpy()
                    )
                    X_test_flat = (
                        X_test.view(-1, self.model.embedding_dim).cpu().numpy()
                    )
                    y_test_flat = (
                        y_test.repeat_interleave(self.config.dataset.group_size)
                        .cpu()
                        .numpy()
                    )

                    # Train classifier
                    evaluator.fit(X_train_flat, y_train_flat)

                    # Batch prediction
                    batch_size = 1000
                    num_batches = (len(X_test_flat) + batch_size - 1) // batch_size
                    y_pred_flat = []
                    for i in range(num_batches):
                        start = i * batch_size
                        end = min(start + batch_size, len(X_test_flat))
                        X_test_batch = X_test_flat[start:end]
                        y_pred_batch = evaluator.predict(X_test_batch)
                        y_pred_flat.append(y_pred_batch)
                    y_pred_flat = np.concatenate(y_pred_flat)

                    # Evaluate per-timeseries accuracy
                    acc = (y_pred_flat == y_test_flat).mean()

                    # Evaluate majority vote accuracy
                    y_pred_groups = y_pred_flat.reshape(
                        -1, self.config.dataset.group_size
                    )
                    majority_pred = np.array(
                        [np.bincount(group).argmax() for group in y_pred_groups]
                    )

                    # Log metrics
                    metrics = {
                        f"training/val_acc__single_{n_samples}": acc,
                        f"training/val_acc_{n_samples}": (
                            majority_pred.astype(np.int32)
                            == y_test.cpu().numpy().astype(np.int32).flatten()
                        ).mean(),
                    }
                    results.update(metrics)
                else:
                    # Convert data to numpy for scikit-learn/cuml
                    X_train_flat = X_train.cpu().numpy()
                    y_train_flat = y_train.cpu().numpy()
                    X_test_flat = X_test.cpu().numpy()
                    y_test_flat = y_test.cpu().numpy()

                    # Train classifier
                    evaluator.fit(X_train_flat, y_train_flat)

                    # Batch prediction
                    batch_size = 1000
                    num_batches = (len(X_test_flat) + batch_size - 1) // batch_size
                    y_pred_flat = []
                    for i in range(num_batches):
                        start = i * batch_size
                        end = min(start + batch_size, len(X_test_flat))
                        X_test_batch = X_test_flat[start:end]
                        y_pred_batch = evaluator.predict(X_test_batch)
                        y_pred_flat.append(y_pred_batch)
                    y_pred_flat = np.concatenate(y_pred_flat)

                    # Evaluate accuracy
                    acc = (y_pred_flat == y_test_flat).mean()

                    # Log metrics
                    metrics = {
                        f"training/val_acc__single_{n_samples}": acc,
                        f"training/val_acc_{n_samples}": acc,
                    }
                    results.update(metrics)

        # Log all metrics
        self.log_metrics(results)

        # Check metrics and save checkpoints
        self.check_and_save_checkpoints(results)

        return results

    def check_and_save_checkpoints(self, metrics):
        """Check if monitored metrics improved and save checkpoints accordingly"""
        # Check if any monitored metrics improved and save checkpoints
        for metric_config in self.monitored_metrics:
            metric_name = metric_config["metric"]
            mode = metric_config["mode"]
            base_filename = metric_config["filename"]

            if metric_name in metrics:
                current_value = metrics[metric_name]
                best_value = self.best_metrics[metric_name]

                improved = (mode == "min" and current_value <= best_value) or (
                    mode == "max" and current_value >= best_value
                )

                if improved:
                    self.best_metrics[metric_name] = current_value
                    # Create filename with step and metric value
                    filename = f"{base_filename[:-3]}_{current_value:.4f}_step_{self.global_step}.pt"
                    checkpoint_path = self.checkpoint_dir / filename

                    # Save new checkpoint
                    self.save_checkpoint(checkpoint_path)
                    self.best_checkpoint_paths[metric_name] = checkpoint_path

                    # Remove previous best checkpoint if it exists
                    prev_checkpoints = list(
                        self.checkpoint_dir.glob(f"{base_filename[:-3]}_*_step_*")
                    )
                    for checkpoint in prev_checkpoints:
                        if checkpoint.name != filename:
                            checkpoint.unlink()

                    self.logger.info(
                        f"New best {metric_name}: {current_value:.4f} "
                        f"(previous best: {best_value:.4f})"
                    )

        # Save last checkpoint if configured
        if self.config.training.checkpoints.save_last:
            last_filename = f"last_step_{self.global_step}.pt"
            last_checkpoint_path = self.checkpoint_dir / last_filename

            # Save new last checkpoint
            self.save_checkpoint(last_checkpoint_path)
            self.last_checkpoint_path = last_checkpoint_path

            # Remove previous last checkpoint if it exists
            prev_last = list(self.checkpoint_dir.glob("last_step*.pt"))
            for checkpoint in prev_last:
                if checkpoint.name != last_filename:
                    checkpoint.unlink()

    def log_metrics(
        self,
        metrics,
        step=None,
        ignore_loggers=None,
    ):
        """Log metrics using the logger manager

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
            ignore_loggers: Optional list of logger names to ignore for this call
        """
        step = step or self.global_step
        self.logger.log_metrics(metrics, step, ignore_loggers=ignore_loggers)

    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def get_best_checkpoint_path(self):
        """Get the path of the best checkpoint based on monitored metrics"""
        # If we have monitored metrics, return the best checkpoint for the first metric
        if self.monitored_metrics:
            metric_name = self.monitored_metrics[0]["metric"]
            if self.best_checkpoint_paths[metric_name]:
                return self.best_checkpoint_paths[metric_name]

        # If no best metric checkpoint or no metrics monitored, return last checkpoint
        return self.last_checkpoint_path
