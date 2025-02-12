from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ts_ssl.data.datamodule import SupervisedGroupedTimeSeriesDataset
from ts_ssl.utils.logger_manager import LoggerManager


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class FineTuner:
    def __init__(
        self,
        encoder,
        train_loader,
        val_loader,
        test_loader,
        device,
        config,
        logger,
    ):
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.logger = logger

        # Get number of classes from the dataset
        self.num_classes = len(torch.unique(torch.tensor(train_loader.dataset.labels)))

        # Initialize classification head
        self.classifier = ClassificationHead(
            input_dim=encoder.embedding_dim,
            hidden_dim=config.finetune.classifier.hidden_dim,
            num_classes=self.num_classes,
            dropout=config.finetune.classifier.dropout,
        ).to(device)

        # Initialize optimizers
        self.encoder_optimizer = None  # Will be initialized after frozen epochs
        self.classifier_optimizer = Adam(
            self.classifier.parameters(),
            lr=config.finetune.learning_rate.classifier,
            weight_decay=config.finetune.weight_decay,
        )

        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()

        # Setup paths for saving checkpoints
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize best metrics
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def save_checkpoint(self, name, metrics=None):
        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, self.checkpoint_dir / f"{name}.pt")

    def train_epoch(self, epoch, unfreeze_encoder=False):
        self.encoder.train()
        self.classifier.train()

        if not unfreeze_encoder:
            self.encoder.eval()

        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", ncols=80):
            x, labels = batch
            x = x.to(self.device)  # B x G x T x C
            labels = labels.to(self.device)

            batch_size = labels.size(0)
            n_samples_per_group = x.size(1)

            # Get features using encoder directly
            with torch.set_grad_enabled(unfreeze_encoder):
                features = self.encoder(x, aggregate=False)  # B x G x embedding_dim

            # Get predictions
            features_flat = features.view(
                -1, features.size(-1)
            )  # (B x G) x embedding_dim
            logits = self.classifier(features_flat)

            # Reshape labels to match predictions
            labels_repeated = labels.repeat_interleave(n_samples_per_group)

            # Calculate loss
            loss = self.criterion(logits, labels_repeated)

            # Backward pass
            loss.backward()

            # Update weights
            if unfreeze_encoder and self.encoder_optimizer is not None:
                self.encoder_optimizer.step()
                self.encoder_optimizer.zero_grad()
            self.classifier_optimizer.step()
            self.classifier_optimizer.zero_grad()

            # Calculate accuracy
            _, predicted = logits.max(1)
            correct += (predicted == labels_repeated).sum().item()
            total += labels_repeated.size(0)
            total_loss += loss.item() * batch_size

        epoch_loss = total_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        self.encoder.eval()
        self.classifier.eval()

        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", ncols=80):
                x, labels = batch
                x = x.to(self.device)  # B x G x T x C
                labels = labels.to(self.device)

                batch_size = labels.size(0)
                n_samples_per_group = x.size(1)

                # Get features using encoder directly
                features = self.encoder(x, aggregate=False)  # B x G x embedding_dim

                # Get predictions
                features_flat = features.view(
                    -1, features.size(-1)
                )  # (B x G) x embedding_dim
                logits = self.classifier(features_flat)

                # Reshape labels to match predictions
                labels_repeated = labels.repeat_interleave(n_samples_per_group)

                # Calculate loss
                loss = self.criterion(logits, labels_repeated)
                total_loss += loss.item() * batch_size

                # Store predictions and labels
                _, predicted = logits.max(1)
                all_predictions.append(predicted.cpu())
                all_labels.append(labels_repeated.cpu())

        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        val_loss = total_loss / len(self.val_loader.dataset)
        val_acc = (all_predictions == all_labels).float().mean().item()

        # Calculate majority vote accuracy
        predictions_grouped = all_predictions.view(-1, n_samples_per_group)
        majority_predictions = torch.mode(predictions_grouped, dim=1).values
        true_labels = all_labels.view(-1, n_samples_per_group)[
            :, 0
        ]  # Take first label from each group

        majority_acc = (majority_predictions == true_labels).float().mean().item()

        # Calculate additional metrics
        kappa = cohen_kappa_score(true_labels.numpy(), majority_predictions.numpy())
        macro_f1 = f1_score(
            true_labels.numpy(), majority_predictions.numpy(), average="macro"
        )

        metrics = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_majority_acc": majority_acc,
            "val_kappa": kappa,
            "val_macro_f1": macro_f1,
        }

        return metrics

    def evaluate(self, checkpoint_path):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])

        self.encoder.eval()
        self.classifier.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", ncols=80):
                x, labels = batch
                x = x.to(self.device)  # B x G x T x C
                labels = labels.to(self.device)

                n_samples_per_group = x.size(1)

                # Process in smaller chunks if needed
                chunk_size = self.config.finetune.eval_batch_size
                n_chunks = (x.size(0) + chunk_size - 1) // chunk_size

                batch_predictions = []
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, x.size(0))
                    x_chunk = x[start_idx:end_idx]  # chunk_size x G x T x C

                    # Get features using encoder directly
                    features = self.encoder(
                        x_chunk, aggregate=False
                    )  # chunk_size x G x embedding_dim
                    features_flat = features.view(
                        -1, features.size(-1)
                    )  # (chunk_size x G) x embedding_dim
                    logits = self.classifier(features_flat)
                    _, predicted = logits.max(1)
                    batch_predictions.append(predicted)

                # Combine chunk predictions
                batch_predictions = torch.cat(batch_predictions)
                all_predictions.append(batch_predictions.cpu())
                all_labels.append(labels.cpu())

        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # Print shapes for debugging
        print("all_predictions shape:", all_predictions.shape)
        print("all_labels shape:", all_labels.shape)
        print("n_samples_per_group:", n_samples_per_group)

        # Calculate accuracy
        try:
            repeated_labels = all_labels.repeat_interleave(n_samples_per_group)
            print("repeated_labels shape:", repeated_labels.shape)
            acc = (all_predictions == repeated_labels).float().mean().item()
        except Exception as e:
            print("Error in accuracy calculation:", str(e))
            acc = 0.0

        # Calculate majority vote accuracy
        try:
            predictions_grouped = all_predictions.view(-1, n_samples_per_group)
            print("predictions_grouped shape:", predictions_grouped.shape)
            majority_predictions = torch.mode(predictions_grouped, dim=1).values
            print("majority_predictions shape:", majority_predictions.shape)
            true_labels = all_labels.flatten()
            print("true_labels shape:", true_labels.shape)
            majority_acc = (majority_predictions == true_labels).float().mean().item()
        except Exception as e:
            print("Error in majority vote calculation:", str(e))
            majority_acc = 0.0

        # Calculate additional metrics
        try:
            kappa = cohen_kappa_score(true_labels.numpy(), majority_predictions.numpy())
            macro_f1 = f1_score(
                true_labels.numpy(), majority_predictions.numpy(), average="macro"
            )
        except Exception as e:
            print("Error in kappa/f1 calculation:", str(e))
            kappa = 0.0
            macro_f1 = 0.0

        metrics = {
            "test_acc": acc,
            "test_majority_acc": majority_acc,
            "test_kappa": kappa,
            "test_macro_f1": macro_f1,
        }

        return metrics

    def train(self):
        # First phase: train only the classifier
        self.logger.info("Phase 1: Training classifier with frozen encoder")
        for epoch in range(self.config.finetune.initial_frozen_epochs):
            train_loss, train_acc = self.train_epoch(epoch, unfreeze_encoder=False)
            val_metrics = self.validate()

            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, val_acc={val_metrics['val_majority_acc']:.4f}"
            )

            # Save best models
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_val_loss", val_metrics)

            if val_metrics["val_majority_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_majority_acc"]
                self.save_checkpoint("best_val_acc", val_metrics)

        # Second phase: fine-tune both encoder and classifier
        self.logger.info("Phase 2: Fine-tuning entire model")
        self.encoder_optimizer = Adam(
            self.encoder.parameters(),
            lr=self.config.finetune.learning_rate.encoder,
            weight_decay=self.config.finetune.weight_decay,
        )

        for epoch in range(
            self.config.finetune.initial_frozen_epochs, self.config.finetune.max_epochs
        ):
            train_loss, train_acc = self.train_epoch(epoch, unfreeze_encoder=True)
            val_metrics = self.validate()

            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, val_acc={val_metrics['val_majority_acc']:.4f}"
            )

            # Save best models
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_val_loss", val_metrics)
                print(
                    f"Saved best_val_loss at epoch {epoch} : {self.best_val_loss:.4f}"
                )
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if val_metrics["val_majority_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_majority_acc"]
                self.save_checkpoint("best_val_acc", val_metrics)
                print(f"Saved best_val_acc at epoch {epoch} : {self.best_val_acc:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping check
            if self.patience_counter >= self.config.finetune.patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Evaluate both checkpoints
        results = []
        for checkpoint_name in ["best_val_loss", "best_val_acc"]:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
            metrics = self.evaluate(checkpoint_path)
            metrics["checkpoint"] = checkpoint_name
            results.append(metrics)

            self.logger.info(
                f"\nResults for {checkpoint_name}:"
                f"\n\tAccuracy: {metrics['test_acc']:.4f}"
                f"\n\tMajority Accuracy: {metrics['test_majority_acc']:.4f}"
                f"\n\tKappa: {metrics['test_kappa']:.4f}"
                f"\n\tMacro F1: {metrics['test_macro_f1']:.4f}"
            )

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            self.output_dir / self.config.finetune.metrics_output_file, index=False
        )

        return results


@hydra.main(config_path="config", config_name="finetune", version_base="1.3")
def main(config):
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Initialize logger
    logger = LoggerManager(
        output_dir=Path(config.output_dir),
        loggers=["logger", "tensorboard", "csv"],
        log_file=Path(config.output_dir) / "finetune.log",
    )

    # Log hyperparameters
    logger.log_hyperparameters(config)

    # Load pretrained model
    logger.info("Loading pretrained model")
    ssl_model = hydra.utils.instantiate(config.model)
    checkpoint = torch.load(config.finetune.checkpoint_path, map_location=config.device)

    # Fix state dict keys by removing _orig_mod prefix
    fixed_state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        if "_orig_mod." in k:
            new_key = k.replace("_orig_mod.", "")
            fixed_state_dict[new_key] = v
        else:
            fixed_state_dict[k] = v

    # Load state dict into SSL model
    ssl_model.load_state_dict(fixed_state_dict)

    # Extract just the encoder for finetuning
    encoder = ssl_model.encoder
    encoder.compile()

    # Initialize datasets
    logger.info("Loading datasets")
    train_dataset = SupervisedGroupedTimeSeriesDataset(
        data_dir=config.dataset.train_data_dir,
        percentiles=config.dataset.percentiles,
        logger=logger,
        normalize_data=config.dataset.normalize,
    )

    val_dataset = SupervisedGroupedTimeSeriesDataset(
        data_dir=config.dataset.validation_data_dir,
        percentiles=config.dataset.percentiles,
        logger=logger,
        normalize_data=config.dataset.normalize,
    )

    test_dataset = SupervisedGroupedTimeSeriesDataset(
        data_dir=config.dataset.test_data_dir,
        percentiles=config.dataset.percentiles,
        logger=logger,
        normalize_data=config.dataset.normalize,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.finetune.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.finetune.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.finetune.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    # Initialize fine-tuner
    logger.info("Initializing fine-tuner")
    finetuner = FineTuner(
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.device,
        config=config,
        logger=logger,
    )

    # Train and evaluate
    logger.info("Starting fine-tuning")
    results = finetuner.train()

    logger.close()
    return results


if __name__ == "__main__":
    main()
