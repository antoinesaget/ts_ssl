from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from sklearn.metrics import cohen_kappa_score, f1_score
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from ts_ssl.data.datamodule import SupervisedGroupedTimeSeriesDataset
from ts_ssl.utils.logger_manager import LoggerManager
from ts_ssl.utils.lr_factory import create_logistic_regression

torch.set_float32_matmul_precision("high")


def evaluate_model(
    model,
    train_features,
    train_labels,
    test_features,
    test_labels,
    n_samples,
    evaluator,
    device,
    use_raw_features=False,
    group_size=100,
):
    """Evaluate model with specific number of samples per class"""
    classes = torch.unique(train_labels)
    train_indices = []

    # Check if we have enough samples for each class
    for c in classes:
        class_indices = torch.where(train_labels == c)[0]
        if len(class_indices) < n_samples:
            return None
        perm = torch.randperm(len(class_indices), device=device)
        selected_indices = class_indices[perm[:n_samples]]
        train_indices.append(selected_indices)

    # Combine all selected indices
    train_indices = torch.cat(train_indices)

    # Get training data with selected samples
    X_train = train_features[train_indices]
    y_train = train_labels[train_indices]

    # Convert data to numpy for scikit-learn/cuml
    if use_raw_features:
        # For raw features, flatten the time series directly
        X_train_flat = (
            X_train.reshape(X_train.shape[0] * X_train.shape[1], -1).cpu().numpy()
        )
        X_test_flat = (
            test_features.reshape(test_features.shape[0] * test_features.shape[1], -1)
            .cpu()
            .numpy()
        )
        y_train_flat = y_train.repeat_interleave(group_size).cpu().numpy()
        y_test_flat = test_labels.repeat_interleave(group_size).cpu().numpy()
    else:
        if len(X_train.shape) == 3:
            # For model features, use the existing flattening logic
            X_train_flat = X_train.view(-1, model.embedding_dim).cpu().numpy()
            y_train_flat = y_train.repeat_interleave(group_size).cpu().numpy()
            X_test_flat = test_features.view(-1, model.embedding_dim).cpu().numpy()
            y_test_flat = test_labels.repeat_interleave(group_size).cpu().numpy()
        else:
            X_train_flat = X_train.cpu().numpy()
            y_train_flat = y_train.cpu().numpy()
            X_test_flat = test_features.cpu().numpy()
            y_test_flat = test_labels.cpu().numpy()

    # Train evaluator
    evaluator.fit(X_train_flat, y_train_flat)

    # Evaluate
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

    acc = (y_pred_flat == y_test_flat).mean()

    if len(X_train.shape) == 3 or use_raw_features:
        # Majority vote accuracy
        y_pred_groups = y_pred_flat.reshape(-1, group_size)
        majority_pred = np.array(
            [np.bincount(group).argmax() for group in y_pred_groups]
        )
        majority_acc = (majority_pred == test_labels.cpu().numpy().flatten()).mean()

        # Calculate additional metrics using majority predictions
        true_labels = test_labels.cpu().numpy().flatten()
        kappa = cohen_kappa_score(true_labels, majority_pred)
        macro_f1 = f1_score(true_labels, majority_pred, average="macro")

        return {
            "accuracy": acc,
            "majority_accuracy": majority_acc,
            "kappa": kappa,
            "macro_f1": macro_f1,
        }
    else:
        # For non-grouped predictions, all metrics are the same
        true_labels = test_labels.cpu().numpy()
        kappa = cohen_kappa_score(true_labels, y_pred_flat)
        macro_f1 = f1_score(true_labels, y_pred_flat, average="macro")

        return {
            "accuracy": acc,
            "majority_accuracy": acc,
            "kappa": kappa,
            "macro_f1": macro_f1,
        }


def extract_features(model, dataloader, device, use_raw_features=False, group_size=100):
    """Extract features from a dataset using the model or raw features"""
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", ncols=80):
            x, labels = batch
            x = x.to(device)
            labels = labels.to(device)

            if use_raw_features:
                # Use raw features directly
                features = x
            else:
                # Use model to extract features
                features = model.get_features(x, aggregate=False)

            features_list.append(features)
            labels_list.append(labels)

    return torch.cat(features_list), torch.cat(labels_list)


def evaluate_main(config, logger=None):
    # Setup
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    device = torch.device(config.device)

    # Validate required data directories are provided
    if not hasattr(config.dataset, "cross_validation_dir"):
        # Classic mode validation
        if not (
            hasattr(config.dataset, "train_data")
            and hasattr(config.dataset, "test_data")
        ):
            raise ValueError(
                "Both train_data and test_data must be provided in config.dataset for classic evaluation mode"
            )
    else:
        # Cross validation mode validation
        if not Path(config.dataset.cross_validation_dir).exists():
            raise ValueError(
                f"Cross validation directory {config.dataset.cross_validation_dir} does not exist"
            )

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = LoggerManager(
            output_dir=Path(config.output_dir),
            loggers=["logger", "csv"],
            log_file=Path(config.output_dir) / "evaluate.log",
        )

    logger.log_hyperparameters(config)

    # Load model only if not using raw features
    model = None
    if config.eval.use_raw_features is False:
        if config.eval.checkpoint_path is None:
            raise ValueError(
                "Checkpoint path is required when using model features. Either set use_raw_features=True or provide a checkpoint path in eval.checkpoint_path"
            )
        model = instantiate(config.model)

        # Log model summary
        model_summary = summary(
            model,
            input_size=(
                config.eval.batch_size,
                config.dataset.group_size,
                config.dataset.n_timesteps,
                config.dataset.n_channels,
            ),
            device=device,
            verbose=0,
            mode="eval",
            depth=4,
        )
        logger.info(f"\nModel Summary:\n{model_summary}")

        logger.info(f"Loading checkpoint from {config.eval.checkpoint_path}")
        checkpoint = torch.load(
            config.eval.checkpoint_path, map_location=device, weights_only=True
        )

        # Fix state dict keys by removing _orig_mod prefix
        fixed_state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            if "_orig_mod." in k:
                new_key = k.replace("_orig_mod.", "")
                fixed_state_dict[new_key] = v
            else:
                fixed_state_dict[k] = v

        model.load_state_dict(fixed_state_dict)
        model.compile()
        model = model.to(device)
        model.eval()

    # Store results for all runs
    results_path = Path(config.output_dir) / "evaluation_results.csv"
    summary_path = Path(config.output_dir) / "evaluation_results_summary.csv"

    # Determine evaluation mode based on config
    is_cross_validation = hasattr(config.dataset, "cross_validation_dir")

    if is_cross_validation:
        # Cross-validation mode headers
        pd.DataFrame(
            columns=[
                "train_fold",
                "test_fold",
                "samples_per_class",
                "evaluator",
                "accuracy",
                "majority_accuracy",
                "kappa",
                "macro_f1",
            ]
        ).to_csv(results_path, index=False)

        pd.DataFrame(
            columns=[
                "samples_per_class",
                "evaluator",
                "mean_accuracy",
                "std_accuracy",
                "mean_majority_accuracy",
                "std_majority_accuracy",
                "mean_kappa",
                "std_kappa",
                "mean_macro_f1",
                "std_macro_f1",
                "num_combinations",
            ]
        ).to_csv(summary_path, index=False)
    else:
        # Classic mode headers
        pd.DataFrame(
            columns=[
                "train_subset",
                "test_dataset",
                "samples_per_class",
                "evaluator",
                "accuracy",
                "majority_accuracy",
                "kappa",
                "macro_f1",
            ]
        ).to_csv(results_path, index=False)

        pd.DataFrame(
            columns=[
                "test_dataset",
                "samples_per_class",
                "evaluator",
                "mean_accuracy",
                "std_accuracy",
                "mean_majority_accuracy",
                "std_majority_accuracy",
                "mean_kappa",
                "std_kappa",
                "mean_macro_f1",
                "std_macro_f1",
                "num_train_subsets",
            ]
        ).to_csv(summary_path, index=False)

    # Extract and cache features
    features_cache = {}

    if is_cross_validation:
        # Cross-validation mode
        data_dirs = sorted(Path(config.dataset.cross_validation_dir).glob("subset_*"))
        if config.dataset.num_train_subsets is not None:
            data_dirs = data_dirs[: config.dataset.num_train_subsets]
            logger.info(f"Using {len(data_dirs)} folds")
    else:
        # Classic mode: Extract training features from subsets
        if hasattr(config.dataset, "num_train_subsets"):
            train_subsets = [
                {**config.dataset.train_data, "split": f"train_subset_{i}"}
                for i in range(config.dataset.num_train_subsets)
            ]
        else:
            train_subsets = [config.dataset.train_data]
        logger.info(f"Using {len(train_subsets)} training subsets")

    if config.eval.use_raw_features:
        logger.info("Using raw features for evaluation")
    else:
        logger.info("Using model features for evaluation")

    # Precompute features for all directories/subsets
    if is_cross_validation:
        for data_dir in data_dirs:
            logger.info(f"\nExtracting features for: {data_dir.name}")
            dataset = SupervisedGroupedTimeSeriesDataset(
                data_dir=data_dir,
                percentiles=config.dataset.percentiles,
                logger=logger,
                normalize_data=config.dataset.normalize,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.eval.batch_size,
                num_workers=config.num_workers,
                shuffle=False,
            )

            # Extract features
            features, labels = extract_features(
                model,
                loader,
                device,
                use_raw_features=config.eval.use_raw_features,
                group_size=config.dataset.group_size,
            )
            # Store features in CPU RAM
            features_cache[data_dir.name] = (features.cpu(), labels.cpu())
            del dataset, loader
    else:
        # Classic mode: Extract features for each training subset
        for i, train_subset in enumerate(train_subsets):
            subset_name = f"subset_{i}"
            logger.info(f"\nExtracting features for training {subset_name}")
            dataset = SupervisedGroupedTimeSeriesDataset(
                data=train_subset,
                percentiles=config.dataset.percentiles,
                logger=logger,
                normalize_data=config.dataset.normalize,
                dataset_type=config.dataset.dataset_type,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.eval.batch_size,
                num_workers=config.num_workers,
                shuffle=False,
            )

            # Extract features
            features, labels = extract_features(
                model,
                loader,
                device,
                use_raw_features=config.eval.use_raw_features,
                group_size=config.dataset.group_size,
            )
            # Store features in CPU RAM
            features_cache[subset_name] = (features.cpu(), labels.cpu())
            del dataset, loader

        # Extract test features
        logger.info("\nExtracting features for test dataset")
        test_dataset = SupervisedGroupedTimeSeriesDataset(
            data=config.dataset.test_data,
            percentiles=config.dataset.percentiles,
            logger=logger,
            normalize_data=config.dataset.normalize,
            dataset_type=config.dataset.dataset_type,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

        # Extract test features
        test_features, test_labels = extract_features(
            model,
            test_loader,
            device,
            use_raw_features=config.eval.use_raw_features,
            group_size=config.dataset.group_size,
        )
        test_features = test_features.cpu()
        test_labels = test_labels.cpu()

        del test_dataset, test_loader

    # For each number of samples per class
    for n_samples in config.eval.samples_per_class:
        # For each evaluator
        for evaluator_name, evaluator_cfg in config.eval.downstream_models.items():
            logger.info(
                f"\nEvaluating with {n_samples} samples per class using {evaluator_name}"
            )

            if is_cross_validation:
                # Cross-validation evaluation
                all_results = []  # Store results for all train/test combinations

                for train_dir in data_dirs:
                    train_features, train_labels = features_cache[train_dir.name]
                    train_features = train_features.to(device)
                    train_labels = train_labels.to(device)

                    # Evaluate on all other folds
                    for test_dir in data_dirs:
                        if test_dir == train_dir:
                            continue  # Skip same fold

                        test_features, test_labels = features_cache[test_dir.name]
                        test_features = test_features.to(device)
                        test_labels = test_labels.to(device)

                        # Evaluate model
                        evaluator = create_logistic_regression(
                            **evaluator_cfg, logger=logger
                        )
                        results = evaluate_model(
                            model,
                            train_features,
                            train_labels,
                            test_features,
                            test_labels,
                            n_samples,
                            evaluator,
                            device,
                            use_raw_features=config.eval.use_raw_features,
                            group_size=config.dataset.group_size,
                        )

                        if results is not None:
                            result_entry = {
                                "train_fold": train_dir.name,
                                "test_fold": test_dir.name,
                                "samples_per_class": n_samples,
                                "evaluator": evaluator_name,
                                **results,
                            }
                            all_results.append(result_entry)

                            # Write single result to CSV
                            pd.DataFrame([result_entry]).to_csv(
                                results_path, mode="a", header=False, index=False
                            )

                            logger.info(
                                f"Results for train:{train_dir.name}, test:{test_dir.name}:"
                                f"\n\tAccuracy: {results['accuracy']:.4f}"
                                f"\n\tMajority Accuracy: {results['majority_accuracy']:.4f}"
                                f"\n\tKappa: {results['kappa']:.4f}"
                                f"\n\tMacro F1: {results['macro_f1']:.4f}"
                            )

                # Calculate and log mean over all train/test combinations
                if all_results:
                    df_all = pd.DataFrame(all_results)
                    mean_acc = df_all["accuracy"].mean()
                    std_acc = df_all["accuracy"].std()
                    mean_maj_acc = df_all["majority_accuracy"].mean()
                    std_maj_acc = df_all["majority_accuracy"].std()
                    mean_kappa = df_all["kappa"].mean()
                    std_kappa = df_all["kappa"].std()
                    mean_macro_f1 = df_all["macro_f1"].mean()
                    std_macro_f1 = df_all["macro_f1"].std()

                    # Save summary statistics
                    summary_entry = {
                        "samples_per_class": n_samples,
                        "evaluator": evaluator_name,
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                        "mean_majority_accuracy": mean_maj_acc,
                        "std_majority_accuracy": std_maj_acc,
                        "mean_kappa": mean_kappa,
                        "std_kappa": std_kappa,
                        "mean_macro_f1": mean_macro_f1,
                        "std_macro_f1": std_macro_f1,
                        "num_combinations": len(all_results),
                    }
                    pd.DataFrame([summary_entry]).to_csv(
                        summary_path, mode="a", header=False, index=False
                    )

                    logger.info(
                        f"\nMean results over {len(all_results)} train/test combinations "
                        f"with {n_samples} samples using {evaluator_name}:"
                        f"\n\tAccuracy: {mean_acc:.4f} ± {std_acc:.4f}"
                        f"\n\tMajority Accuracy: {mean_maj_acc:.4f} ± {std_maj_acc:.4f}"
                        f"\n\tKappa: {mean_kappa:.4f} ± {std_kappa:.4f}"
                        f"\n\tMacro F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}"
                    )
            else:
                # Classic evaluation mode
                current_results = []  # Store results for current configuration

                # Move test data to device
                test_features_gpu = test_features.to(device)
                test_labels_gpu = test_labels.to(device)

                # Process each training subset
                for i, train_subset in enumerate(train_subsets):
                    subset_name = f"subset_{i}"
                    # Get cached training features and move to GPU
                    train_features, train_labels = features_cache[subset_name]
                    train_features = train_features.to(device)
                    train_labels = train_labels.to(device)

                    # Evaluate model
                    evaluator = create_logistic_regression(
                        **evaluator_cfg, logger=logger
                    )
                    results = evaluate_model(
                        model,
                        train_features,
                        train_labels,
                        test_features_gpu,
                        test_labels_gpu,
                        n_samples,
                        evaluator,
                        device,
                        use_raw_features=config.eval.use_raw_features,
                        group_size=config.dataset.group_size,
                    )

                    if results is not None:
                        result_entry = {
                            "train_subset": subset_name,
                            "test_dataset": config.dataset.name,
                            "samples_per_class": n_samples,
                            "evaluator": evaluator_name,
                            **results,
                        }
                        current_results.append(result_entry)

                        # Write single result to CSV
                        pd.DataFrame([result_entry]).to_csv(
                            results_path, mode="a", header=False, index=False
                        )

                        logger.info(
                            f"Results for {subset_name}:"
                            f"\n\tAccuracy: {results['accuracy']:.4f}"
                            f"\n\tMajority Accuracy: {results['majority_accuracy']:.4f}"
                            f"\n\tKappa: {results['kappa']:.4f}"
                            f"\n\tMacro F1: {results['macro_f1']:.4f}"
                        )

                # Calculate and log mean over training subsets
                if current_results:
                    df_current = pd.DataFrame(current_results)
                    mean_acc = df_current["accuracy"].mean()
                    std_acc = df_current["accuracy"].std()
                    mean_maj_acc = df_current["majority_accuracy"].mean()
                    std_maj_acc = df_current["majority_accuracy"].std()
                    mean_kappa = df_current["kappa"].mean()
                    std_kappa = df_current["kappa"].std()
                    mean_macro_f1 = df_current["macro_f1"].mean()
                    std_macro_f1 = df_current["macro_f1"].std()

                    # Save summary statistics
                    summary_entry = {
                        "test_dataset": config.dataset.name,
                        "samples_per_class": n_samples,
                        "evaluator": evaluator_name,
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                        "mean_majority_accuracy": mean_maj_acc,
                        "std_majority_accuracy": std_maj_acc,
                        "mean_kappa": mean_kappa,
                        "std_kappa": std_kappa,
                        "mean_macro_f1": mean_macro_f1,
                        "std_macro_f1": std_macro_f1,
                        "num_train_subsets": len(current_results),
                    }
                    pd.DataFrame([summary_entry]).to_csv(
                        summary_path, mode="a", header=False, index=False
                    )

                    logger.info(
                        f"\nMean results over {len(current_results)} training subsets for {config.dataset.name} "
                        f"with {n_samples} samples using {evaluator_name}:"
                        f"\n\tAccuracy: {mean_acc:.4f} ± {std_acc:.4f}"
                        f"\n\tMajority Accuracy: {mean_maj_acc:.4f} ± {std_maj_acc:.4f}"
                        f"\n\tKappa: {mean_kappa:.4f} ± {std_kappa:.4f}"
                        f"\n\tMacro F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}"
                    )


@hydra.main(config_path="config", config_name="eval", version_base="1.3")
def main(config):
    evaluate_main(config)


if __name__ == "__main__":
    main()
