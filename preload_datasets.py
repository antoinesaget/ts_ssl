import argparse
from pathlib import Path

from omegaconf import OmegaConf

from datasets import load_dataset


def load_dataset_config(dataset_name):
    """Load dataset configuration from YAML file."""
    config_path = Path("ts_ssl/config")

    # Load location config first for base_path
    location_config = OmegaConf.load(config_path / "location/local.yaml")

    # Load dataset config
    dataset_config = OmegaConf.load(config_path / f"dataset/{dataset_name}.yaml")

    # Merge configs to resolve base_path
    config = OmegaConf.merge({"location": location_config}, {"dataset": dataset_config})
    return config


def preload_dataset(dataset_name):
    """Download and cache the dataset."""
    config = load_dataset_config(dataset_name)

    print(f"Loading dataset: {dataset_name}")
    print(f"Cache directory: {config.dataset.ssl_data.cache_dir}")

    # Load the dataset using Hugging Face's load_dataset
    if config.dataset.dataset_type == "huggingface":
        dataset = load_dataset(
            path=config.dataset.ssl_data.path,
            cache_dir=config.dataset.ssl_data.cache_dir,
            split=config.dataset.ssl_data.split,
        )
        print(f"Successfully loaded dataset: {dataset}")
    else:
        raise ValueError(f"Unsupported dataset type: {config.dataset.dataset_type}")


def main():
    parser = argparse.ArgumentParser(description="Preload datasets for TS-SSL")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset configuration (without .yaml extension)",
    )
    args = parser.parse_args()

    try:
        preload_dataset(args.dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


if __name__ == "__main__":
    main()
