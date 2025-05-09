import logging
from pathlib import Path
from typing import Literal, Optional

import hydra
import mmap_ninja
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from ts_ssl.data.augmentations import Compose, sample_groups
from ts_ssl.utils.logger_manager import LoggerManager


class GroupedTimeSeriesDataset(Dataset):
    """Base dataset class for time series data stored in memmap format.

    Expects data in shape (N, G, T, C) where:
    - N is number of parcels
    - G is number of pixels per parcel
    - T is number of timesteps
    - C is number of channels
    """

    def __init__(
        self,
        data: str | dict,
        percentiles: list,
        transform: Optional[Compose] = None,
        logger: Optional[LoggerManager] = None,
        normalize_data: bool = True,
    ):
        super().__init__()

        self.data_dir = Path(data) if isinstance(data, str) else Path(data["cache_dir"])
        self.transform = transform
        self._length = -1
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.normalize_data = normalize_data

        # Set normalization values
        self.percentile_low = torch.tensor(percentiles[0])
        self.percentile_high = torch.tensor(percentiles[1])

    def __len__(self):
        return self._length

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the data using pre-computed percentiles."""
        x = x - self.percentile_low
        x = x / (self.percentile_high - self.percentile_low)
        return x - 0.5


class SupervisedGroupedTimeSeriesDataset(GroupedTimeSeriesDataset):
    """Dataset for supervised learning with time series data."""

    def __init__(
        self,
        data: str | dict,
        percentiles: list,
        transform: Optional[Compose] = None,
        logger: Optional[LoggerManager] = None,
        normalize_data: bool = True,
        dataset_type: Literal["mmap_ninja", "huggingface"] = "mmap_ninja",
    ):
        super().__init__(data, percentiles, transform, logger, normalize_data)

        self.logger.info(f"Loading {dataset_type} data from {self.data_dir}")

        # Load data based on dataset type
        if dataset_type == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError(
                    "Hugging Face datasets package is not installed. "
                    "Please install it with `pip install datasets`"
                )
            self.dataset = load_dataset(
                data["path"],
                cache_dir=data["cache_dir"],
                split=data["split"],
            ).with_format(
                "torch",
                columns=["x", "y"],
                dtype=torch.float32,
            )
            self._length = len(self.dataset)
            self.logger.info(
                f"Loaded {len(self.dataset)} samples from Hugging Face dataset"
            )
        else:  # mmap_ninja
            self.data = np.load(self.data_dir / "x.npy")
            self.labels = np.load(self.data_dir / "y.npy")
            self._length = len(self.labels)
            self.logger.info(
                f"Loaded {len(self.data)} samples with shape {self.data.shape}"
            )
            self.logger.info(
                f"Loaded {len(self.labels)} labels with shape {self.labels.shape}"
            )
            assert len(self.labels) == len(self.data), (
                "Labels and data must have the same length"
            )

        self.dataset_type = dataset_type

    def __getitem__(self, idx):
        if self.dataset_type == "huggingface":
            # Get data from HuggingFace dataset
            sample = self.dataset[idx]
            x = sample["x"]
            y = sample["y"].long()
        else:
            # Get data from numpy arrays
            x = torch.from_numpy(self.data[idx].copy()).float()
            y = torch.tensor(self.labels[idx])

        # Normalize the data
        if self.normalize_data:
            x = self.normalize(x)

        # Apply additional transforms if specified
        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self._length


class SSLGroupedTimeSeriesDataset(GroupedTimeSeriesDataset):
    """Dataset for self-supervised learning with time series data."""

    def __init__(
        self,
        data: str | dict,
        n_samples_per_group: int,
        percentiles: list,
        config: dict,
        transform: Optional[Compose] = None,
        logger: Optional[LoggerManager] = None,
        normalize_data: bool = True,
        dataset_type: Literal["mmap_ninja", "huggingface"] = "mmap_ninja",
    ):
        super().__init__(data, percentiles, transform, logger, normalize_data)

        self.logger.info(f"Loading {dataset_type} data from {self.data_dir}")

        # Load data based on dataset type
        if dataset_type == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError(
                    "Hugging Face datasets package is not installed. "
                    "Please install it with `pip install datasets`"
                )
            self.data = load_dataset(
                data["path"],
                cache_dir=data["cache_dir"],
                split=data["split"],
            ).with_format("torch", columns=["x", "y"], dtype=torch.float32)
            self._length = len(self.data)
        else:  # mmap_ninja
            self.data = mmap_ninja.np_open_existing(self.data_dir)
            self._length = len(self.data)

        self.dataset_type = dataset_type
        self.augmentations = config.name if hasattr(config, 'name') else None
        self.logger.info(
            f"Loaded {len(self.data)} parcels with shape {self.data.shape if hasattr(self.data, 'shape') else 'N/A'}"
        )

        self.n_samples_per_group = n_samples_per_group
        self.sampling_strategy = config.views_sampling_strategy

        # Initialize transforms for each view
        self.view1_transform = self._setup_transforms(config.view1.transforms)
        self.view2_transform = self._setup_transforms(config.view2.transforms)

        # Initialize multi-view transforms that process both views at once
        self.multiview_transforms = (
            self._setup_transforms(config.multiview_transforms)
            if hasattr(config, "multiview_transforms")
            else None
        )

    def _setup_transforms(self, transform_configs) -> Optional[Compose]:
        """Setup transforms from config using Hydra's instantiation"""
        transforms = [hydra.utils.instantiate(t) for t in transform_configs.values()]
        return Compose(transforms) if transforms else None

    def __getitem__(self, idx):
        # Handle different types of indices
        if isinstance(idx, (list, tuple, range)):
            indices = idx
            is_batch = True
        elif isinstance(idx, slice):
            # Convert slice to list of indices
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1
            indices = range(start, stop, step)
            is_batch = True
        elif isinstance(idx, int):
            indices = [idx]
            is_batch = False
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Break down data loading steps
        if self.dataset_type == "huggingface":
            x = self.data[indices]["x"] if is_batch else self.data[idx]["x"]
        else:
            x = self.data[indices] if is_batch else self.data[idx]
            x = x.copy()
            x = torch.from_numpy(x)
            x = x.float()

        # Sample groups using specified strategy
        if is_batch:
            # For batches, we need to handle each sample
            B = len(indices)  # Batch size
            view1_list = []
            view2_list = []
            for i in range(B):
                v1, v2 = sample_groups(
                    x[i], self.n_samples_per_group, self.sampling_strategy
                )
                view1_list.append(v1)
                view2_list.append(v2)
            view1 = torch.stack(view1_list)
            view2 = torch.stack(view2_list)
        else:
            view1, view2 = sample_groups(
                x, self.n_samples_per_group, self.sampling_strategy
            )

        if self.normalize_data:
            view1 = self.normalize(view1)
            view2 = self.normalize(view2)

        # Apply multi-view transforms that process both views at once
        if self.multiview_transforms is not None:
            if is_batch:
                # Apply transforms to each sample in the batch
                B = len(indices)
                for i in range(B):
                    view1[i], view2[i] = self.multiview_transforms([view1[i], view2[i]])
            else:
                view1, view2 = self.multiview_transforms([view1, view2])

        # Apply view-specific transforms
        if is_batch:
            # Apply transforms to each sample in the batch
            B = len(indices)
            for i in range(B):
                if self.view1_transform is not None:
                    view1[i] = self.view1_transform(view1[i])
                if self.view2_transform is not None:
                    view2[i] = self.view2_transform(view2[i])
        else:
            if self.view1_transform is not None:
                view1 = self.view1_transform(view1)
            if self.view2_transform is not None:
                view2 = self.view2_transform(view2)

        return [view1, view2]

    def get_dtype(self):
        return self.data.dtype if self.dataset_type == "mmap_ninja" else self.data[0]["x"].dtype
    
    def get_augmentations(self):
        return self.augmentations
    
    def get_raw_item(self, idx):
        # Handle different types of indices
        if isinstance(idx, (list, tuple, range)):
            indices = idx
            is_batch = True
        elif isinstance(idx, slice):
            # Convert slice to list of indices
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1
            indices = range(start, stop, step)
            is_batch = True
        elif isinstance(idx, int):
            indices = [idx]
            is_batch = False
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Break down data loading steps
        if self.dataset_type == "huggingface":
            x = self.data[indices]["x"] if is_batch else self.data[idx]["x"]
        else:
            x = self.data[indices] if is_batch else self.data[idx]
            x = x.copy()
            x = torch.from_numpy(x)
            x = x.float()

        # Sample groups using specified strategy
        if is_batch:
            # For batches, we need to handle each sample
            B = len(indices)  # Batch size
            view1_list = []
            view2_list = []
            for i in range(B):
                v1, v2 = sample_groups(
                    x[i], self.n_samples_per_group, self.sampling_strategy
                )
                view1_list.append(v1)
                view2_list.append(v2)
            view1 = torch.stack(view1_list)
            view2 = torch.stack(view2_list)
        else:
            view1, view2 = sample_groups(
                x, self.n_samples_per_group, self.sampling_strategy
            )

        if self.normalize_data:
            view1 = self.normalize(view1)
            view2 = self.normalize(view2)

        # Skip transformations

        return [view1, view2]
class GCDTimeSeries(GroupedTimeSeriesDataset):
    """Dataset for GCD with time series data."""

    def __init__(
        self,
        data: str | dict,
        n_samples_per_group: int,
        percentiles: list,
        config: dict,
        transform: Optional[Compose] = None,
        logger: Optional[LoggerManager] = None,
        normalize_data: bool = True,
        known_classes: list = [0,1,2,3,4,5,6,7,8,9],
        labeled_fraction: float = 0.1,
        dataset_type: Literal["mmap_ninja", "huggingface"] = "huggingface",  
    ):
        super().__init__(data, percentiles, transform, logger, normalize_data)

        self.logger.info(f"Loading {dataset_type} data from {self.data_dir}")

        # Load data based on dataset type
        if dataset_type == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError(
                    "Hugging Face datasets package is not installed. "
                    "Please install it with `pip install datasets`"
                )
            self.dataset = load_dataset(
                data["path"],
                cache_dir=data["cache_dir"],
                split=data["split"],
            ).with_format(
                "torch",
                columns=["x", "y"],
                dtype=torch.float32,
            )
            self._length = len(self.dataset)
            self.logger.info(
                f"Loaded {len(self.dataset)} samples from Hugging Face dataset"
            )
            # For HuggingFace datasets, we'll use this to get labels
            self.all_labels = torch.tensor(self.dataset["y"])
        else:  # mmap_ninja
            self.data = np.load(self.data_dir / "x.npy")
            self.labels = np.load(self.data_dir / "y.npy")
            self._length = len(self.labels)
            self.logger.info(
                f"Loaded {len(self.data)} samples with shape {self.data.shape}"
            )
            self.logger.info(
                f"Loaded {len(self.labels)} labels with shape {self.labels.shape}"
            )
            assert len(self.labels) == len(self.data), (
                "Labels and data must have the same length"
            )
            self.all_labels = self.labels

        self.sampled_label_counts = {cls: 0 for cls in known_classes}
        self.dataset_type = dataset_type
        self.augmentations = config.name if hasattr(config, 'name') else None
        
        self.logger.info(
            f"Loaded {len(self.dataset if self.dataset_type == 'huggingface' else self.data)} parcels with shape "
            f"{self.dataset['x'].shape if self.dataset_type == 'huggingface' else self.data.shape}"
        )

        self.n_samples_per_group = n_samples_per_group
        self.sampling_strategy = config.views_sampling_strategy

        # Initialize transforms for each view
        self.view1_transform = self._setup_transforms(config.view1.transforms)
        self.view2_transform = self._setup_transforms(config.view2.transforms)

        # Initialize multi-view transforms that process both views at once
        self.multiview_transforms = (
            self._setup_transforms(config.multiview_transforms)
            if hasattr(config, "multiview_transforms")
            else None
        )
        self.number_of_remaining_samples = []
        num_classes = int(self.all_labels.max()) + 1  # Use all_labels instead of self.labels
        self.known_classes = known_classes
        self.labeled_fraction = labeled_fraction

        for cls in range(num_classes):
            if cls in self.known_classes:
                cls_indices = np.where(self.all_labels == cls)[0]
                n_labeled = int(len(cls_indices) * self.labeled_fraction)
            else:
                n_labeled = 0
            self.number_of_remaining_samples.append(n_labeled)

    def _setup_transforms(self, transform_configs) -> Optional[Compose]:
        transforms = [hydra.utils.instantiate(t) for t in transform_configs.values()]
        return Compose(transforms) if transforms else None

    def _get_two_samples_from_same_class(self, class_label):
        """Return two views (each of shape [n_samples_per_group, T, C]) from same class"""

        class_indices = np.where(self.all_labels == class_label)[0]
        selected_indices = np.random.choice(class_indices, size=2 * self.n_samples_per_group, replace=False)

        # Split indices into two sets
        indices_view1 = selected_indices[:self.n_samples_per_group]
        indices_view2 = selected_indices[self.n_samples_per_group:]

        # Load samples
        if self.dataset_type == "huggingface":
            view1 = torch.stack([self.dataset[int(i)]["x"] for i in indices_view1])
            view2 = torch.stack([self.dataset[int(i)]["x"] for i in indices_view2])
        else:
            view1 = torch.stack([torch.from_numpy(self.data[int(i)].copy()).float() for i in indices_view1])
            view2 = torch.stack([torch.from_numpy(self.data[int(i)].copy()).float() for i in indices_view2])

        return view1, view2
    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, range)):
            indices = idx
            is_batch = True
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1
            indices = range(start, stop, step)
            is_batch = True
        elif isinstance(idx, int):
            indices = [idx]
            is_batch = False
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Load data based on dataset type
        if self.dataset_type == "huggingface":
            x = self.dataset[indices]["x"] if is_batch else self.dataset[idx]["x"]
            y = self.dataset[indices]["y"].long() if is_batch else self.dataset[idx]["y"].long()
        else:
            x = self.data[indices] if is_batch else self.data[idx]
            x = torch.from_numpy(x.copy()).float()
            y = self.labels[indices] if is_batch else self.labels[idx]
            y = torch.tensor(y).long()

        if is_batch:
            B = len(indices)
            view1_list, view2_list, labels_list = [], [], []
            
            for i in range(B):
                label = y[i].item()
                
                # Check if sample is labeled (known class and within quota)
                if label in self.known_classes and self.sampled_label_counts[label] < self.number_of_remaining_samples[label]:
                    self.sampled_label_counts[label] += 1
                    
                    # Option 1: Get view1 from current sample and find matching view2
                    view1, _ = sample_groups(x[i], self.n_samples_per_group, cop)
                    
                    # Find next sample with same label
                    view2 = None
                    for j in range(i+1, B):  # Search remaining samples
                        if y[j].item() == label:
                            view2, _ = sample_groups(x[j], self.n_samples_per_group, "copy")
                            break
                    
                    # If no match found, use augmentation instead
                    if view2 is None:
                        view1, view2 = sample_groups(x[i], self.n_samples_per_group, "copy")
                    
                    labels_list.append(torch.tensor(label))
                    
                else:  # Unlabeled case
                    view1, view2 = sample_groups(x[i], self.n_samples_per_group, self.sampling_strategy)
                    labels_list.append(torch.tensor(-1))
                
                view1_list.append(view1)
                view2_list.append(view2)

            view1 = torch.stack(view1_list)
            view2 = torch.stack(view2_list)
            labels = torch.stack(labels_list)

        else:  # Single sample
            label = y.item()
            
            if label in self.known_classes and self.sampled_label_counts[label] < self.number_of_remaining_samples[label]:
                self.sampled_label_counts[label] += 1
                # Labeled sample - get two different samples from same class
                view1, view2 = sample_groups(x, self.n_samples_per_group, "copy")
            else:
                # Unlabeled sample - apply different augmentations to same sample
                view1, view2 = sample_groups(x, self.n_samples_per_group, self.sampling_strategy)
                label = -1
            
            labels = torch.tensor(label)

        # Normalization
        if self.normalize_data:
            view1 = self.normalize(view1)
            view2 = self.normalize(view2)

        # Apply multiview transforms if specified
        if self.multiview_transforms is not None:
            if is_batch:
                for i in range(B):
                    view1[i], view2[i] = self.multiview_transforms([view1[i], view2[i]])
            else:
                view1, view2 = self.multiview_transforms([view1, view2])

        # Apply view-specific transforms
        if is_batch:
            for i in range(B):
                if self.view1_transform is not None:
                    view1[i] = self.view1_transform(view1[i])
                if self.view2_transform is not None:
                    view2[i] = self.view2_transform(view2[i])
        else:
            if self.view1_transform is not None:
                view1 = self.view1_transform(view1)
            if self.view2_transform is not None:
                view2 = self.view2_transform(view2)

        return view1, view2, labels

