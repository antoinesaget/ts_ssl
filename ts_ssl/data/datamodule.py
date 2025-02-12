import logging
from pathlib import Path
from typing import Optional

import hydra
import mmap_ninja
import numpy as np
import torch
from torch.utils.data import Dataset

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
        data_dir: str,
        percentiles: list,
        transform: Optional[Compose] = None,
        logger: Optional[LoggerManager] = None,
        normalize_data: bool = True,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.transform = transform
        self._length = None
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.normalize_data = normalize_data

        # Set normalization values
        self.percentile_low = torch.tensor(percentiles[0])
        self.percentile_high = torch.tensor(percentiles[1])

    def __len__(self):
        return self._length

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the data using pre-computed percentiles."""
        if not self.normalize_data:
            return x
        x = x - self.percentile_low
        x = x / (self.percentile_high - self.percentile_low)
        return x - 0.5


class SupervisedGroupedTimeSeriesDataset(GroupedTimeSeriesDataset):
    """Dataset for supervised learning with time series data."""

    def __init__(
        self,
        data_dir: str,
        percentiles: list,
        transform: Optional[Compose] = None,
        logger: Optional[LoggerManager] = None,
        normalize_data: bool = True,
    ):
        super().__init__(data_dir, percentiles, transform, logger, normalize_data)

        self.data = np.load(self.data_dir / "x.npy")
        self.logger.info(
            f"Loaded {len(self.data)} samples with shape {self.data.shape}"
        )

        self.labels = np.load(self.data_dir / "y.npy")
        self.logger.info(
            f"Loaded {len(self.labels)} labels with shape {self.labels.shape}"
        )

        self._length = len(self.labels)

        assert len(self.labels) == len(self.data), (
            "Labels and data must have the same length"
        )

    def __getitem__(self, idx):
        # Get all pixels for this parcel
        x = torch.from_numpy(self.data[idx].copy()).float()  # Shape: (G, T, C)
        # y = torch.tensor(self.labels[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx])

        # Normalize the data
        x = self.normalize(x)

        # Apply additional transforms if specified
        if self.transform is not None:
            x = self.transform(x)

        return x, y


class SSLGroupedTimeSeriesDataset(GroupedTimeSeriesDataset):
    """Dataset for self-supervised learning with time series data."""

    def __init__(
        self,
        data_dir: str,
        n_samples_per_group: int,
        percentiles: list,
        config: dict,
        transform: Optional[Compose] = None,
        logger: Optional[LoggerManager] = None,
        normalize_data: bool = True,
    ):
        super().__init__(data_dir, percentiles, transform, logger, normalize_data)

        self.logger.info(f"Loading memmap data from {self.data_dir}")

        # Load memmap data
        self.data = mmap_ninja.np_open_existing(self.data_dir)
        self.logger.info(
            f"Loaded {len(self.data)} parcels with shape {self.data.shape}"
        )

        self._length = len(self.data)
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
        # Get all pixels for this parcel
        x = torch.from_numpy(self.data[idx].copy()).float()  # Shape: (G, T, C)

        # Sample groups using specified strategy
        view1, view2 = sample_groups(
            x, self.n_samples_per_group, self.sampling_strategy
        )

        # Normalize both views
        view1 = self.normalize(view1)
        view2 = self.normalize(view2)

        # Apply multi-view transforms that process both views at once
        if self.multiview_transforms is not None:
            view1, view2 = self.multiview_transforms([view1, view2])

        # Apply view-specific transforms
        if self.view1_transform is not None:
            view1 = self.view1_transform(view1)
        if self.view2_transform is not None:
            view2 = self.view2_transform(view2)

        return [view1, view2]
