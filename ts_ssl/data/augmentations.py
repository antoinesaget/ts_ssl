from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F


class TimeSeriesTransform:
    """Base class for all time series transforms.

    All transforms should inherit from this class and implement the forward method.
    The base class handles input validation and provides a standard interface.

    TODO(augmentations):
        Currently, each transform treats each sample in the group independently.
        We might want to add support for group-level operations where the same
        transform is applied consistently across all samples in a group.
        This would require:
        1. A way to specify if a transform should operate at group or sample level
        2. Different implementations for group-level operations
        3. Careful consideration of how this interacts with the sampling strategy
    """

    def validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor shape and values

        Args:
            x: Input tensor expected to be of shape (G, T, C) where:
               - G is the group size (number of samples in the group)
               - T is the number of timesteps
               - C is the number of channels
        """
        if not (
            isinstance(x, torch.Tensor)
            or (isinstance(x, list) and all(isinstance(e, torch.Tensor) for e in x))
        ):
            raise TypeError(
                f"Input must be a torch.Tensor or list of torch.Tensor, got {type(x)}"
            )

        if (isinstance(x, torch.Tensor) and x.dim() != 3) or (
            isinstance(x, list) and not all(e.dim() == 3 for e in x)
        ):
            raise ValueError(
                f"Input tensor must have 3 dimensions (G, T, C), got shape {x.shape}"
            )

        if (isinstance(x, torch.Tensor) and not torch.isfinite(x).all()) or (
            isinstance(x, list) and not all(torch.isfinite(e).all() for e in x)
        ):
            raise ValueError("Input tensor contains NaN or infinite values")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform implementation to be defined by subclasses"""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Detailed string representation for debugging"""
        return self.__class__.__name__

    def __str__(self) -> str:
        """Simple string representation for logging"""
        return self.__class__.__name__


class Jittering(TimeSeriesTransform):
    """Add Gaussian noise to the input for jittering effect.

    This transform adds random noise sampled from N(0, sigma²) to create
    a jittering effect that helps with robustness to noise in the data.

    Important:
        This transform assumes the input data is normalized with approximately unit
        standard deviation. If your data has a different scale, the sigma parameter
        should be adjusted accordingly, or the jittering effect might be too strong
        or too weak.

        In the context of this codebase, this transform is designed to be applied
        after the normalization step in SSLGroupedTimeSeriesDataset, which normalizes
        the data using pre-computed percentiles.
    """

    def __init__(self, sigma: float = 0.1):
        """Initialize the jittering transform.

        Args:
            sigma: Standard deviation of the Gaussian noise relative to unit-scaled data.
                  Must be non-negative. A value of 0.1 means the noise standard deviation
                  will be 10% of the data's expected standard deviation.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return x + noise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(σ={self.sigma:.3f})"


class Flipping(TimeSeriesTransform):
    """Flip the time series along the time axis.

    This transform randomly flips the time series horizontally (along time axis)
    with a given probability. When applied, it reverses the temporal order of
    the sequence while maintaining the shape and scale of the data.

    Note:
        This transform should be used with caution as it may not be appropriate
        for all types of time series data. For example, in cases where the temporal
        ordering is crucial for the task (e.g., forecasting), flipping might
        destroy important patterns in the data.
    """

    def __init__(self, p: float = 0.5):
        """Initialize the flipping transform.

        Args:
            p: Probability of applying the flip. Must be between 0 and 1.
        """
        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample a mask of shape (group_size, 1, 1) to decide which sequences to flip
        flip_mask = torch.rand(x.shape[0], 1, 1, device=x.device) < self.p
        # Flip along time dimension (dim=1) where mask is True
        x = torch.where(
            flip_mask,
            x.flip(dims=[1]),  # Flipped version
            x,  # Original version
        )
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p:.2f})"


class Scaling(TimeSeriesTransform):
    """Randomly scale the time series by a factor.

    This transform multiplies the input by a random scaling factor sampled from
    a uniform distribution between [1-magnitude, 1+magnitude].
    """

    def __init__(self, magnitude: float = 0.2):
        """Initialize the scaling transform.

        Args:
            magnitude: Maximum scaling deviation from 1. The actual scaling factor
                      will be uniformly sampled from [1-magnitude, 1+magnitude].
                      Must be in (0, 1) to ensure positive scaling factors.
        """
        if not 0 < magnitude < 1:
            raise ValueError(f"magnitude must be between 0 and 1, got {magnitude}")
        self.magnitude = magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample scaling factor for each sample in the group
        scale = (
            1.0
            + (2 * torch.rand(x.shape[0], 1, 1, device=x.device) - 1) * self.magnitude
        )
        return x * scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(magnitude={self.magnitude})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(m={self.magnitude:.3f})"


class Resizing(TimeSeriesTransform):
    """Resize the time series by cropping and scaling.

    This transform changes the temporal resolution of the time series by randomly
    cropping a portion of it and then scaling that portion back to the original length.
    The crop size is determined by a factor sampled uniformly from [1-magnitude, 1].
    A resize factor of 1 means no cropping, while smaller values mean taking a smaller crop.

    Note:
        This transform first crops a portion of the time series and then uses linear
        interpolation to scale it back to the original length. This creates a different
        effect than pure interpolation-based resizing, as it focuses on a specific
        temporal segment of the data.
    """

    def __init__(self, magnitude: float = 0.2):
        """Initialize the resizing transform.

        Args:
            magnitude: Maximum amount to crop. The resize factor will be sampled
                      uniformly from [1-magnitude, 1]. Must be between 0 and 1.
                      A larger magnitude means more aggressive cropping.
        """
        if not 0 < magnitude < 1:
            raise ValueError(f"magnitude must be between 0 and 1, got {magnitude}")
        self.magnitude = magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original shape
        group_size, time_steps, channels = x.shape

        # Sample resize factor between [1-magnitude, 1]
        # This is the portion of the sequence to keep
        resize_factor = 1.0 - torch.rand(group_size, device=x.device) * self.magnitude

        # Calculate crop sizes - ensure at least 1 timestep
        crop_sizes = (time_steps * resize_factor).round().long().clamp(min=1)

        # Prepare output tensor
        out = torch.zeros_like(x)

        # Process each sample in the group
        for i in range(group_size):
            crop_size = crop_sizes[i].item()

            # Randomly select start position for crop
            if crop_size < time_steps:
                start = torch.randint(0, time_steps - crop_size + 1, (1,)).item()
            else:
                start = 0
                crop_size = time_steps

            # Extract crop
            crop = x[
                i : i + 1, start : start + crop_size
            ]  # Keep batch dim for interpolate

            # Scale back to original size
            temp = F.interpolate(
                crop.transpose(1, 2),  # Shape: (1, C, crop_size)
                size=time_steps,
                mode="linear",
                align_corners=True,
            )

            out[i] = temp.transpose(1, 2).squeeze(0)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(magnitude={self.magnitude})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(m={self.magnitude:.3f})"


class TimeMasking(TimeSeriesTransform):
    """Mask random segments of the time series with zeros.

    This transform randomly selects segments of the time series and sets them to zero,
    effectively creating "masked" regions. The number of masks and their lengths are
    configurable parameters.

    Note:
        The masking is applied independently to each sequence in the batch. The masks
        may overlap, effectively creating longer masked regions. The total proportion
        of masked timesteps will be approximately num_masks * mask_length / sequence_length.
    """

    def __init__(self, num_masks: int = 1, mask_length: float = 0.1):
        """Initialize the time masking transform.

        Args:
            num_masks: Number of masks to apply. Must be positive.
            mask_length: Length of each mask as a fraction of sequence length.
                        Must be between 0 and 1.
        """
        if num_masks <= 0:
            raise ValueError(f"num_masks must be positive, got {num_masks}")
        if not 0 < mask_length < 1:
            raise ValueError(f"mask_length must be between 0 and 1, got {mask_length}")
        self.num_masks = num_masks
        self.mask_length = mask_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size, time_steps, channels = x.shape

        # Calculate mask length in timesteps
        mask_size = max(1, int(time_steps * self.mask_length))

        # Create mask tensor (1 for keeping values, 0 for masking)
        mask = torch.ones(group_size, time_steps, channels, device=x.device)

        # Apply multiple masks
        for _ in range(self.num_masks):
            # Random start position for the mask
            start = torch.randint(0, time_steps - mask_size + 1, (1,)).item()
            # Set the mask region to 0
            mask[:, start : start + mask_size] = 0

        return x * mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_masks={self.num_masks}, mask_length={self.mask_length})"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(n={self.num_masks}, l={self.mask_length:.2f})"
        )


class Permutation(TimeSeriesTransform):
    """Randomly permute segments of the time series.

    This transform divides the time series into n_segments of equal length and
    randomly permutes a subset of these segments. This creates a new sequence that
    preserves some of the original temporal structure while modifying others.

    Note:
        This transform preserves local temporal structure within segments while
        disrupting the global temporal structure. The segment_size parameter
        controls this trade-off: larger segments preserve more local structure
        but allow for fewer permutations.

        When n_permute < n_segments, only a random subset of segments will be
        permuted, while others remain in their original positions. This allows
        for more subtle temporal modifications.
    """

    def __init__(self, n_segments: int = 5, n_permute: Optional[int] = None):
        """Initialize the permutation transform.

        Args:
            n_segments: Number of segments to divide the sequence into.
                       Must be at least 2 (otherwise no permutation is possible).
            n_permute: Number of segments to permute. Must be at least 2 and not greater
                      than n_segments. If None, all segments will be permuted.
        """
        if n_segments < 2:
            raise ValueError(
                f"n_segments must be at least 2 for permutation, got {n_segments}"
            )

        # If n_permute is not specified, permute all segments
        n_permute = n_permute if n_permute is not None else n_segments

        if n_permute < 2:
            raise ValueError(
                f"n_permute must be at least 2 for permutation, got {n_permute}"
            )
        if n_permute > n_segments:
            raise ValueError(
                f"n_permute ({n_permute}) cannot be greater than n_segments ({n_segments})"
            )

        self.n_segments = n_segments
        self.n_permute = n_permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size, time_steps, channels = x.shape

        # Calculate segment size (handle cases where time_steps isn't perfectly divisible)
        segment_size = time_steps // self.n_segments
        if segment_size == 0:
            raise ValueError(
                f"Time series length ({time_steps}) is too short to be divided "
                f"into {self.n_segments} segments"
            )

        # Adjust n_segments if necessary to handle remainder
        actual_n_segments = time_steps // segment_size
        if actual_n_segments < 2:
            raise ValueError(
                f"Time series length ({time_steps}) is too short for permutation "
                f"with segment size {segment_size}"
            )

        # Adjust n_permute if necessary
        actual_n_permute = min(self.n_permute, actual_n_segments)

        # Prepare output tensor
        out = x.clone()  # Start with a copy since we might keep some segments unchanged

        # Process each sample in the group
        for i in range(group_size):
            # Randomly select segments to permute
            segment_indices = torch.randperm(actual_n_segments)[:actual_n_permute]
            # Generate permutation for selected segments
            perm = torch.randperm(actual_n_permute)

            # Create temporary storage for segments to be permuted
            temp = torch.zeros(
                (actual_n_permute, segment_size, channels),
                device=x.device,
                dtype=x.dtype,
            )

            # Collect segments to be permuted
            for j, idx in enumerate(segment_indices):
                start = idx * segment_size
                end = start + segment_size
                temp[j] = x[i, start:end]

            # Place permuted segments back
            for j, idx in enumerate(segment_indices):
                start = idx * segment_size
                end = start + segment_size
                out[i, start:end] = temp[perm[j]]

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_segments={self.n_segments}, n_permute={self.n_permute})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n_segments}, p={self.n_permute})"


class Compose:
    """Composes several transforms together to be applied sequentially.

    This class allows you to chain multiple transforms together in a fixed order.
    Each transform is applied to the output of the previous transform.

    Example:
        >>> transforms = Compose([
        ...     Jittering(sigma=0.1),
        ...     AnotherTransform(),
        ... ])
        >>> output = transforms(input)  # Applies transforms in order
    """

    def __init__(self, transforms: List[TimeSeriesTransform]):
        """Initialize the compose transform.

        Args:
            transforms: List of transforms to apply in order. Must not be empty.
        """
        if not transforms:
            raise ValueError("transforms list must not be empty")
        if not all(isinstance(t, TimeSeriesTransform) for t in transforms):
            raise TypeError("All transforms must inherit from TimeSeriesTransform")
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {repr(t)}"
        format_string += "\n)"
        return format_string

    def __str__(self) -> str:
        return " → ".join(str(t) for t in self.transforms)


def sample_groups(
    x: torch.Tensor, n_samples: int, strategy: Literal["group", "copy"]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample groups from input tensor using specified strategy

    Args:
        x: Input tensor of shape (G, T, C) where:
            - G is the number of samples in the group
            - T is number of timesteps
            - C is number of channels
        n_samples: Number of samples to select per view
        strategy: Sampling strategy:
            - "group": Sample two different sets of n_samples
            - "copy": Sample one set of n_samples and copy it

    Returns:
        Tuple of tensors (view1, view2), each of shape (n_samples, T, C)
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")

    if x.dim() != 3:
        raise ValueError(
            f"Input tensor must have 3 dimensions (G, T, C), got shape {x.shape}"
        )

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if strategy not in ["group", "copy"]:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    # Calculate required number of samples based on strategy
    required_samples = n_samples * (2 if strategy == "group" else 1)
    if required_samples > x.shape[0]:
        raise ValueError(
            f"Not enough samples in group for '{strategy}' strategy. "
            f"Group has {x.shape[0]} samples, but need {required_samples} "
            f"({n_samples} per view)"
        )

    # Sample indices for both strategies
    indices = torch.randperm(x.shape[0])[:required_samples]

    if strategy == "group":
        # Split indices into two views
        view1 = x[indices[:n_samples]]
        view2 = x[indices[n_samples:]]
    else:  # strategy == "copy"
        # Use same indices for both views
        view1 = x[indices]
        view2 = view1.clone()

    return view1, view2


class ResamplingAugmentation(TimeSeriesTransform):
    """Resampling augmentation that processes both views at once.

    This transform performs temporal resampling in three steps:
    1. Upsamples the original time series to T_up timesteps using linear interpolation
    2. Samples two disjoint subsequences that maintain temporal coverage
    3. Resamples both sequences back to the original temporal resolution
    """

    def __init__(
        self,
        upsampling_factor: float = 2.0,
        subsequence_length_ratio: float = 0.5,
    ):
        """Initialize the resampling augmentation.

        Args:
            upsampling_factor: Factor to upsample the original time series (typically 2.0)
            subsequence_length_ratio: Length of each subsequence as a ratio of upsampled length (typically 0.5)
        """
        if upsampling_factor <= 1.0:
            raise ValueError(
                f"upsampling_factor must be > 1.0, got {upsampling_factor}"
            )
        if not 0.0 < subsequence_length_ratio < 1.0:
            raise ValueError(
                f"subsequence_length_ratio must be between 0 and 1, got {subsequence_length_ratio}"
            )

        self.upsampling_factor = upsampling_factor
        self.subsequence_length_ratio = subsequence_length_ratio

    def forward(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process two input views, returning two augmented views.

        Args:
            views: List containing two tensors [view1, view2], each of shape (G, T, C) where:
                - G is the number of samples in the group
                - T is number of timesteps
                - C is number of channels

        Returns:
            List of tensors [view1, view2], each of shape (G, T, C)
        """
        if not isinstance(views, list) or len(views) != 2:
            raise ValueError(f"Input must be a list of two tensors, got {type(views)}")

        view1, view2 = views
        group_size, time_steps, channels = view1.shape

        # Step 1: Upsample both views to higher temporal resolution
        up_steps = int(time_steps * self.upsampling_factor)
        view1_up = torch.nn.functional.interpolate(
            view1.transpose(1, 2),  # (G, C, T)
            size=up_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)  # Back to (G, T, C)

        view2_up = torch.nn.functional.interpolate(
            view2.transpose(1, 2),  # (G, C, T)
            size=up_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)  # Back to (G, T, C)

        # Step 2: Sample subsequences
        subseq_length = int(up_steps * self.subsequence_length_ratio)
        quarters = up_steps // 4
        points_per_quarter = subseq_length // 4

        # Create disjoint indices for both views ensuring coverage of all quarters
        all_indices = torch.randperm(up_steps)
        indices1 = []
        indices2 = []

        # Ensure each quarter contributes proportionally
        for q in range(4):
            quarter_start = q * quarters
            quarter_end = (q + 1) * quarters
            quarter_indices = all_indices[
                (all_indices >= quarter_start) & (all_indices < quarter_end)
            ]

            # Split indices between views
            indices1.extend(quarter_indices[:points_per_quarter].tolist())
            indices2.extend(
                quarter_indices[points_per_quarter : 2 * points_per_quarter].tolist()
            )

        # Sort indices to maintain temporal order
        indices1.sort()
        indices2.sort()

        # Extract subsequences for all samples at once
        subseq1 = view1_up[:, indices1]  # Shape: (G, subseq_length, C)
        subseq2 = view2_up[:, indices2]  # Shape: (G, subseq_length, C)

        # Resample back to original resolution
        view1_aug = torch.nn.functional.interpolate(
            subseq1.transpose(1, 2),  # (G, C, subseq_length)
            size=time_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)  # Back to (G, T, C)

        view2_aug = torch.nn.functional.interpolate(
            subseq2.transpose(1, 2),
            size=time_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

        return [view1_aug, view2_aug]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"upsampling_factor={self.upsampling_factor}, "
            f"subsequence_length_ratio={self.subsequence_length_ratio})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"up={self.upsampling_factor:.1f}, "
            f"len={self.subsequence_length_ratio:.2f})"
        )
