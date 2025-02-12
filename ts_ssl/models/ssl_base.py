import torch
from torch import nn


class SSLBase(nn.Module):
    """Base class for SSL models implementing common functionalities"""

    def __init__(
        self,
        encoder,
        name: str,
        n_samples_per_group: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = encoder.embedding_dim
        self.name = name
        self.n_samples_per_group = n_samples_per_group

    def compile(self):
        self.encoder = torch.compile(self.encoder)

    def _get_features(self, x, aggregate=True):
        """Extract features from input using encoder"""
        h = self.encoder(x, aggregate=aggregate)
        return h

    def forward(self, x):
        """Forward pass through the model"""
        raise NotImplementedError("Subclasses must implement forward method")

    def training_step(self, batch):
        """Perform a training step"""
        raise NotImplementedError("Subclasses must implement training_step method")

    def get_features(self, x, aggregate=True):
        """Public method to get features for validation

        Args:
            x: Input tensor
            aggregate: Whether to apply aggregation layer (default: True)

        Returns:
            Features from encoder if aggregate=False, or aggregated features if aggregate=True
        """
        with torch.no_grad():
            return self._get_features(x, aggregate=aggregate)
