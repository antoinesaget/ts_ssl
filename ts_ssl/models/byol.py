import copy

import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import BYOLPredictionHead, BYOLProjectionHead

from ts_ssl.models.ssl_base import SSLBase


def update_moving_average(target, source, momentum):
    """Update target network using exponential moving average"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data = (
            momentum * target_param.data + (1 - momentum) * source_param.data
        )


class BYOL(SSLBase):
    """BYOL implementation for time series data"""

    def __init__(
        self,
        encoder,
        projection_dim: int,
        prediction_dim: int,
        momentum: float = 0.99,
        n_samples_per_group: int = 4,
        name: str = "byol",
    ):
        super().__init__(
            encoder,
            n_samples_per_group=n_samples_per_group,
            name=name,
        )

        # Online network components
        self.projector = BYOLProjectionHead(
            self.embedding_dim, self.embedding_dim, projection_dim
        )
        self.predictor = BYOLPredictionHead(
            projection_dim, self.embedding_dim, prediction_dim
        )

        # Target network components
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)

        # Disable gradient updates for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.criterion = NegativeCosineSimilarity()

        self.momentum = momentum

    def compile(self):
        super().compile()
        self.projector = torch.compile(self.projector)
        self.predictor = torch.compile(self.predictor)
        self.criterion = torch.compile(self.criterion)
        self.target_encoder = torch.compile(self.target_encoder)
        self.target_projector = torch.compile(self.target_projector)

    def _get_features(self, x, encoder=None, aggregate=True):
        """Extract features from input using specified encoder

        Args:
            x: Input tensor
            encoder: Encoder to use (default: self.encoder)
        """
        encoder = encoder or self.encoder
        h = encoder(x, aggregate=aggregate)
        return h

    def forward(self, x, target=False):
        """Forward pass through the network

        Args:
            x: Input tensor
            target: Whether to use target encoder (default: False)
        """
        # Choose encoder based on target flag
        encoder = self.target_encoder if target else self.encoder
        projector = self.target_projector if target else self.projector

        # Forward pass with correct encoder
        h = self._get_features(x, encoder=encoder)
        z = projector(h)

        if not target:
            z = self.predictor(z)
        else:
            z = z.detach()
        return z

    def training_step(self, batch):
        """Compute BYOL's loss"""
        x1, x2 = batch  # Two augmented views

        # Online network forward passes
        q1 = self.forward(x1, target=False)  # Prediction of view 1
        q2 = self.forward(x2, target=False)  # Prediction of view 2

        # Target network forward passes
        with torch.no_grad():
            z1 = self.forward(x1, target=True)  # Projection of view 1
            z2 = self.forward(x2, target=True)  # Projection of view 2

        # Compute loss for both directions
        loss = 0.5 * (self.criterion(q1, z2.detach()) + self.criterion(q2, z1.detach()))

        # Update target network
        update_moving_average(self.target_encoder, self.encoder, self.momentum)
        update_moving_average(self.target_projector, self.projector, self.momentum)

        return loss
