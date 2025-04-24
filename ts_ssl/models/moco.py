import copy

import torch
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum

from ts_ssl.models.ssl_base import SSLBase


class MoCo(SSLBase):
    """MoCo implementation for time series data"""

    def __init__(
        self,
        encoder,
        projection_dim: int,
        memory_bank_size: int = 4096,
        momentum: float = 0.99,
        temperature: float = 0.07,
        n_samples_per_group: int = 4,
        name: str = "moco",
    ):
        super().__init__(
            encoder,
            n_samples_per_group=n_samples_per_group,
            name=name,
        )

        # Online network components
        self.projector = MoCoProjectionHead(
            self.embedding_dim, self.embedding_dim, projection_dim
        )

        # Momentum network components
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)

        # Disable gradient updates for momentum network
        deactivate_requires_grad(self.momentum_encoder)
        deactivate_requires_grad(self.momentum_projector)

        # Loss function
        self.criterion = NTXentLoss(
            memory_bank_size=(memory_bank_size, projection_dim),
            temperature=temperature,
        )

        self.momentum = momentum

    def compile(self):
        super().compile()
        self.projector = torch.compile(self.projector)
        self.momentum_encoder = torch.compile(self.momentum_encoder)
        self.momentum_projector = torch.compile(self.momentum_projector)
        self.criterion = torch.compile(self.criterion)

    def _get_features(self, x, encoder=None, aggregate=True):
        """Extract features from input using specified encoder

        Args:
            x: Input tensor
            encoder: Encoder to use (default: self.encoder)
        """
        encoder = encoder or self.encoder
        h = encoder(x, aggregate=aggregate)
        return h

    def forward(self, x, momentum=False):
        """Forward pass through the network

        Args:
            x: Input tensor
            momentum: Whether to use momentum encoder (default: False)
        """
        # Choose encoder and projector based on momentum flag
        encoder = self.momentum_encoder if momentum else self.encoder
        projector = self.momentum_projector if momentum else self.projector

        # Forward pass with correct encoder
        h = self._get_features(x, encoder=encoder)
        z = projector(h)

        if momentum:
            z = z.detach()
        return z

    def training_step(self, batch):
        """Compute MoCo's loss with momentum update"""
        x_q, x_k = batch  # Query and key views

        # Update momentum encoder
        update_momentum(self.encoder, self.momentum_encoder, m=self.momentum)
        update_momentum(self.projector, self.momentum_projector, m=self.momentum)

        # Forward passes
        query = self.forward(x_q, momentum=False)
        key = self.forward(x_k, momentum=True)

        # Compute loss
        loss = self.criterion(query, key)
        return loss
