import torch
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from ts_ssl.models.ssl_base import SSLBase


class SimCLR(SSLBase):
    """SimCLR implementation for time series data"""

    def __init__(
        self,
        encoder,
        projection_dim: int,
        temperature: float = 0.07,
        n_samples_per_group: int = 4,
        name: str = "simclr",
    ):
        super().__init__(
            encoder,
            n_samples_per_group=n_samples_per_group,
            name=name,
        )
        self.projector = SimCLRProjectionHead(
            self.embedding_dim, self.embedding_dim, projection_dim
        )
        self.temperature = temperature
        self.criterion = NTXentLoss(temperature=temperature)

    def compile(self):
        super().compile()
        self.projector = torch.compile(self.projector)
        self.criterion = torch.compile(self.criterion)

    def forward(self, x, aggregate=True):
        h = self._get_features(x, aggregate=aggregate)
        z = self.projector(h)
        return z

    def training_step(self, batch):
        """Compute NT-Xent loss for SimCLR"""
        x1, x2 = batch  # Two augmented views

        # Get projections of both views
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        loss = self.criterion(z1, z2)
        return loss
