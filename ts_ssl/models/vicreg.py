import torch
from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead

from ts_ssl.models.ssl_base import SSLBase


class VICReg(SSLBase):
    """VICReg implementation for time series data"""

    def __init__(
        self,
        encoder,
        projection_dim: int,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        n_samples_per_group: int = 4,
        name="vicreg",
    ):
        super().__init__(encoder, n_samples_per_group=n_samples_per_group, name=name)
        self.projector = VICRegProjectionHead(
            input_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            output_dim=projection_dim,
            num_layers=2,
        )
        self.criterion = VICRegLoss(
            lambda_param=sim_coeff,
            mu_param=std_coeff,
            nu_param=cov_coeff,
        )

    def compile(self):
        super().compile()
        self.projector = torch.compile(self.projector)
        self.criterion = torch.compile(self.criterion)

    def forward(self, x):
        h = self._get_features(x)
        z = self.projector(h)
        return z

    def training_step(self, batch):
        """Compute VICReg loss"""
        x1, x2 = batch  # Two augmented views

        # Get projections of both views
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        loss = self.criterion(z1, z2)
        return loss
