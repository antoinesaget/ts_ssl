# gcd.py
import torch
from torch import nn
from ts_ssl.models.ssl_base import SSLBase
from lightly.loss import NTXentLoss
import torch.nn.functional as F


class GCD(SSLBase):
    """GCD model adapted for time series using SSLBase"""

    def __init__(
        self,
        encoder,
        projection_dim: int,
        n_clusters: int,
        temperature: float = 0.07,
        n_samples_per_group: int = 4,
        name: str = "gcd",
    ):
        super().__init__(encoder, name=name, n_samples_per_group=n_samples_per_group)

        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, projection_dim),
        )

        self.cluster_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, n_clusters)
        )

        self.temperature = temperature
        self.contrastive_loss = NTXentLoss(temperature=temperature)

    def compile(self):
        super().compile()
        self.projector = torch.compile(self.projector)
        self.cluster_head = torch.compile(self.cluster_head)

    def forward(self, x, aggregate=True):
        """Forward pass"""
        h = self._get_features(x, aggregate=aggregate)
        z = F.normalize(self.projector(h), dim=-1)  
        logits = self.cluster_head(z)
        return z, logits

    def entropy_loss(self, logits):
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-4)  
        avg_probs = probs.mean(dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs))
        return -entropy

    def training_step(self, batch):
        if len(batch) == 3:
            x1, x2, y = batch
        else:
            x1, x2 = batch
            y = None

        z1, logits1 = self.forward(x1)
        z2, logits2 = self.forward(x2)

        contrastive = self.contrastive_loss(z1, z2)
        entropy1 = self.entropy_loss(logits1)
        entropy2 = self.entropy_loss(logits2)

        total_loss = contrastive + 0.1 * (entropy1 + entropy2)

        # Handle partially labeled batch
        if y is not None:
            mask = y != -1  # labeled indices
            if mask.sum() > 0:
                y_labeled = y[mask]
                logits1_labeled = logits1[mask]
                logits2_labeled = logits2[mask]

                supervised_loss1 = F.cross_entropy(logits1_labeled, y_labeled)
                supervised_loss2 = F.cross_entropy(logits2_labeled, y_labeled)
                supervised_loss = supervised_loss1 + supervised_loss2

                total_loss += 0.5 * supervised_loss
        return total_loss
