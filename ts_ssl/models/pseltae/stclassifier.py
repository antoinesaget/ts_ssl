import torch
import torch.nn as nn

from ts_ssl.models.pseltae.ltae import LTAE
from ts_ssl.models.pseltae.pse import PixelSetEncoder


class PseLTae(nn.Module):
    """
    Pixel-Set encoder + Lightweight Temporal Attention Encoder sequence classifier
    """

    def __init__(
        self,
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[132, 128],
        n_head=16,
        d_k=8,
        d_model=256,
        mlp3=[256, 128],
        dropout=0.2,
        T=1000,
        len_max_seq=24,
        positions=None,
        mlp4=[128, 64, 32, 20],
        embedding_dim=128,
    ):
        super(PseLTae, self).__init__()
        self.embedding_dim = embedding_dim
        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=[input_dim] + mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=False,
            extra_size=0,
        )
        self.temporal_encoder = LTAE(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3 + [embedding_dim],
            dropout=dropout,
            T=T,
            len_max_seq=len_max_seq,
            positions=positions,
            return_att=False,
        )

    def forward(self, input, aggregate=True):
        """
        Args:
           input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
           Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
           Pixel-Mask : Batch_size x Sequence length x Number of pixels
        """
        # input is of shape B x G x T x C
        input = input.permute(0, 2, 3, 1)  # B x T x C x G
        mask = torch.ones_like(input[:, :, 0, :])  # B x T x G

        # spatial encoder takes
        # (B x T x C x G, B x T x G)
        out = self.spatial_encoder((input, mask))
        out = self.temporal_encoder(out)

        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}%".format(
                s / total * 100, t / total * 100
            )
        )

        return total


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
