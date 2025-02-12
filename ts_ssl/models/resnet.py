from torch import nn

from ts_ssl.models.base import Add, ConvBlock, Squeeze


class ResBlock(nn.Module):
    """Residual block for 1D time series data"""

    def __init__(
        self, ni, nf, kernel_sizes=(7, 5, 3), use_identity_shortcut=False, stride=1
    ):
        super().__init__()
        self.convblock1 = ConvBlock(ni, nf, kernel_sizes[0], stride=stride)
        self.convblock2 = ConvBlock(nf, nf, kernel_sizes[1])
        self.convblock3 = ConvBlock(nf, nf, kernel_sizes[2], act=False)

        # Shortcut connection
        if use_identity_shortcut and ni == nf and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvBlock(ni, nf, 1, stride=stride, act=False)

        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResNet(nn.Module):
    """Original ResNet implementation for time series data

    Receptive Field (RF) and shape progression for input B x 12 x 60:
    Layer            RF    Total Stride    Output Shape
    ------------    ----   ------------    ------------
    Input            1          1          B x 12 x 60
    resblock1       13          1          B x 256 x 60
    resblock2       25          1          B x 512 x 60
    resblock3       37          1          B x 512 x 60
    gap            37          -          B x 512 x 1
    squeeze        37          -          B x 512
    """

    def __init__(self, n_channels, n_filters=256, embedding_dim=512):
        super().__init__()
        kernel_sizes = (7, 5, 3)
        self.resblock1 = ResBlock(n_channels, n_filters, kernel_sizes)
        self.resblock2 = ResBlock(n_filters, n_filters * 2, kernel_sizes)
        self.resblock3 = ResBlock(n_filters * 2, embedding_dim, kernel_sizes)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)

    def forward(self, x):  # x of shape (B x G) x C x T
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.squeeze(self.gap(x))
        return x


class ResNetGrouped(nn.Module):
    """Original ResNet implementation for time series data

    Receptive Field (RF) and shape progression for input B x 12 x 60:
    Layer            RF    Total Stride    Output Shape
    ------------    ----   ------------    ------------
    Input            1          1          B x 12 x 60
    resblock1       13          1          B x 256 x 60
    resblock2       25          1          B x 512 x 60
    resblock3       37          1          B x 512 x 60
    gap            37          -          B x 512 x 1
    squeeze        37          -          B x 512
    """

    def __init__(self, n_channels, n_filters=256, embedding_dim=512):
        super().__init__()
        kernel_sizes = (7, 5, 3)

        self.n_channels = n_channels
        self.n_filters = n_filters
        self.embedding_dim = embedding_dim

        self.resblock1 = ResBlock(n_channels, n_filters, kernel_sizes)
        self.resblock2 = ResBlock(n_filters, n_filters * 2, kernel_sizes)
        self.resblock3 = ResBlock(n_filters * 2, embedding_dim, kernel_sizes)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.aggregation_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, aggregate=True):  # x of shape B x G x T x C
        group_size = x.shape[1]
        x = x.view(x.shape[0] * group_size, *x.shape[2:])  # (B x G) x T x C
        x = x.transpose(1, 2)  # (B x G) x C x T
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.squeeze(self.gap(x))
        x = x.view(x.shape[0] // group_size, group_size, -1)
        if aggregate:
            x = x.transpose(1, 2)  # B x F x G
            x = self.aggregation_layer(x).flatten(start_dim=1)
        return x
