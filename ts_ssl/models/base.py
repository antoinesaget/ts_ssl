from torch import nn


class Add(nn.Module):
    """Simple addition module for residual connections"""

    def forward(self, x, y):
        return x + y


class Squeeze(nn.Module):
    """Squeeze module to remove dimensions of size 1"""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation"""

    def __init__(self, ni, nf, kernel_size, stride=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(
            ni, nf, kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm1d(nf)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MLP(nn.Module):
    """MLP for BYOL's projection and prediction heads"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
