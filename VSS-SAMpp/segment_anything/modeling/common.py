import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class MLPBlock_2(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin3 = nn.Linear(embedding_dim, mlp_dim)
        self.lin4 = nn.Linear(mlp_dim, embedding_dim)
        self.act1 = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin4(self.act1(self.lin3(x)))

class MLPBlock_3(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin5 = nn.Linear(embedding_dim, mlp_dim)
        self.lin6 = nn.Linear(mlp_dim, embedding_dim)
        self.act2 = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin6(self.act2(self.lin5(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class BasicBlock(nn.Module):
    """Two convolution layers with GroupNorm and ReLU"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.conv(x)



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            BasicBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, dropout_p, trilinear=True):
        super(UpBlock, self).__init__()
        self.trilinear = trilinear

        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)


        self.conv = BasicBlock(in_channels2 * 2, in_channels2, dropout_p)

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)

        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x



class PrUpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, trilinear=True):
        super(PrUpBlock, self).__init__()

        self.trilinear = trilinear

        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)


        self.refine = nn.Sequential(
            nn.Conv3d(in_channels2, in_channels2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=in_channels2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        if self.trilinear:
            x1 = self.conv1x1(x1)

        x = self.up(x1)
        x = self.refine(x)
        return x



class CenterBasicBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(CenterBasicBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.conv_conv(x)


class CenterDownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(CenterDownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            BasicBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class CenterUpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, dropout_p,
                 trilinear=True):
        super(CenterUpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.conv1 = BasicBlock(in_channels2 * 2, in_channels2, dropout_p)
        self.conv2 = BasicBlock(in_channels2, in_channels2, dropout_p)

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class CenterPrUpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, trilinear=True):
        super(CenterPrUpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=(2, 2, 1), stride=(2, 2, 1))


    def forward(self, x1):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x = self.up(x1)

        return x








