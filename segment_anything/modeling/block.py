import torch.nn as nn
from .common import BasicBlock,UpBlock,PrUpBlock
from .common import CenterUpBlock, CenterPrUpBlock, CenterDownBlock, CenterBasicBlock


class SAMBasicBlock (nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_p: float
    ):
        super(SAMBasicBlock,self).__init__()
        self.convs1 = BasicBlock(in_channels,out_channels,dropout_p)
        self.convs2 = BasicBlock(out_channels, out_channels, dropout_p)


    def forward(self, x):
        x1 = self.convs1(x)
        x2 = self.convs2(x1)

        return x2



class SAMPrUpBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layer: int,
            dropout_p: float,
            trilinear: bool
    ):
        super(SAMPrUpBlock, self).__init__()
        layers = []
        for _ in range(num_layer):
            layers.append(nn.Sequential(
                PrUpBlock(in_channels, out_channels, trilinear=trilinear),
                BasicBlock(out_channels, out_channels, dropout_p)
            ))
            in_channels = out_channels
        self.PrUpBlock = nn.Sequential(*layers)

    def forward(self, x):
        return self.PrUpBlock(x)


class SAMUpBlock(nn.Module):
    def __init__(
            self,
            in_channels1: int,
            in_channels2: int,
            dropout_p: float,
            trilinear: bool
    ):
        super(SAMUpBlock, self).__init__()

        self.UpBlock=UpBlock(in_channels1, in_channels2, dropout_p, trilinear=trilinear)

    def forward(self, x1,x2):
        return self.UpBlock(x1,x2)


class CenterSAMBasicBlock (nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_p: float
    ):
        super(CenterSAMBasicBlock,self).__init__()
        self.convs1 = CenterBasicBlock(in_channels,out_channels,dropout_p)
        self.convs2 = CenterBasicBlock(out_channels, out_channels, dropout_p)


    def forward(self, x):
        x1 = self.convs1(x)
        x2 = self.convs2(x1)

        return x2



class CenterSAMPrUpBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layer: int,
            dropout_p: float,
            trilinear: bool
    ):
        super(CenterSAMPrUpBlock, self).__init__()
        layers = []
        for _ in range(num_layer):
            layers.append(nn.Sequential(
                CenterPrUpBlock(in_channels, out_channels, trilinear=trilinear),
                CenterBasicBlock(out_channels, out_channels, dropout_p)
            ))
            in_channels = out_channels
        self.PrUpBlock = nn.Sequential(*layers)

    def forward(self, x):
        return self.PrUpBlock(x)


class CenterSAMUpBlock(nn.Module):
    def __init__(
            self,
            in_channels1: int,
            in_channels2: int,
            dropout_p: float,
            trilinear: bool
    ):
        super(CenterSAMUpBlock, self).__init__()

        self.UpBlock=CenterUpBlock(in_channels1, in_channels2, dropout_p, trilinear=trilinear)

    def forward(self, x1,x2):
        return self.UpBlock(x1,x2)

