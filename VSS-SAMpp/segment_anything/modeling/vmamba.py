
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from mamba_ssm import Mamba
from typing import Optional, Tuple, Type

from .common import MLPBlock_2

class VMAmba(nn.Module):

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        d_size: int = 5,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,

    ) -> None:

        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )


        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)



    def forward(self, x: torch.Tensor,d_size) :
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        mamba_hidden_out = []
        for blk in self.blocks:
            x = blk(x,d_size)
            mamba_hidden_out.append(x)

        return x, mamba_hidden_out


class Block(nn.Module):

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:


        super().__init__()

        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.mlp = MLPBlock_2(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)





    def forward(self, x: torch.Tensor,d_size) -> torch.Tensor:
        b_size, hw_size, dim = x.shape[0], x.shape[2], x.shape[3]
        shortcut = x
        x = self.norm3(x)
        x = x.contiguous().view(int(b_size / d_size), -1,dim)
        x = self.mamba(x)
        x = x.contiguous().view(b_size, hw_size, hw_size, dim)
        x = shortcut + x

        x = x + self.mlp(self.norm4(x))

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x