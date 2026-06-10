import torch
from torch import nn
from torch.nn import functional as F


from typing import List, Tuple, Type


from .block  import SAMUpBlock,SAMPrUpBlock,SAMBasicBlock
from .common import MLPBlock_3
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        mlp_ratio: float = 4.0,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.MLP = MLPBlock_3(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.use_rel_pos = use_rel_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0,2,3,4,1)
        shortcut = x
        x = self.norm1(x)
        B, H, W, D, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W * D, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, H, W, D, -1)
        x = self.proj(x)
        x = x + shortcut

        x = x.permute(0,4,1,2,3)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim_a, dim_b, dim_out):
        super(CrossAttention, self).__init__()
        self.dim_out = dim_out
        self.query_transform = nn.Linear(dim_a, dim_out)
        self.key_transform = nn.Linear(dim_b, dim_out)
        self.value_transform = nn.Linear(dim_b, dim_out)
        self.output_transform = nn.Linear(dim_out, dim_a)
        self.att = Attention(dim=dim_a)



        # door3
        self.gate1 = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.ReLU(),
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(dim_b, dim_b),
            nn.ReLU(),
            nn.Linear(dim_b, dim_b),
            nn.Sigmoid()
        )

    def forward(self, feature_a, feature_b):
        batch_size, channels_a, height, width, depth = feature_a.shape
        _, channels_b, _, _, _ = feature_b.shape

        feature_a_flat = feature_a.permute(0, 2, 3, 4, 1).reshape(-1, channels_a)
        feature_b_flat = feature_b.permute(0, 2, 3, 4, 1).reshape(-1, channels_b)
        G1 = self.gate1(feature_a_flat)
        G2 = self.gate2(feature_b_flat)
        feature_a_flat = feature_a_flat*G1
        feature_b_flat = feature_b_flat*G2



        Q = self.query_transform(feature_a_flat)
        K = self.key_transform(feature_b_flat)
        V = self.value_transform(feature_b_flat)


        scaling_factor = torch.sqrt(torch.tensor(self.dim_out, dtype=torch.float32))
        attention_scores = Q @ K.transpose(-2, -1) / scaling_factor
        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_output = attention_scores @ V


        attention_output_transformed = self.output_transform(attention_output)
        attention_output_reshaped = attention_output_transformed.reshape(batch_size, height, width, depth, channels_a).permute(0, 4, 1, 2, 3)

        combined_features = feature_a + attention_output_reshaped
        combined_features = self.att(combined_features)

        return combined_features
# #
class MultiLayerCrossAttention(nn.Module):
    def __init__(self, num_layers, dim_a, dim_b, dim_out, mlp_hidden_dim):
        super(MultiLayerCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttention(dim_a, dim_b, dim_out) for _ in range(num_layers)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(dim_a, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim_a)
        )

    def forward(self, feature_a, feature_b):
        for layer in self.layers:
            feature_a = layer(feature_a, feature_b)
        combined_features_flat = feature_a.permute(0, 2, 3, 4, 1).reshape(-1, feature_a.size(1))
        enhanced_features_flat = self.mlp(combined_features_flat)
        enhanced_features = enhanced_features_flat.reshape(feature_a.size(0), feature_a.size(2), feature_a.size(3),feature_a.size(4), feature_a.size(1)).permute(0, 4, 1, 2, 3)
        return enhanced_features





class Decoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        out_channels: int,
        dropout_p: float,
        trilinear: bool,
        hidden_states,

    ):
        super().__init__()



        self.cross_att4 = MultiLayerCrossAttention(num_layers=4,dim_a=1024, dim_b=384, dim_out=256,mlp_hidden_dim=1024)




        self.hidden_states = hidden_states
        self.encoder1 = SAMBasicBlock(
            in_channels=3,
            out_channels=base_channels,
            dropout_p=dropout_p,
        )
        self.encoder2 = SAMPrUpBlock(
            in_channels=in_channels,
            out_channels=2*base_channels,
            num_layer=3,
            dropout_p=dropout_p,
            trilinear=trilinear,
        )
        self.encoder3 = SAMPrUpBlock(
            in_channels=in_channels,
            out_channels=4 * base_channels,
            num_layer=2,
            dropout_p=dropout_p,
            trilinear=trilinear,
        )
        self.encoder4 = SAMPrUpBlock(
            in_channels=in_channels,
            out_channels=8 * base_channels,
            num_layer=1,
            dropout_p=dropout_p,
            trilinear=trilinear,
        )
        self.decoder4 = SAMUpBlock(
            in_channels1=in_channels,
            in_channels2=8 * base_channels,
            dropout_p=dropout_p,
            trilinear=trilinear,

        )
        self.decoder3 = SAMUpBlock(
            in_channels1=8 * base_channels,
            in_channels2=4 * base_channels,
            dropout_p=dropout_p,
            trilinear=trilinear,

        )
        self.decoder2 = SAMUpBlock(
            in_channels1=4 * base_channels,
            in_channels2=2 * base_channels,
            dropout_p=dropout_p,
            trilinear=trilinear,

        )
        self.decoder1 = SAMUpBlock(
            in_channels1=2 * base_channels,
            in_channels2=base_channels,
            dropout_p=dropout_p,
            trilinear=trilinear,

        )
        self.out = nn.Conv3d(in_channels=base_channels, out_channels=out_channels,kernel_size=1, stride=1)




    def proj_feat(self,x, b_size, d_size):
        x = x.view(b_size, d_size, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        return x


    def forward(
        self,
        batched_input: torch.Tensor,
        hidden_states_out: torch.Tensor,
        image_embeddings,
         up5,
        b_size,  d_size,

    ):
        x1 = batched_input.view(b_size,d_size,batched_input.size(1),batched_input.size(2),batched_input.size(3))
        x1 = x1.permute(0,2,3,4,1)
        enc1 = self.encoder1(x1)
        x2 = hidden_states_out[self.hidden_states[0]]
        x2 = self.proj_feat(x2,b_size, d_size)
        enc2 = self.encoder2(x2)
        x3 = hidden_states_out[self.hidden_states[1]]
        x3 = self.proj_feat(x3,b_size, d_size)
        enc3 = self.encoder3(x3)
        x4 = hidden_states_out[self.hidden_states[2]]
        x4 = self.proj_feat(x4,b_size, d_size)
        enc4 = self.encoder4(x4)
        x5 = hidden_states_out[self.hidden_states[3]]
        dec5 = self.proj_feat(x5,b_size,  d_size)
        dec5 = self.cross_att4(dec5,up5)
        dec4 = self.decoder4(dec5,enc4)
        dec3 = self.decoder3(dec4,enc3)
        dec2 = self.decoder2(dec3,enc2)
        dec1 = self.decoder1(dec2,enc1)
        output = self.out(dec1)


        return output


