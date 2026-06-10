import torch
from torch import nn
from .block  import CenterSAMUpBlock,CenterSAMPrUpBlock,CenterSAMBasicBlock




class CenterDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_states,
    ):
        super().__init__()

        self.hidden_states = hidden_states

    def proj_feat(self, x, b_size, d_size):
        x = x.view(b_size, d_size, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        return x

    def forward(
        self,
        batched_input: torch.Tensor,
        hidden_states_out: torch.Tensor,
        b_size,
        d_size,
    ):

        x5 = hidden_states_out[self.hidden_states[3]]
        dec5 = self.proj_feat(x5, b_size, d_size)

        return dec5


