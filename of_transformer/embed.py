import torch.nn as nn

from . import input_distributed

# Should "input_dim" here be the number of particle features or the maximum
# number of particles? I think the latter, and then we mask after the
# embedding layer.


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu"):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend(
                [
                    nn.Linear(input_dim, dim),
                    # nn.BatchNorm1d(dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
            input_dim = dim
        embed_one_element = nn.Sequential(*module_list)
        self.embed = input_distributed.InputDistributed(embed_one_element)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        if self.input_bn is not None:
            x = self.input_bn(x)
        # x: (batch, seq_len, dims[-1])
        return self.embed(x)
