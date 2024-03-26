""" 
input_distributed.py - This file implements a pytorch module that applies a module to each 
element of a sequence. Inspired by the keras TimeDistributed layer. Humbly pillaged from
https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/3

Author: Kevin Greif
01/25/2024
python3
"""

import torch.nn as nn


class InputDistributed(nn.Module):

    def __init__(self, module):
        super(InputDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x -> (batch_len, input_size, seq_len)
        # If not, then just apply the module to the sequence
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis, taking care that we retain placing
        # the features of each element next to each other.
        x_reshape = x.permute(0, 2, 1).flatten(end_dim=1)  # (batch_len * seq_len, input_size)

        # Apply the module
        y = self.module(x_reshape) # (batch_len * seq_len, output_size)

        # Reshape Y to the correct output shape, again taking care to maintain placement
        y = y.reshape(x.size(0), x.size(-1), y.size(-1)).permute(0, 2, 1) # (batch_len, output_size, seq_len)

        return y