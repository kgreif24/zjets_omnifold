import torch
import torch.nn as nn
from functools import partial

from . import net_utils


class Conv1dBlock(nn.Module):
    """ Conv1dBlock - This class defines a block in the network
    which perform the embedding of pairwise features into a space
    which is added to the attention weights inside of the attention blocks.

    Refactored from the original ParT code to allow masking of padded inputs
    in between each convolutional block. Prevents the padded features from
    affecting the statistics of the batch normalization layers.
    """

    def __init__(self, input_dim, output_dim, activation=True):
        super().__init__()
        module_list = []
        module_list.extend(
            [
                nn.Conv1d(input_dim, output_dim, 1),
                nn.BatchNorm1d(output_dim),
            ]
        )
        if activation:
            module_list.append(nn.GELU())
        self.block = nn.Sequential(*module_list)

    def forward(self, x):
        out = self.block(x)
        return out


class PairEmbed(nn.Module):
    def __init__(
        self,
        pairwise_lv_dim,
        dims,
        remove_self_pair=False,
        normalize_input=True,
        eps=1e-8,
        for_onnx=False,
    ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.is_symmetric = (pairwise_lv_dim <= 5)
        self.remove_self_pair = remove_self_pair
        self.normalize_input = normalize_input
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(
            net_utils.pairwise_lv_fts,
            num_outputs=pairwise_lv_dim,
            eps=eps,
            for_onnx=for_onnx,
        )
        self.out_dim = dims[-1]

        # Define blocks of embedding network
        input_dim = pairwise_lv_dim
        self.input_norm = nn.BatchNorm1d(pairwise_lv_dim) if normalize_input else None
        self.embed_blocks = nn.ModuleList()
        for i, dim in enumerate(dims):
            use_act = i < len(dims) - 1
            self.embed_blocks.append(Conv1dBlock(input_dim, dim, activation=use_act))
            input_dim = dim

    def forward(self, x, mask=None):

        # Calculate pairwise features
        # Do not evaluate gradients here as there are no trainable weights
        # in the calculation of pairwise features.
        with torch.no_grad():
            batch_size, _, seq_len = x.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(
                    seq_len,
                    seq_len,
                    offset=-1 if self.remove_self_pair else 0,
                    device=x.device,
                )
                x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                xj = x[:, :, j, i]
                maski = mask[:, :, i, j]  # (batch, seq_len*(seq_len+1)/2)
                maskj = mask[:, :, j, i]
                pair_mask = maski & maskj
                x = self.pairwise_lv_fts(xi, xj)
                x = x.masked_fill(~pair_mask, 0)
            else:
                x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                if self.remove_self_pair:
                    i = torch.arange(0, seq_len, device=x.device)
                    x[:, :, i, i] = 0
                x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)

        # Embed the pairwise features
        elements = x
        if self.input_norm is not None:
            elements = self.input_norm(elements)
        for block in self.embed_blocks:
            elements = block(elements)
            elements = elements.masked_fill(~pair_mask, 0)

        # If the features are symmetric, we don't want to waste time calculating
        # the same features twice, so we just calculate the lower triangular part
        # of the matrix and then fill in the upper triangular part with the same values.
        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(
                batch_size,
                self.out_dim,
                seq_len,
                seq_len,
                dtype=elements.dtype,
                device=elements.device,
            )
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)

        return y
