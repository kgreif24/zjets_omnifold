"""of_transformer.py - First implementation of a transformer for use in
Omnifold. Modeled after the Particle Transformer (ParT).

Author: Kevin Greif
12/12/23
python3
"""

# Standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Import network pieces
from . import embed
from . import pair_embed
from . import block

# Import utilities
from . import net_utils


class OfTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=1,
        # network configurations
        pair_input_dim=3,
        global_input_dim=0,
        remove_self_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        global_embed_dims=[],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_nodes=[],
        fc_dropout=0.0,
        activation="gelu",
        # misc
        **kwargs
    ) -> None:
        super(OfTransformer, self).__init__(**kwargs)

        # Set instance variables
        self.num_heads = num_heads

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        global_embed_dims.append(embed_dim)
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        # Embedding layers
        self.embed = (
            embed.Embed(
                input_dim, embed_dims, activation=activation, normalize_input=False
            )
            if len(embed_dims) > 0
            else nn.Identity()
        )

        self.pair_embed = pair_embed.PairEmbed(
            # The number of heads is added to the pair_embed_dims since we add
            # the pairwise features to the weights for **each** head.
            pair_input_dim,
            pair_embed_dims + [cfg_block["num_heads"]],
            remove_self_pair=remove_self_pair,
            normalize_input=False,
        )

        global_embed = nn.ModuleList()
        for dim in global_embed_dims:
            global_embed.extend(
                [
                    nn.Linear(global_input_dim, dim),
                    nn.GELU(),
                ]
            )
            global_input_dim = dim
        self.global_embed = nn.Sequential(*global_embed)

        # Transformer layers
        self.blocks = nn.ModuleList(
            [block.Block(**cfg_block) for _ in range(num_layers)]
        )
        self.cls_blocks = nn.ModuleList(
            [block.Block(**cfg_cls_block) for _ in range(num_cls_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Fully connected layers
        if fc_nodes is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim in fc_nodes:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(fc_dropout)
                    )
                )
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        net_utils.trunc_normal_(self.cls_token, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, x, v=None, mask=None, glb_features=None):
        # x: (N, C, P)
        # v: (N, 3, P) [pT, eta, phi]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # global: (N, G) -- global features

        # input embedding
        x = self.embed(x)
        x = x.masked_fill(~mask, 0)  # (batch, embed_dim, seq_len)

        # pairwise embedding
        pairwise_embedding = self.pair_embed(v, mask=mask).view(
            -1, v.size(-1), v.size(-1)
        )
        # ^ (batch*num_heads, seq_len, seq_len))

        # global embedding
        if glb_features is not None:
            glb_features = self.global_embed(glb_features)
            glb_features = glb_features.unsqueeze(-1)  # (batch, embed_dim, 1)
            glb_features = glb_features.permute(2, 0, 1)  # (1, batch, embed_dim)

        # After embeddings, need to alter masks and pairwise features
        # if we have global features. Add 1's to the mask tensor
        # so they participate in the attention blocks, and add 0's to the
        # pairwise features since we don't want to shift attention weights for them
        qmask, kmask = mask, mask
        if glb_features is not None:
            kmask = torch.cat((torch.ones_like(kmask[:, :, :1]), kmask), dim=2)
            # ^ (N, 1, P+1)
            pairwise_embedding = F.pad(pairwise_embedding, (1, 0))

        # For class attention blocks, we need a padding mask
        # Note padding is denoted by a 1 in pytorch MHA
        with torch.no_grad():
            padding_mask = kmask.squeeze(1)  # (N, P)
            padding_mask = torch.cat(
                (torch.ones_like(padding_mask[:, :1]), padding_mask), dim=1
            )
            # Padding is marked by a 1 in pytorch MHA
            padding_mask = ~padding_mask
            # ^ (N, P+1)

        # Reshape inputs to (seq_len, batch, embed_dim)
        # for attention layers
        x = x.permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        qmask = qmask.permute(0, 2, 1)  # (batch, seq_len, 1)
        kmask = kmask.permute(0, 2, 1)  # (batch, seq_len + 1, 1)

        # Build mask for pairwise features
        pair_mask = qmask.float() @ kmask.float().transpose(-1, -2)
        # ^ (batch, seq_len, seq_len)
        pair_mask = ~(pair_mask.bool()).repeat_interleave(self.num_heads, dim=0)
        # ^ (batch*num_heads, seq_len, seq_len)
        attn_mask = pairwise_embedding + (-1e9 * pair_mask.float())

        # transform
        for blk in self.blocks:
            x = blk(x, x_cls=None, attn_mask=attn_mask, glb_features=glb_features)

        # extract class token
        cls_tokens = self.cls_token.expand(1, x.size(1), -1)
        # ^ (1, batch, embed_dim)
        for blk in self.cls_blocks:
            cls_tokens = blk(
                x,
                x_cls=cls_tokens,
                padding_mask=padding_mask,
                glb_features=glb_features,
            )

        x_cls = self.norm(cls_tokens).squeeze(0)

        # fc
        if self.fc is None:
            return x_cls
        output = self.fc(x_cls)
        return output
