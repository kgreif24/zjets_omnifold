""" of_transformer.py - First implementation of a transformer for use in
Omnifold. Modeled after the Particle Transformer (ParT).

Author: Kevin Greif
12/12/23
python3
"""

# Standard imports
import torch
import torch.nn as nn
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
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_nodes=[],
        fc_dropout=0.0,
        activation="gelu",
        # misc
        for_inference=False,
        use_amp=False,
        **kwargs
    ) -> None:
        super(OfTransformer, self).__init__(**kwargs)

        # Set instance variables
        self.num_heads = num_heads
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
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

        # Confused why we need a copy? Maybe just for using this logger thing?
        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        # Embedding layers
        self.pair_extra_dim = pair_extra_dim
        print("Batch normalization disabled in embeddings!")
        self.embed = (
            embed.Embed(
                input_dim, embed_dims, activation=activation, normalize_input=False
            )
            if len(embed_dims) > 0
            else nn.Identity()
        )

        self.pair_embed = (
            pair_embed.PairEmbed(
                # The number of heads is added to the pair_embed_dims since we add
                # the pairwise features to the weights for **each** head.
                pair_input_dim,
                pair_extra_dim,
                pair_embed_dims + [cfg_block["num_heads"]],
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair,
                normalize_input=False,
                for_onnx=for_inference,
            )
            if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0
            else None
        )

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

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 3, P) [pT, eta, phi]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        # Get padding mask for use in attention blocks
        # Note padding is denoted by a 1 in pytorch MHA
        with torch.no_grad():
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):

            # input embedding
            x = self.embed(x).masked_fill(~mask, 0)  # (batch, embed_dim, seq_len)

            # after input embedding, reshape to (seq_len, batch, embed_dim)
            # for attention layers
            x = x.permute(2, 0, 1)  # (seq_len, batch, embed_dim)
            mask = mask.permute(0, 2, 1)  # (seq_len, batch, 1)

            pair_mask = mask.float() @ mask.float().transpose(-1, -2)  
            # ^ (batch, seq_len, seq_len)
            pair_mask = ~(pair_mask.bool()).repeat_interleave(self.num_heads, dim=0)
            # ^ (batch*num_heads, seq_len, seq_len)
            attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))
            # ^ (batch*num_heads, seq_len, seq_len)
            attn_mask = attn_mask + (-1e9*pair_mask.float())

            # transform
            for blk in self.blocks:
                x = blk(x, x_cls=None, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  
            # ^ (1, batch, embed_dim)
            for blk in self.cls_blocks:
                cls_tokens = blk(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            return output
