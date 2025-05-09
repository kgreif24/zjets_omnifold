import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        add_bias_kv=False,
        activation="gelu",
        scale_fc=True,
        scale_attn=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True)
            if scale_heads
            else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True)
            if scale_resids
            else None
        )

    def forward(
        self, x, x_cls=None, padding_mask=None, attn_mask=None, glb_features=None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer
                with shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len+1)` where padding
                elements are indicated by ``1``.
            attn_mask (Tensor, optional): FloatTensor of shape
                `(batch * num_heads, seq_len, seq_len)` where
                attention weights are masked with ``-inf``. Pairwise features
                can be added to attention weights with non-zero values.
            glb_features (Tensor, optional): FloatTensor of shape
                `(1, batch, embed_dim)` where global features are appended to
                the sequence before attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        # Default QKV settings for no class attention and no global features
        residual = x
        x = self.pre_attn_norm(x)
        q = x
        kv = x

        # If we have global features, prepend them to KV but queries are still
        # the original input
        if glb_features is not None:
            kv = torch.cat((glb_features, kv), dim=0)
            # ^ (seq_len+1, batch, embed_dim)

        # If we have class attention, prepend the class token to KV and
        # the queries become the class token alone
        if x_cls is not None:
            # Note assumed padding mask already has pre-pended zeros
            # for the class token
            residual = x_cls
            q = x_cls
            kv = torch.cat((x_cls, kv), dim=0)
            # ^ (seq_len+1, batch, embed_dim)

        # Compute attention
        x = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x
