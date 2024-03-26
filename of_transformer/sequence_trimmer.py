import torch
import torch.nn as nn

import random

class SequenceTrimmer(nn.Module):
    """ SequenceTrimmer - This class shuffles and truncates the input particles
    during model training. 
    
    The "perm" tensor contains a random permutation of the 
    particles in an input, that is applied to the x, v, and uu inputs.

    The "maxlen" integer tells the layer how many particles to allow in the output.
    It is set by first taking the minimum of 1 and some number drawn from a random
    distribution between 0.9 and 1.02. These numbers are set by the "target" kwarg.
    Then this quantile of the distribution of the # of particles is evaluated.
    This is the maximum number of particles allowed to pass the trimmer.

    If the model is not training, then the input particles are not permuted
    or trimmed.

    Also the permuting and trimming only occurs if the trimmer has already been
    called 5 times. Unclear why we do this as of now.

    """

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            # Only apply the trimming after the first 5 times the trimmer is called
            # In practice this is on the 6th forward pass of a given network
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1) # Masked particles are given a random number of -1
                    # Then they are sorted last here, this way masked particles are always truncated
                    # before real particles.
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu