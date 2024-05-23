""" net_utils.py - This file defines utility functions for training
the OF particle transformer.

Author: Kevin Greif
12/12/23
python3
"""

import math
import warnings
import torch

@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(xi, xj, eps=1e-8):
    """ to_m2 - This function calculates the invariant mass squared of the sum
    of the four vectors given by xi and xj. The xi and xj are assumed to have shape
    (n_events, 3 + n_onehot, n_particles) where the 3 dimensions are (log(pT), eta, phi).

    The n_particles in xi will be added elementwise to the n_particles in xj.

    Arguments:
    xi - The first set of three vectors
    xj - The second set of three vectors
    eps - The minimum value for the invariant mass squared. Useful if you
    are taking the log of the invariant mass squared.

    Returns:
    m2 - The invariant mass squared of the sum of the four vectors. In shape
    (n_events, n_particles)
    """

    # Separate the pT, eta, and phi from the input
    pti, etai, phii, onehotsi = torch.split(xi, [1, 1, 1, xi.shape[1] - 3], dim=1) # pT, eta, phi, onehots
    ptj, etaj, phij, onehotsj = torch.split(xj, [1, 1, 1, xj.shape[1] - 3], dim=1) # pT, eta, phi, onehots

    # Remember we take the log of the pTs, so we need to exponentiate them
    pti = torch.exp(pti)
    ptj = torch.exp(ptj)

    # Determine masses for the mi, mj based on the onehot encodings
    # muon mass = 0.11, pion mass = 0.14, so not so huge of a difference anyway
    mi = torch.cat((0.11 * onehots[:,:2,:], 0.14 * onehots[:,2:,:]), dim=1).sum(dim=1, keepdim=True)
    mj = torch.cat((0.11 * onehots[:,:2,:], 0.14 * onehots[:,2:,:]), dim=1).sum(dim=1, keepdim=True)

    # Calculate px, py, pz, and E
    pxi = pti * torch.cos(phii)
    pxj = ptj * torch.cos(phij)
    pyi = pti * torch.sin(phii)
    pyj = ptj * torch.sin(phij)
    pzi = pti * torch.sinh(etai)
    pzj = ptj * torch.sinh(etaj)
    Ei = torch.sqrt((pti * torch.cosh(etai))**2 + mi**2)
    Ej = torch.sqrt((ptj * torch.cosh(etaj))**2 + mj**2)

    # Calculate the invariant mass
    m2 = (Ei + Ej)**2 - (pxi + pxj)**2 - (pyi + pyj)**2 - (pzi + pzj)**2

    # Clamp the invariant mass to the minimum value
    if eps is not None:
        m2 = m2.clamp(min=eps)

    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    """ pairwise_lv_fts - This function calculates the pairwise level features
    given a set of input particles which is assumed to be in the shape:

    (n_events, 3, n_particles) -> dim 1 w/ len=3 is (pT, eta, phi)

    This is modified from the original ParT version which takes 4-vectors with
    E, px, py, pz as input.
    """


    pti, etai, phii = xi.split((1, 1, 1), dim=1)
    ptj, etaj, phij = xj.split((1, 1, 1), dim=1)

    delta = delta_r2(etai, phii, etaj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        lnm2 = torch.log(to_m2(xi, xj, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    print("idx: ", idx)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor