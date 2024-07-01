r"""Core functions of :math:`O(2)`
"""
import functools
import math
from typing import Union

import torch
from e3nn.util import explicit_default_types


def decomp_3m(m1: int, m2: int, m3: int, p1=0, p2=0, p3=0, dtype=None, device=None) -> torch.Tensor:
    r"""Decompose :math:`m_2 \otimes m_3` rep to to :math:`m_1`

    Parameters
    ----------
    m1 : int
        :math:`m_1`

    m2 : int
        :math:`m_2`

    m3 : int
        :math:`m_3`

    p1 : int
        :math:`p_1`

    p2 : int
        :math:`p_2`

    p3 : int
        :math:`p_3`

    dtype : torch.dtype or None
        ``dtype`` of the returned tensor. If ``None`` then set to ``torch.get_default_dtype()``.

    device : torch.device or None
        ``device`` of the returned tensor. If ``None`` then set to the default device of the current context.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`C` of shape
        :math:`(2 (m1>0) or 1 (m1=0), 2 (m2 > 0) or 1 (m2=0), 2 (m3 > 0) or 1 (m3=0))`
    """
    assert m1 == abs(m2 - m3) or m1 == m2 + m3

    C = _o2_clebsch_gordan(m1, m2, m3, p1, p2, p3)

    dtype, device = explicit_default_types(dtype, device)
    # make sure we always get:
    # 1. a copy so mutation doesn't ruin the stored tensors
    # 2. a contiguous tensor, regardless of what transpositions happened above
    return C.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)


@functools.lru_cache(maxsize=None)
def _o2_clebsch_gordan(m1: int, m2: int, m3: int, p1=0, p2=0, p3=0) -> torch.Tensor:
    assert m1 == abs(m2 - m3) or m1 == m2 + m3
    p = p1 * p2 * p3
    assert p == 0 or p == 1
    d1 = 2 if m1 != 0 else 1
    d2 = 2 if m2 != 0 else 1
    d3 = 2 if m3 != 0 else 1
    if m1 == 0 and m2 == 0 and m3 == 0:
        return torch.ones([1, 1, 1], dtype=torch.float64)
    r2_2 = math.sqrt(2) / 2.0
    h = 1 / 2.0
    if min([m1, m2, m3]) == 0:
        p_zero = [p1, p2, p3][[m1, m2, m3].index(0)]
        if p_zero == -1:
            return torch.tensor([0, r2_2, -r2_2, 0], dtype=torch.float64).reshape(d1, d2, d3)
        else:
            return torch.tensor([r2_2, 0, 0, r2_2], dtype=torch.float64).reshape(d1, d2, d3)
    if m1 == m2 + m3:
        return torch.tensor([h, 0, 0, -h, 0, h, h, 0], dtype=torch.float64).reshape(d1, d2, d3)

    if m1 + m2 == m3:
        return torch.tensor([h, 0, 0, h, 0, h, -h, 0], dtype=torch.float64).reshape(d1, d2, d3)

    if m1 + m3 == m2:
        return torch.tensor([h, 0, 0, h, 0, -h, h, 0], dtype=torch.float64).reshape(d1, d2, d3)
