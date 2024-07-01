import math

import torch

# matrix


def rand_matrix(*shape, requires_grad: bool = False, dtype=None, device=None):
    r"""random rotation matrix

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape}, 2, 2)`
    """
    R = angles_to_matrix(*rand_angles(*shape, dtype=dtype, device=device))
    return R.detach().requires_grad_(requires_grad)


# angles


def identity_angles(*shape, requires_grad: bool = False, dtype=None, device=None):
    r"""angles of the identity rotation

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`

    """
    a = torch.zeros(1, *shape, dtype=dtype, device=device)
    return a[0].requires_grad_(requires_grad)


def rand_angles(*shape, requires_grad: bool = False, dtype=None, device=None):
    r"""random rotation angles

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`

    """
    alpha = 2 * math.pi * torch.rand(1, *shape, dtype=dtype, device=device)
    alpha = alpha.detach().requires_grad_(requires_grad)
    return alpha


def compose_angles(a1, a2):
    r"""compose angles

    Computes :math:`(a)` such that :math:`R(a) = R(a_1) \circ R(a_2)`

    Parameters
    ----------
    a1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    a2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    """
    a1, a2 = torch.broadcast_tensors(a1, a2)
    return matrix_to_angles(angles_to_matrix(a1) @ angles_to_matrix(a2))


def inverse_angles(a):
    r"""angles of the inverse rotation

    Parameters
    ----------
    a : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    """
    return -a


# conversions


def matrix(angle: torch.Tensor, m=None) -> torch.Tensor:
    r"""matrix of 2D rotation

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 2, 2)`
    """
    if m is None:
        m = 1
    elif m == 0:
        return torch.ones_like(angle).reshape(-1, 1, 1)
    c = torch.cos(m * angle)
    s = torch.sin(m * angle)
    return torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)


def reflection(angle: torch.Tensor, m=None, p=None):
    r"""reflection across line at angle from x axis

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 2, 2)`
    """
    if m is None:
        m = 1
    if m == 0:
        assert p in (-1, 1)
        if p == -1:
            return -torch.ones_like(angle).reshape(-1, 1, 1)
        else:
            return torch.ones_like(angle).reshape(-1, 1, 1)
    x_reflection = angle.new_tensor([[1.0, 0.0], [0.0, -1.0]])
    rot = matrix(angle, m)
    inv_rot = matrix(-angle, m)
    return rot @ x_reflection @ inv_rot


def matrix_to_angle(R, atol=1e-6):
    r"""conversion from matrix to angles

    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 2, 2)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    dets = torch.det(R)
    assert torch.allclose(torch.det(R).abs(), R.new_tensor(1))
    mask1 = torch.isclose(dets, torch.tensor(1.0), atol=tol)
    # rotation
    a_rot = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    # reflection
    a_refl = 0.5 * torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.where(mask1, a_rot, a_refl)


# point on the circle


def angle_to_xy(alpha) -> torch.Tensor:
    r"""convert :math:`(\alpha)` into a point :math:`(x, y)` on the circle

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 2)`

    Examples
    --------

    >>> angles_to_xy(torch.tensor(1.7)).abs()
    tensor([0., 1.])
    """
    alpha = torch.broadcast_tensors(alpha)
    x = torch.cos(alpha)
    y = torch.sin(alpha)
    return torch.stack([x, y], dim=-1)


def xy_to_angle(xy):
    r"""convert a point :math:`\vec r = (x, y)` on the circle into angle :math:`(\alpha)`

    Parameters
    ----------
    xy : `torch.Tensor`
        tensor of shape :math:`(..., 2)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    xy = torch.nn.functional.normalize(xy, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
    xy = xy.clamp(-1, 1)

    alpha = torch.atan2(xy[..., 1], xy[..., 0])
    return alpha


def D(m, p, angle: torch.Tensor, k) -> torch.Tensor:
    r"""matrix of 2D rotation or reflection for irrep m

    Parameters
    ----------
    m : `torch.Tensor`
        tensor of any shape :math:`(...)`

    p : `torch.Tensor`
        tensor of any shape :math:`(...)`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        If k=0: Rotation by :math:`\alpha`.
        If k=1: Reflection along line at angle alpha

    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`
        whether reflection is applied


    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 2, 2)`
    """

    if m == 0 and p == 1:
        return torch.ones_like(angle).reshape(-1, 1, 1)
    elif m == 0 and p == -1:
        return torch.where(k.bool(), -1 * torch.ones_like(angle), torch.ones_like(angle)).reshape(-1, 1, 1)

    rot = matrix(angle, m)
    refl = reflection(angle, m, p)
    return torch.where(k.bool().reshape(-1, 1, 1), refl, rot)
