# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools
from functools import partial

import pytest
import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel, GroupKernel
from e3nn.point.message_passing import Convolution, WTPConv, KMeans, SymmetricKMeans, Pooling, get_new_edge_index
from e3nn.radial import ConstantRadialModel


@pytest.mark.parametrize('Rs_in, Rs_out, n_source, n_target, n_edge', itertools.product([[1]], [[2]], [2, 3], [1, 3], [0, 3]))
def test_equivariance(Rs_in, Rs_out, n_source, n_target, n_edge):
    torch.set_default_dtype(torch.float64)

    mp = Convolution(Kernel(Rs_in, Rs_out, ConstantRadialModel))
    groups = 4
    mp_group = Convolution(GroupKernel(Rs_in, Rs_out, partial(Kernel, RadialModel=ConstantRadialModel), groups))

    features = rs.randn(n_target, Rs_in)
    features2 = rs.randn(n_target, Rs_in * groups)

    edge_index = torch.stack([
        torch.randint(n_source, size=(n_edge,)),
        torch.randint(n_target, size=(n_edge,)),
    ])
    size = (n_target, n_source)

    edge_r = torch.randn(n_edge, 3)

    out1 = mp(features, edge_index, edge_r, size=size)
    out1_groups = mp(features2, edge_index, edge_r, size=size, groups=groups)
    out1_kernel_groups = mp_group(features2, edge_index, edge_r, size=size, groups=groups)

    angles = o3.rand_angles()
    D_in = rs.rep(Rs_in, *angles)
    D_out = rs.rep(Rs_out, *angles)
    D_in_groups = rs.rep(Rs_in * groups, *angles)
    D_out_groups = rs.rep(Rs_out * groups, *angles)
    R = o3.rot(*angles)

    out2 = mp(features @ D_in.T, edge_index, edge_r @ R.T, size=size) @ D_out
    out2_groups = mp(features2 @ D_in_groups.T, edge_index, edge_r @ R.T, size=size, groups=groups) @ D_out_groups
    out2_kernel_groups = mp_group(features2 @ D_in_groups.T, edge_index, edge_r @ R.T, size=size, groups=groups) @ D_out_groups

    assert (out1 - out2).abs().max() < 1e-10
    assert (out1_groups - out2_groups).abs().max() < 1e-10
    assert (out1_kernel_groups - out2_kernel_groups).abs().max() < 1e-10


@pytest.mark.parametrize('Rs_in, Rs_out, n_source, n_target, n_edge', itertools.product([[1]], [[2]], [2, 3], [1, 3], [0, 3]))
def test_equivariance_wtp(Rs_in, Rs_out, n_source, n_target, n_edge):
    torch.set_default_dtype(torch.float64)

    mp = WTPConv(Rs_in, Rs_out, 3, ConstantRadialModel)

    features = rs.randn(n_target, Rs_in)

    edge_index = torch.stack([
        torch.randint(n_source, size=(n_edge,)),
        torch.randint(n_target, size=(n_edge,)),
    ])
    size = (n_target, n_source)

    edge_r = torch.randn(n_edge, 3)
    if n_edge > 1:
        edge_r[0] = 0

    out1 = mp(features, edge_index, edge_r, size=size)

    angles = o3.rand_angles()
    D_in = rs.rep(Rs_in, *angles)
    D_out = rs.rep(Rs_out, *angles)
    R = o3.rot(*angles)

    out2 = mp(features @ D_in.T, edge_index, edge_r @ R.T, size=size) @ D_out

    assert (out1 - out2).abs().max() < 1e-10


def test_flow():
    """
    This test checks that information is flowing as expected from target to source.
    edge_index[0] is source (convolution center)
    edge_index[1] is target (neighbors)
    """

    edge_index = torch.LongTensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ])
    features = torch.tensor(
        [-1., 1., 1., 1., 1.]
    )
    features = features.unsqueeze(-1)
    edge_r = torch.ones(edge_index.shape[-1], 3)

    Rs = [0]
    conv = Convolution(Kernel(Rs, Rs, ConstantRadialModel))
    conv.kernel.R.weight.data.fill_(1.)  # Fix weight to 1.

    output = conv(features, edge_index, edge_r)
    torch.allclose(output, torch.tensor([4., 0., 0., 0., 0.]).unsqueeze(-1))

    edge_index = torch.LongTensor([
        [1, 2, 3, 4],
        [0, 0, 0, 0]
    ])
    output = conv(features, edge_index, edge_r)
    torch.allclose(output, torch.tensor(
        [0., -1., -1., -1., -1.]).unsqueeze(-1))


def test_KMeans():
    pos = torch.tensor([[0., 0., 0.], [0.2, 0., 0.], [1.0, 0., 0.], [1.2, 0., 0.]])
    centers = torch.tensor([[0.1, 0., 0.], [1.1, 0., 0.]])
    pos = torch.cat([pos, pos], dim=0)
    batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
    kmeans = KMeans()
    classification, centroids, centroids_batch = kmeans.forward(pos, batch, fps_ratio=0.5)
    # Check centroids_batch
    assert torch.allclose(centroids_batch, torch.LongTensor([0, 0, 1, 1]))
    # Check groupings
    assert torch.allclose(classification[torch.LongTensor([0, 2, 4, 6])], classification[torch.LongTensor([1, 3, 5, 7])])
    # Check cluster centers
    assert torch.allclose(torch.sort(centroids, dim=0)[0], torch.tensor([[0.1, 0., 0.], [0.1, 0., 0.], [1.1, 0., 0.], [1.1, 0., 0.]]))


def test_SymmetricKMeans():
    kmeans = SymmetricKMeans(rand_iter=10)

    batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])

    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])  # Square
    pos = torch.cat([pos, pos], dim=0)
    labels = kmeans.forward(pos, batch)
    truth = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).to(torch.int64)
    print(labels, truth)
    assert torch.allclose(labels, truth)

    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [2., 1., 0.]])  # Zig
    pos = torch.cat([pos, pos], dim=0)
    labels = kmeans.forward(pos, batch)
    truth = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(torch.int64)
    print(labels, truth)
    assert torch.allclose(labels, truth)


def test_get_new_edge_index():
    N, B, C = 3, 6, 2
    edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    bloom_batch = torch.LongTensor([0, 0, 1, 1, 2, 2])
    cluster = torch.LongTensor([0, 0, 0, 1, 1, 1])
    assert(B == len(bloom_batch))
    assert(C == max(cluster + 1))

    new_edge_index = get_new_edge_index(N, edge_index, bloom_batch, cluster)
    assert torch.allclose(new_edge_index, torch.LongTensor([[0, 0, 1, 1], [0, 1, 0, 1]]))

    N, B, C = 4, 6, 2
    edge_index = torch.LongTensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
                                   [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]])
    bloom_batch = torch.LongTensor([0, 1, 1, 2, 2, 3])
    cluster = torch.LongTensor([0, 0, 1, 0, 1, 1])
    assert(B == len(bloom_batch))
    assert(C == max(cluster + 1))

    new_edge_index = get_new_edge_index(N, edge_index, bloom_batch, cluster)
    assert torch.allclose(new_edge_index, torch.LongTensor([[0, 0, 1, 1], [0, 1, 0, 1]]))
