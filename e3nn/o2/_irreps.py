import itertools
import collections
from typing import List, Union

import torch

from e3nn.math import direct_sum, perm

# These imports avoid cyclic reference from o2 itself
from . import _rotation

class Irrep(tuple):
    r"""Irreducible representation of :math:`O(2)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of
    functions. The irreps of :math:`O(2)` are two-dimensional for all integer angular frequency :math:`m`,
    except for :math:`m=0` of which there are two one-dimensional irreps, the scalar (`0e`) and 
    pseudoscalar (`0o`) which changes signs under 2D reflections.

    Parameters
    ----------

    m : int
        non-negative integer, the degree of the representation, :math:`m = 0, 1, \dots`

    p : {1, -1}
        the parity of the representation only for m=0

    Examples
    --------
    Create a scalar representation (:math:`m=0`) of even parity.

    >>> Irrep(0, 1)
    0e

    Create a pseudoscalar representation (:math:`m=0`) of odd parity.

    >>> Irrep(0, -1)
    0o

    Create a 2D vector representation (:math:`m=1`).
    >>> Irrep("1")
    1

    >>> Irrep("2").dim
    2

    >>> Irrep("2") in Irrep("1") * Irrep("1")
    True

    >>> Irrep("1") + Irrep("2")
    1x1+1x2
    """

    def __new__(cls, m: Union[int, "Irrep", str, tuple], p=None):
        if p is None:
            if isinstance(m, Irrep):
                return m

            if isinstance(m, str):
                name = m.strip()
                try:
                    m = int(name)
                    p = 0
                except Exception:
                    if name in {"0e", "0o"}:
                        m = 0
                        p = {"e": 1, "o": -1}[name[-1]]
                    else:
                        raise ValueError(f'unable to convert string "{name}" into an Irrep')
            if isinstance(m, int):
                if m > 0:
                    m, p = m, 0
                elif m == 0:
                    if p not in (-1, 1):
                        raise ValueError(f'm=0 must have parity -1 or 1 not "{p}"')
                else:
                    raise ValueError(f'm must be non-negative integer not "{m}"')
            elif isinstance(m, tuple):
                m, p = m
                if m != 0 and p != 0:
                    raise ValueError(f'm > 0 does not have specific parity and must be None or 0')
        if not isinstance(m, int) or m < 0:
            raise ValueError(f"l must be positive integer, got {l}")
        if p not in (-1, 1, 0):
            raise ValueError(f"parity must be on of (-1, 1, 0), got {p}")
        return super().__new__(cls, (m, p))

    @property
    def m(self) -> int:  # noqa: E743
        r"""The degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[0]

    @property
    def p(self) -> int:
        r"""The parity of the representation, :math:`p = \pm 1`."""
        return self[1]

    def __repr__(self) -> str:
        if self.p == 0:
            p = ""
        else:
            p = {+1: "e", -1: "o"}[self.p]
        return f"{self.m}{p}"

    @classmethod
    def iterator(cls, mmax=None):
        r"""Iterator through all the irreps of :math:`O(2)`

        Examples
        --------
        >>> it = Irrep.iterator()
        >>> next(it), next(it), next(it), next(it)
        (0e, 0o, 1, 2)
        """
        for m in itertools.count():
            if m == 0:
                yield Irrep(0, 1)
                yield Irrep(0, -1)
            else:
                yield Irrep(m)

            if m == mmax:
                break

    def D_from_angle(self, alpha, k=None) -> torch.Tensor:
        r"""Matrix :math:`p^k D^l(\alpha)`

        (matrix) Representation of :math:`O(2)`. :math:`D` is the representation of :math:`SO(2)`, see `D`.

        Parameters
        ----------
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
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        See Also
        --------
        o2.D
        Irreps.D_from_angles
        """
        if k is None:
            k = torch.zeros_like(alpha)

        alpha, k = torch.broadcast_tensors(alpha, k)

        # Need to change
        return _rotation.D(self.m, self.p, alpha, k)

    def D_from_matrix(self, R) -> torch.Tensor:
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        Examples
        --------
        >>> m = Irrep(1, -1).D_from_matrix(-torch.eye(3))
        >>> m.long()
        tensor([[-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1]])
        """
        k = torch.det(R).sign()
        return self.D_from_angle(*_rotation.matrix_to_angle(R), k)

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 if self.m > 0 else 1

    def is_scalar(self) -> bool:
        """Equivalent to ``m == 0 and p == 1``"""
        return self.m == 0 and self.p == 1

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.

        Returns
        -------
        generator of `e3nn.o2.Irrep`
        """
        other = Irrep(other)
        mmin = abs(self.m - other.m)
        mmax = self.m + other.m
        if self.m == 0 and other.m == 0:
            yield Irrep(0, self.p * other.p)
        elif self.m == 0:
            yield Irrep(other.m)
        elif other.m == 0:
            yield Irrep(self.m)
        elif mmin == 0:
            yield Irrep(0, 1)
            yield Irrep(0, -1)
        else:
            yield Irrep(mmin)
            yield Irrep(mmax)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1')
        3x1
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self) -> str:
        return f"{self.mul}x{self.ir}"

    def __getitem__(self, item) -> Union[int, Irrep]:  # pylint: disable=useless-super-delegation
        return super().__getitem__(item)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

# I'm pretty sure I didn't change anything here... may want to just reuse o3.Irreps
class Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`O(2)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of
    functions.

    Attributes
    ----------
    dim : int
        the total dimension of the representation

    num_irreps : int
        number of irreps. the sum of the multiplicities

    ls : list of int
        list of :math:`l` values

    lmax : int
        maximum :math:`l` value

    Examples
    --------
    Create a representation of 100 :math:`m=0` of even parity and 50 vectors.

    >>> x = Irreps([(100, (0, 1)), (50, (1, 0))])
    >>> x
    100x0e+50x1

    >>> x.dim
    200

    >>> Irreps("100x0e + 50x1 + 0x2")
    100x0e+50x1+0x2

    >>> Irreps("100x0e + 50x1e + 0x2e").mmax
    1

    >>> Irrep("2") in Irreps("0e + 2")
    True

    Empty Irreps

    >>> Irreps(), Irreps("")
    (, )
    """

    def __new__(cls, irreps=None) -> Union[_MulIr, "Irreps"]:
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, Irrep):
            out.append(_MulIr(1, Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                mul = None
                ir = None

                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    def slices(self):
        r"""List of slices corresponding to indices for each irrep.

        Examples
        --------

        >>> Irreps('2x0e + 1').slices()
        [slice(0, 2, None), slice(2, 4, None)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(
        self, *size: int, normalization: str = "component", requires_grad: bool = False, dtype=None, device=None
    ) -> torch.Tensor:
        r"""Random tensor.

        Parameters
        ----------
        *size : list of int
            size of the output tensor, needs to contains a ``-1``

        normalization : {'component', 'norm'}

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``size`` where ``-1`` is replaced by ``self.dim``

        Examples
        --------

        >>> Irreps("5x0e + 10x1").randn(5, -1, 5, normalization='norm').shape
        torch.Size([5, 25, 5])

        >>> random_tensor = Irreps("2").randn(2, -1, 3, normalization='norm')
        >>> random_tensor.norm(dim=1).sub(1).abs().max().item() < 1e-5
        True
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]

        if normalization == "component":
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            with torch.no_grad():
                for s, (mul, ir) in zip(self.slices(), self):
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i) -> Union[_MulIr, "Irreps"]:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        r"""Multiplicity of ``ir``.

        Parameters
        ----------
        ir : `e3nn.o2.Irrep`

        Returns
        -------
        `int`
            total multiplicity of ``ir``
        """
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps) -> "Irreps":
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other) -> "Irreps":
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o2.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other) -> "Irreps":
        r"""
        >>> 2 * Irreps('0e + 1')
        1x0e+1x1+1x0e+1x1
        """
        return Irreps(super().__rmul__(other))

    def simplify(self) -> "Irreps":
        """Simplify the representations.

        Returns
        -------
        `e3nn.o2.Irreps`

        Examples
        --------

        Note that simplify does not sort the representations.

        >>> Irreps("1 + 1 + 0e").simplify()
        2x1+1x0e

        Equivalent representations which are separated from each other are not combined.

        >>> Irreps("1 + 1 + 0e + 1").simplify()
        2x1+1x0e+1x1
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self) -> "Irreps":
        """Remove any irreps with multiplicities of zero.

        Returns
        -------
        `e3nn.o3.Irreps`

        Examples
        --------

        >>> Irreps("4x0e + 0x1 + 2x3").remove_zero_multiplicities()
        4x0e+2x3

        """
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return Irreps(out)

    def sort(self):
        r"""Sort the representations.

        Returns
        -------
        irreps : `e3nn.o3.Irreps`
        p : tuple of int
        inv : tuple of int

        Examples
        --------

        >>> Irreps("1 + 0e + 1").sort().irreps
        1x0e+1x1+1x1

        >>> Irreps("2 + 1 + 0e + 1").sort().p
        (3, 1, 0, 2)

        >>> Irreps("2 + 1 + 0e + 1").sort().inv
        (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ms(self) -> List[int]:
        return [m for mul, (m, p) in self for _ in range(mul)]

    @property
    def mmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get mmax of empty Irreps")
        return max(self.ms)

    def __repr__(self) -> str:
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D_from_angle(self, alpha, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return direct_sum(*[ir.D_from_angle(alpha, k) for mul, ir in self for _ in range(mul)])

    def D_from_matrix(self, R):
        r"""Matrix of the representation

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angle(*_rotation.matrix_to_angle(R), k)
    
    @classmethod
    def o3_to_o2_mapping(cls, o3_irreps):
        o2_list = []
        blocks = []
        o2_ir_strs = []
        for o3_ir in o3_irreps:
            mul, l, p = o3_ir.mul, o3_ir.ir.l, o3_ir.ir.p
            if (l % 2 == 0 and p == 1) or (l % 2 == 1 and p == -1):
                # Same parity as SH
                scalar = '0e'
                odd = torch.eye(2)
                even = torch.tensor([[0, 1], [-1, 0]])
            elif (l % 2 == 0 and p == -1) or (l % 2 == 1 and p == 1):
                scalar = '0o'
                odd = torch.tensor([[0, 1], [-1, 0]])
                even = torch.eye(2)
            else:
                raise ValueError('Something is wrong')
            o2_ir_strs += list([scalar] + list(map(str, range(1, l+1)))) * mul
            blocks += list([torch.ones(1, 1)] + [odd if m % 2 == 1 else even for m in range(1, l+1)]) * mul
        return Irreps("+".join(o2_ir_strs)), direct_sum(*blocks)
