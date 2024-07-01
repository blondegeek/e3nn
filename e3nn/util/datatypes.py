from typing import NamedTuple, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from e3nn import o3, o2


class Chunk(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None


class Path(NamedTuple):
    input_1_chunk: Chunk
    input_2_chunk: Chunk
    output_chunk: Chunk


class IrrepsContext(ABC):
    def __init__(self, irreps_class) -> None:
        self._Irreps = irreps_class

    @abstractmethod
    def clebsch_gordan(self, ir_1, ir_2, ir_out) -> np.ndarray:
        pass

    @abstractmethod
    def path_hash(self, ir_1, ir_2, ir_out) -> Tuple:
        pass


class O3Context(IrrepsContext):
    def __init__(self):
        super().__init__(o3.Irreps)

    def clebsch_gordan(self, ir_1, ir_2, ir_out):
        return o3.wigner_3j(ir_1.l, ir_2.l, ir_out.l)

    def path_hash(self, ir_1, ir_2, ir_out):
        return (ir_1.l, ir_2.l, ir_out.l)


class O2Context(IrrepsContext):
    def __init__(self):
        super().__init__(o2.Irreps)

    def clebsch_gordan(self, ir_1, ir_2, ir_out):
        return o2.decomp_3m(ir_1.m, ir_2.m, ir_out.m, ir_1.p, ir_2.p, ir_out.p)

    def path_hash(self, ir_1, ir_2, ir_out):
        return (ir_1.m, ir_2.m, ir_out.m, ir_1.p, ir_2.p, ir_out.p)
