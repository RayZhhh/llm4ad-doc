from __future__ import annotations

import dataclasses
from typing import Type

from .sampler import EoHSampler
from ...base import Evaluator

@dataclasses.dataclass(frozen=True)
class EoHConfig:
    pop_size: int = 50
    selection_num = 2
    # use_i1_operator: bool = True
    # use_e1_operator: bool = True
    use_e2_operator: bool = True
    use_m1_operator: bool = True
    use_m2_operator: bool = True
    # num_samplers: int = 4
    # num_evaluators: int = 4