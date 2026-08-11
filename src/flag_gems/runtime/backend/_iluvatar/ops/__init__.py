# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ._native_batch_norm_legit_functional import _native_batch_norm_legit_functional
from .adaptive_max_pool3d_backward import run as adaptive_max_pool3d_backward
from .addmm import addmm, addmm_, addmm_out
from .arccosh_ import arccosh_
from .avg_pool3d import avg_pool3d_backward
from .broadcast_tensors import broadcast_tensors
from .broadcast_to import broadcast_to
from .conv_depthwise2d import _conv_depthwise2d
from .conv_transpose1d import conv_transpose1d
from .diagonal_scatter import diagonal_scatter
from .div import div_mode, div_mode_
from .gcd_ import gcd_
from .hadamard_transform import hadamard_transform
from .histc import histc
from .index_select_backward import index_select_backward
from .linalg_eigvals import run as _linalg_eigvals
from .linalg_ldl_factor_ex import run as ldl_factor_ex
from .linalg_svdvals import run as linalg_svdvals
from .linear import linear
from .linear_backward import run as linear_backward
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .mm import mm, mm_out
from .nextafter import run as nextafter
from .nonzero_numpy import nonzero_numpy
from .repeat import repeat
from .scatter_add import scatter_add_
from .softplus import softplus_backward
from .special_modified_bessel_k1 import (
    special_modified_bessel_k1,
    special_modified_bessel_k1_out,
)
from .special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from .thnn_fused_lstm_cell_backward_impl import (
    run as _thnn_fused_lstm_cell_backward_impl,
)
from .tile import tile
from .unsafe_masked_index_put_accumulate import (
    run as _unsafe_masked_index_put_accumulate,
)
from .var import var, var_correction, var_dim

__all__ = [
    "_conv_depthwise2d",
    "_linalg_eigvals",
    "_native_batch_norm_legit_functional",
    "_thnn_fused_lstm_cell_backward_impl",
    "_unsafe_masked_index_put_accumulate",
    "adaptive_max_pool3d_backward",
    "addmm",
    "addmm_",
    "addmm_out",
    "arccosh_",
    "avg_pool3d_backward",
    "broadcast_tensors",
    "broadcast_to",
    "conv_transpose1d",
    "diagonal_scatter",
    "div_mode",
    "div_mode_",
    "gcd_",
    "hadamard_transform",
    "histc",
    "index_select_backward",
    "ldl_factor_ex",
    "linalg_svdvals",
    "linear",
    "linear_backward",
    "matmul_bf16",
    "matmul_int8",
    "mm",
    "mm_out",
    "nextafter",
    "nonzero_numpy",
    "repeat",
    "scatter_add_",
    "softplus_backward",
    "special_modified_bessel_k1",
    "special_modified_bessel_k1_out",
    "special_shifted_chebyshev_polynomial_w",
    "tile",
    "var",
    "var_correction",
    "var_dim",
]
