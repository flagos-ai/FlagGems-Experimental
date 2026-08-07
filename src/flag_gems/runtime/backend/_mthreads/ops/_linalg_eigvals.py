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

import logging

import torch

from flag_gems.ops._linalg_eigvals import _linalg_eigvals as default_linalg_eigvals

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware supports neither fp64 compute nor on-device complex
# tensors, so the eigenvalue solve must run on the host (torch_musa's own
# _linalg_eigvals does the same). The generic implementation additionally
# launches a no-op Triton "proxy" copy kernel and juggles the tensor across
# devices twice; on MUSA that kernel launch is pure overhead. This
# specialization keeps the required host solve but drops the proxy kernel and
# performs a single round-trip. fp64 / unsupported dtypes defer to the generic.
_SUPPORTED_DTYPES = {torch.float32}


def _linalg_eigvals(inp):
    logger.debug("GEMS_MTHREADS _LINALG_EIGVALS")

    if inp.device.type != "musa" or inp.dtype not in _SUPPORTED_DTYPES:
        return default_linalg_eigvals(inp)

    # Route the eigenvalue computation through the CPU LAPACK path (musa has no
    # on-device complex eigensolver) and move the complex result back in one hop.
    return torch.linalg.eigvals(inp.cpu()).to(inp.device)
