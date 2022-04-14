# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations  # must import to defer parsing of annotations
import tvm
import tvm.testing
from tvm import relax
import numpy as np

import tvm.script
from tvm.script import tir as T, relax as R


def test_bind_params():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16))
            C = T.match_buffer(z, (16, 16))
            for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
                with T.block("matmul"):
                    vi = T.axis.S(16, i0 * 4 + i1)
                    vj = T.axis.S(16, j)
                    vk = T.axis.R(16, k0 * 4 + k1)
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.call_tir(tir_matmul, (x, w), (16, 16), dtype="float32")
            return gv0

    w_tvm = tvm.nd.array(np.random.rand(16, 16).astype(np.float32))
    x_tvm = tvm.nd.array(np.random.rand(16, 16).astype(np.float32))
    mod = relax.transform.BindParams("main", {"x": x_tvm})(InputModule)
    assert len(mod["main"].params) == 1

    target = tvm.target.Target("llvm", host="llvm")
    ex_after = relax.vm.build(mod, target)
    vm_after = relax.VirtualMachine(ex_after, tvm.cpu())
    res_after = vm_after["main"](w_tvm)

    ex_before = relax.vm.build(InputModule, target)
    vm_before = relax.VirtualMachine(ex_before, tvm.cpu())
    res_before = vm_before["main"](x_tvm, w_tvm)

    tvm.testing.assert_allclose(res_before.numpy(), res_after.numpy())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
