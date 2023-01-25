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
from __future__ import annotations

import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script.highlight import cprint


def test_highlight_script():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            k = T.var("int32")
            A = T.match_buffer(x, (32, 32))
            B = T.match_buffer(y, (32, 32))
            C = T.match_buffer(z, (32, 32))

            for (i0, j0, k0) in T.grid(32, 32, 32):
                with T.block():
                    i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                    with T.init():
                        C[i, j] = 0.0
                    C[i, j] += A[i, k] * B[j, k]

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = R.call_tir(tir_matmul, (x, w), R.Tensor((32, 32), dtype="float32"))
                R.output(lv0)
            return lv0

    Module.show()
    Module["main"].show()
    Module["tir_matmul"].show()
    Module["main"].show(style="light")
    Module["main"].show(style="dark")
    Module["main"].show(style="ansi")


def test_cprint():
    # Print string
    cprint("a + 1")

    # Print nodes with `script` method, e.g. PrimExpr
    cprint(tvm.tir.Var("v", "int32") + 1)

    # Cannot print non-Python-style codes if black installed
    try:
        import black

        with pytest.raises(ValueError):
            cprint("if (a == 1) { a +=1; }")
    except ImportError:
        pass

    # Cannot print unsupported nodes (nodes without `script` method)
    with pytest.raises(TypeError):
        cprint(relay.const(1))


if __name__ == "__main__":
    tvm.testing.main()
