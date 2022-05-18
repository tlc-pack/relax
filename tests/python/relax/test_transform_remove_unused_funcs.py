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
from tvm import relax
import tvm.script
from tvm.script import tir as T, relax as R


def check_if_func_exists(mod, func_name):
    gvs = [str(gv) for gv in mod.get_global_vars()]
    return ("@" + func_name) in gvs


def test_unused_relax_func():
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
        def unused_func(x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.call_tir(tir_matmul, (x, w), (16, 16), dtype="float32")
            return gv0

    mod = InputModule
    assert mod
    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_matmul")
    assert not check_if_func_exists(new_mod, "unused_func")

    # Test with relax function w/ symbolic shape.
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def unused_func(x: Tensor((m, n), "float32"), w: Tensor((n, k), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def foo(x: Tensor((m, n), "float32"), w: Tensor((n, k), "float32")):
            gv0 = R.call_tir(tir_matmul, (x, w), (m, k), dtype="float32")
            return gv0

    mod = InputModule
    assert mod

    # Remove unused function before shape lowering.
    # Test entry function other than "main".
    new_mod = relax.transform.RemoveUnusedFunctions(entry_functions=["foo"])(mod)
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_matmul")
    assert not check_if_func_exists(new_mod, "unused_func")

    # Remove unused function after shape lowering.
    # Shape lowering will inject several shape-related global functions.
    # We need to make sure unused function removal pass does not remove those functions.
    shape_lowered_mod = relax.transform.VMShapeLower()(mod)
    new_mod = relax.transform.RemoveUnusedFunctions(entry_functions=["foo"])(shape_lowered_mod)
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_matmul")
    assert check_if_func_exists(new_mod, "shape_func")  # injected by VMShapeLower pass
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_prim_func():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def unused_func(x: T.handle, y: T.handle, z: T.handle) -> None:
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
        def relax_add(x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = relax_add(x, w)
            return gv0

    mod = InputModule
    assert mod

    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "relax_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_multiple_unused_funcs():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def unused_func1(x: T.handle, y: T.handle, z: T.handle) -> None:
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
        def unused_func2(x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = relax.add(x, w)
            return gv0

    mod = InputModule
    assert mod
    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert not check_if_func_exists(new_mod, "unused_func1")
    assert not check_if_func_exists(new_mod, "unused_func2")


if __name__ == "__main__":
    pytest.main([__file__])
