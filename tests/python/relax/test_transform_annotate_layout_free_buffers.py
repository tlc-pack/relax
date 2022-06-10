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

import pytest
import sys
import tvm
from tvm import relax
import numpy as np
import tvm.script
from tvm.script import tir as T, relax as R


def gen_mod(mod, binding):
    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    return relax.transform.BindParams("main", binding)(mod)


def update_attr_layout_free_buffers(mod, attr_map):
    funcs = {}
    for gv, func in mod.functions.items():
        attr = attr_map.get(gv.name_hint, None)
        if attr:
            func = func.with_attr("layout_free_buffers", attr)
        funcs[gv] = func
    return tvm.IRModule(funcs)


def test_annotate_simple():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def tir_matmul(
            A: T.Buffer[(16, 16), "float32"],
            B: T.Buffer[(16, 16), "float32"],
            C: T.Buffer[(16, 16), "float32"],
        ):
            for i, j, k in T.grid(16, 16, 16):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            x: Tensor((16, 16), "float32"),
            w: Tensor((16, 16), "float32"),
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.call_tir(tir_matmul, (x, w), (16, 16), dtype="float32")
            return gv0

    w_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)

    before = gen_mod(Before, {"w": w_np})
    after = relax.transform.AnnotateLayoutFreeBuffers()(before)
    expected = update_attr_layout_free_buffers(before, {"tir_matmul": [1]})

    tvm.ir.assert_structural_equal(after, expected)


def test_annotate_intersection():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def tir_matmul(
            A: T.Buffer[(16, 16), "float32"],
            B: T.Buffer[(16, 16), "float32"],
            C: T.Buffer[(16, 16), "float32"],
        ):
            for i, j, k in T.grid(16, 16, 16):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            x: Tensor((16, 16), "float32"),
            w: Tensor((16, 16), "float32"),
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.call_tir(tir_matmul, (x, w), (16, 16), dtype="float32")
            gv1 = R.call_tir(tir_matmul, (gv0, x), (16, 16), dtype="float32")
            return gv1

    w_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)

    before = gen_mod(Before, {"w": w_np})
    after = relax.transform.AnnotateLayoutFreeBuffers()(before)
    expected = before

    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
