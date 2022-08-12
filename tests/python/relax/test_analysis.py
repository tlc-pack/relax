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
from tvm import tir
from tvm import relax as rx
from tvm.relax.analysis import udchain, remove_all_unused, name_to_binding
from tvm.script import relax as R


def test_dispatch_var():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    type_anno0 = rx.DynTensorType(ndim=2, dtype="float16")
    type_anno1 = rx.DynTensorType(ndim=1, dtype="float16")
    v0 = rx.Var("v0", [m, n], type_anno0)
    v1 = rx.DataflowVar("v1", [n], type_anno1)
    t = None

    def fvisit(e):
        nonlocal t
        t = type(e)

    rx.analysis.post_order_visit(v0, fvisit)
    assert t == type(v0)
    rx.analysis.post_order_visit(v1, fvisit)
    assert t == type(v1)


def test_post_order_visit():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    type_anno0 = rx.DynTensorType(ndim=2, dtype="float16")
    type_anno1 = rx.DynTensorType(ndim=1, dtype="float16")
    x = rx.Var("x", [m, n], type_anno0)
    y = rx.Var("y", [n], type_anno1)
    ib = rx.BlockBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_func_output(gv0)
    expr = ib.get()["func"]

    names = []

    def fvisit(e):
        nonlocal names
        if isinstance(e, tvm.ir.op.Op):
            names.append(e.name)

    rx.analysis.post_order_visit(expr.body, fvisit)
    assert names == ["relax.add", "relax.multiply"]


def test_use_def():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    type_anno0 = rx.DynTensorType(ndim=2, dtype="float16")
    type_anno1 = rx.DynTensorType(ndim=1, dtype="float16")
    x = rx.Var("x", [m, n], type_anno0)
    y = rx.Var("y", [n], type_anno1)
    ib = rx.BlockBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow():
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_func_output(gv0)
    dfb = ib.get()["func"].body.blocks[0]
    udc = udchain(dfb)
    assert set(udc[x]) == {lv0}
    assert set(udc[y]) == {lv0, lv1}
    assert set(udc[lv0]) == {lv1}
    assert set(udc[lv1]) == {gv0}
    assert set(udc[gv0]) == set()


def test_chained_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_tir(my_sigmoid, (x,), (32, 32), dtype="float32")
                unused1 = R.call_tir(my_sigmoid, (unused0,), (32, 32), dtype="float32")
                R.output(lv0)
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_tir(my_sigmoid, (x,), (32, 32), dtype="float32")
                unused1 = R.call_tir(my_sigmoid, (unused0,), (32, 32), dtype="float32")
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(Tensor((32, 32), "float32")))
            return z

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(Tensor((32, 32), "float32")))
            return z

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_fake_unused_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(Tensor((32, 32), "float32")))
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            # This might bring side effect so cannot be removed.
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(Tensor((32, 32), "float32")))
            return lv0

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_edge_binding_block_fake_unused_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor((32, 32), "float32"):
            z = R.call_packed("vm.builtin.copy", x, type_args=(Tensor((32, 32), "float32")))
            return x

    optimized = remove_all_unused(IdentityUnused["main"])
    tvm.ir.assert_structural_equal(optimized, IdentityUnused["main"])


def test_name_to_binding_var_shadowing():
    @R.function
    def main(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            lv1 = lv0
            R.output(lv1)

        with R.dataflow():
            lv0 = lv1  # shadowing
            lv2 = lv0
            R.output(lv2)
        return lv2

    n2binding = name_to_binding(main)

    assert "lv0" in n2binding
    assert "lv1" in n2binding
    assert "lv2" in n2binding

    assert len(n2binding["lv0"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])
