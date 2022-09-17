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

import pytest
from typing import Union
import tvm
import tvm.testing

from tvm import relax, tir
from tvm import IRModule
from tvm.script.parser import ir as I, tir as T, relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Union[relax.Function, IRModule],
):
    # TODO(siyuan): add round-trip tests
    tvm.ir.assert_structural_equal(parsed, expect)


def test_simple_func():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        R.func_attr({"Primitive": 1})
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        return gv0

    x = relax.Var("x", [128, 128], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), attrs={"Primitive": 1}):
        out = bb.emit(relax.call_tir("extern_func", x, (128, 128), dtype="float32"))
        bb.emit_func_output(out)

    _check(foo, bb.get()["foo"])


def test_error_report():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = gv1 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            return gv0


def test_simple_module():
    @I.ir_module
    class TestModule:
        @T.prim_func
        def tir_func(x: T.Buffer((128, 128), "float32"), y: T.Buffer((128, 128), "float32")):
            T.func_attr({"global_symbol": "tir_func", "tir.noalias": True})
            for i, j in T.grid(128, 128):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            # TODO(Siyuan): Need to change to `TestModule.tir_func`
            gv0 = R.call_tir(tir_func, x, (128, 128), dtype="float32")
            return gv0

    x = relax.Var("x", [128, 128], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)

    _check(TestModule, bb.get())


def test_relax_tensor_op():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
        y = R.add(x, x)
        z = R.multiply(x, y)
        return z

    x = relax.Var("x", [4, 4], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.op.add(x, x))
        z = bb.emit(relax.op.multiply(x, y))
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_relax_base_op():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        alloc = R.builtin.alloc_tensor((4, 4), runtime_device_index=0, dtype="float32")
        shape = R.shape_of(alloc)
        return shape

    x = relax.Var("x", [4, 4], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        alloc = bb.emit(relax.op.builtin.alloc_tensor(relax.ShapeExpr((4, 4)), "float32", 0))
        shape = bb.emit(relax.op.shape_of(alloc))
        bb.emit_func_output(shape)

    _check(foo, bb.get()["foo"])


def test_symbolic_shape():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
        m = T.var("int64", "m")
        n = T.var("int64", "n")
        gv0 = R.call_tir("extern_func", x, (m, n), dtype="float32")
        return gv0

    @R.function
    def bar(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
        m = T.var("int64")
        n = T.var("int64")
        gv0 = R.call_tir("extern_func", x, (m, n), dtype="float32")
        return gv0

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def mismatch_dtype(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
            m = T.var("int64")
            n = T.var("int32")  # The shape dtype should be int64
            gv0 = R.call_tir("extern_func", x, (m, n), dtype="float32")
            return gv0

    def _expected(name: str):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = relax.Var("x", [m, n], relax.DynTensorType(2, "float32"))
        bb = relax.BlockBuilder()
        with bb.function(name, (x,)):
            out = bb.emit(relax.call_tir("extern_func", x, (m, n), dtype="float32"))
            bb.emit_func_output(out)
        return bb.get()[name]

    _check(foo, _expected("foo"))
    _check(bar, _expected("bar"))


def test_shadowing():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        y = R.add(x, x)
        z = R.multiply(x, y)
        y = R.add(x, y)
        y = z
        y = R.multiply(y, x)
        z = y
        return z

    x = relax.Var("x", [4, 4], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.op.add(x, x))
        z = bb.emit(relax.op.multiply(x, y))
        y = bb.emit(relax.op.add(x, y))
        y = bb.emit(z)
        y = bb.emit(relax.op.multiply(y, x))
        z = bb.emit(y)
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_match_shape():
    @R.function
    def foo(x: R.Tensor(None, "float32"), y: R.Tensor(None, "float32")):
        m = T.var("int64")
        n = T.var("int64")
        R.match_shape(x, (m,))
        y1 = R.match_shape(y, (n,))
        return (m, n * 2)

    x = relax.Var("x", type_annotation=relax.DynTensorType(-1, "float32"))
    y = relax.Var("y", type_annotation=relax.DynTensorType(-1, "float32"))
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        bb.match_shape_binding(relax.MatchShape(x, (m,), var=None))
        y1 = bb.match_shape(y, (n,))
        bb.emit_func_output(relax.ShapeExpr([m, n * 2]))
    _check(foo, bb.get()["foo"])


def test_tuple_return():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        gv0 = R.call_tir("extern_func_0", x, (4, 4), dtype="float32")
        gv1 = R.call_tir("extern_func_1", x, (4, 4), dtype="float32")
        return (gv0, gv1)

    x = relax.Var("x", [4, 4], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        gv0 = bb.emit(relax.call_tir("extern_func_0", x, (4, 4), dtype="float32"))
        gv1 = bb.emit(relax.call_tir("extern_func_1", x, (4, 4), dtype="float32"))
        bb.emit_func_output(relax.Tuple((gv0, gv1)))

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
