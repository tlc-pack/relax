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

from typing import Union

import pytest
import tvm
import tvm.testing
from tvm import IRModule, relax, tir
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T


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


def test_dataflow_block():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        with R.dataflow():
            lv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            lv1 = R.call_tir("extern_func", lv0, (128, 128), dtype="float32")
            gv = lv1
            R.output(gv)
        return gv

    x = relax.Var("x", (128, 128), relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        with bb.dataflow():
            lv0 = bb.emit(relax.call_tir("extern_func", x, (128, 128), dtype="float32"))
            lv1 = bb.emit(relax.call_tir("extern_func", lv0, (128, 128), dtype="float32"))
            gv = bb.emit_output(lv1)
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_dataflow_block_advanced():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        gv1 = R.call_tir("extern_func", gv0, (128, 128), dtype="float32")
        with R.dataflow():
            m = T.var("int64")
            n = T.var("int64")
            lv0 = R.call_tir("extern_func", gv1, (128, 128), dtype="float32")
            lv1 = R.match_shape(lv0, (m, n))
            gv2 = R.call_tir("extern_func", lv0, (128, 128), dtype="float32")
            gv2 = R.call_tir("extern_func", gv2, (128, 128), dtype="float32")
            gv3 = R.match_shape(gv2, (m, n))
            gv3 = R.match_shape(lv0, (m, n))
            gv4 = gv3
            gv5 = gv2
            R.output(gv5, gv4)
        gv6 = R.call_tir("extern_func", gv5, (128, 128), dtype="float32")
        gv7 = R.call_tir("extern_func", gv6, (128, 128), dtype="float32")
        return gv7

    x = relax.Var("x", (128, 128), relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    with bb.function("foo", (x,)):
        gv0 = bb.emit(relax.call_tir("extern_func", x, (128, 128), dtype="float32"))
        gv1 = bb.emit(relax.call_tir("extern_func", gv0, (128, 128), dtype="float32"))
        with bb.dataflow():
            lv0 = bb.emit(relax.call_tir("extern_func", gv1, (128, 128), dtype="float32"))
            lv1 = bb.match_shape(lv0, (m, n))
            gv2 = bb.emit(relax.call_tir("extern_func", lv0, (128, 128), dtype="float32"))
            gv21 = bb.emit(relax.call_tir("extern_func", gv2, (128, 128), dtype="float32"))
            gv3 = bb.match_shape(gv21, (m, n))
            gv31 = bb.match_shape(lv0, (m, n))
            gv32 = bb.emit_output(gv31)
            gv22 = bb.emit_output(gv21)
        gv4 = bb.emit(relax.call_tir("extern_func", gv22, (128, 128), dtype="float32"))
        gv5 = bb.emit(relax.call_tir("extern_func", gv4, (128, 128), dtype="float32"))
        bb.emit_func_output(gv5)

    _check(foo, bb.get()["foo"])


def test_dataflow_binding_after_output():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv = R.call_tir("extern_func", x, (128, 128), dtype="float32")
                R.output(gv)
                lv = R.call_tir("extern_func", gv, (128, 128), dtype="float32")
            return gv


def test_dataflow_output_global_var():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            with R.dataflow():
                gv1 = R.call_tir("extern_func", gv0, (128, 128), dtype="float32")
                R.output(gv0, gv1)
            return gv1


def test_dataflow_multiple_output():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv = R.call_tir("extern_func", x, (128, 128), dtype="float32")
                R.output(gv)
                R.output(gv)
            return gv


def test_dataflow_output_outside_dataflow_block():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            R.output(gv)
            return gv


def test_return_without_binding():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")):
        return x

    x = relax.Var("x", (128, 128), relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        bb.emit_func_output(x)

    _check(foo, bb.get()["foo"])


def test_multiple_return():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            return x
            return x


def test_function_without_return():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")


def test_tensor_type_without_args():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        v = R.call_tir("tir_relu", x, (32, 32), dtype="float32")
        return v

    x = relax.Var("x", (32, 32), relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        v = bb.emit(relax.call_tir("tir_relu", x, (32, 32), dtype="float32"))
        bb.emit_func_output(v)

    _check(foo, bb.get()["foo"])


def test_direct_return():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
        return x

    x = relax.Var("x", (32, 32), relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        bb.emit_func_output(x)

    _check(foo, bb.get()["foo"])


def test_call_packed():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        z = R.call_packed("vm.builtin.copy", x, type_args=R.Tensor((32, 32), "float32"))
        return z

    x = relax.Var("x", (32, 32), relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        z = bb.emit(
            relax.Call(
                relax.ExternFunc("vm.builtin.copy"),
                (x,),
                None,
                type_args=[relax.DynTensorType(2, "float32")],
            )
        )
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_annotation():
    @R.function
    def foo(
        x: R.Tensor((32, "m"), "float32"),
        y: R.Tensor(("m"), "float32"),
        r: R.Tensor(dtype="int64"),
    ) -> R.Object:
        m = T.var("int64")
        z: R.Tensor((32, m), "float32") = R.multiply(x, y)
        w: R.Tensor = R.multiply(z, z)
        q: R.Tensor(ndim=2) = R.add(w, w)
        t = R.add(w, z)
        sh: R.Shape = R.shape_of(t)
        o: R.Object = R.call_packed("contrib.tensor_array_stack", x, y, type_args=R.Object)
        return o

    m = tir.Var("m", "int64")
    x = relax.Var("x", (32, m), relax.DynTensorType(2, "float32"))
    y = relax.Var("y", (m,), relax.DynTensorType(1, "float32"))
    r = relax.Var("r", None, relax.DynTensorType(-1, "int64"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y, r)):
        z = bb.emit(R.multiply(x, y))
        w = bb.emit(R.multiply(z, z))
        q = bb.emit(R.add(w, w))
        t = bb.emit(R.add(w, z))
        sh = bb.emit(R.shape_of(t))
        o = bb.emit(
            relax.Call(
                relax.ExternFunc("contrib.tensor_array_stack"),
                [x, y],
                None,
                type_args=[relax.ObjectType()],
            )
        )
        bb.emit_func_output(o)

    _check(foo, bb.get()["foo"])


def test_empty_shape():
    @R.function
    def foo(x: R.Tensor((), "float32")):
        z = R.call_tir("scalar_add", x, (), dtype="float32")
        return z

    (z_bind,) = foo.body.blocks[0].bindings
    shape_expr = z_bind.value.args[2]

    assert isinstance(shape_expr, relax.ShapeExpr)
    assert len(shape_expr.values) == 0


def test_local_function():
    @R.function
    def main(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        @R.function
        def outer_func(
            c1: R.Tensor((2, 3), "float32")
        ) -> R.Callable((R.Tensor(None, "float32", ndim=2),), R.Tensor(None, "float32", ndim=2)):
            @R.function
            def inner_func(x1: R.Tensor((2, 3), "float32")):
                s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                return s

            return inner_func

        in_call = outer_func(x)
        res = in_call(y)
        return res

    main_bindings = main.body.blocks[0].bindings
    assert len(main_bindings) == 3
    outer_func = main_bindings[0].value
    assert isinstance(outer_func, relax.Function)

    outer_func_bindings = outer_func.body.blocks[0].bindings
    assert len(outer_func_bindings) == 1
    inner_func = outer_func_bindings[0].value
    assert isinstance(inner_func, relax.Function)


def test_other_cases():
    # They are corner case tests, which is only to check if it can be parsed.
    # No need to add structural equal checks here
    @R.function
    def foo(x: R.Tensor):
        return R.unique(x, sorted=True)

    @R.function
    def bar(x: R.Tensor):
        return R.print(x, format="{}")


if __name__ == "__main__":
    tvm.testing.main()
