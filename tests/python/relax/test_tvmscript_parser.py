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

from typing import Optional, Union, List

import pytest
import tvm
import tvm.testing
from tvm import IRModule, relax, tir
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.relax import RuntimeDepShape, DynTensorType


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.parse(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_simple_func():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
        R.func_attr({"Primitive": 1})
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
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
        def tir_func(
            x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
            y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            # TODO(Siyuan): Need to change to `TestModule.tir_func`
            gv0 = R.call_tir(tir_func, x, (128, 128), dtype="float32")
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)

    _check(TestModule, bb.get())


def test_relax_tensor_op():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")) -> R.Tensor((4, 4), "float32"):
        y = R.add(x, x)
        z = R.multiply(x, y)
        return z

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
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

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        alloc = bb.emit(relax.op.builtin.alloc_tensor(relax.ShapeExpr((4, 4)), "float32", 0))
        shape = bb.emit(relax.op.shape_of(alloc))
        bb.emit_func_output(shape)

    _check(foo, bb.get()["foo"])


def test_symbolic_shape():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
        m = T.var("int64", "m")
        n = T.var("int64", "n")
        gv0 = R.call_tir("extern_func", x, (m, n), dtype="float32")
        return gv0

    @R.function
    def bar(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
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
        x = relax.Var("x", R.Tensor([m, n], "float32"))
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

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
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

    x = relax.Var("x", R.Tensor("float32", ndim=-1))
    y = relax.Var("y", R.Tensor("float32", ndim=-1))
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

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        gv0 = bb.emit(relax.call_tir("extern_func_0", x, (4, 4), dtype="float32"))
        gv1 = bb.emit(relax.call_tir("extern_func_1", x, (4, 4), dtype="float32"))
        bb.emit_func_output(relax.Tuple((gv0, gv1)))

    _check(foo, bb.get()["foo"])


def test_tuple_return_2():
    @R.function
    def foo(x: R.Tensor("float32", ndim=2)):
        n, m = T.var("int64"), T.var("int64")
        x0 = R.match_shape(x, (n, m))
        return (x0, (n + 1, m, 1))

    x = relax.Var("x", R.Tensor("float32", ndim=2))
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        x0 = bb.match_shape(x, (n, m))
        bb.emit_func_output(relax.Tuple([x0, relax.ShapeExpr([n + 1, m, 1])]))

    _check(foo, bb.get()["foo"])


def test_tuple_binding():
    @R.function
    def foo(x: R.Tensor("float32", ndim=2)):
        n, m = T.var("int64"), T.var("int64")
        x0 = R.match_shape(x, (n, m))
        t0 = (x, x0)
        t1 = (x, (n, m), t0)
        return t1

    x = relax.Var("x", R.Tensor("float32", ndim=2))
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        x0 = bb.match_shape(x, (n, m))
        t0 = bb.emit(relax.Tuple([x, x0]))
        t1 = bb.emit(relax.Tuple([x, relax.ShapeExpr([n, m]), t0]))
        bb.emit_func_output(t1)

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

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
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

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
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

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
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

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        v = bb.emit(relax.call_tir("tir_relu", x, (32, 32), dtype="float32"))
        bb.emit_func_output(v)

    _check(foo, bb.get()["foo"])


def test_direct_return():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
        return x

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        bb.emit_func_output(x)

    _check(foo, bb.get()["foo"])


def test_call_packed():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        z = R.call_packed("vm.builtin.copy", x, type_args=R.Tensor((32, 32), "float32"))
        return z

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        z = bb.emit(
            relax.Call(
                relax.ExternFunc("vm.builtin.copy"),
                (x,),
                None,
                type_args=[DynTensorType(2, "float32")],
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
        m = T.var("int64", "m")
        z: R.Tensor((32, m), "float32") = R.multiply(x, y)
        w: R.Tensor = R.multiply(z, z)
        q: R.Tensor(ndim=2) = R.add(w, w)
        t = R.add(w, z)
        sh: R.Shape = R.shape_of(t)
        _: R.Tensor((1, 1), "int8") = R.builtin.alloc_tensor(
            (1, 1), dtype="int8", runtime_device_index=0
        )
        o: R.Object = R.call_packed("contrib.tensor_array_stack", x, y, type_args=R.Object)
        return o

    def _check_type_shape(binding, expected_type, expected_shape):
        tvm.ir.assert_structural_equal(binding.var.checked_type, expected_type)
        tvm.ir.assert_structural_equal(binding.var.shape_, expected_shape)

    # Cannot use block builder here because we need to check the annotated type,
    # which may be inconsistent with deduced type.
    assert isinstance(foo.ret_struct_info, relax.ObjectStructInfo)
    m = foo.params[0].shape[1]
    bindings = foo.body.blocks[0].bindings
    _check_type_shape(
        bindings[0],
        relax.DynTensorType(ndim=2, dtype="float32"),
        relax.ShapeExpr([tvm.tir.IntImm("int64", 32), m]),
    )
    _check_type_shape(bindings[1], relax.DynTensorType(dtype=""), RuntimeDepShape())
    _check_type_shape(bindings[2], relax.DynTensorType(ndim=2, dtype=""), RuntimeDepShape())
    _check_type_shape(bindings[3], relax.DynTensorType(dtype=""), RuntimeDepShape())
    _check_type_shape(bindings[4], relax.ShapeType(), None)
    _check_type_shape(
        bindings[5],
        relax.DynTensorType(ndim=2, dtype="int8"),
        relax.ShapeExpr([tvm.tir.IntImm("int64", 1), tvm.tir.IntImm("int64", 1)]),
    )
    _check_type_shape(bindings[6], relax.ObjectType(), None)


def test_annotate_override():
    @R.function
    def foo(x: R.Tensor):
        y = x
        # z will be treated as object type even though it's a tensor
        z: R.Object = y
        return z

    assert isinstance(foo.ret_struct_info, relax.ObjectStructInfo)
    y_bind, z_bind = foo.body.blocks[0].bindings
    assert isinstance(y_bind.var.checked_type, relax.DynTensorType)
    assert isinstance(z_bind.var.checked_type, relax.ObjectType)


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

    @I.ir_module
    class TestModule:
        @R.function
        def f(x: R.Tensor((128, 128), "float32"), y: R.Tensor((128, 128), "float32")):
            @T.prim_func
            def my_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                C = T.match_buffer(c, (128, 128))

                for i, j, k in T.grid(128, 128, 128):
                    with T.block():
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] += A[vi, vk] * B[vj, vk]

            z = relax.call_tir(my_matmul, (x, y), (128, 128), dtype="float32")
            return z

    bindings = TestModule["f"].body.blocks[0].bindings
    assert len(bindings) == 2
    tir_func = bindings[0].value
    assert isinstance(tir_func, tir.PrimFunc)


def test_cross_function_call():
    @I.ir_module
    class Mod0:
        @R.function
        def foo(x: R.Tensor((10, 5), "float32")):
            s = R.add(x, x)
            return s

        @R.function
        def main(x: R.Tensor((10, 5), "float32")):
            inner = foo
            gv1 = inner(x)
            gv2 = foo(x)
            return (inner, gv1, gv2)

    @I.ir_module
    class Mod1:
        @R.function
        def main(x: R.Tensor((10, 5), "float32")):
            inner = foo
            gv1 = inner(x)
            gv2 = foo(x)
            return (inner, gv1, gv2)

        @R.function
        def foo(x: R.Tensor((10, 5), "float32")) -> R.Tensor((10, 5), "float32"):
            s = R.add(x, x)
            return s

    # TODO(relax-team): enable it after fix block builder
    # Current error: `gv2.shape` is different: (10, 5) vs RuntimeDepShape()
    # tvm.ir.assert_structural_equal(Mod0, Mod1)

    with pytest.raises(OSError):

        @I.ir_module
        class ErrorMod:
            @R.function
            def main(x: R.Tensor((10, 5), "float32")):
                inner = foo
                gv1 = inner(x)
                gv2 = foo(x)
                return (inner, gv1, gv2)

            @R.function
            def foo(
                x: R.Tensor((10, 5), "float32")
            ):  # need function ret info since it is parse later than `main`
                s = R.add(x, x)
                return s


def test_if_branch():
    @R.function
    def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")) -> R.Tensor((1,), "float32"):
        if cond:
            w = R.add(x, x)
            y = R.multiply(w, w)
        else:
            w = R.multiply(x, x)
            y = R.add(w, w)
        return y

    cond, x = foo.params
    y_bind = foo.body.blocks[0].bindings[0]
    y, ite = y_bind.var, y_bind.value

    assert isinstance(y, relax.Var)
    assert y.name_hint == "y"

    assert isinstance(ite, relax.If)
    assert isinstance(ite.true_branch, relax.SeqExpr)
    assert isinstance(ite.false_branch, relax.SeqExpr)

    def check_call(call, op, args):
        assert isinstance(call, relax.Call)
        if isinstance(op, str):
            assert str(call.op) == op
        else:
            assert call.op == op
        tvm.ir.assert_structural_equal(call.args, args)

    w_bind = ite.true_branch.blocks[0].bindings[0]
    # the seq exprts in the branches are normalized to bind any call
    # in the seq expr "body" to a var
    y_bind = ite.true_branch.blocks[-1].bindings[-1]
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "relax.add", [x, x])
    check_call(y_bind.value, "relax.multiply", [w_bind.var, w_bind.var])

    w_bind = ite.false_branch.blocks[0].bindings[0]
    y_bind = ite.false_branch.blocks[-1].bindings[-1]
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "relax.multiply", [x, x])
    check_call(y_bind.value, "relax.add", [w_bind.var, w_bind.var])


def test_if_inside_dataflow():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(
            cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")
        ) -> R.Tensor((1,), "float32"):
            with R.dataflow():
                if cond:
                    w = R.add(x, x)
                    y = R.multiply(w, w)
                else:
                    w = R.multiply(x, x)
                    y = R.add(w, w)
                R.output(y)
            return y


def test_if_branch_output_name():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(
            cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")
        ) -> R.Tensor((1,), "float32"):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                z = R.add(w, w)
            return y


def test_if_branch_var_scope():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(
            cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")
        ) -> R.Tensor((1,), "float32"):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                y = R.add(w, w)
            return w


def test_other_cases():
    # They are corner case tests, which is only to check if it can be parsed.
    # No need to add structural equal checks here
    @R.function
    def foo(x: R.Tensor):
        return R.unique(x, sorted=True)

    @R.function
    def bar(x: R.Tensor):
        return R.print(x, format="{}")


def test_erase_to_well_defined():
    @R.function
    def foo(x: R.Tensor):
        q = x
        m, n = T.var("int64"), T.var("int64")
        z = R.match_shape(q, (m, n))
        w = z
        return w

    tvm.ir.assert_structural_equal(foo.ret_struct_info, R.Tensor(ndim=2))
    _check(foo, None)


@pytest.mark.skip(reason="potential upstream Metadata changes.")
def test_meta():
    metadata = tvm.ir.load_json(
        {
            "root": 1,
            "nodes": [
                {"type_key": ""},
                {"type_key": "Map", "keys": ["relax.expr.Constant"], "data": [2]},
                {"type_key": "Array", "data": [3, 12]},
                {
                    "type_key": "relax.expr.Constant",
                    "attrs": {
                        "_checked_type_": "6",
                        "data": "0",
                        "shape_": "7",
                        "span": "0",
                        "virtual_device_": "4",
                    },
                },
                {
                    "type_key": "VirtualDevice",
                    "attrs": {
                        "device_type_int": "-1",
                        "memory_scope": "5",
                        "target": "0",
                        "virtual_device_id": "-1",
                    },
                },
                {"type_key": "runtime.String"},
                {
                    "type_key": "relax.DynTensorType",
                    "attrs": {"dtype": "float32", "ndim": "2", "span": "0"},
                },
                {
                    "type_key": "relax.expr.ShapeExpr",
                    "attrs": {"_checked_type_": "11", "shape_": "0", "span": "0", "values": "8"},
                },
                {"type_key": "Array", "data": [9, 10]},
                {"type_key": "IntImm", "attrs": {"dtype": "int64", "span": "0", "value": "2"}},
                {"type_key": "IntImm", "attrs": {"dtype": "int64", "span": "0", "value": "3"}},
                {"type_key": "relax.ShapeType", "attrs": {"span": "0"}},
                {
                    "type_key": "relax.expr.Constant",
                    "attrs": {
                        "_checked_type_": "13",
                        "data": "1",
                        "shape_": "14",
                        "span": "0",
                        "virtual_device_": "4",
                    },
                },
                {
                    "type_key": "relax.DynTensorType",
                    "attrs": {"dtype": "float32", "ndim": "2", "span": "0"},
                },
                {
                    "type_key": "relax.expr.ShapeExpr",
                    "attrs": {"_checked_type_": "18", "shape_": "0", "span": "0", "values": "15"},
                },
                {"type_key": "Array", "data": [16, 17]},
                {"type_key": "IntImm", "attrs": {"dtype": "int64", "span": "0", "value": "2"}},
                {"type_key": "IntImm", "attrs": {"dtype": "int64", "span": "0", "value": "3"}},
                {"type_key": "relax.ShapeType", "attrs": {"span": "0"}},
            ],
            "b64ndarrays": [
                "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAgAAAAIgAQACAAAAAAAAAAMAAAAAAAAAGAAAAAAAAADNzMw9zcyMP2ZmBkBmZkZAMzODQDMzo0A=",
                "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAgAAAAIgAQACAAAAAAAAAAMAAAAAAAAAGAAAAAAAAAAAAEBAAABAQAAAQEAAAEBAAABAQAAAQEA=",
            ],
        }
    )

    @R.function
    def my_const(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
        # block 0
        y1: R.Tensor((2, 3), dtype="float32") = metadata["relax.expr.Constant"][0]
        y2: R.Tensor((), dtype="float32") = 2.1
        y3: R.Tensor((2, 3), dtype="float32") = metadata["relax.expr.Constant"][1]
        z: R.Tensor((2, 3), dtype="float32") = R.add(x, y1)
        r: R.Tensor((2, 3), dtype="float32") = R.add(z, y2)
        w: R.Tensor((2, 3), dtype="float32") = R.add(r, y3)
        return w

    _check(my_const, None)


if __name__ == "__main__":
    tvm.testing.main()
