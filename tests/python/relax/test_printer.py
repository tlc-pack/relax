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
import tvm

from tvm import relax
from tvm import tir, relay
from tvm.ir import structural_equal, assert_structural_equal

import tvm.script
from tvm.relax.utils import metadata_partitioner
from tvm.script import tir as T, relax as R


def check_roundtrip(f_pre):
    relax_text = R.parser.astext(f_pre, show_meta_data=True)
    f_post = R.parser.from_source(input_func=relax_text)
    if isinstance(f_pre, tvm.IRModule) and not isinstance(f_post, tvm.IRModule):
        global_vars = f_pre.get_global_vars()
        f_post = tvm.IRModule({global_vars[0]: f_post}, attrs=metadata)
    assert_structural_equal(f_pre, f_post, map_free_vars=True)


def test_annotations():
    @R.function
    def foo(x: Tensor((32, m), "float32"), y: Tensor((m, k), "float32")) -> Tensor:
        z: Tensor((32, k), "float32") = nn.matmul(x, y, units=None)
        w: Tensor(_, _) = multiply(z, z)
        t = subtract(w, z)
        sh: Shape = t.shape
        return t

    check_roundtrip(foo)


def test_ndim_annotations():
    @R.function
    def foo(
        x: Tensor((2, 3, 5), "float32", ndim=3),
        y: Tensor(_, "float32", ndim=-1),
        z: Tensor(_, "float32", ndim=2),
    ):
        w: Tensor(None, "float32", ndim=-1) = x + x
        return w

    check_roundtrip(foo)


def test_match_shape():
    @R.function
    def foo(x: Tensor(_, "float32")):
        relax.match_shape(x.shape, (n, m))
        y: Tensor((n, m), "float32") = add(x, x)
        return x

    check_roundtrip(foo)


def test_if():
    @R.function
    def foo(cond: Tensor((), "bool"), x: Tensor((1,), "float32")):
        if cond:
            w = add(x, x)
            y = multiply(w, w)
        else:
            w = multiply(x, x)
            y = add(w, w)
        return y

    check_roundtrip(foo)


def test_tuple():
    @R.function
    def foo(x: Tensor(_, _), y: Tensor((32,), "float32")):
        t: Tuple(Tensor(_, _), Tensor((32,), "float32")) = (x, y)
        return t

    check_roundtrip(foo)


def test_tuplegetitem():
    @R.function
    def foo(x: Tensor(_, _)):
        y = add(x, x)
        z = multiply(y, x)
        t = relax.Tuple((y, z))
        a = relax.TupleGetItem(t, 0)
        b = relax.TupleGetItem(t, 1)
        c = divide(a, b)
        return c

    check_roundtrip(foo)


def test_local_func():
    @R.function
    def foo(x: Tensor(_, _)):
        @R.function
        def bar(y: Tensor(_, _)):
            return y

        y = bar(x)  # tests local function variable scoping
        return y

    check_roundtrip(foo)


def test_dataflow():
    @R.function
    def foo(x: Tensor(_, _)):
        with relax.dataflow():
            # TODO: parse this
            # nonlocal y, w
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(y, w)
        t = divide(y, w)
        return t

    check_roundtrip(foo)


def test_dataflow_match_shape():
    @R.function
    def foo(x: Tensor(_, _)):
        with relax.dataflow():
            x2: Tensor((n, m), _) = relax.match_shape(x, (n, m))
            y = add(x2, x2)
            z = multiply(y, x)
            relax.match_shape(z.shape, (n, m))
            w: Tensor((n, m), _) = subtract(z, x)
            relax.output(y, w, x2)
        t: Tensor((n, m), _) = divide(y, w)
        q: Tensor((n, m), _) = add(t, x2)
        return q

    check_roundtrip(foo)


def test_inline_tir():
    @R.function
    def foo(x: Tensor((B, 128), "float32"), y: Tensor((128, 128), "float32")):
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

        z = relax.call_tir(my_matmul, (x, y), (B, 128), dtype="float32")
        return z

    check_roundtrip(foo)


def test_call_packed():
    @R.function
    def foo(x: Tensor((3, 3), "float32")):
        # test that we can intro dim vars
        z: Tensor((n, m), "float32") = relax.call_packed(
            "contrib.my_matmul", x, x, mp=False, type_args=(Tensor(ndim=2, dtype="float32"))
        )
        w = relax.call_packed(
            "contrib.my_shape_of",
            x,
            dtype="int32",
            attrs_type_key="relay.attrs.ShapeOfAttrs",
            type_args=(Shape),
        )
        o = relax.call_packed("contrib.tensor_array_stack", x, z, type_args=(Object))
        return z

    check_roundtrip(foo)


def test_primexpr_arithmetic():
    @R.function
    def foo(x: Tensor((n, m), "float32")):
        z: Tensor((n * m,), "float32") = relax.call_packed(
            "my_flatten", (x,), type_args=(Tensor(ndim=2, dtype="float32"))
        )
        sh: Shape = (n + m, n // m)
        return z

    check_roundtrip(foo)


def test_call_tir_extern():
    @R.function
    def foo(x: Tensor):
        z = relax.call_tir("my_extern", (x,), (10,), dtype="float32")
        return z

    check_roundtrip(foo)


def test_const_irmodule():
    def _gen_meta_data():
        @tvm.script.ir_module
        class Module:
            @R.function
            def my_const(x: Tensor((2, 3), "float32")):
                y: Tensor((2, 3), "float32") = relax.const(
                    [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]], dtype="float32"
                )
                z: Tensor((2, 3), "float32") = relax.add(x, y)
                return z

        mod = Module
        relax_text = R.parser.astext(mod, show_meta_data=True)
        texts = metadata_partitioner(relax_text)
        return texts[1]

    json_str = _gen_meta_data()

    @tvm.script.ir_module(metadata=json_str)
    class MyModule:
        @R.function
        def my_const(x: Tensor((2, 3), "float32")):
            z: Tensor((2, 3), "float32") = relax.add(x, meta[relay.Constant][0])
            return z

    my_module = MyModule

    check_roundtrip(my_module)


def test_const():
    @R.function
    def my_const(x: Tensor((2, 3), "float32")):
        y1 = relax.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]])
        y2 = relax.const(2.1, dtype="float32")
        y3 = relax.const([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
        z = relax.add(x, y1)
        r = relax.add(z, y2)
        w = relax.add(r, y3)
        return w

    check_roundtrip(my_const)


def test_const_meta():
    def _get_meta_data():
        @R.function
        def my_const(x: Tensor((2, 3), "float32")):
            y1: Tensor((2, 3), "float32") = relax.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]])
            y2 = relax.const(2.1, dtype="float32")
            y3: Tensor((2, 3), "float32") = relax.const([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
            z: Tensor((2, 3), "float32") = relax.add(x, y1)
            r: Tensor((2, 3), "float32") = relax.add(z, y2)
            w: Tensor((2, 3), "float32") = relax.add(r, y3)
            return w

        relax_text = R.parser.astext(my_const, show_meta_data=True)
        texts = metadata_partitioner(relax_text)
        return texts[1]

    json_str = _get_meta_data()

    @R.function(metadata=json_str)
    def my_const(x: Tensor((2, 3), "float32")):
        y2 = relax.const(2.1, dtype="float32")
        z: Tensor((2, 3), "float32") = relax.add(x, meta[relay.Constant][0])
        r: Tensor((2, 3), "float32") = relax.add(z, y2)
        w: Tensor((2, 3), "float32") = relax.add(r, meta[relay.Constant][1])
        return w

    check_roundtrip(my_const)


def test_class_irmodule():
    @tvm.script.ir_module
    class MyModule:
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

        @R.function
        def f(x: Tensor((n, n), _)) -> Tensor:
            # todo(@yongwww): Update the check_type_ function's body is a call_node
            r = g(x)
            return r

        @R.function
        def g(y: Tensor((n, n), _)) -> Tensor:
            r = relax.call_tir(my_matmul, (y, y), (n, n), dtype="float32")
            return r

        @R.function
        def h(x: Tensor((n, n), _), y: Tensor((n, n), _), z: Tensor((n, n), _)) -> Tensor:
            _ = my_matmul(x, y, z)
            return z

    my_module = MyModule
    check_roundtrip(my_module)


def test_tir_max():
    @R.function
    def tir_max(x: Tensor((m, n), "float32")):
        gv = relax.call_tir("my_extern", (x,), (tir.max(n, m),), dtype="float32")
        return gv

    check_roundtrip(tir_max)


def test_tir_cast():
    @R.function
    def tir_cast(x: Tensor((m,), "float32")):
        gv = relax.call_tir("my_extern", (x,), (tir.cast("int32", m),), dtype="float32")
        return gv

    check_roundtrip(tir_cast)


def test_dyntensor_type():
    x = relax.DynTensorType(ndim=3, dtype="float32")
    assert x.__str__() == 'Tensor[ndim=3, dtype="float32"]'


def test_object_type():
    x = relax.ObjectType()
    assert x.__str__() == "Object"


def test_shape_expr():
    x = relax.ShapeExpr([tir.IntImm("int64", 10), tir.IntImm("int64", 5)])
    assert x.__str__() == "(10, 5)"


def test_runtime_dep_shape():
    x = relax.RuntimeDepShape()
    assert x.__str__() == "_"


def test_func_type():
    @tvm.script.ir_module
    class TestFuncType:
        @R.function
        def global_func_1(
            x: Tensor((m, n), "float32")
        ) -> Callable((Tensor((m, n), "float32")), Tensor((m, n), "float32")):
            @R.function
            def local_func_1(y: Tensor((m, n), "float32")) -> Tensor((m, n), "float32"):
                s = relax.add(x, y)
                return s

            return local_func_1

        @R.function
        def global_func_2(
            x: Tensor((m, n), "float32")
        ) -> Callable(
            (Tensor(None, "float32", ndim=2)),
            Callable((Tensor((m, n), "float32"),), Tensor((m, n), "float32")),
        ):
            @R.function
            def local_func_1(
                y: Tensor((m, n), "float32")
            ) -> Callable((Tensor((m, n), "float32"),), Tensor((m, n), "float32")):
                @R.function
                def local_func_2(z: Tensor((m, n), "float32")) -> Tensor(None, "float32", ndim=2):
                    s1 = relax.add(x, y)
                    s2 = relax.add(z, s1)
                    return s2

                return local_func_2

            return local_func_1

    func_type = TestFuncType
    check_roundtrip(func_type)


if __name__ == "__main__":
    pytest.main([__file__])
