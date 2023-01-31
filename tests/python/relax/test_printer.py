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
import tvm
import tvm.testing

from tvm import TVMError
from tvm import relax
from tvm import tir
from tvm.ir import assert_structural_equal

import tvm.script
from tvm.relax.utils import metadata_partitioner
from tvm.script import tir as T, relax as R


def check_roundtrip(f_pre):
    relax_text = f_pre.script(show_meta=True)
    f_post = tvm.script.parse(relax_text)

    if isinstance(f_pre, tvm.IRModule) and not isinstance(f_post, tvm.IRModule):
        global_vars = f_pre.get_global_vars()
        f_post = tvm.IRModule({global_vars[0]: f_post}, attrs=metadata)
    assert_structural_equal(f_pre, f_post, map_free_vars=True)


def test_annotations():
    @R.function
    def foo(x: R.Tensor((32, "m"), "float32"), y: R.Tensor(("m",), "float32")) -> R.Tensor:
        m = T.var("int64")
        z: R.Tensor((32, m), "float32") = R.multiply(x, y)
        w = R.multiply(z, z)
        t = R.add(w, z)
        sh: R.Shape = R.shape_of(t)
        return t

    check_roundtrip(foo)


def test_ndim_annotations():
    @R.function
    def foo(
        x: R.Tensor((2, 3, 5), "float32"),
        y: R.Tensor(dtype="float32", ndim=-1),
        z: R.Tensor(dtype="float32", ndim=2),
    ):
        w: R.Tensor(None, "float32", ndim=-1) = R.add(x, x)
        return w

    check_roundtrip(foo)


def test_match_cast():
    @R.function
    def foo(x: R.Tensor(dtype="float32")):
        n, m = T.var("int64"), T.var("int64")
        _ = R.match_cast(R.shape_of(x), R.Shape((n, m)))
        y: R.Tensor((n, m), "float32") = R.add(x, x)
        return x

    check_roundtrip(foo)


def test_if():
    @R.function
    def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")) -> R.Tensor:
        if cond:
            w = R.add(x, x)
            y = R.multiply(w, w)
        else:
            w = R.multiply(x, x)
            y = R.add(w, w)
        return y

    check_roundtrip(foo)


def test_tuple():
    @R.function
    def foo(x: R.Tensor(ndim=2), y: R.Tensor((32,), "float32")):
        t: R.Tuple(R.Tensor(ndim=2), R.Tensor((32,), "float32")) = (x, y)
        return t

    check_roundtrip(foo)


def test_tuplegetitem():
    @R.function
    def foo(x: R.Tensor(ndim=2)):
        y = R.add(x, x)
        z = R.multiply(y, x)
        t = (y, z)
        a = R.TupleGetItem(t, 0)
        b = R.TupleGetItem(t, 1)
        c = R.multiply(a, b)
        return c

    check_roundtrip(foo)


def test_local_func():
    @R.function
    def foo(x: R.Tensor(ndim=2)):
        @R.function
        def bar(y: R.Tensor(ndim=2)):
            return y

        y = bar(x)  # tests local function variable scoping
        return y

    check_roundtrip(foo)


def test_dataflow():
    @R.function
    def foo(x: R.Tensor(ndim=2)):
        with R.dataflow():
            # TODO: parse this
            # nonlocal y, w
            y = R.add(x, x)
            z = R.multiply(y, x)
            w = R.add(z, x)
            R.output(y, w)
        t = R.multiply(y, w)
        return t

    check_roundtrip(foo)


def test_dataflow_match_cast():
    @R.function
    def foo(x: R.Tensor(ndim=2)):
        n, m = T.var("int64"), T.var("int64")
        with R.dataflow():
            x2: R.Tensor((n, m)) = R.match_cast(x, R.Tensor((n, m)))
            y = R.add(x2, x2)
            z = R.multiply(y, x)
            _ = R.match_cast(R.shape_of(z), R.Shape((n, m)))
            w: R.Tensor((n, m)) = R.add(z, x)
            R.output(y, w, x2)
        t: R.Tensor((n, m)) = R.multiply(y, w)
        q: R.Tensor((n, m)) = R.add(t, x2)
        return q

    check_roundtrip(foo)


def test_inline_tir():
    @R.function
    def foo(x: R.Tensor(("B", 128), "float32"), y: R.Tensor((128, 128), "float32")):
        B = T.var("int64")

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

        z = R.call_tir(my_matmul, (x, y), R.Tensor((B, 128), dtype="float32"))
        return z

    check_roundtrip(foo)


def test_call_packed():
    @R.function
    def foo(x: R.Tensor((3, 3), "float32")):
        # test that we can intro dim vars
        n, m = T.var("int64"), T.var("int64")
        z: R.Tensor((n, m), "float32") = R.call_packed(
            "contrib.my_matmul", x, x, mp=False, sinfo_args=R.Tensor((n, m), dtype="float32")
        )
        w = R.call_packed(
            "contrib.my_shape_of",
            x,
            dtype="int32",
            attrs_type_key="relay.attrs.ShapeOfAttrs",
            sinfo_args=R.Shape(),
        )
        o = R.call_packed("contrib.tensor_array_stack", x, z, sinfo_args=R.Object())
        return z

    check_roundtrip(foo)


def test_relax_base_op():
    @R.function
    def foo(x: R.Tensor((2, 4), dtype="float32")):
        gv = R.call_builtin_with_ctx("test_intrin", [x], sinfo_args=R.Object)
        return gv

    check_roundtrip(foo)


def test_primexpr_arithmetic():
    @R.function
    def foo(x: R.Tensor(("n", "m"), "float32")):
        n, m = T.var("int64"), T.var("int64")
        z: R.Tensor((n * m,), "float32") = R.call_packed(
            "my_flatten", (x,), sinfo_args=R.Tensor((n * m,), dtype="float32")
        )
        sh: R.Shape = (n + m, n // m)
        return z

    check_roundtrip(foo)


def test_call_tir_extern():
    @R.function
    def foo(x: R.Tensor):
        z = R.call_tir("my_extern", (x,), R.Tensor((10,), dtype="float32"))
        return z

    check_roundtrip(foo)


def test_const_irmodule():
    def _gen_meta_data():
        @tvm.script.ir_module
        class Module:
            @R.function
            def my_const(x: R.Tensor((2, 3), "float32")):
                y: R.Tensor((2, 3), "float32") = R.const(
                    [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]], dtype="float32"
                )
                z: R.Tensor((2, 3), "float32") = R.add(x, y)
                return z

        mod = Module
        relax_text = mod.script(show_meta=True)
        texts = metadata_partitioner(relax_text)
        return texts[1]

    json_str = _gen_meta_data()
    metadata = tvm.ir.load_json(json_str)

    @tvm.script.ir_module
    class MyModule:
        @R.function
        def my_const(x: R.Tensor((2, 3), "float32")):
            z: R.Tensor((2, 3), "float32") = R.add(x, metadata["relax.expr.Constant"][0])
            return z

    my_module = MyModule

    check_roundtrip(my_module)


def test_const():
    @R.function
    def my_const(x: R.Tensor((2, 3), "float32")):
        y1 = R.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]])
        y2 = R.const(2.1, dtype="float32")
        y3 = R.const([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
        z = R.add(x, y1)
        r = R.add(z, y2)
        w = R.add(r, y3)
        return w

    check_roundtrip(my_const)


def test_scalar_const():
    x = relax.Var("x", relax.TensorStructInfo((8,)))
    const_one = relax.const(1, "float32")

    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        v0 = bb.emit(R.add(x, const_one))
        bb.emit_func_output(v0)

    check_roundtrip(bb.get())


def test_const_meta():
    def _get_meta_data():
        @R.function
        def my_const(x: R.Tensor((2, 3), "float32")):
            y1: R.Tensor((2, 3), "float32") = R.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]])
            y2 = R.const(2.1, dtype="float32")
            y3: R.Tensor((2, 3), "float32") = R.const([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
            z: R.Tensor((2, 3), "float32") = R.add(x, y1)
            r: R.Tensor((2, 3), "float32") = R.add(z, y2)
            w: R.Tensor((2, 3), "float32") = R.add(r, y3)
            return w

        relax_text = my_const.script(show_meta=True)

        texts = metadata_partitioner(relax_text)
        return texts[1]

    json_str = _get_meta_data()
    metadata = tvm.ir.load_json(json_str)

    @R.function
    def my_const(x: R.Tensor((2, 3), "float32")):
        y2 = R.const(2.1, dtype="float32")
        z: R.Tensor((2, 3), "float32") = R.add(x, metadata["relax.expr.Constant"][0])
        r: R.Tensor((2, 3), "float32") = R.add(z, y2)
        w: R.Tensor((2, 3), "float32") = R.add(r, metadata["relax.expr.Constant"][1])
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
        def f(x: R.Tensor(("n", "n"))) -> R.Tensor:
            # todo(@yongwww): Update the check_type_ function's body is a call_node
            r = g(x)
            return r

        @R.function
        def g(y: R.Tensor(("n", "n"))) -> R.Tensor(("n", "n"), "float32"):
            n = T.var("int64")
            r = relax.call_tir(my_matmul, (y, y), R.Tensor((n, n), dtype="float32"))
            return r

        @R.function
        def h(
            x: R.Tensor(("n", "n")), y: R.Tensor(("n", "n")), z: R.Tensor(("n", "n"))
        ) -> R.Tensor:
            n = T.var("int64")
            _ = R.call_tir(my_matmul, (x, y), R.Tensor((n, n), dtype="float32"))
            return z

    my_module = MyModule
    check_roundtrip(my_module)


def test_tir_max():
    @R.function
    def tir_max(x: R.Tensor(("m", "n"), "float32")):
        m, n = T.var("int64"), T.var("int64")
        gv = relax.call_tir("my_extern", (x,), R.Tensor((T.max(n, m),), dtype="float32"))
        return gv

    check_roundtrip(tir_max)


def test_tir_cast():
    @R.function
    def tir_cast(x: R.Tensor(("m",), "float32")):
        m = T.var("int64")
        gv = R.call_tir(
            "my_extern", (x,), R.Tensor((T.cast(T.cast(m, "int32"), "int64"),), dtype="float32")
        )
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


def test_func_type():
    # Since current all functions have "global_symbol" attribute, we can't
    # use the same name for different functions, even it's a local function.

    # TODO(relax-team): remove "global_symbol" and enable the same-name local
    # function under different scope.
    @tvm.script.ir_module
    class TestFuncType:
        @R.function
        def global_func_1(
            x: R.Tensor(("m", "n"), "float32")
        ) -> R.Callable((R.Tensor(("m", "n"), "float32"),), R.Tensor(("m", "n"), "float32")):
            m, n = T.var("int64"), T.var("int64")

            @R.function
            def local_func_1(y: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
                s = R.add(x, y)
                return s

            return local_func_1

        @R.function
        def global_func_2(
            x: R.Tensor(("m", "n"), "float32")
        ) -> R.Callable(
            (R.Tensor(None, "float32", ndim=2),),
            R.Callable((R.Tensor(("m", "n"), "float32"),), R.Tensor(("m", "n"), "float32")),
        ):
            m, n = T.var("int64"), T.var("int64")

            @R.function
            def local_func_2(
                y: R.Tensor((m, n), "float32")
            ) -> R.Callable((R.Tensor((m, n), "float32"),), R.Tensor((m, n), "float32")):
                @R.function
                def local_func_3(
                    z: R.Tensor((m, n), "float32")
                ) -> R.Tensor(None, "float32", ndim=2):
                    s1 = R.add(x, y)
                    s2 = R.add(z, s1)
                    return s2

                return local_func_3

            return local_func_2

    func_type = TestFuncType
    check_roundtrip(func_type)


def test_tuple_pretty_print():
    x = relax.Var("x", R.Tensor(ndim=2))
    y = relax.Var("y", R.Tensor((32,)))
    assert relax.Tuple([x]).__str__() == "(x,)"
    assert relax.Tuple([x, y]).__str__() == "(x, y)"


def test_tuple_get_item_pretty_print():
    t = relax.Var(
        "t", relax.TupleStructInfo([relax.TensorStructInfo((32,)), relax.TensorStructInfo(ndim=2)])
    )
    assert relax.TupleGetItem(t, 0).__str__() == "t[0]"
    assert relax.TupleGetItem(t, 1).__str__() == "t[1]"


def test_call_pretty_print():
    x = relax.Var("x", R.Tensor((32,), "float32"))

    extern_func = relax.ExternFunc("my_func")
    call = relax.Call(extern_func, [x], sinfo_args=[R.Tensor(ndim=1, dtype="float32")])
    assert (
        call.__str__()
        == 'R.call_packed("my_func", x, sinfo_args=[R.Tensor(dtype="float32", ndim=1)])'
    )


def test_if_pretty_print():
    @R.function
    def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")) -> R.Tensor:
        if cond:
            w = R.add(x, x)
            y = R.multiply(w, w)
        else:
            w = R.multiply(x, x)
            y = R.add(w, w)
        return y

    if_node = foo.body.blocks[0].bindings[0].value
    assert isinstance(foo.body.blocks[0].bindings[0].value, relax.If)

    # Relax script printer doesn't support printing a standalone IfNode.
    # Catch the error thrown by Relax script printer
    with pytest.raises(TVMError):
        if_node.__str__()


def test_constant_pretty_print():
    import numpy as np  # pylint: disable=import-outside-toplevel

    const0 = relax.const(1.0, "float32")
    assert isinstance(const0, relax.Constant)
    assert const0.__str__() == 'R.const(1, "float32")'

    const1 = relax.const(np.array([[2, 3], [1, 4]], "float32"), "float32")
    assert isinstance(const1, relax.Constant)
    assert const1.__str__() == 'metadata["relax.expr.Constant"][0]\n'


def test_binding_pretty_print():
    @R.function
    def foo0(
        x: R.Tensor((2, 3, 5), "float32"),
    ):
        y: R.Tensor((2, 3, 5), "float32") = R.add(x, x)
        return y

    binding = foo0.body.blocks[0].bindings[0]
    assert isinstance(binding, relax.VarBinding)
    assert binding.__str__() == 'y: R.Tensor((2, 3, 5), dtype="float32") = R.add(x, x)'

    @R.function
    def foo1(x: R.Tensor(dtype="float32")):
        n, m = T.var("int64"), T.var("int64")
        _ = R.match_cast(R.shape_of(x), R.Shape((n, m)))
        y: R.Tensor((n, m), "float32") = R.add(x, x)
        return x

    binding = foo1.body.blocks[0].bindings[0]
    assert isinstance(binding, relax.MatchCast)
    assert binding.__str__() == "_: R.Shape([n, m]) = R.match_cast(R.shape_of(x), R.Shape([n, m]))"


def test_binding_block_pretty_print():
    @R.function
    def foo0(
        x: R.Tensor((2, 3, 5), "float32"),
    ):
        y: R.Tensor((2, 3, 5), "float32") = R.add(x, x)
        z: R.Tensor((2, 3, 5), "float32") = R.multiply(x, y)
        return z

    block = foo0.body.blocks[0]
    assert isinstance(block, relax.BindingBlock)
    assert block.__str__() == (
        'y: R.Tensor((2, 3, 5), dtype="float32") = R.add(x, x)\n'
        'z: R.Tensor((2, 3, 5), dtype="float32") = R.multiply(x, y)\n'
    )

    @R.function
    def foo1(
        x: R.Tensor((2, 3, 5), "float32"),
    ):
        with R.dataflow():
            y: R.Tensor((2, 3, 5), "float32") = R.add(x, x)
            z: R.Tensor((2, 3, 5), "float32") = R.multiply(x, y)
            R.output(z)
        return z

    block = foo1.body.blocks[0]
    assert isinstance(block, relax.DataflowBlock)
    assert block.__str__() == (
        "with R.dataflow():\n"
        '    y: R.Tensor((2, 3, 5), dtype="float32") = R.add(x, x)\n'
        '    z: R.Tensor((2, 3, 5), dtype="float32") = R.multiply(x, y)\n'
        "    R.output(z)\n"
    )


def test_seq_expr_pretty_print():
    @R.function
    def foo(
        x: R.Tensor((2, 3, 5), "float32"),
    ):
        y: R.Tensor((2, 3, 5), "float32") = R.add(x, x)
        z: R.Tensor((2, 3, 5), "float32") = R.multiply(x, y)
        return z

    seq_expr = foo.body
    assert isinstance(seq_expr, relax.SeqExpr)
    assert seq_expr.__str__() == (
        "# block 0\n"
        'y: R.Tensor((2, 3, 5), dtype="float32") = R.add(x, x)\n'
        'z: R.Tensor((2, 3, 5), dtype="float32") = R.multiply(x, y)\n'
    )


def test_extern_func_pretty_print():
    extern_func = relax.ExternFunc("my_func")
    assert extern_func.__str__() == '"my_func"'


def test_primvalue_pretty_print():
    int_prim_value = relax.PrimValue(1)
    assert int_prim_value.__str__() == "R.prim_value(1)"


def test_stringimm_pretty_print():
    strimm = relax.StringImm("test")
    assert strimm.__str__() == 'R.str("test")'


def test_dtypeimm_pretty_print():
    dtypeimm = relax.DataTypeImm("float32")
    assert dtypeimm.__str__() == 'R.dtype("float32")'


if __name__ == "__main__":
    tvm.testing.main()
