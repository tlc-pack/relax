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
from tvm import tir, relay, relax
from tvm.ir import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R

# TODO: replace xfails with proper diagnostics checking.
#       c.f. tests/python/unittest/test_tvmscript_error_report.py


def check_shape(e, s):

    if isinstance(e, relax.Call):
        e = e.shape
    elif isinstance(e, relax.Expr):
        e = e.shape_

    if s is None:
        assert e is None
        return

    assert len(e) == len(s)

    for edim, sdim in zip(e, s):
        if isinstance(sdim, str):
            assert isinstance(edim, tir.Var)
            assert edim.name == sdim
        else:
            assert isinstance(edim, tir.IntImm)
            assert edim.value == sdim


def check_tensor_var(v, s, d, rank=None):
    assert isinstance(v.type_annotation, relax.ty.DynTensorType)
    assert v.type_annotation.dtype == d
    if isinstance(s, (list, tuple)):
        assert v.type_annotation.rank == len(s)
    if rank is not None:
        assert v.type_annotation.rank == rank
    check_shape(v, s)


def check_call(call, op, args):
    assert isinstance(call, relax.Call)
    if isinstance(op, str):
        op = relay.op.get(op)
    assert call.op == op
    assert_structural_equal(call.args, args)


def test_annotations():
    @R.function
    def f(x: Tensor[(32, m), "float32"], y: Tensor[(m, k), "float32"]) -> Tensor:
        z: Tensor[(32, k), "float32"] = nn.matmul(x, y, units=None)
        w: Tensor[_, _] = multiply(z, z)
        q: Tensor[(_, _), _] = add(w, w)
        t = subtract(w, z)
        sh: Shape = t.shape
        return t

    x, y = f.params
    z_bind, w_bind, q_bind, t_bind, sh_bind = f.body.blocks[0].bindings
    z, mm = z_bind.var, z_bind.value
    w, mul = w_bind.var, w_bind.value
    q, add = q_bind.var, w_bind.value
    t, sub = t_bind.var, t_bind.value
    sh, shape_of = sh_bind.var, sh_bind.value

    check_tensor_var(x, (32, "m"), "float32")
    check_tensor_var(y, ("m", "k"), "float32")
    check_tensor_var(z, (32, "k"), "float32")
    check_tensor_var(w, None, "")
    check_tensor_var(q, None, "", rank=2)
    assert t.type_annotation is None
    assert isinstance(sh.type_annotation, relax.ty.ShapeType)

    check_call(mm, "nn.matmul", [x, y])
    check_call(mul, "multiply", [z, z])
    check_call(sub, "subtract", [w, z])
    check_call(shape_of, "relax.shape_of", [t])

    assert f.body.body == t

    assert isinstance(f.ret_type, relax.ty.DynTensorType)


def test_match_shape():
    @R.function
    def f(x: Tensor[_, "float32"]):
        relax.match_shape(x.shape, (n, m))
        y: Tensor[(n, m), "float32"] = add(x, x)
        return x

    match_sh = f.body.blocks[0].bindings[0]
    pattern, value = match_sh.pattern, match_sh.value

    check_shape(pattern, ("n", "m"))
    check_call(value, "relax.shape_of", [f.params[0]])


def test_dim_var_intro_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor[_, _]):
            y: Tensor[(n, m), "float32"] = x
            return y


def test_if():
    @R.function
    def f(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
        if cond:
            w = add(x, x)
            y = multiply(w, w)
        else:
            w = multiply(x, x)
            y = add(w, w)
        return y

    cond, x = f.params
    y_bind = f.body.blocks[0].bindings[0]
    y, ite = y_bind.var, y_bind.value

    check_tensor_var(cond, tuple(), "bool")
    check_tensor_var(x, (1,), "float32")

    assert isinstance(y, relax.Var)
    assert y.name_hint == "y"

    assert isinstance(ite, relax.If)
    assert isinstance(ite.true_branch, relax.SeqExpr)
    assert isinstance(ite.false_branch, relax.SeqExpr)

    w_bind = ite.true_branch.blocks[0].bindings[0]
    body = ite.true_branch.body
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "add", [x, x])
    check_call(body, "multiply", [w_bind.var, w_bind.var])

    w_bind = ite.false_branch.blocks[0].bindings[0]
    body = ite.false_branch.body
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "multiply", [x, x])
    check_call(body, "add", [w_bind.var, w_bind.var])


# TODO: figure out if-else binding type and shape


def test_var_redefine_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x, y):
            z = add(x, y)
            y = z
            return y


def test_var_redefine_fail_if():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
            y = x
            if cond:
                w = add(x, x)
                y = multiply(w, w)
            else:
                w = multiply(x, x)
                y = add(w, w)
            return y


@pytest.mark.xfail
def test_var_if_scoping_fail():
    # TODO: fix this
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
            if cond:
                w = add(x, x)
                y = multiply(w, w)
            else:
                w = multiply(x, x)
                y = add(w, w)
            return w


def test_if_mismatch_var_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
            if cond:
                w = add(x, x)
                y = multiply(w, w)
            else:
                w = multiply(x, x)
                z = add(w, w)
            return z


def test_unassigned_call_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor[_, _]):
            add(x, x)
            return x


def test_tuple():
    @R.function
    def f(x: Tensor[_, _], y: Tensor[(32,), "float32"]):
        t: Tuple[Tensor[_, _], Tensor[(32,), "float32"]] = (x, y)
        return t

    x, y = f.params
    t_bind = f.body.blocks[0].bindings[0]
    t, tup = t_bind.var, t_bind.value

    assert isinstance(t.type_annotation, relay.TupleType)
    annot = t.type_annotation
    assert isinstance(annot.fields[0], relax.ty.DynTensorType) and annot.fields[0].dtype == ""
    assert (
        isinstance(annot.fields[1], relax.ty.DynTensorType) and annot.fields[1].dtype == "float32"
    )

    assert t.shape_ is None

    assert isinstance(tup, relax.Tuple)
    assert_structural_equal(tup.fields, [x, y])
    assert tup.shape_ is None
    check_shape(tup.fields[0], None)
    check_shape(tup.fields[1], (32,))


def test_tuplegetitem():
    @R.function
    def f(x: Tensor[_, _], y: Tensor[_, _]):
        t1 = relax.Tuple((x, y))
        t2 = (x, y)
        a = t1[0]
        b = relax.TupleGetItem(t2, 1)
        c = add(a, b)
        return c

    x, y = f.params
    bind_0 = f.body.blocks[0].bindings[0]
    bind_1 = f.body.blocks[0].bindings[1]
    bind_2 = f.body.blocks[0].bindings[2]
    bind_3 = f.body.blocks[0].bindings[3]
    bind_4 = f.body.blocks[0].bindings[4]
    assert_structural_equal(bind_0.value.fields, [x, y])
    assert_structural_equal(bind_1.value.fields, [x, y])
    assert isinstance(bind_0.value, relax.expr.Tuple)
    assert isinstance(bind_1.value, relax.expr.Tuple)
    assert isinstance(bind_2.value, relax.TupleGetItem)
    assert isinstance(bind_3.value, relax.TupleGetItem)
    assert bind_2.value.index == 0
    assert bind_3.value.index == 1
    assert bind_2.var.name_hint == "a"
    assert bind_3.var.name_hint == "b"
    check_call(bind_4.value, "add", [bind_2.var, bind_3.var])


def test_local_func():
    @R.function
    def f(x: Tensor[_, _]):
        @R.function
        def bar(y: Tensor[_, _]):
            return y

        y = bar(x)  # tests local function variable scoping
        return y

    bar_bind, y_bind = f.body.blocks[0].bindings
    bar, bar_fn = bar_bind.var, bar_bind.value
    bar_x = y_bind.value

    assert isinstance(bar_fn, relax.Function)
    assert bar_fn.body.body == bar_fn.params[0]

    assert bar_x.op == bar


def test_dataflow():
    @R.function
    def f(x: Tensor[_, _]):
        with relax.dataflow():
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(y, w)
        t = divide(y, w)
        return t

    assert len(f.body.blocks) == 2
    df_block = f.body.blocks[0]
    y_bind, z_bind, w_bind = df_block.bindings
    (t_bind,) = f.body.blocks[1].bindings
    x = f.params[0]
    y, z, w, t = map(lambda b: b.var, [y_bind, z_bind, w_bind, t_bind])

    assert isinstance(y, relax.Var)
    assert isinstance(z, relax.DataflowVar)
    assert isinstance(w, relax.Var)

    check_call(y_bind.value, "add", [x, x])
    check_call(z_bind.value, "multiply", [y, x])
    check_call(w_bind.value, "subtract", [z, x])
    check_call(t_bind.value, "divide", [y, w])

    assert f.body.body == t


def test_dataflow_match_shape():
    @R.function
    def f(x: Tensor[_, _]):
        with relax.dataflow():
            x2: Tensor[(n, m), _] = relax.match_shape(x, (n, m))
            y = add(x2, x2)
            z = multiply(y, x)
            relax.match_shape(z.shape, (n, m))
            w: Tensor[(n, m), _] = subtract(z, x)
            relax.output(y, w, x2)
        t: Tensor[(n, m), _] = divide(y, w)
        q: Tensor[(n, m), _] = add(t, x2)
        return q

    x = f.params[0]
    df_block = f.body.blocks[0]
    x2_bind = df_block.bindings[0]
    z_shape_bind = df_block.bindings[3]
    q_bind = f.body.blocks[1].bindings[1]

    assert x2_bind.var.name_hint == "x2"
    check_tensor_var(x2_bind.var, ("n", "m"), "")
    check_shape(x2_bind.pattern, ("n", "m"))
    assert x2_bind.value == x

    check_shape(z_shape_bind.pattern, ("n", "m"))

    assert q_bind.value.args[1] == x2_bind.var


@pytest.mark.xfail
def test_dataflow_scope_fail():
    with pytest.raises(tvm.error.DiagnosticError):
        # FIXME
        @R.function
        def f(x: Tensor[_, _]):
            with relax.dataflow():
                y = add(x, x)
                z = multiply(y, x)
                w = subtract(z, x)
                relax.output(y, w)
            t = divide(y, z)
            return t


def test_dataflow_syntax_fail_pattern():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor[_, _]):
            with relax.dataflow() as df:
                y = add(x, x)
                z = multiply(y, x)
                w = subtract(z, x)
                relax.output(y, z)
            t = divide(y, z)
            return t


def test_dataflow_syntax_fail_params():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor[_, _]):
            with relax.dataflow(x) as df:
                y = add(x, x)
                z = multiply(y, x)
                w = subtract(z, x)
                relax.output(y, w)
            t = divide(y, z)
            return t


def test_dataflow_unbound_outputs():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor[_, _]):
            with relax.dataflow():
                y = add(x, x)
                z = multiply(y, x)
                w = subtract(z, x)
                relax.output(x, y, w, q)
            t = divide(y, z)
            return t


def test_invalid_special_op_dataflow():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor):
            y = add(x, x)
            z = relax.dataflow()
            return z


def test_invalid_special_op_output():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor):
            y = add(x, x)
            z = relax.output(y)
            return z


def test_func_no_return_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: Tensor[_, _]):
            y = add(x, x)


def test_call_tir():
    @R.function
    def foo(x: Tensor[(m, n), "float32"]):
        gv0 = relax.call_tir("test.op.identity", (x,), (m, n), dtype="float32")
        return gv0

    call_tir_node = foo.body.blocks[0].bindings[0].value
    assert call_tir_node.attrs is None
    assert_structural_equal(
        call_tir_node.type_args[0], relax.DynTensorType(rank=2, dtype="float32")
    )


def test_inline_tir():
    @R.function
    def f(x: Tensor[(B, 128), "float32"], y: Tensor[(128, 128), "float32"]):
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

    x, y = f.params
    B = x.shape_[0]
    mm_bind, z_bind = f.body.blocks[0].bindings

    assert mm_bind.var.name_hint == "my_matmul"
    assert isinstance(mm_bind.value, tir.PrimFunc)

    check_call(
        z_bind.value,
        "relax.call_tir",
        [mm_bind.var, relax.Tuple([x, y]), relax.ShapeExpr([B, tir.IntImm("int64", 128)])],
    )


def test_call_packed():
    @R.function
    def f(x: Tensor[(3, 3), "float32"]):
        # test that we can intro dim vars
        z: Tensor[(n, m), "float32"] = relax.call_packed("contrib.my_matmul", x, x, mp=False)
        w = relax.call_packed(
            "contrib.my_shape_of", x, dtype="int32", attrs_type_key="relay.attrs.ShapeOfAttrs"
        )
        return z

    x = f.params[0]
    (z_bind, w_bind) = f.body.blocks[0].bindings
    check_tensor_var(z_bind.var, ("n", "m"), "float32")

    assert isinstance(z_bind.value.op, relax.ExternFunc)
    assert z_bind.value.op.global_symbol == "contrib.my_matmul"
    assert "mp" in z_bind.value.attrs and z_bind.value.attrs["mp"] == False
    assert_structural_equal(z_bind.value.args, [x, x])

    assert isinstance(w_bind.value.attrs, relay.op.op_attrs.ShapeOfAttrs)


def test_constant():
    @R.function
    def f(x: Tensor[(2, 3), "float32"]):
        y1 = relax.const(2, dtype="float32")
        y2 = relax.const([[3.1, 4.0, 5.0], [6.0, 7.1, 9.0]])
        z = add(x, y1)
        r = add(z, y2)
        return r

    x = f.params[0]
    bind_0 = f.body.blocks[0].bindings[0]
    assert bind_0.var.name_hint == "y1"
    bind_1 = f.body.blocks[0].bindings[1]
    assert bind_1.var.name_hint == "y2"
    bind_2 = f.body.blocks[0].bindings[2]
    assert bind_2.var.name_hint == "z"
    bind_3 = f.body.blocks[0].bindings[3]
    assert bind_3.var.name_hint == "r"
    check_call(bind_2.value, "add", [x, bind_0.var])
    check_call(bind_3.value, "add", [bind_2.var, bind_1.var])


def test_primexpr_arithmetic():
    @R.function
    def f(x: Tensor[(n, m), "float32"]):
        z: Tensor[(n * m,), "float32"] = relax.call_packed("my_flatten", (x,))
        sh: Shape = (n + m, n // m)
        return z

    x = f.params[0]
    n, m = x.shape_
    z_bind, sh_bind = f.body.blocks[0].bindings

    assert_structural_equal(z_bind.var.shape_.values, [tir.Mul(n, m)])
    assert_structural_equal(sh_bind.value.values, [tir.Add(n, m), tir.FloorDiv(n, m)])


def test_call_tir_extern():
    @R.function
    def f(x: Tensor):
        z = relax.call_tir("my_extern", (x,), (10,), dtype="float32")
        return z

    x = f.params[0]
    (z_bind,) = f.body.blocks[0].bindings

    check_call(
        z_bind.value,
        "relax.call_tir",
        [
            relax.ExternFunc("my_extern"),
            relax.Tuple([x]),
            relax.ShapeExpr([tir.IntImm("int64", 10)]),
        ],
    )


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
        def f(x: Tensor[(n, n), _]) -> Tensor:
            return g(x)

        @R.function
        def g(y: Tensor[(n, n), _]) -> Tensor:
            return relax.call_tir(my_matmul, (y, y), (n, n), dtype="float32")

        @R.function
        def j(y: Tensor[(n, n), _]) -> Tensor:
            with relax.dataflow():
                gv = relax.call_tir(my_matmul, (y, y), (n, n), dtype="float32")
                relax.output(gv)
            return gv

        @R.function
        def h(x, y, z):
            _ = my_matmul(x, y, z)
            return z

    my_module = MyModule
    assert isinstance(my_module, tvm.IRModule)

    var_f = my_module.get_global_var("f")
    var_g = my_module.get_global_var("g")
    var_j = my_module.get_global_var("j")
    var_my_matmul = my_module.get_global_var("my_matmul")
    f = my_module[var_f]
    g = my_module[var_g]
    j = my_module[var_j]

    assert f.body.body.op == var_g
    assert g.body.body.args[0] == var_my_matmul

    gv_bind = j.body.blocks[0].bindings[0]
    assert gv_bind.value.checked_type.rank == 2
    assert gv_bind.value.checked_type.dtype == "float32"
    assert gv_bind.var.checked_type.rank == 2
    assert gv_bind.var.checked_type.dtype == "float32"
    check_shape(gv_bind.value, ("n", "n"))
    check_shape(gv_bind.var, ("n", "n"))


if __name__ == "__main__":
    pytest.main([__file__])
