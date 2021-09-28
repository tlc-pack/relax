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
from tvm import relax as rx
from tvm import tir, relay
from tvm.ir import structural_equal

# TODO: replace xfails with proper diagnostics checking.
#       c.f. tests/python/unittest/test_tvmscript_error_report.py


def rx_func(func):
    return func.module[func.fn_name]


def check_shape(e, s):
    if isinstance(e, rx.Expr):
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
    assert isinstance(v.type_annotation, rx.ty.DynTensorType)
    assert v.type_annotation.dtype == d
    if isinstance(s, (list, tuple)):
        assert v.type_annotation.rank == len(s)
    if rank is not None:
        assert v.type_annotation.rank == rank
    check_shape(v, s)


def check_call(call, op, args):
    assert isinstance(call, rx.Call)
    if isinstance(op, str):
        op = relay.op.get(op)
    assert call.op == op
    assert structural_equal(call.args, args)


def test_annotations():
    @rx.script
    def foo(x: Tensor[(32, m), "float32"], y: Tensor[(m, k), "float32"]) -> Tensor:
        z: Tensor[(32, k), "float32"] = nn.matmul(x, y, units=None)
        w: Tensor[_, _] = multiply(z, z)
        q: Tensor[(_, _), _] = add(w, w)
        t = subtract(w, z)
        sh: Shape = t.shape
        return t

    f = rx_func(foo)
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
    assert isinstance(sh.type_annotation, rx.ty.ShapeType)

    check_call(mm, "nn.matmul", [x, y])
    check_call(mul, "multiply", [z, z])
    check_call(sub, "subtract", [w, z])
    check_call(shape_of, "shape_of", [t])

    assert f.body.body == t

    assert isinstance(f.ret_type, rx.ty.DynTensorType)


def test_match_shape():
    @rx.script
    def foo(x: Tensor[_, "float32"]):
        relax.match_shape(x.shape, (n, m))
        y: Tensor[(n, m), "float32"] = add(x, x)
        return x

    f = rx_func(foo)
    match_sh = f.body.blocks[0].bindings[0]
    pattern, value = match_sh.pattern, match_sh.value

    check_shape(pattern, ("n", "m"))
    check_call(value, "shape_of", [f.params[0]])


@pytest.mark.xfail
def test_dim_var_intro_fail():
    @rx.script
    def foo(x: Tensor[_, _]):
        y: Tensor[(n, m), "float32"] = x
        return y


def test_if():
    @rx.script
    def foo(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
        if cond:
            w = add(x, x)
            y = multiply(w, w)
        else:
            w = multiply(x, x)
            y = add(w, w)
        return y

    f = rx_func(foo)
    cond, x = f.params
    y_bind = f.body.blocks[0].bindings[0]
    y, ite = y_bind.var, y_bind.value

    check_tensor_var(cond, tuple(), "bool")
    check_tensor_var(x, (1,), "float32")

    assert isinstance(y, rx.Var)
    assert y.name_hint == "y"

    assert isinstance(ite, rx.If)
    assert isinstance(ite.true_branch, rx.SeqExpr)
    assert isinstance(ite.false_branch, rx.SeqExpr)

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


@pytest.mark.xfail
def test_var_redefine_fail():
    @rx.script
    def foo(x, y):
        z = add(x, y)
        y = z
        return y


@pytest.mark.xfail
def test_var_redefine_fail_if():
    @rx.script
    def foo(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
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
    @rx.script
    def foo(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
        if cond:
            w = add(x, x)
            y = multiply(w, w)
        else:
            w = multiply(x, x)
            y = add(w, w)
        return w


@pytest.mark.xfail
def test_if_mismatch_var_fail():
    @rx.script
    def foo(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
        if cond:
            w = add(x, x)
            y = multiply(w, w)
        else:
            w = multiply(x, x)
            z = add(w, w)
        return z


@pytest.mark.xfail
def test_unassigned_call_fail():
    @rx.script
    def foo(x: Tensor[_, _]):
        add(x, x)
        return x


def test_tuple():
    @rx.script
    def foo(x: Tensor[_, _], y: Tensor[(32,), "float32"]):
        t: Tuple[Tensor[_, _], Tensor[(32,), "float32"]] = (x, y)
        return t

    f = rx_func(foo)
    x, y = f.params
    t_bind = f.body.blocks[0].bindings[0]
    t, tup = t_bind.var, t_bind.value

    assert isinstance(t.type_annotation, relay.TupleType)
    annot = t.type_annotation
    assert isinstance(annot.fields[0], rx.ty.DynTensorType) and annot.fields[0].dtype == ""
    assert isinstance(annot.fields[1], rx.ty.DynTensorType) and annot.fields[1].dtype == "float32"

    assert t.shape_ is None

    assert isinstance(tup, rx.Tuple)
    assert structural_equal(tup.fields, [x, y])
    assert tup.shape_ is None
    check_shape(tup.fields[0], None)
    check_shape(tup.fields[1], (32,))


def test_local_func():
    @rx.script
    def foo(x: Tensor[_, _]):
        def bar(y: Tensor[_, _]):
            return y

        y = bar(x)  # tests local function variable scoping
        return y

    f = rx_func(foo)
    bar_bind, y_bind = f.body.blocks[0].bindings
    bar, bar_fn = bar_bind.var, bar_bind.value
    bar_x = y_bind.value

    assert isinstance(bar_fn, rx.Function)
    assert bar_fn.body.body == bar_fn.params[0]

    assert bar_x.op == bar


def test_dataflow():
    @rx.script
    def foo(x: Tensor[_, _]):
        with relax.dataflow():
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(y, w)
        t = divide(y, w)
        return t

    f = rx_func(foo)
    assert len(f.body.blocks) == 2
    df_block = f.body.blocks[0]
    y_bind, z_bind, w_bind = df_block.bindings
    (t_bind,) = f.body.blocks[1].bindings
    x = f.params[0]
    y, z, w, t = map(lambda b: b.var, [y_bind, z_bind, w_bind, t_bind])

    assert isinstance(y, rx.Var)
    assert isinstance(z, rx.DataflowVar)
    assert isinstance(w, rx.Var)

    check_call(y_bind.value, "add", [x, x])
    check_call(z_bind.value, "multiply", [y, x])
    check_call(w_bind.value, "subtract", [z, x])
    check_call(t_bind.value, "divide", [y, w])

    assert f.body.body == t


def test_dataflow_match_shape():
    @rx.script
    def foo(x: Tensor[_, _]):
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

    f = rx_func(foo)
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
    @rx.script
    def foo(x: Tensor[_, _]):
        with relax.dataflow():
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(y, w)
        t = divide(y, z)
        return t


@pytest.mark.xfail
def test_dataflow_syntax_fail_pattern():
    @rx.script
    def foo(x: Tensor[_, _]):
        with relax.dataflow() as df:
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(y, z)
        t = divide(y, z)
        return t


@pytest.mark.xfail
def test_dataflow_syntax_fail_params():
    @rx.script
    def foo(x: Tensor[_, _]):
        with relax.dataflow(x) as df:
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(y, w)
        t = divide(y, z)
        return t


@pytest.mark.xfail
def test_dataflow_unbound_outputs():
    @rx.script
    def foo(x: Tensor[_, _]):
        with relax.dataflow():
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(x, y, w, q)
        t = divide(y, z)
        return t


@pytest.mark.xfail
def test_invalid_special_op_dataflow():
    @rx.script
    def foo(x: Tensor):
        y = add(x, x)
        z = relax.dataflow()
        return z


@pytest.mark.xfail
def test_invalid_special_op_output():
    @rx.script
    def foo(x: Tensor):
        y = add(x, x)
        z = relax.output(y)
        return z


@pytest.mark.xfail
def test_func_no_return_fail():
    @rx.script
    def foo(x: Tensor[_, _]):
        y = add(x, x)


def test_inline_tir():
    @rx.script
    def foo(x: Tensor[(B, 128), "float32"], y: Tensor[(128, 128), "float32"]):
        @tvm.script.tir
        def my_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
            A = tir.match_buffer(a, [128, 128])
            B = tir.match_buffer(b, [128, 128])
            C = tir.match_buffer(c, [128, 128])

            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                with tir.init():
                    C[vi, vj] = tir.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        z = relax.call_dps((B, 128), my_matmul, (x, y))
        return z

    f = rx_func(foo)
    x, y = f.params
    B = x.shape_[0]
    mm_bind, z_bind = f.body.blocks[0].bindings

    assert mm_bind.var.name_hint == "my_matmul"
    assert isinstance(mm_bind.value, tir.PrimFunc)

    check_call(
        z_bind.value,
        "relax.call_dps",
        [rx.ShapeExpr([B, tir.IntImm("int32", 128)]), mm_bind.var, rx.Tuple([x, y])],
    )


def test_call_packed():
    @rx.script
    def foo(x: Tensor[(3, 4), "float32"]):
        # test that we can intro dim vars
        z: Tensor[(n, m), "float32"] = relax.call_packed("contrib.my_matmul", (x, x), mp=False)
        return z

    f = rx_func(foo)
    x = f.params[0]
    (z_bind,) = f.body.blocks[0].bindings
    check_tensor_var(z_bind.var, ("n", "m"), "float32")

    assert isinstance(z_bind.value.op, rx.ExternFunc)
    assert z_bind.value.op.global_symbol == "contrib.my_matmul"
    assert "mp" in z_bind.value.attrs and z_bind.value.attrs["mp"] == False
    assert structural_equal(z_bind.value.args, [rx.Tuple([x, x])])


def test_primexpr_arithmetic():
    @rx.script
    def foo(x: Tensor[(n, m), "float32"]):
        z: Tensor[(n * m,), "float32"] = relax.call_packed("my_flatten", (x,))
        sh: Shape = (n + m, n // m)
        return z

    f = rx_func(foo)
    x = f.params[0]
    n, m = x.shape_
    z_bind, sh_bind = f.body.blocks[0].bindings

    assert structural_equal(z_bind.var.shape_.values, [tir.Mul(n, m)])
    assert structural_equal(sh_bind.value.values, [tir.Add(n, m), tir.FloorDiv(n, m)])
