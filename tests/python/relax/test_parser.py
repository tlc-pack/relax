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
from tvm import tir, relay, relax
from tvm.ir import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R

# TODO: replace xfails with proper diagnostics checking.
#       c.f. tests/python/unittest/test_tvmscript_error_report.py


def check_shape(e, s):
    if isinstance(e, relax.ShapeExpr):
        pass
    elif isinstance(e, relax.Call):
        e = e.shape
    elif isinstance(e, relax.Expr):
        e = e.shape_

    if s is None:
        assert e is None
        return

    if isinstance(s, relax.RuntimeDepShape):
        assert isinstance(e, relax.RuntimeDepShape)
        return

    assert len(e) == len(s)

    for edim, sdim in zip(e, s):
        if isinstance(sdim, str):
            assert isinstance(edim, tir.Var)
            assert edim.name == sdim
        else:
            assert isinstance(edim, tir.IntImm)
            assert edim.value == sdim


def check_tensor_var(v, s, d, ndim=None):
    assert isinstance(v._checked_type_, relax.ty.DynTensorType)
    assert v._checked_type_.dtype == d
    if isinstance(s, (list, tuple)):
        assert v._checked_type_.ndim == len(s)
    if ndim is not None:
        assert v._checked_type_.ndim == ndim
    check_shape(v, s)


def check_call(call, op, args):
    assert isinstance(call, relax.Call)
    if isinstance(op, str):
        op = relay.op.get(op)
    assert call.op == op
    assert_structural_equal(call.args, args)


def test_annotations():
    @R.function
    def f(
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

    x, y, r = f.params
    z_bind, w_bind, q_bind, t_bind, sh_bind, o_bind = f.body.blocks[0].bindings
    z, mm = z_bind.var, z_bind.value
    w, mul = w_bind.var, w_bind.value
    q, add = q_bind.var, w_bind.value
    t, sub = t_bind.var, t_bind.value
    sh, shape_of = sh_bind.var, sh_bind.value
    o, o_call_packed = o_bind.var, o_bind.value

    check_tensor_var(x, (32, "m"), "float32")
    check_tensor_var(y, ("m",), "float32")
    check_tensor_var(r, relax.RuntimeDepShape(), "int64")
    check_tensor_var(z, (32, "m"), "float32")
    check_tensor_var(w, relax.RuntimeDepShape(), "")
    check_tensor_var(q, relax.RuntimeDepShape(), "", ndim=2)
    assert isinstance(t._checked_type_, relax.ty.DynTensorType)
    assert isinstance(sh._checked_type_, relax.ty.ShapeType)

    check_call(mm, "relax.multiply", [x, y])
    check_call(mul, "relax.multiply", [z, z])
    check_call(sub, "relax.add", [w, z])
    check_call(shape_of, "relax.shape_of", [t])

    assert f.body.body == o

    assert isinstance(f.ret_struct_info, relax.ObjectStructInfo)

    assert isinstance(o._checked_type_, relax.ty.ObjectType)
    assert len(o_call_packed.type_args) == 1


def test_mismatch_shape_dims_and_ndim():
    with pytest.raises(Exception):
        # TODO: replace with DiagnosticError once we have better error reporting.
        # with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor((2, 3), "float32", ndim=3)):
            return x


def test_unexpected_num_kw_args():
    with pytest.raises(Exception):
        # TODO: replace with DiagnosticError once we have better error reporting.
        # with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(dtype="float32", ndim=1, foo=2)):
            return x


def test_unexpected_kw_arg():
    with pytest.raises(Exception):
        # TODO: replace with DiagnosticError once we have better error reporting.
        # with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(dtype="float32", foo=1)):
            return x


def test_unexpected_ndim():
    with pytest.raises(Exception):
        # TODO: replace with DiagnosticError once we have better error reporting.
        # with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(dtype="float32", ndim=-2)):
            return x


def test_unexpected_ndim_type():
    with pytest.raises(Exception):
        # TODO: replace with DiagnosticError once we have better error reporting.
        # with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(dtype="float32", ndim="1")):
            return x


def test_unexpected_tir_cast_args():
    # tir.cast expects 2 arguments, but got 3
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(("m",), "float32")):
            m = T.var("int64")
            return R.call_tir("foo", (x,), (T.cast("int32", m, 1),), dtype="float32")


def test_unexpected_tir_max_args():
    # tir.max expects 2 arguments, but got 1
    with pytest.raises(Exception):

        @R.function
        def f(x: R.Tensor(("m", "n"), "float32")):
            m = T.var("int64")
            return relax.call_tir("foo", (x,), (T.max(m),), dtype="float32")


def test_match_shape():
    @R.function
    def f(x: R.Tensor(dtype="float32")):
        n, m = T.var("int64"), T.var("int64")
        R.match_shape(R.shape_of(x), (n, m))
        y: R.Tensor((n, m), "float32") = R.add(x, x)
        return x

    match_sh = f.body.blocks[0].bindings[0]
    pattern, value = match_sh.pattern, match_sh.value

    check_shape(pattern, ("n", "m"))
    check_call(value, "relax.shape_of", [f.params[0]])


def test_if():
    @R.function
    def f(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")) -> R.Tensor:
        if cond:
            w = R.add(x, x)
            y = R.multiply(w, w)
        else:
            w = R.multiply(x, x)
            y = R.add(w, w)
        return y

    cond, x = f.params
    y_bind = f.body.blocks[0].bindings[0]
    y, ite = y_bind.var, y_bind.value

    check_tensor_var(cond, tuple(), "bool")
    check_tensor_var(x, (1,), "float32")

    assert isinstance(y, relax.Var)
    assert y.name_hint == "y"

    assert isinstance(ite, relax.If)
    assert ite.checked_type == relax.DynTensorType(1, "float32")
    check_shape(ite.shape, (1,))
    assert y.checked_type == relax.DynTensorType(1, "float32")
    check_shape(y.shape, (1,))

    assert isinstance(ite.true_branch, relax.SeqExpr)
    assert isinstance(ite.false_branch, relax.SeqExpr)

    w_bind = ite.true_branch.blocks[0].bindings[0]
    body = ite.true_branch.body
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "relax.add", [x, x])
    body_bind = ite.true_branch.blocks[0].bindings[1]
    check_call(body_bind.value, "relax.multiply", [w_bind.var, w_bind.var])
    assert ite.true_branch.body == body_bind.var

    w_bind = ite.false_branch.blocks[0].bindings[0]
    body = ite.false_branch.body
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "relax.multiply", [x, x])
    body_bind = ite.false_branch.blocks[0].bindings[1]
    check_call(body_bind.value, "relax.add", [w_bind.var, w_bind.var])
    assert ite.false_branch.body == body_bind.var


def test_func_type_annotation_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x, y):
            z = R.add(x, y)
            y = z
            return y


def test_var_if_scoping_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                y = R.add(w, w)
            return w


def test_if_mismatch_var_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                z = R.add(w, w)
            return z


def test_unassigned_call_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor):
            R.add(x, x)
            return x


def test_tuple():
    @R.function
    def f(x: R.Tensor, y: R.Tensor((32,), "float32")):
        t: R.Tuple(R.Tensor(), R.Tensor((32,), "float32")) = (x, y)
        return t

    x, y = f.params
    t_bind = f.body.blocks[0].bindings[0]
    t, tup = t_bind.var, t_bind.value

    annot = t._checked_type_
    assert isinstance(annot, relay.TupleType)
    assert isinstance(annot.fields[0], relax.ty.DynTensorType) and annot.fields[0].dtype == ""
    assert (
        isinstance(annot.fields[1], relax.ty.DynTensorType) and annot.fields[1].dtype == "float32"
    )

    assert isinstance(t.shape_, relax.Tuple)
    assert isinstance(tup, relax.Tuple)
    assert_structural_equal(tup.fields, [x, y])

    assert isinstance(tup.shape_, relax.Tuple)
    check_shape(tup.fields[0], relax.RuntimeDepShape())
    check_shape(tup.fields[1], (32,))


def test_tuplegetitem():
    @R.function
    def f(x: R.Tensor, y: R.Tensor):
        t1 = R.Tuple((x, y))
        t2 = (x, y)
        a = t1[0]
        b = R.TupleGetItem(t2, 1)
        c = R.add(a, b)
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
    check_call(bind_4.value, "relax.add", [bind_2.var, bind_3.var])


def test_local_func():
    @R.function
    def f(x: R.Tensor):
        @R.function
        def bar(y: R.Tensor):
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
    def f(x: R.Tensor):
        with R.dataflow():
            y = R.add(x, x)
            z = R.multiply(y, x)
            w = R.multiply(z, x)
            R.output(y, w)
        t = R.add(y, w)
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

    check_call(y_bind.value, "relax.add", [x, x])
    check_call(z_bind.value, "relax.multiply", [y, x])
    check_call(w_bind.value, "relax.multiply", [z, x])
    check_call(t_bind.value, "relax.add", [y, w])

    assert f.body.body == t


def test_dataflow_match_shape():
    @R.function
    def f(x: R.Tensor):
        n, m = T.var("int64"), T.var("int64")
        with R.dataflow():
            x2: R.Tensor((n, m)) = R.match_shape(x, (n, m))
            y = R.add(x2, x2)
            z = R.multiply(y, x)
            R.match_shape(R.shape_of(z), (n, m))
            w: R.Tensor((n, m)) = R.add(z, x)
            R.output(y, w, x2)
        t: R.Tensor((n, m)) = R.multiply(y, w)
        q: R.Tensor((n, m)) = R.add(t, x2)
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


def test_dataflow_scope_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(ndim=2)):
            with R.dataflow():
                y = R.add(x, x)
                z = R.multiply(y, x)
                w = R.add(z, x)
                R.output(y, w)
            t = R.multiply(y, z)
            return t


def test_dataflow_unbound_outputs():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(ndim=2)):
            with R.dataflow():
                y = R.add(x, x)
                z = R.multiply(y, x)
                w = R.add(z, x)
                R.output(x, y, w, q)
            t = R.multiply(y, z)
            return t


def test_invalid_special_op_dataflow():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor):
            y = add(x, x)
            z = relax.dataflow()
            return z


def test_invalid_special_op_output():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor):
            y = add(x, x)
            z = relax.output(y)
            return z


def test_func_no_return_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor):
            y = R.add(x, x)


def test_call_tir():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")):
        m, n = T.var("int64"), T.var("int64")
        gv0 = relax.call_tir("test.op.identity", (x,), (m, n), dtype="float32")
        return gv0

    call_tir_node = foo.body.blocks[0].bindings[0].value
    assert call_tir_node.attrs is None
    assert_structural_equal(
        call_tir_node.type_args[0], relax.DynTensorType(ndim=2, dtype="float32")
    )


def test_inline_tir():
    @R.function
    def f(x: R.Tensor(("B", 128), "float32"), y: R.Tensor((128, 128), "float32")):
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

        B = T.var("int64")
        z = R.call_tir(my_matmul, (x, y), (B, 128), dtype="float32")
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
    def f(x: R.Tensor((3, 3), "float32")):
        n, m = T.var("int64"), T.var("int64")
        z: R.Tensor((n, m), "float32") = R.call_packed(
            "contrib.my_matmul",
            x,
            x,
            mp=False,
            type_args=(R.Tensor(ndim=2, dtype="float32")),
        )

        w = R.call_packed(
            "contrib.my_shape_of",
            x,
            dtype="int32",
            attrs_type_key="relay.attrs.ShapeOfAttrs",
            type_args=(R.Shape),
        )

        o = R.call_packed("contrib.tensor_array_stack", x, z, type_args=(R.Object))

        k = R.call_packed(
            "contrib.construct_tuple",
            x,
            x,
            type_args=(R.Tuple(R.Tuple(R.Tensor(ndim=2, dtype="float32"), R.Tensor), R.Tensor)),
        )
        return k

    x = f.params[0]
    (z_bind, w_bind, o_bind, k_bind) = f.body.blocks[0].bindings

    z_var, z_value = z_bind.var, z_bind.value
    check_tensor_var(z_var, ("n", "m"), "float32")

    assert isinstance(z_value.op, relax.ExternFunc)
    assert z_value.op.global_symbol == "contrib.my_matmul"
    assert "mp" in z_value.attrs and z_value.attrs["mp"] == False
    assert_structural_equal(z_value.args, [x, x])
    assert len(z_value.type_args) == 1
    assert_structural_equal(z_value.type_args[0], relax.ty.DynTensorType(2, "float32"))

    w_value = w_bind.value
    assert isinstance(w_value.attrs, relay.op.op_attrs.ShapeOfAttrs)
    assert_structural_equal(w_value.type_args[0], relax.ty.ShapeType())

    o_value = o_bind.value
    assert_structural_equal(o_value.type_args[0], relax.ty.ObjectType())

    k_value = k_bind.value
    assert_structural_equal(
        k_value.type_args[0],
        relax.ty.TupleType(
            [
                relax.TupleType(
                    [relax.ty.DynTensorType(2, "float32"), relax.ty.DynTensorType(-1, None)]
                ),
                relax.ty.DynTensorType(-1, None),
            ]
        ),
    )


def test_call_packed_no_type_args_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor((3, 3), "float32")):
            n, m = T.var("int64"), T.var("int64")
            z: R.Tensor((n, m), "float32") = relax.call_packed("contrib.my_matmul", x, x)
            return z


def test_call_packed_wrong_type_args_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor((3, 3), "float32")):
            z: R.Tensor((n, m), "float32") = relax.call_packed(
                "contrib.my_matmul", x, x, type_args=(Tuple)
            )
            return z


def test_constant():
    @R.function
    def f(x: R.Tensor((2, 3), "float32")):
        y1 = R.const(2, dtype="float32")
        y2 = R.const([[3.1, 4.0, 5.0], [6.0, 7.1, 9.0]])
        z = R.add(x, y1)
        r = R.add(z, y2)
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
    check_call(bind_2.value, "relax.add", [x, bind_0.var])
    check_call(bind_3.value, "relax.add", [bind_2.var, bind_1.var])


def test_primexpr_arithmetic():
    @R.function
    def f(x: R.Tensor(("n", "m"), "float32")):
        n, m = T.var("int64"), T.var("int64")
        z: R.Tensor((n * m,), "float32") = R.call_packed(
            "my_flatten", (x,), type_args=(R.Tensor(ndim=1, dtype="float32"))
        )
        sh: R.Shape = (n + m, n // m)
        return z

    x = f.params[0]
    n, m = x.shape_
    z_bind, sh_bind = f.body.blocks[0].bindings

    assert_structural_equal(z_bind.var.shape_.values, [tir.Mul(n, m)])
    assert_structural_equal(sh_bind.value.values, [tir.Add(n, m), tir.FloorDiv(n, m)])


def test_call_tir_extern():
    @R.function
    def f(x: R.Tensor) -> R.Tensor:
        z = R.call_tir("my_extern", (x,), (10,), dtype="float32")
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


def test_empty_shape():
    @R.function
    def f(x: R.Tensor((), "float32"), y: R.Tensor((), "float32")):
        @T.prim_func
        def scalar_add(a: T.handle, b: T.handle, c: T.handle) -> None:
            A = T.match_buffer(a, ())
            B = T.match_buffer(b, ())
            C = T.match_buffer(c, ())

            with T.block("add"):
                C[()] = A[()] + B[()]

        z = relax.call_tir(scalar_add, (x, y), (), dtype="float32")
        return z

    x, y = f.params
    add_bind, z_bind = f.body.blocks[0].bindings

    assert add_bind.var.name_hint == "scalar_add"
    assert isinstance(add_bind.value, tir.PrimFunc)

    check_call(
        z_bind.value,
        "relax.call_tir",
        [add_bind.var, relax.Tuple([x, y]), relax.ShapeExpr([])],
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
        def f(x: R.Tensor(("n", "n"))) -> R.Tensor:
            return g(x)

        @R.function
        def g(y: R.Tensor(("n", "n"))) -> R.Tensor:
            n = T.var("int64")
            return R.call_tir(my_matmul, (y, y), (n, n), dtype="float32")

        @R.function
        def j(y: R.Tensor(("n", "n"))) -> R.Tensor:
            n = T.var("int64")
            with R.dataflow():
                gv = R.call_tir(my_matmul, (y, y), (n, n), dtype="float32")
                gv1 = (gv, gv)
                gv2 = gv1[1]
                R.output(gv2)
            return gv2

        @R.function
        def k(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
            gv0 = R.call_packed("test.vm.mul", x, w, type_args=(R.Tensor(ndim=2, dtype="float32")))
            return gv0

    my_module = MyModule
    assert isinstance(my_module, tvm.IRModule)

    my_module.script()
    # check that we can print TIR and Relax functions too using the same api.
    my_module["my_matmul"].script()
    my_module["f"].script()

    var_f = my_module.get_global_var("f")
    var_g = my_module.get_global_var("g")
    var_j = my_module.get_global_var("j")
    var_k = my_module.get_global_var("k")
    var_my_matmul = my_module.get_global_var("my_matmul")
    func_f = my_module[var_f]
    func_g = my_module[var_g]
    func_j = my_module[var_j]
    func_k = my_module[var_k]

    assert len(func_f.body.blocks) == 1
    assert len(func_f.body.blocks[0].bindings) == 1
    f_call_var = func_f.body.blocks[0].bindings[0].var
    assert func_f.body.blocks[0].bindings[0].value.op == var_g
    assert func_f.body.body == f_call_var

    g_call_var = func_g.body.blocks[0].bindings[-1].var
    assert func_g.body.blocks[0].bindings[-1].value.args[0] == var_my_matmul
    assert func_g.body.body == g_call_var

    gv_bind = func_j.body.blocks[0].bindings[0]
    assert gv_bind.value.checked_type.ndim == 2
    assert gv_bind.value.checked_type.dtype == "float32"
    assert gv_bind.var.checked_type.ndim == 2
    assert gv_bind.var.checked_type.dtype == "float32"
    check_shape(gv_bind.value, ("n", "n"))
    check_shape(gv_bind.var, ("n", "n"))

    # check call_packed checked_type_
    gv0_bind = func_k.body.blocks[0].bindings[0]
    assert gv0_bind.value.checked_type.dtype == "float32"
    assert gv0_bind.value.checked_type.ndim == 2
    assert gv0_bind.var.checked_type.dtype == "float32"
    assert gv0_bind.var.checked_type.ndim == 2

    # check function type
    j_type = func_j.checked_type
    assert isinstance(j_type, relax.FuncType)
    assert isinstance(j_type.ret_type, relax.DynTensorType)
    assert j_type.ret_type.ndim == 2
    assert j_type.ret_type.dtype == "float32"
    assert len(j_type.arg_types) == 1
    assert isinstance(j_type.arg_types[0], relax.DynTensorType)
    assert j_type.arg_types[0].ndim == 2

    # check SeqExpr type/shape
    assert isinstance(func_j.body, relax.SeqExpr)
    assert func_j.body.checked_type.dtype == "float32"
    assert func_j.body.checked_type.ndim == 2
    check_shape(func_j.body, ("n", "n"))

    # check tuple type/shape
    gv1_bind = func_j.body.blocks[0].bindings[1]
    isinstance(gv1_bind.value, relax.Tuple)
    isinstance(gv1_bind.value.checked_type, relax.TupleType)
    isinstance(gv1_bind.var.checked_type, relax.TupleType)
    assert gv1_bind.var.checked_type.fields[0].ndim == 2
    assert gv1_bind.var.checked_type.fields[0].dtype == "float32"
    isinstance(gv1_bind.var.shape, relax.Tuple)
    isinstance(gv1_bind.value.shape, relax.Tuple)
    check_shape(gv1_bind.value.shape.fields[0], ("n", "n"))
    check_shape(gv1_bind.value.shape.fields[1], ("n", "n"))
    check_shape(gv1_bind.var.shape.fields[0], ("n", "n"))
    check_shape(gv1_bind.var.shape.fields[1], ("n", "n"))

    # check TupleGetItem type/shape
    gv2_bind = func_j.body.blocks[0].bindings[2]
    isinstance(gv2_bind.value, relax.TupleGetItem)
    assert gv2_bind.value.checked_type.ndim == 2
    assert gv2_bind.value.checked_type.dtype == "float32"
    assert gv2_bind.var.checked_type.ndim == 2
    assert gv2_bind.var.checked_type.dtype == "float32"
    check_shape(gv2_bind.value.shape, ("n", "n"))
    check_shape(gv2_bind.var, ("n", "n"))


def test_function_attrs():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int64")
            n = T.var("int64")
            k = T.var("int64")
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
        def main(
            x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")
        ) -> R.Tensor:
            R.func_attr({"global_symbol": "main"})
            m, n, k = T.var("int64"), T.var("int64"), T.var("int64")
            gv0 = R.call_tir("tir_matmul", (x, w), (m, k), dtype="float32")
            return gv0

    assert InputModule["main"].attrs["global_symbol"] == "main"
    assert InputModule["tir_matmul"].attrs["global_symbol"] == "tir_matmul"


def test_class_normalize():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def mul_add(x: R.Tensor) -> R.Tensor:
            return R.multiply(R.add(x, x), R.add(x, x))

    # The parser automatically normalizes the input AST to the following ANF form
    @tvm.script.ir_module
    class OutputModule:
        @R.function
        def mul_add(x: R.Tensor) -> R.Tensor:
            gv = R.add(x, x)
            gv1 = R.add(x, x)
            return R.multiply(gv, gv1)

    assert_structural_equal(InputModule, OutputModule)


def test_memory_op():
    @R.function
    def memory(x: R.Tensor) -> R.Tensor:
        storage = R.memory.alloc_storage((1024,), -1, "global", "float32")
        alloca = R.memory.alloc_tensor(storage, (1, 256), 0, "float32")
        _ = R.memory.kill_tensor(alloca)
        _ = R.memory.kill_storage(storage)
        return alloca

    b0, b1, b2, b3 = memory.body.blocks[0].bindings
    assert b0.value.op.name == "relax.memory.alloc_storage"
    assert isinstance(b0.value.args[0], relax.ShapeExpr)
    check_shape(b0.value.args[0], (1024,))
    assert isinstance(b0.value.attrs, relax.op.MemAllocStorageAttrs)

    assert b1.value.op.name == "relax.memory.alloc_tensor"
    assert isinstance(b1.value.attrs, relax.op.MemAllocTensorAttrs)

    assert b2.value.op.name == "relax.memory.kill_tensor"
    assert b3.value.op.name == "relax.memory.kill_storage"


if __name__ == "__main__":
    pytest.main([__file__])
