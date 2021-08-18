from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
from tvm import relax as rx
from tvm import tir


def rx_func(func):
    return func.module[func.fn_name]


def check_shape(e, s):
    if not isinstance(e, (list, tuple)) and e is not None:
        e = e._shape

    if s is None:
        assert e is None
        return

    assert isinstance(e, (list, tuple))
    assert len(e) == len(s)

    for edim, sdim in zip(e, s):
        if isinstance(sdim, str):
            assert isinstance(edim, tir.Var)
            assert edim.name == sdim
        else:
            assert isinstance(edim, tir.IntImm)
            assert edim.value == sdim


def check_tensor_var(v, s, d):
    assert isinstance(v.type_annotation, rx.ty.rxTensor)
    assert v.type_annotation.dtype == d
    check_shape(v, s)


def test_annotations():
    @rx.script
    def foo(x: Tensor[(32, m), "float32"], y: Tensor[(m, k), "float32"]) -> Tensor:
        z: Tensor[(32, k), "float32"] = matmul(x, y)
        w: Tensor[_, _] = mul(z, z)
        t = sub(w, z)
        sh: Shape = t.shape
        return t

    f = rx_func(foo)
    x, y = f.params
    z_bind, w_bind, t_bind, sh_bind = f.body.blocks[0]
    z, mm = z_bind.var, z_bind.value
    w, mul = w_bind.var, w_bind.value
    t, sub = t_bind.var, t_bind.value
    sh, shape_of = sh_bind.var, sh_bind.value

    check_tensor_var(x, (32, "m"), "float32")
    check_tensor_var(y, ("m", "k"), "float32")
    check_tensor_var(z, (32, "k"), "float32")
    check_tensor_var(w, None, None)
    assert t.type_annotation is None
    assert isinstance(sh.type_annotation, rx.ty.rxShape)

    assert mm.op == "matmul"
    assert mm.args == [x, y]

    assert mul.op == "mul"
    assert mul.args == [z, z]

    assert sub.op == "sub"
    assert sub.args == [w, z]

    assert shape_of.op == "rx.shape_of"
    assert shape_of.args == [t]

    assert f.body.body == t

    assert isinstance(f.ret_type, rx.ty.rxTensor)


def test_match_shape():
    @rx.script
    def foo(x: Tensor[_, "float32"]):
        rx.match_shape((n, m), x.shape)
        y: Tensor[(n, m), "float32"] = refine(x)
        return x

    f = rx_func(foo)
    match_sh = f.body.blocks[0][0]
    pattern, value = match_sh.pattern, match_sh.value

    check_shape(pattern, ("n", "m"))
    assert isinstance(value, rx.expr.rxCall)
    assert value.op == "rx.shape_of"
    assert value.args == [f.params[0]]


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
            y = mul(w, w)
        else:
            w = mul(x, x)
            y = add(w, w)
        return y

    f = rx_func(foo)
    cond, x = f.params
    y_bind = f.body.blocks[0][0]
    y, ite = y_bind.var, y_bind.value

    check_tensor_var(cond, tuple(), "bool")
    check_tensor_var(x, (1,), "float32")

    assert isinstance(y, rx.expr.rxVar)
    assert y.id == "y"

    assert isinstance(ite, rx.expr.rxIfThenElse)
    assert isinstance(ite.true_branch, rx.expr.rxSeqExpr)
    assert isinstance(ite.false_branch, rx.expr.rxSeqExpr)

    w_bind = ite.true_branch.blocks[0][0]
    body = ite.true_branch.body
    assert w_bind.var.id == "w"
    assert isinstance(w_bind.value, rx.expr.rxCall)
    assert w_bind.value.op == "add" and w_bind.value.args == [x, x]
    assert isinstance(body, rx.expr.rxCall)
    assert body.op == "mul" and body.args == [w_bind.var, w_bind.var]

    w_bind = ite.false_branch.blocks[0][0]
    body = ite.false_branch.body
    assert w_bind.var.id == "w"
    assert isinstance(w_bind.value, rx.expr.rxCall)
    assert w_bind.value.op == "mul" and w_bind.value.args == [x, x]
    assert isinstance(body, rx.expr.rxCall)
    assert body.op == "add" and body.args == [w_bind.var, w_bind.var]


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
            y = mul(w, w)
        else:
            w = mul(x, x)
            y = add(w, w)
        return y


@pytest.mark.xfail
def test_var_if_scoping_fail():
    @rx.script
    def foo(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
        if cond:
            w = add(x, x)
            y = mul(w, w)
        else:
            w = mul(x, x)
            y = add(w, w)
        return w


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
    t_bind = f.body.blocks[0][0]
    t, tup = t_bind.var, t_bind.value

    assert isinstance(t.type_annotation, rx.ty.rxTupleType)
    annot = t.type_annotation
    assert isinstance(annot.fields[0], rx.ty.rxTensor) and annot.fields[0].dtype is None
    assert isinstance(annot.fields[1], rx.ty.rxTensor) and annot.fields[1].dtype == "float32"

    assert isinstance(t._shape, list) and len(t._shape) == 2
    check_shape(t._shape[0], None)
    check_shape(t._shape[1], (32,))

    assert isinstance(tup, rx.expr.rxTuple)
    assert tup.fields == [x, y]
    assert isinstance(tup._shape, list) and len(tup._shape) == 2
    check_shape(tup._shape[0], None)
    check_shape(tup._shape[1], (32,))
