import tvm
from tvm import tir
from tvm import relax as rx
from tvm.ir import TensorType
import numpy as np


def test_var() -> None:
    v0 = rx.Var("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0.type_annotation is None
    shape_anno = [54, 96]
    type_anno = TensorType(shape_anno, "float32")
    v1 = rx.Var("v1", shape_anno, type_anno)
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape_anno):
        assert s0 == s1
    assert v1.type_annotation == type_anno


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0.type_annotation is None
    shape_anno = [54, 96]
    type_anno = TensorType(shape_anno, "float16")
    v1 = rx.DataflowVar("v1", shape_anno, type_anno)
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape_anno):
        assert s0 == s1
    assert v1.type_annotation == type_anno
    assert isinstance(v1, rx.DataflowVar)


def test_match_shape() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32") 
    b0 = rx.MatchShape([m, n], shape)
    assert b0.pattern[0] == m
    assert b0.pattern[1] == n
    assert b0.value == shape


def test_var_binding() -> None:
    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    assert b0.var.name_hint == "v0"
    assert b0.value == val


def test_binding_block() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32") 
    b0 = rx.MatchShape([m, n], shape)

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.BindingBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1


def test_dataflow_block() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32") 
    b0 = rx.MatchShape([m, n], shape)

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.DataflowBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1
    assert isinstance(block0, rx.DataflowBlock)


def test_seq_expr() -> None:
    x = rx.Var("foo")
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    assert seqe.blocks[0] == blocks[0]
    assert seqe.body == x


def test_shape_expr() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    s = rx.ShapeExpr([m, n])
    assert s.values[0] == m
    assert s.values[1] == n


def test_func():
    x = rx.Var("foo")
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    ret_type = TensorType(None, "float32")
    func = rx.Function([x], seqe, ret_type, rx.GlobalVar("func"))
    assert func.params[0] == x
    assert func.body == seqe
    assert func.ret_type == ret_type
    assert func.name.name_hint == "func"


def test_shape_of():
    v0 = rx.Var("v0")
    s0 = v0.shape
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.shape_of"

    shape_anno = [96, 54]
    v1 = rx.Var("v1", shape_anno)
    s1 = v1.shape
    for x, y in zip(shape_anno, s1):
        assert x == y


if __name__ == "__main__":
    test_var()
    test_dataflow_var()
    test_match_shape()
    test_var_binding()
    test_binding_block()
    test_dataflow_block()
    test_seq_expr()
    test_shape_expr()
    test_func()
    test_shape_of()
