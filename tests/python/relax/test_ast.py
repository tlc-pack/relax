import tvm
from tvm import tir
from tvm import relax as rx
from tvm.ir import TensorType
from tvm.relax import Expr, Span, Var, DataflowVar, const, SeqExpr, VarBinding, BasicBlock
import numpy as np


def test_var() -> None:
    v0 = Var("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0.type_annotation is None
    shape_anno = [54, 96]
    type_anno = TensorType(shape_anno, "float32")
    v1 = Var("v1", shape_anno, type_anno)
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape_anno):
        assert s0 == s1
    assert v1.type_annotation == type_anno


def test_df_var() -> None:
    v0 = DataflowVar("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0.type_annotation is None
    shape_anno = [54, 96]
    type_anno = TensorType(shape_anno, "float16")
    v1 = DataflowVar("v1", shape_anno, type_anno)
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape_anno):
        assert s0 == s1
    assert v1.type_annotation == type_anno
    assert isinstance(v1, DataflowVar)


def test_match_shape() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = const([16, 8], "int32") 
    b0 = rx.MatchShape([m, n], shape)
    assert b0.pattern[0] == m
    assert b0.pattern[1] == n
    assert b0.value == shape

    # should we support this?
    # b0 = rx.MatchShape([m, n], [x, y]) 


def test_var_binding() -> None:
    v0 = Var("v0")
    val = const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    assert b0.var.name_hint == "v0"
    assert b0.value == val


def test_basic_block() -> None:
    pass

def test_df_block() -> None:
    pass

def test_seq_expr() -> None:
    id = Id("foo")
    x = Var(id, None, None)
    bindings = [VarBinding(x, const(1))]
    blocks = [BasicBlock(bindings)]
    seqe = SeqExpr(blocks, x)
    assert seqe.blocks == blocks
    assert seqe.body == x

def test_func():
    pass

# TODO(@relax-team): we should port tests from relay/test_ast.py for nodes we will share.

if __name__ == "__main__":
    test_var()
    test_df_var()
    test_match_shape()
    test_var_binding()

