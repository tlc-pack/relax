import tvm
from tvm import relax as rx
from tvm.ir import TensorType
from tvm.relax import Expr, Span, Var, DataflowVar, const, SeqExpr, VarBinding, BasicBlock

def check_span(exp: Expr, sp: Span) -> None:
    assert exp.id.name_hint == "foo"
    assert exp.span.column == 0
    assert exp.span.line == 0

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
    pass

def test_var_binding() -> None:
    pass

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
    # test_df_var()
    # test_seq_expr()

