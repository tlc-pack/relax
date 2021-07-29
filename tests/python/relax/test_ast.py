import tvm
from tvm.relax import Expr, Span, Id, Var, DataflowVar, const, SeqExpr, VarBinding, BasicBlock

def check_span(exp: Expr, sp: Span) -> None:
    assert exp.id.name_hint == "foo"
    assert exp.span.column == 0
    assert exp.span.line == 0


def test_var() -> None:
    id = Id("foo")
    sp = Span("foo", 0, 1, 2, 3)
    # The variant with None, None
    v1 = Var(id, None, None, sp)
    check_span(v1, sp)
    sh = None # todo: make ty
    ty = None # todo: make ty
    v2 = Var(id, sh, ty, sp)
    check_span(v2, sp)
    print(v2)

def test_df_var() -> None:
    id = Id("foo")
    sp = Span("foo", 0, 1, 2, 3)
    # The variant with None, None
    v1 = DataflowVar(id, None, None, sp)
    check_span(v1, sp)
    sh = None # todo: make ty
    ty = None # todo: make ty
    v2 = Var(id, sh, ty, sp)
    check_span(v2, sp)

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
