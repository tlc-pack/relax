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
import re
from functools import partial
from typing import Dict

import numpy as np
import tvm
import tvm.testing
from tvm import relax as rx
from tvm import tir
from tvm.relax.testing import dump_ast
from tvm.relax.testing.ast_printer import ASTPrinter
from tvm.script import relax as R
from tvm.script import tir as T

# Overload dump_ast to test both struct info and type annotations
dump_ast = partial(dump_ast, include_struct_info_annotations=True, include_type_annotations=True)


def strip_whitespace(text: str) -> str:
    """
    Remove all whitespace to avoid reasoning about newlines and indents
    """
    return re.sub(r"\s", "", text)


def normalize(func: rx.Function) -> rx.Function:
    """
    Normalize the expr to fill in the checked_type_ and shape_ fields everywhere
    """
    # using a default mutator to use the BlockBuilder's normalizer,
    # which oddly differs from the Normalize pass
    @rx.expr_functor.mutator
    class DefaultMutator(rx.PyExprMutator):
        pass

    mod = tvm.IRModule()
    mod["main"] = func
    mut = DefaultMutator(mod)
    mod["main"] = mut.visit_expr(func)
    return mod["main"]


def assert_fields(nodename: str, fields: Dict[str, str], target: str) -> None:
    """
    Given a target string, ensure that the string defines the specified node
    and that the given mappings of fields to values are present in the string.
    Strips all whitespace in the target and fields.
    Does not assume any particular ordering for the fields.
    """
    stripped_target = strip_whitespace(target)
    assert stripped_target.startswith(f"{nodename}(")
    for field, value in fields.items():
        assert f"{field}={strip_whitespace(value)}" in stripped_target


# test cases are mostly adapted from text_expr, only testing very basic properties


def test_var() -> None:
    v0 = rx.Var("v0")
    v0_str = dump_ast(v0)
    assert v0_str == 'Var(name_hint="v0")'

    v1 = rx.Var("v1", R.Tensor([54, 96], "float32"))
    v1_no_annos = dump_ast(
        v1, include_struct_info_annotations=False, include_type_annotations=False
    )
    assert v1_no_annos == 'Var(name_hint="v1")'
    v1_annos = dump_ast(v1)
    assert v1_annos != v1_no_annos
    assert "PrimExpr" in v1_annos
    assert "struct_info" in v1_annos
    assert "checked_type_" in v1_annos


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    v0_str = dump_ast(v0)
    assert v0_str == 'DataflowVar(name_hint="v0")'

    v1 = rx.DataflowVar("v1", R.Tensor([54, 96], "float16"))
    v1_no_annos = dump_ast(
        v1, include_struct_info_annotations=False, include_type_annotations=False
    )
    assert v1_no_annos == 'DataflowVar(name_hint="v1")'
    v1_annos = dump_ast(v1)
    assert v1_annos != v1_no_annos
    assert "PrimExpr" in v1_annos
    assert "struct_info" in v1_annos
    assert "checked_type_" in v1_annos


def test_match_shape() -> None:
    # match_shape([16, 8], [m, n])
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    var = rx.Var("v0", R.Shape())
    b0 = rx.MatchShape(shape, [m, n], var)
    b0_str = dump_ast(b0)
    assert b0_str.startswith("MatchShape(")
    assert "Constant" in b0_str
    assert "PrimExpr(value=`m: int64`)" in b0_str
    assert "PrimExpr(value=`n: int64`)" in b0_str
    assert "16" in b0_str
    assert "8" in b0_str
    assert b0_str != dump_ast(b0, include_type_annotations=False)

    # var1: Tensor((m, n), "float32") =
    #   match_shape(var0: R.Tensor("float32"), [m, n])
    value = rx.Var("value", R.Tensor("float32"))
    var = rx.Var("v1", R.Tensor([m, n], "float32"))
    b1 = rx.MatchShape(value, [m, n], var)
    b1_str = dump_ast(b1)
    assert b1_str.startswith("MatchShape(")
    assert "PrimExpr(value=`m: int64`)" in b1_str
    assert "PrimExpr(value=`n: int64`)" in b1_str
    assert b1_str != dump_ast(
        b1, include_type_annotations=False, include_struct_info_annotations=False
    )


def test_match_shape_unbound() -> None:
    @R.function
    def func(x: R.Tensor) -> R.Tensor:
        R.match_shape(x, (1, 1))
        return x

    # no var field on the match shape!
    func_str = strip_whitespace(dump_ast(func))
    assert "MatchShape" in func_str
    assert "value=Var(" in func_str
    assert "pattern=[PrimExpr(" in func_str
    assert "var=" not in func_str


def test_var_binding() -> None:
    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    b0_str = dump_ast(b0, include_type_annotations=False, include_struct_info_annotations=False)
    assert b0_str.startswith("VarBinding(")
    assert 'var=Var(name_hint="v0")' in b0_str
    assert "value=" in b0_str
    assert "Constant(" in b0_str


def test_binding_block() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchShape(shape, [m, n], rx.Var("v0"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.BindingBlock([b0, b1])
    block0_str = dump_ast(block0)
    assert block0_str.startswith("BindingBlock(")
    assert "bindings=" in block0_str
    assert "VarBinding(" in block0_str
    assert "MatchShape(" in block0_str
    assert '"v0"' in block0_str


def test_dataflow_block() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchShape(shape, [m, n], rx.Var("v0"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.DataflowBlock([b0, b1])
    block0_str = dump_ast(block0)
    assert block0_str.startswith("DataflowBlock(")
    assert "bindings=" in block0_str
    assert "VarBinding(" in block0_str
    assert "MatchShape(" in block0_str
    assert '"v0"' in block0_str


def test_seq_expr() -> None:
    x = rx.Var("foo")
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    seqe_str = dump_ast(seqe)
    assert seqe_str.startswith("SeqExpr(")
    assert "blocks=" in seqe_str
    assert "BindingBlock(" in seqe_str
    assert "VarBinding(" in seqe_str
    assert "Constant(" in seqe_str
    assert 'var=Var(name_hint="foo")' in seqe_str
    assert "value=Constant(data" in strip_whitespace(seqe_str)
    assert "body=" in seqe_str


def test_shape_expr() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    s = rx.ShapeExpr([m, n])
    s_str = dump_ast(s)
    assert s_str.startswith("ShapeExpr(")
    assert "values=" in s_str
    assert "PrimExpr(value=`m: int32`)" in s_str
    assert "PrimExpr(value=`n: int32`)" in s_str


def test_func():
    x = rx.Var("foo", R.Tensor("float32", ndim=2))
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    func = rx.Function([x], seqe, R.Tensor("float32"))
    func = func.with_attr("global_symbol", "func")

    func_str = dump_ast(func)
    assert func_str.startswith("Function(")
    assert "params=" in func_str
    assert "body=" in func_str
    assert "ret_struct_info=" in func_str
    assert "attrs=" in func_str
    assert '"global_symbol": "func"' in func_str
    assert "SeqExpr(" in func_str
    assert "blocks=" in func_str
    assert "VarBinding(" in func_str
    assert func_str != dump_ast(func, include_type_annotations=False)


def test_shape_of():
    v0 = rx.Var("v0")
    s0 = v0.shape
    s0_str = dump_ast(s0)
    assert s0_str.startswith("Call(")
    assert 'op=Op(name="relax.shape_of")' in s0_str
    assert "args=" in s0_str
    assert 'Var(name_hint="v0")' in s0_str

    v1 = rx.Var("v1", R.Tensor([96, 54]))
    s1 = v1.shape
    s1_str = dump_ast(s1)
    assert s1_str.startswith("ShapeExpr("), s1_str
    assert "values=" in s1_str
    assert "PrimExpr(value=`96i64`)" in s1_str
    assert "PrimExpr(value=`54i64`)" in s1_str


def test_shape_expr():
    shape_expr = rx.ShapeExpr([10, 20])
    shape_expr_str = dump_ast(shape_expr)
    assert shape_expr_str.startswith("ShapeExpr(")
    assert "values" in shape_expr_str
    assert "PrimExpr(value=`10i64`)" in shape_expr_str
    assert "PrimExpr(value=`20i64`)" in shape_expr_str


def test_types():
    printer = ASTPrinter()
    shape_type = rx.ShapeType()
    assert strip_whitespace(printer.visit_type_(shape_type)) == "ShapeType()"
    object_type = rx.ObjectType()
    assert strip_whitespace(printer.visit_type_(object_type)) == "ObjectType()"
    packed_type = rx.PackedFuncType()
    assert strip_whitespace(printer.visit_type_(packed_type)) == "PackedFuncType()"
    tensor_type = rx.DynTensorType(ndim=2, dtype="int32")
    assert strip_whitespace(printer.visit_type_(tensor_type)) == "DynTensorType(ndim=2,dtype=int32)"
    unit_type = rx.TupleType([])
    assert strip_whitespace(printer.visit_type_(unit_type)) == "TupleType(fields=[])"
    tuple_type = rx.TupleType([shape_type, object_type])
    assert_fields(
        "TupleType", {"fields": "[ShapeType(), ObjectType()]"}, printer.visit_type_(tuple_type)
    )

    func_type = rx.FuncType([tensor_type], unit_type)
    assert_fields(
        "FuncType",
        {"arg_types": "[DynTensorType(ndim=2, dtype=int32)]", "ret_type": "TupleType(fields=[])"},
        printer.visit_type_(func_type),
    )


def test_call_packed():
    # test case from test_parser
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
        o: R.Object = R.call_packed(
            "contrib.tensor_array_stack", x, y, type_args=R.Object, test_attr=True
        )
        return o

    # checking that the call_packed call is turned into a call to an extern func
    f_str = strip_whitespace(
        dump_ast(
            f,
            include_type_annotations=False,
            include_struct_info_annotations=False,
            include_call_attrs=True,
        )
    )

    # the function has an annotated return type
    assert "ret_struct_info=ObjectStructInfo()" in f_str

    assert isinstance(f.body, rx.SeqExpr)
    extern_call = f.body.blocks[0].bindings[-1].value
    extern_call_text = dump_ast(
        extern_call,
        include_type_annotations=False,
        include_struct_info_annotations=False,
        include_call_attrs=True,
    )
    assert strip_whitespace(extern_call_text) in f_str
    assert_fields(
        "Call",
        {
            "op": 'ExternFunc(global_symbol="contrib.tensor_array_stack")',
            "args": '[Var(name_hint="x"), Var(name_hint="y")]',
            "type_args": "[ObjectType()]",
            "attrs": '{"test_attr": 1}',
        },
        extern_call_text,
    )

    # check that the op call is there too
    op_call = f.body.blocks[0].bindings[0].value
    op_call_text = dump_ast(
        op_call,
        include_type_annotations=False,
        include_struct_info_annotations=False,
        include_call_attrs=True,
    )
    assert strip_whitespace(op_call_text) in f_str
    assert_fields(
        "Call",
        {
            "op": 'Op(name="relax.multiply")',
            "args": '[Var(name_hint="x"), Var(name_hint="y")]',
        },
        op_call_text,
    )

    # TODO: add testcase for op attrs


def test_call_tir():
    # also from test_parser
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")):
        m, n = T.var("int64"), T.var("int64")
        gv0 = R.call_tir("test.op.identity", (x,), (m, n), dtype="float32")
        return gv0

    foo_str = strip_whitespace(
        dump_ast(
            foo,
            include_type_annotations=False,
            include_struct_info_annotations=False,
            include_call_attrs=False,
        )
    )
    assert foo_str.startswith('Function(params=[Var(name_hint="x")]')

    # call_tir is an op in Relax and it takes an extern func as an argument
    assert isinstance(foo.body, rx.SeqExpr)
    tir_call = foo.body.blocks[0].bindings[0].value
    tir_call_text = dump_ast(
        tir_call,
        include_type_annotations=False,
        include_struct_info_annotations=False,
        include_call_attrs=False,
    )
    assert_fields(
        "Call",
        {
            "op": 'Op(name="relax.call_tir")',
            "args": """[
                ExternFunc(global_symbol="test.op.identity"),
                Tuple(fields=[
                    Var(name_hint="x")]),
                    ShapeExpr(values=[PrimExpr(value=`m: int64`),
                    PrimExpr(value=`n: int64`)
                ])
            ]""",
            "type_args": "[DynTensorType(ndim=2, dtype=float32)]",
        },
        tir_call_text,
    )
    assert strip_whitespace(tir_call_text) in foo_str


def test_operators():
    # the operator attributes need to be registered to work in the printer

    @R.function
    def foo(x: R.Tensor):
        return R.unique(x, sorted=True)

    foo_str = strip_whitespace(
        dump_ast(
            foo,
            include_type_annotations=False,
            include_struct_info_annotations=False,
        )
    )
    # checking that the attributes are present
    assert '"sorted":1' in foo_str
    assert '"return_inverse"' in foo_str
    assert '"return_counts"' in foo_str
    assert '"dim"' in foo_str

    @R.function
    def bar(x: R.Tensor):
        return R.print(x, format="{}")

    bar_str = strip_whitespace(
        dump_ast(
            bar,
            include_type_annotations=False,
            include_struct_info_annotations=False,
        )
    )
    print_attrs_str = strip_whitespace('{"format": "{}"}')
    assert print_attrs_str in bar_str


def test_print_shape_annotation_non_var():
    @R.function
    def f() -> R.Tensor:
        return R.const([1, 2])

    body = normalize(f).body
    body_str = strip_whitespace(dump_ast(body))
    # the constant has a shape of (2,)
    struct_info = strip_whitespace(
        """
        struct_info=TensorStructInfo(
            dtype=int32,
            shape=ShapeExpr(
                values=[PrimExpr(value=`2i64`)],
                struct_info=ShapeStructInfo(
                    ndim=1,
                    values=[PrimExpr(value=`2i64`)]
                ),
                checked_type_=ShapeType()
            )
        )
        """
    )
    assert struct_info in body_str


def test_print_type_annotation_non_var():
    @R.function
    def f() -> R.Shape:
        return R.shape_of(R.const(1))

    body = normalize(f).body
    assert isinstance(body, rx.SeqExpr)
    call = body.blocks[-1].bindings[-1].value
    assert isinstance(call, rx.Call)
    arg = call.args[0]
    arg_str = strip_whitespace(dump_ast(arg))
    # the constant should have a tensor type
    assert "checked_type_=DynTensorType(ndim=0" in arg_str

    call_str = strip_whitespace(dump_ast(call))
    # we expect the shape_of call to have a checked_type_ of ShapeType
    type_str = "checked_type_=ShapeType()"
    assert type_str in call_str


def test_if():
    @R.function
    def f(cond: R.Tensor((), dtype="bool")) -> R.Tensor((), dtype="int32"):
        if cond:
            x = R.const(1)
        else:
            x = R.const(2)
        return x

    body = normalize(f).body
    assert isinstance(body, rx.SeqExpr)
    body_str = strip_whitespace(dump_ast(body))
    # we expect both branches to be seq exprs
    assert "If" in body_str
    assert "true_branch=SeqExpr(" in body_str
    assert "false_branch=SeqExpr(" in body_str


def test_tuple_get_item():
    @R.function
    def f(x: R.Tuple(R.Tensor((), dtype="int32"))) -> R.Tensor((), dtype="int32"):
        return x[0]

    body = normalize(f).body
    assert isinstance(body, rx.SeqExpr)
    body_str = strip_whitespace(dump_ast(body))

    assert "TupleGetItem" in body_str
    assert 'tuple_value=Var(name_hint="x"' in body_str
    assert "index=0" in body_str


if __name__ == "__main__":
    tvm.testing.main()
