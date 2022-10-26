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
import re

import tvm
from tvm import tir
from tvm import relax as rx
from tvm.relax.testing import dump_ast
from tvm.script import tir as T, relax as R

import numpy as np


def strip_whitespace(text: str) -> str:
    """
    Remove all whitespace to avoid reasoning about newlines and indents
    """
    return re.sub(r"\s", "", text)


# test cases are mostly adapted from text_expr, only testing very basic properties


def test_var() -> None:
    v0 = rx.Var("v0")
    v0_str = dump_ast(v0)
    assert v0_str == 'Var(name_hint="v0")'

    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float32")
    v1 = rx.Var("v1", shape_anno, type_anno)
    v1_no_annos = dump_ast(v1, include_shape_annotations=False, include_type_annotations=False)
    assert v1_no_annos == 'Var(name_hint="v1")'
    v1_annos = dump_ast(v1)
    assert v1_annos != v1_no_annos
    assert "PrimExpr" in v1_annos
    assert "shape_" in v1_annos
    assert "_checked_type_" in v1_annos


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    v0_str = dump_ast(v0)
    assert v0_str == 'DataflowVar(name_hint="v0")'

    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float16")
    v1 = rx.DataflowVar("v1", shape_anno, type_anno)
    v1_no_annos = dump_ast(v1, include_shape_annotations=False, include_type_annotations=False)
    assert v1_no_annos == 'DataflowVar(name_hint="v1")'
    v1_annos = dump_ast(v1)
    assert v1_annos != v1_no_annos
    assert "PrimExpr" in v1_annos
    assert "shape_" in v1_annos
    assert "_checked_type_" in v1_annos


def test_match_shape() -> None:
    # match_shape([16, 8], [m, n])
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32")
    var = rx.Var("v0", type_annotation=rx.ShapeType())
    b0 = rx.MatchShape(shape, [m, n], var)
    b0_str = dump_ast(b0)
    assert b0_str.startswith("MatchShape(")
    assert "Constant" in b0_str
    assert "PrimExpr(value=`m: int32`)" in b0_str
    assert "PrimExpr(value=`n: int32`)" in b0_str
    assert "16" in b0_str
    assert "8" in b0_str
    assert b0_str != dump_ast(b0, include_type_annotations=False)

    # var1: Tensor((m, n), "float32") =
    #   match_shape(var0: Tensor(_, "float32"), [m, n])
    type_anno0 = rx.DynTensorType(-1, "float32")
    value = rx.Var("value", type_annotation=type_anno0)

    shape_anno = [m, n]
    type_anno = rx.DynTensorType(2, "float32")
    var = rx.Var("v1", shape_anno, type_anno)
    b1 = rx.MatchShape(value, [m, n], var)
    b1_str = dump_ast(b1)
    assert b1_str.startswith("MatchShape(")
    assert "PrimExpr(value=`m: int32`)" in b1_str
    assert "PrimExpr(value=`n: int32`)" in b1_str
    assert b1_str != dump_ast(b1, include_type_annotations=False, include_shape_annotations=False)


def test_var_binding() -> None:
    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    b0_str = dump_ast(b0)
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
    assert "value=Constant(data=1)" in seqe_str
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
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("foo", type_annotation=type_anno)
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    ret_type = rx.DynTensorType(-1, "float32")
    ret_shape = rx.RuntimeDepShape()
    func = rx.Function([x], seqe, ret_type, ret_shape)
    func = func.with_attr("global_symbol", "func")

    func_str = dump_ast(func)
    assert func_str.startswith("Function(")
    assert "params=" in func_str
    assert "body=" in func_str
    assert "ret_type=" in func_str
    assert "ret_shape=" in func_str
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

    shape_anno = [96, 54]
    v1 = rx.Var("v1", shape_anno)
    s1 = v1.shape
    s1_str = dump_ast(s1)
    assert s1_str.startswith("ShapeExpr("), s1_str
    assert "values=" in s1_str
    assert "PrimExpr(value=`96`)" in s1_str
    assert "PrimExpr(value=`54`)" in s1_str


def test_shape_expr():
    shape_expr = rx.ShapeExpr([10, 20])
    shape_expr_str = dump_ast(shape_expr)
    assert shape_expr_str.startswith("ShapeExpr(")
    assert "values" in shape_expr_str
    assert "PrimExpr(value=`10`)" in shape_expr_str
    assert "PrimExpr(value=`20`)" in shape_expr_str


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
        o: R.Object = R.call_packed("contrib.tensor_array_stack", x, y, type_args=R.Object)
        return o

    # checking that the call_packed call is turned into a call to an extern func
    f_str = strip_whitespace(
        dump_ast(
            f,
            include_type_annotations=False,
            include_shape_annotations=False,
            include_call_attrs=False,
        )
    )
    extern_call = strip_whitespace(
        """
        Call(
            op=ExternFunc(global_symbol="contrib.tensor_array_stack"),
            args=[
                Var(name_hint="x"),
                Var(name_hint="y")
            ],
            type_args=[ObjectType()]
        )
        """
    )
    assert extern_call in f_str
    # check that the op call is there too
    op_call = strip_whitespace(
        """
        Call(
            op=Op(name="relax.multiply"),
            args=[
                Var(name_hint="x"),
                Var(name_hint="y")
            ]
        )
        """
    )
    assert op_call in f_str
    # the function has an annotated return type
    assert "ret_type=ObjectType()" in f_str

    # the op call has attributes so let's check those too
    f_str_complete = strip_whitespace(dump_ast(f))
    assert f_str != f_str_complete
    attrs_str = strip_whitespace(
        """
        attrs={
            "units": None,
            "out_dtype": "",
            "transpose_a": 0,
            "transpose_b": 0
        }
        """
    )
    assert attrs_str in f_str_complete


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
            include_shape_annotations=False,
            include_call_attrs=False,
        )
    )
    # call_tir is an op in Relax and it takes an extern func as an argument
    call_tir_text = strip_whitespace(
        """
        Call(
            op=Op(name="relax.call_tir"),
            args=[
                ExternFunc(global_symbol="test.op.identity"),
                Tuple(fields=[Var(name_hint="x")]),
                ShapeExpr(values=[
                    PrimExpr(value=`m: int64`),
                    PrimExpr(value=`n: int64`)
                ])
            ],
            type_args=[DynTensorType(ndim=2, dtype=float32)]
        )
        """
    )
    assert foo_str.startswith('Function(params=[Var(name_hint="x")]')
    assert call_tir_text in foo_str


def test_operators():
    # the operator attributes need to be registered to work in the printer

    @R.function
    def foo(x: R.Tensor):
        return R.unique(x, sorted=True)

    foo_str = strip_whitespace(
        dump_ast(
            foo,
            include_type_annotations=False,
            include_shape_annotations=False,
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
            include_shape_annotations=False,
        )
    )
    print_attrs_str = strip_whitespace('{"format": "{}"}')
    assert print_attrs_str in bar_str


if __name__ == "__main__":
    pytest.main([__file__])
