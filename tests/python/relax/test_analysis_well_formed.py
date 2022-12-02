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
from tvm import tir
from tvm import relax as rx

m = tir.Var("m", "int64")
n = tir.Var("n", "int64")
type_anno = rx.DynTensorType(ndim=2, dtype="float16")
bool_type_anno = rx.DynTensorType(ndim=0, dtype="bool")
x = rx.Var("x", [m, n], type_anno)
cond = rx.Var("cond", [], bool_type_anno)


def build_function(blocks, params=[]):
    """Returns relax.function with given blocks"""
    seq_expr = rx.SeqExpr(blocks, blocks[-1].bindings[-1].var)
    ret_type = rx.DynTensorType(ndim=-1, dtype="float32")
    ret_shape = rx.RuntimeDepShape()
    func = rx.Function([x, cond] + params, seq_expr, ret_type, ret_shape).with_attr(
        "global_symbol", "foo"
    )
    return func


def test_var():
    # Error: Var gv0 is not defined
    gv0 = rx.Var("gv0", [m, n], type_anno)
    gv1 = rx.Var("gv1", [m, n], type_anno)
    call_node = rx.op.add(x, gv0)
    bindings = [rx.VarBinding(gv1, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)

    # Error: Var gv0 is defined more than once
    gv0 = rx.Var("gv0", [m, n], type_anno)
    call_node = rx.op.add(x, x)
    call_node2 = rx.op.multiply(x, x)
    bindings = [rx.VarBinding(gv0, call_node), rx.VarBinding(gv0, call_node2)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)


def test_dataflow_var():
    # Error: DataflowVar lv0 is not defined
    lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
    gv0 = rx.Var("gv0", [m, n], type_anno)
    call_node = rx.op.add(x, lv0)
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.DataflowBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)

    # Error: DataflowVar gv0 is defined more than once
    lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
    call_node = rx.op.add(x, x)
    call_node2 = rx.op.multiply(x, x)
    bindings = [rx.VarBinding(lv0, call_node), rx.VarBinding(lv0, call_node2)]
    blocks = [rx.DataflowBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)

    # Error: DataflowVar lv0 is defined outside DataflowBlock
    lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
    call_node = rx.op.add(x, x)
    bindings = [rx.VarBinding(lv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)

    # Error: DataflowVar lv0 is used outside DataflowBlock
    lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
    gv0 = rx.Var("gv0", [m, n], type_anno)
    call_node = rx.op.add(lv0, x)
    bindings = [rx.VarBinding(lv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)


def test_global_var():
    # Error: GlobalVar GlobalVar0 is not defined
    gv0 = rx.Var("gv0", [m, n], type_anno)
    globalvar = rx.GlobalVar("GlobalVar0")
    call_node = rx.Call(
        op=tvm.ir.Op.get("relax.call_tir"),
        args=[globalvar, rx.Tuple([x]), rx.ShapeExpr([m, n])],
    )
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)


def test_symbolic_var():
    # Error: Symbolic Var new_s is not defined
    new_s = tir.Var("new_s", "int64")
    gv0 = rx.Var("gv0", [m, new_s], type_anno)
    call_node = rx.op.add(x, x)
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)


def test_symbolic_var_invalid_type():
    with pytest.raises(tvm.TVMError, match="the value in ShapeExpr can only have dtype of int64"):
        dim = tir.Var("dim", "float32")
        type_anno = rx.DynTensorType(ndim=1, dtype="float32")
        y = rx.Var("y", [dim], type_anno)
        gv0 = rx.Var("gv0", [dim], type_anno)
        call_node = rx.op.add(y, y)
        bindings = [rx.VarBinding(gv0, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks, [y])
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        assert not rx.analysis.well_formed(mod)


def test_seq_expr():
    # Error: SeqExpr in VarBinding
    gv0 = rx.Var("gv0", [m, n], type_anno)
    # build a SeqExpr
    gv1 = rx.Var("gv1", [m, n], type_anno)
    call_node = rx.op.add(x, gv0)
    _bindings = [rx.VarBinding(gv1, call_node)]
    _blocks = [rx.BindingBlock(_bindings)]
    _seq_expr = rx.SeqExpr(_blocks, gv1)
    # build a Binding with the SeqExpr as value
    bindings = [rx.VarBinding(gv0, _seq_expr)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)


def test_if():
    # Error: Var defined in true/false branch is invisible in the outer scope
    # except the return Var, i.e the var in the last stmt
    # v_in_if is invisible in the outer scope
    v_in_if = rx.Var("v_in_if", [m, n], type_anno)
    # gv0 is visible in the outer scope
    gv0 = rx.Var("gv0", [m, n], type_anno)
    # build true branch
    true_bindings = [
        rx.VarBinding(v_in_if, rx.op.add(x, x)),
        rx.VarBinding(gv0, rx.op.multiply(x, x)),
    ]
    true_blocks = [rx.BindingBlock(true_bindings)]
    true_seq_expr = rx.SeqExpr(true_blocks, true_blocks[-1].bindings[-1].var)
    # build false branch
    false_bindings = [
        rx.VarBinding(v_in_if, rx.op.multiply(x, x)),
        rx.VarBinding(gv0, rx.op.add(x, x)),
    ]
    false_blocks = [rx.BindingBlock(false_bindings)]
    false_seq_expr = rx.SeqExpr(false_blocks, false_blocks[-1].bindings[-1].var)
    # build If node
    if_node = rx.If(cond=cond, true_branch=true_seq_expr, false_branch=false_seq_expr)
    gv1 = rx.Var("gv1", [m, n], type_anno)
    # try to call v_in_if defined in the true/false branch
    bindings = [rx.VarBinding(gv0, if_node), rx.VarBinding(gv1, v_in_if)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)


def test_if_non_seq_body():
    # Error: If node has a body that is not a seq node
    if_node = rx.If(cond=cond, true_branch=x, false_branch=x)
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", [m, n], type_anno),
                    if_node,
                )
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm.IRModule.from_expr(func)
    assert not rx.analysis.well_formed(mod)

    # on the other hand, if they're wrapped in a seq node, it's fine
    seq = rx.SeqExpr([], x)
    new_if_node = rx.If(cond=cond, true_branch=seq, false_branch=seq)
    new_blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", [m, n], type_anno),
                    new_if_node,
                )
            ]
        )
    ]
    new_func = build_function(new_blocks)
    new_mod = tvm.IRModule.from_expr(new_func)
    assert rx.analysis.well_formed(new_mod)


def test_if_complex_condition():
    # Error: If condition must be a leaf expression
    cond_tuple = rx.Tuple([cond])
    cond_idx = rx.TupleGetItem(cond_tuple, 0)
    if_node = rx.If(cond_idx, rx.SeqExpr([], x), rx.SeqExpr([], x))
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", [m, n], type_anno),
                    if_node,
                )
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm.IRModule.from_expr(func)
    assert not rx.analysis.well_formed(mod)

    cond_var = rx.Var("q", [], bool_type_anno)
    new_if = rx.If(cond_var, rx.SeqExpr([], x), rx.SeqExpr([], x))
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(cond_var, cond_idx),
                rx.VarBinding(
                    rx.Var("gv1", [m, n], type_anno),
                    new_if,
                ),
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm.IRModule.from_expr(func)
    assert rx.analysis.well_formed(mod)


def test_tuple_get_item_nested():
    # Error: The tuple value in tuple get item must be a leaf expression
    nested_tup = rx.Var(
        "t",
        type_annotation=rx.TupleType(
            [
                rx.TupleType(
                    [
                        rx.DynTensorType(ndim=0, dtype="int32"),
                    ]
                )
            ]
        ),
    )
    double_idx = rx.TupleGetItem(rx.TupleGetItem(nested_tup, 0), 0)
    ret_var = rx.Var("r", [], rx.DynTensorType(ndim=0, dtype="int32"))
    f = rx.Function(
        [nested_tup],
        rx.SeqExpr([rx.BindingBlock([rx.VarBinding(ret_var, double_idx)])], ret_var),
        ret_type=rx.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=rx.RuntimeDepShape(),
    )
    f = f.with_attr("global_symbol", "f")
    mod = tvm.IRModule.from_expr(f)
    assert not rx.analysis.well_formed(mod)

    # okay with an intermediate binding
    first_idx = rx.TupleGetItem(nested_tup, 0)
    idx_var = rx.Var("v", type_annotation=rx.TupleType([rx.DynTensorType(ndim=0, dtype="int32")]))
    second_idx = rx.TupleGetItem(idx_var, 0)
    new_f = rx.Function(
        [nested_tup],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [rx.VarBinding(idx_var, first_idx), rx.VarBinding(ret_var, second_idx)]
                )
            ],
            ret_var,
        ),
        ret_type=rx.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=rx.RuntimeDepShape(),
    )
    new_f = new_f.with_attr("global_symbol", "new_f")
    mod = tvm.IRModule.from_expr(new_f)
    assert rx.analysis.well_formed(mod)


def test_complex_seq_body():
    # Error: seq expr with a body that is not a leaf expression is not permitted
    x = rx.Var("x", [], rx.DynTensorType(ndim=0, dtype="int32"))
    y = rx.Var("y", [], rx.DynTensorType(ndim=0, dtype="int32"))
    ret_type = rx.DynTensorType(ndim=0, dtype="float32")
    ret_shape = rx.RuntimeDepShape()
    func = rx.Function(
        [x, y],
        rx.SeqExpr([], rx.op.add(x, y)),
        ret_type,
        ret_shape,
    ).with_attr("global_symbol", "foo")
    mod = tvm.IRModule.from_expr(func)
    assert not rx.analysis.well_formed(mod)

    # but if the result is bound, then it's okay
    z = rx.Var("z", [], rx.DynTensorType(ndim=0, dtype="int32"))
    new_func = rx.Function(
        [x, y],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [
                        rx.VarBinding(
                            var=z,
                            value=rx.op.add(x, y),
                        )
                    ]
                )
            ],
            z,
        ),
        ret_type,
        ret_shape,
    ).with_attr("global_symbol", "foo")
    new_mod = tvm.IRModule.from_expr(new_func)
    assert rx.analysis.well_formed(new_mod)


def test_ANF():
    # Error: Nested Call
    gv0 = rx.Var("gv0", [m, n], type_anno)
    call_node = rx.op.add(x, rx.op.add(x, x))
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)

    # Error: Call Node in Tuple
    gv0 = rx.Var("gv0", [m, n], type_anno)
    bindings = [rx.VarBinding(gv0, rx.Tuple((x, rx.op.add(x, x))))]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod)


if __name__ == "__main__":
    pytest.main([__file__])
