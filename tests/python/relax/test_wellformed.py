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

m = tir.Var("m", "int32")
n = tir.Var("n", "int32")
type_anno = rx.DynTensorType(ndim=2, dtype="float16")
x = rx.Var("x", [m, n], type_anno)


def build_function(blocks):
    """Returns relax.function with given blocks"""
    seq_expr = rx.SeqExpr(blocks, blocks[-1].bindings[-1].var)
    ret_type = rx.DynTensorType(-1, "float32")
    func = rx.Function([x], seq_expr, ret_type).with_attr("global_symbol", "foo")
    return func


def test_var():
    # Error: Var gv0 is not defined
    with pytest.raises(tvm.TVMError):
        gv0 = rx.Var("gv0", [m, n], type_anno)
        gv1 = rx.Var("gv1", [m, n], type_anno)
        call_node = rx.op.add(x, gv0)
        bindings = [rx.VarBinding(gv1, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)

    # Error: Var gv0 is defined more than once
    with pytest.raises(tvm.TVMError):
        gv0 = rx.Var("gv0", [m, n], type_anno)
        call_node = rx.op.add(x, x)
        call_node2 = rx.op.multiply(x, x)
        bindings = [rx.VarBinding(gv0, call_node), rx.VarBinding(gv0, call_node2)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)


def test_dataflowvar():
    # Error: DataflowVar lv0 is not defined
    with pytest.raises(tvm.TVMError):
        lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
        gv0 = rx.Var("gv0", [m, n], type_anno)
        call_node = rx.op.add(x, lv0)
        bindings = [rx.VarBinding(gv0, call_node)]
        blocks = [rx.DataflowBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)

    # Error: DataflowVar gv0 is defined more than once
    with pytest.raises(tvm.TVMError):
        lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
        call_node = rx.op.add(x, x)
        call_node2 = rx.op.multiply(x, x)
        bindings = [rx.VarBinding(lv0, call_node), rx.VarBinding(lv0, call_node2)]
        blocks = [rx.DataflowBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)

    # Error: DataflowVar lv0 is defined outside DataflowBlock
    with pytest.raises(tvm.TVMError):
        lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
        call_node = rx.op.add(x, x)
        bindings = [rx.VarBinding(lv0, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)

    # Error: DataflowVar lv0 is used outside DataflowBlock
    with pytest.raises(tvm.TVMError):
        lv0 = rx.DataflowVar("lv0", [m, n], type_anno)
        gv0 = rx.Var("gv0", [m, n], type_anno)
        call_node = rx.op.add(lv0, x)
        bindings = [rx.VarBinding(lv0, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)


def test_globalvar():
    # Error: GlobalVar GlobalVar0 is not defined
    with pytest.raises(tvm.TVMError):
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
        rx.analysis.well_formed(mod)


def test_symbolicvar():
    # Error: Symbolic Var new_s is not defined
    with pytest.raises(tvm.TVMError):
        new_s = tir.Var("new_s", "int32")
        gv0 = rx.Var("gv0", [m, new_s], type_anno)
        call_node = rx.op.add(x, x)
        bindings = [rx.VarBinding(gv0, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)


def test_seqexpr():
    # Error: SeqExpr in VarBinding
    with pytest.raises(tvm.TVMError):
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
        rx.analysis.well_formed(mod)


def test_ANF():
    # Error: Nested Call
    with pytest.raises(tvm.TVMError):
        gv0 = rx.Var("gv0", [m, n], type_anno)
        call_node = rx.op.add(x, rx.op.add(x, x))
        bindings = [rx.VarBinding(gv0, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)

    # Error: Call Node in Tuple
    with pytest.raises(tvm.TVMError):
        gv0 = rx.Var("gv0", [m, n], type_anno)
        bindings = [rx.VarBinding(gv0, rx.Tuple((x, rx.op.add(x, x))))]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks)
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        rx.analysis.well_formed(mod)


if __name__ == "__main__":
    pytest.main([__file__])
