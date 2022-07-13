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

from __future__ import annotations
import pytest

from tvm._ffi.base import TVMError
from tvm.relax.dataflow_pattern import *
from tvm.relax.analysis import get_var2val
from tvm import relax as rx, tir
from tvm.script import relax as R, tir as T


@tvm.script.ir_module
class Module:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        k = T.var("int32")
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        C = T.match_buffer(z, (32, 32))

        for (i0, j0, k0) in T.grid(32, 32, 32):
            with T.block():
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]

    @T.prim_func
    def tir_relu(x: T.handle, y: T.handle):
        T.func_attr({"global_symbol": "tir_relu"})
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        for (i, j) in T.grid(32, 32):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x: Tensor((32, 32), "float32"), w: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
            lv1 = R.call_tir(tir_relu, (lv0), (32, 32), dtype="float32")
            relax.output(lv1)
        return lv1


main_fn = Module["main"]
bindings = main_fn.body.blocks[0].bindings

## Node-wise Matching
def test_expr_pattern():
    ep = is_expr(rx.Var("x"))
    assert isinstance(ep, ExprPattern)
    assert isinstance(ep.expr, rx.Var)


def test_var_pattern():
    v = is_var("x")
    assert isinstance(v, VarPattern)
    assert v.name == "x"
    assert v.match_expr(rx.Var("x"))
    assert is_var().match_expr(rx.Var("x"))
    assert is_var().match_expr(rx.DataflowVar("x"))  # DataflowVar is also a Var
    assert not v.match_expr(rx.GlobalVar("x"))


def test_dataflow_var_pattern():
    v = is_dfv("x")
    assert isinstance(v, DataflowVarPattern)
    assert v.name == "x"
    assert v.match_expr(rx.DataflowVar("x"))
    assert not v.match_expr(rx.GlobalVar("x"))
    assert is_dfv().match_expr(bindings[0].var)


def test_global_var_pattern():
    assert is_gv("x").match_expr(rx.GlobalVar("x"))
    assert is_gv().match_expr(rx.GlobalVar("x"))
    assert not is_gv("x").match_expr(rx.GlobalVar("y"))
    assert not is_gv("x").match_expr(rx.Var("x"))


def test_constant_pattern():
    c = is_constant()
    assert isinstance(c, ConstantPattern)
    assert c.match_expr(rx.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]]))


def test_wildcard_pattern():
    wc = wildcard()
    assert isinstance(wc, WildcardPattern)
    assert wc.match_expr(rx.Var("x"))


def test_call_pattern():
    wc1 = wildcard()
    wc2 = wildcard()
    c = is_op("relax.add")(wc1, wc2)
    assert isinstance(c, CallPattern)
    assert isinstance(c.args[0], WildcardPattern)
    assert isinstance(c.args[1], WildcardPattern)
    assert c.match_expr(rx.op.add(rx.Var("x"), rx.Var("y")))


def test_function_pattern():
    wc1 = wildcard()
    wc2 = wildcard()
    f = FunctionPattern([wc1, wc2], is_op("relax.add")(wc1, wc2))
    assert isinstance(f, FunctionPattern)
    assert isinstance(f.params[0], WildcardPattern)
    assert isinstance(f.params[1], WildcardPattern)
    assert isinstance(f.body, CallPattern)
    assert isinstance(f.body.args[0], WildcardPattern)
    assert isinstance(f.body.args[1], WildcardPattern)
    ttype = rx.DynTensorType(-1, "float32")
    x = rx.Var("x", type_annotation=ttype)
    y = rx.Var("y", type_annotation=ttype)
    assert f.match_expr(rx.Function([x, y], rx.op.add(x, y), ret_type=ttype))
    assert not f.match_expr(rx.Function([x, y], rx.op.multiply(x, y), ret_type=ttype))


def test_tuple_pattern():
    wc1 = wildcard()
    wc2 = is_dfv()
    t = is_tuple([wc1, wc2])
    assert isinstance(t, TuplePattern)
    assert isinstance(t.fields[0], WildcardPattern)
    assert isinstance(t.fields[1], DataflowVarPattern)
    assert t.match_expr(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]))
    assert not t.match_expr(rx.Tuple([rx.DataflowVar("x"), rx.GlobalVar("y")]))


def test_tuple_get_item_pattern():
    assert is_tuple_get_item(is_tuple([is_gv("x"), is_dfv("y")]), 0).match_expr(
        rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0)
    )
    assert is_tuple_get_item(is_tuple([is_gv("x"), is_dfv("y")]), 0).match_expr(
        rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0)
    )


def test_or_pattern():
    dfv_or_gv = is_dfv("x") | is_gv("x")
    assert isinstance(dfv_or_gv, OrPattern)
    assert dfv_or_gv.match_expr(rx.DataflowVar("x"))
    assert dfv_or_gv.match_expr(rx.GlobalVar("x"))
    assert not dfv_or_gv.match_expr(rx.Var("x"))
    assert not dfv_or_gv.match_expr(rx.DataflowVar("y"))
    assert not dfv_or_gv.match_expr(rx.GlobalVar("y"))


def test_and_pattern():
    # float[2, 3, 3]
    f32_233 = has_shape((2, 3, 3)) & has_dtype("float32")
    assert isinstance(f32_233, AndPattern)
    assert f32_233.match_expr(rx.Var("x", (2, 3, 3), rx.DynTensorType(3, "float32")))
    assert not f32_233.match_expr(rx.Var("x", (3, 3, 3), rx.DynTensorType(3, "float32")))
    assert not f32_233.match_expr(rx.Var("x", rx.RuntimeDepShape(), rx.DynTensorType(3, "float32")))


def test_not_pattern():
    no_shape233 = ~has_shape((2, 3, 3))
    assert isinstance(no_shape233, NotPattern)
    assert no_shape233.match_expr(rx.Var("x", (3, 3, 3), rx.DynTensorType(3, "float32")))
    assert not no_shape233.match_expr(rx.Var("x", (2, 3, 3), rx.DynTensorType(3, "float32")))


def test_type_pattern():
    assert has_type(rx.DynTensorType(2, "float32")).match_expr(bindings[0].var)


def test_dtype_pattern():
    dtype = "float16"
    pattern = has_dtype(dtype)
    assert isinstance(pattern, DataTypePattern)
    assert pattern.dtype == dtype
    assert has_dtype("float32").match_expr(bindings[0].var)


def test_shape_pattern():
    shape = [32, 32]
    pattern = has_shape(shape)
    assert isinstance(pattern, ShapePattern)
    tvm.ir.structural_equal(pattern.shape, shape)
    assert pattern.match_expr(bindings[0].var)
    assert has_shape(32, 32).match_expr(bindings[0].var)
    n, m = tir.Var("n", dtype="int32"), tir.Var("m", dtype="int32")
    symbolic_shape = rx.ShapeExpr([n, m, n + m])
    symsh_var = rx.Var("x", symbolic_shape, rx.DynTensorType(3, "float32"))
    assert has_shape(n, m, n + m).match_expr(symsh_var)
    assert has_shape(n, m, m + n).match_expr(symsh_var)  # + is commutative.
    assert not has_shape(1, 2, 3).match_expr(symsh_var)
    assert not has_shape(m, n, n + m).match_expr(symsh_var)


def test_prim_arr_pattern():
    pattern = is_shape(32, 32)
    assert isinstance(pattern, PrimArrPattern)
    assert pattern.match_expr(bindings[0].var.shape)
    n, m = tir.Var("n", dtype="int32"), tir.Var("m", dtype="int32")
    symbolic_shape = rx.ShapeExpr([n, m, n + m])
    assert is_shape([n, m, n + m]).match_expr(symbolic_shape)
    assert not is_shape([n, m, n * m]).match_expr(symbolic_shape)


def test_rt_dep_shape_pattern():
    # runtime-dep-shape var
    rts_var = rx.Var("rts_var", rx.RuntimeDepShape(), rx.DynTensorType(4, "float32"))
    # static-shape var
    ss_var = rx.Var("ss_var", rx.ShapeExpr([32, 32]), rx.DynTensorType(4, "float32"))
    assert isinstance(has_rt_dep_shape(), RuntimeDepShapePattern)
    assert has_rt_dep_shape().match_expr(rts_var)
    assert not has_rt_dep_shape().match_expr(ss_var)


def test_extern_fn_pattern():
    pattern = ExternFuncPattern("test.blockbuilder.nop")
    assert pattern.match_expr(rx.ExternFunc("test.blockbuilder.nop"))


def test_match_call_attr():
    ttype = rx.DynTensorType(-1, "float32")
    x = rx.Var("x", type_annotation=ttype)
    y = rx.Var("y", type_annotation=ttype)
    fn = rx.Function([x, y], rx.op.add(x, y), ret_type=ttype)
    annotated_fn = fn.with_attr({"Codegen": "test-codegen", "global_symbol": "test-symbol"})
    xp = is_var("x")
    yp = is_var("y")
    root_pattern = FunctionPattern([xp, yp], is_op("relax.add")(xp, yp))
    assert root_pattern.has_attr(
        {"Codegen": "test-codegen", "global_symbol": "test-symbol"}
    ).match_expr(annotated_fn)

    assert root_pattern.has_attr({"Codegen": "test-codegen"}).match_expr(annotated_fn)
    assert not root_pattern.has_attr({"ping": "pong"}).match_expr(annotated_fn)
    assert root_pattern.has_attr({}).match_expr(annotated_fn)


def test_is_call_tir():
    lv1_val = bindings[1].value
    var2val = get_var2val(Module["main"])
    assert is_call_tir("tir_relu").match_expr(lv1_val)
    assert is_call_tir("tir_relu", is_call_tir("tir_matmul")).match_expr(lv1_val, var2val=var2val)
    assert not is_call_tir("tir_relu", is_call_tir("tir_relu")).match_expr(lv1_val, var2val=var2val)


## Graph-wise Matching
def test_single_node_cannot():
    with pytest.raises(TVMError):
        # Match a DFPattern without graph constraints (e.g., single node) is not allowed.
        wildcard().match_dfb(main_fn.body.blocks[0])


def test_simple_edge():
    n0 = wildcard()
    n1 = wildcard()
    n0.used_by(n1)
    dfb = main_fn.body.blocks[0]
    matched = n0.match_dfb(dfb)
    assert matched
    assert matched[n0] == dfb.bindings[0]
    assert matched[n1] == dfb.bindings[1]


def test_simple_call_tir_edge():
    n0 = is_call_tir("tir_matmul")
    n1 = is_call_tir("tir_relu")
    n0.used_by(n1)
    dfb = main_fn.body.blocks[0]
    matched = n0.match_dfb(dfb)
    assert matched
    assert matched[n0] == dfb.bindings[0]
    assert matched[n1] == dfb.bindings[1]


def test_simple_used_by():
    n0 = wildcard()
    n1 = wildcard()
    n0 ^ n1
    dfb = main_fn.body.blocks[0]
    matched = n0.match_dfb(dfb)
    assert matched
    assert matched[n0] == dfb.bindings[0]
    assert matched[n1] == dfb.bindings[1]


def test_simple_oub():
    n0 = is_call_tir("tir_matmul")
    n1 = is_call_tir("tir_relu")
    n0 >> n1
    dfb = main_fn.body.blocks[0]
    matched = n0.match_dfb(dfb)
    assert matched
    assert matched[n0] == dfb.bindings[0]
    assert matched[n1] == dfb.bindings[1]


def test_counter_oub_syntax():
    n0 = is_call_tir("tir_matmul")
    n1 = is_call_tir("tir_impossible")
    n0 >> n1
    dfb = main_fn.body.blocks[0]
    matched = n0.match_dfb(dfb)
    assert not matched


@tvm.script.ir_module
class Diamond:
    @R.function
    def main(x: Tensor((32, 32), "float32"), w: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
            lv1 = R.call_tir(tir_relu, (lv0), (32, 32), dtype="float32")
            lv2 = R.call_tir(tir_sigmoid, (lv0), (32, 32), dtype="float32")
            lv3 = R.call_tir(tir_add, (lv1, lv2), (32, 32), dtype="float32")
            relax.output(lv3)
        return lv3


def test_diamond():
    n0 = is_call_tir("tir_matmul")
    n1 = is_call_tir("tir_relu")
    n2 = is_call_tir("tir_sigmoid")
    n3 = is_call_tir("tir_add")

    n0 ^ n1
    n0 ^ n2
    n1 >> n3
    n2 >> n3

    dfb = Diamond["main"].body.blocks[0]
    matched = n0.match_dfb(dfb)
    assert matched


def test_diamond_counter_oub():
    n0 = is_call_tir("tir_matmul")
    n1 = is_call_tir("tir_relu")
    n2 = is_call_tir("tir_sigmoid")
    n3 = is_call_tir("tir_add")

    n0 >> n1
    n0 >> n2
    n1 >> n3
    n2 >> n3

    dfb = Diamond["main"].body.blocks[0]
    matched = n0.match_dfb(dfb)
    assert not matched


@tvm.script.ir_module
class CBRx2:
    @R.function
    def main(
        x: Tensor((32, 32), "float32"),
        w0: Tensor((1, 1), "float32"),
        bias0: Tensor((32, 32), "float32"),
        w1: Tensor((1, 1), "float32"),
        bias1: Tensor((32, 32), "float32"),
    ) -> Tensor:
        with R.dataflow():
            lv0 = R.call_tir(conv1x1, (x, w0), (32, 32), dtype="float32")
            lv1 = R.call_tir(bias_add, (lv0, bias0), (32, 32), dtype="float32")
            lv2 = R.call_tir(relu, (lv1), (32, 32), dtype="float32")
            #     CBR_0
            #   /       \
            # X        concat
            #   \       /
            #     CBR_1
            lv3 = R.call_tir(conv1x1, (x, w1), (32, 32), dtype="float32")
            lv4 = R.call_tir(bias_add, (lv3, bias1), (32, 32), dtype="float32")
            lv5 = R.call_tir(relu, (lv4), (32, 32), dtype="float32")

            lv6 = R.call_tir(concat, (lv2, lv5), (32, 64), dtype="float32")
            relax.output(lv6)
        return lv6


def test_single_cbr():
    cbr = is_call_tir("conv1x1") >> is_call_tir("bias_add") >> is_call_tir("relu")
    dfb = CBRx2["main"].body.blocks[0]
    assert cbr.patterns[0].match_dfb(dfb)


def test_counter_single_crb():
    crb = is_call_tir("conv1x1") >> is_call_tir("relu") >> is_call_tir("bias_add")
    dfb = CBRx2["main"].body.blocks[0]
    assert not crb.patterns[0].match_dfb(dfb)