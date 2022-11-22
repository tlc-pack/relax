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
import sys
import tvm
from tvm import topi
from tvm import relax


def _check(mod_before, mod_expected):
    mod = relax.transform.FuseTIR()(mod_before)
    print(mod["fused_add_exp_squeeze"].script())
    print(mod_expected["fused_add_exp_squeeze"].script())
    tvm.ir.assert_structural_equal(mod["fused_add_exp_squeeze"], mod_expected["fused_add_exp_squeeze"])
    tvm.ir.assert_structural_equal(mod, mod_expected)


def test_simple():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        p0 = relax.Var("p0", (), relax.DynTensorType(0, "float32"))

        with bb.function("fused_add_exp_squeeze", [x, p0], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add_exp_squeeze = bb.get().get_global_var("fused_add_exp_squeeze")

        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_add_exp_squeeze, [x, p0]))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        def fused_add_exp_squeeze(x, p0):
            add = topi.add(x, p0)
            exp = topi.exp(add)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        p0 = relax.Var("p0", (), relax.DynTensorType(0, "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(fused_add_exp_squeeze, x, p0))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_conv2d_fuse():
    def before(dtype):
        bb = relax.BlockBuilder()
        tensor_type = relax.DynTensorType(4, dtype)

        # Grouped function 1
        x = relax.Var("x", (1, 16, 64, 64), tensor_type)
        w = relax.Var("w", (16, 16, 3, 3), tensor_type)
        p0 = relax.Var("p0", (), relax.DynTensorType(0, dtype))
        with bb.function("fused_conv2d_add1_add2", [x, w, p0], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=1,
                    dilation=1,
                    primfunc_name_hint="conv2d",
                )
                lv1 = bb.emit_te(topi.add, p0, lv0, primfunc_name_hint="add1")
                gv = bb.emit_output(bb.call_te(topi.add, lv0, lv1, primfunc_name_hint="add2"))
            bb.emit_func_output(gv)

        # Grouped function 2
        x = relax.Var("x", (1, 16, 64, 64), tensor_type)
        w = relax.Var("w", (16, 16, 1, 1), tensor_type)
        y = relax.Var("y", (1, 16, 64, 64), tensor_type)
        with bb.function("fused_conv2d1_add2", [x, w, y], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=0,
                    dilation=1,
                    primfunc_name_hint="conv2d1",
                )
                gv = bb.emit_output(bb.call_te(topi.add, lv0, y, primfunc_name_hint="add2"))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        mod = bb.get()
        fused_conv2d_add1_add2 = mod.get_global_var("fused_conv2d_add1_add2")
        fused_conv2d1_add2 = mod.get_global_var("fused_conv2d1_add2")

        # Main function
        x = relax.Var("x", (1, 16, 64, 64), tensor_type)
        w1 = relax.Var("w1", (16, 16, 3, 3), tensor_type)
        w2 = relax.Var("w2", (16, 16, 1, 1), tensor_type)
        w3 = relax.Var("w3", (16, 16, 3, 3), tensor_type)
        with bb.function("main", [x, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, dtype))
                lv1 = bb.emit(relax.Call(fused_conv2d_add1_add2, [lv0, w1, relax.const(1, dtype)]))
                lv2 = bb.emit_te(
                    topi.nn.conv2d,
                    lv1,
                    w3,
                    strides=1,
                    padding=1,
                    dilation=1,
                )
                gv = bb.emit_output(relax.Call(fused_conv2d1_add2, [lv1, w2, lv2]))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dtype):
        def fused_conv2d_add1_add2(x, w, p):
            conv = topi.nn.conv2d(x, w, strides=1, padding=1, dilation=1)
            add = topi.add(p, conv)
            return topi.add(conv, add)

        def fused_conv2d1_add2(x, w, p):
            conv = topi.nn.conv2d(x, w, strides=1, padding=0, dilation=1)
            return topi.add(conv, p)

        bb = relax.BlockBuilder()
        tensor_type = relax.DynTensorType(4, dtype)

        # Main function
        x = relax.Var("x", (1, 16, 64, 64), tensor_type)
        w1 = relax.Var("w1", (16, 16, 3, 3), tensor_type)
        w2 = relax.Var("w2", (16, 16, 1, 1), tensor_type)
        w3 = relax.Var("w3", (16, 16, 3, 3), tensor_type)
        with bb.function("main", [x, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, dtype))
                lv1 = bb.emit_te(fused_conv2d_add1_add2, lv0, w1, relax.const(1, dtype))
                lv2 = bb.emit_te(
                    topi.nn.conv2d,
                    lv1,
                    w3,
                    strides=1,
                    padding=1,
                    dilation=1,
                )
                gv = bb.emit_output(bb.call_te(fused_conv2d1_add2, lv1, w2, lv2))
            bb.emit_func_output(gv)

        print(bb.get()["fused_conv2d_add1_add2"].script())
        return bb.get()

    _check(before("float32"), expected("float32"))


def test_two_subfunction():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("fused_exp_squeeze", [x1], attrs={"Primitive": True}):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_squeeze")
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(relax.Call(func_gv, [x]))
                lv2 = bb.emit(relax.Call(func_gv, [lv]))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_squeeze(x):
            exp = topi.exp(x)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_squeeze, x)
                lv2 = bb.emit_te(fused_exp_squeeze, lv)
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_same_primfunc():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("fused_exp_exp_squeeze", [x1], attrs={"Primitive": True}):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                lv2 = bb.emit_te(topi.exp, lv1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv2))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_exp_squeeze")
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(relax.Call(func_gv, [x]))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_exp_squeeze(x):
            exp = topi.exp(x)
            exp = topi.exp(exp)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_exp_squeeze, x)
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_tuple_as_param():
    dyn_tensor_type = relax.DynTensorType(1, "float32")
    tuple_type = relax.TupleType([dyn_tensor_type, dyn_tensor_type])
    tuple_shape = relax.Tuple([relax.ShapeExpr([10]), relax.ShapeExpr([10])])

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", tuple_shape, tuple_type)
        with bb.function("fused_exp_add", [x], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                lv2 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.add, lv2, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_add")
        x = relax.Var("x", tuple_shape, tuple_type)
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(func_gv, [x]))
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_add(x1, x2):
            exp = topi.exp(x1)
            return topi.add(exp, x2)

        bb = relax.BlockBuilder()
        dyn_tensor_type = relax.DynTensorType(1, "float32")
        tuple_type = relax.TupleType([dyn_tensor_type, dyn_tensor_type])
        tuple_shape = relax.Tuple([relax.ShapeExpr([10]), relax.ShapeExpr([10])])
        x = relax.Var("x", tuple_shape, tuple_type)
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                gv = bb.emit_output(bb.call_te(fused_exp_add, lv0, lv1))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_nested_tuple_as_param():
    dyn_tensor_type = relax.DynTensorType(1, "float32")
    tuple_type = relax.TupleType(
        [dyn_tensor_type, relax.TupleType([dyn_tensor_type, dyn_tensor_type])]
    )
    shape = relax.ShapeExpr([10])
    tuple_shape = relax.Tuple([shape, relax.Tuple([shape, shape])])

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", tuple_shape, tuple_type)
        with bb.function("fused_exp_add_add", [x], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv0_exp = bb.emit_te(topi.exp, lv0)
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                lv1_0 = bb.emit(relax.TupleGetItem(lv1, 0))
                lv1_1 = bb.emit(relax.TupleGetItem(lv1, 1))
                lv2 = bb.emit_te(topi.add, lv1_0, lv1_1)
                gv = bb.emit_output(bb.call_te(topi.add, lv0_exp, lv2))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_add_add")
        x = relax.Var("x", tuple_shape, tuple_type)
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(func_gv, [x]))
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_add_add(x1, x2, x3):
            exp = topi.exp(x1)
            add = topi.add(x2, x3)
            return topi.add(exp, add)

        bb = relax.BlockBuilder()
        x = relax.Var("x", tuple_shape, tuple_type)
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                lv2 = bb.emit(relax.TupleGetItem(lv1, 0))
                lv3 = bb.emit(relax.TupleGetItem(lv1, 1))
                gv = bb.emit_output(bb.call_te(fused_exp_add_add, lv0, lv2, lv3))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_call_tir_in_main():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("fused_exp_squeeze", [x1], attrs={"Primitive": True}):
            with bb.dataflow():
                lv = bb.emit_te(topi.exp, x1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_squeeze")
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(relax.Call(func_gv, [x]))
                lv1 = bb.emit_te(topi.add, lv0, relax.const(1, "float32"))
                gv = bb.emit_output(lv1)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_squeeze(x):
            exp = topi.exp(x)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_squeeze, x)
                lv2 = bb.emit_te(topi.add, lv, relax.const(1, "float32"))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_const_in_argument():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", [10, 20], relax.DynTensorType(2, "float32"))
        x2 = relax.Var("x2", [], relax.DynTensorType(0, "float32"))
        with bb.function("fused_add_exp_squeeze", [x1, x2], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x1, x2)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_add_exp_squeeze")
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(relax.Call(func_gv, [x, relax.const(1, "float32")]))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_add_exp_squeeze(x, y):
            add = topi.add(x, y)
            exp = topi.exp(add)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_add_exp_squeeze, x, relax.const(1, "float32"))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_tuple_output():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        p0 = relax.Var("p0", (), relax.DynTensorType(0, "float32"))

        with bb.function("fused_add_exp", [x, p0], attrs={"Primitive": True}):
            with bb.dataflow():
                gv0 = bb.emit_output(bb.call_te(topi.add, x, p0))
                gv1 = bb.emit_output(bb.call_te(topi.exp, gv0))
            bb.emit_func_output(relax.Tuple([gv0, gv1]))
        fused_add_exp = bb.get().get_global_var("fused_add_exp")

        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_add_exp, [x, p0]))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        def fused_add_exp(x, p0):
            add = topi.add(x, p0)
            exp = topi.exp(add)
            return add, exp

        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        p0 = relax.Var("p0", (), relax.DynTensorType(0, "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(fused_add_exp, x, p0))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_immediate_tuple():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        y = relax.Var("y", [10, 20], relax.DynTensorType(2, "float32"))

        with bb.function("fused_add", [x, y], attrs={"Primitive": True}):
            with bb.dataflow():
                lv_tuple = bb.emit(relax.Tuple([x, relax.Tuple([x, y])]))
                lv_x = bb.emit(relax.TupleGetItem(lv_tuple, 0))
                lv0 = bb.emit(relax.TupleGetItem(lv_tuple, 1))
                lv_y = bb.emit(relax.TupleGetItem(lv0, 1))
                gv = bb.emit_output(bb.call_te(topi.add, lv_x, lv_y))
            bb.emit_func_output(gv)
        fused_add = bb.get().get_global_var("fused_add")

        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        y = relax.Var("y", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x, y]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_add, [x, y]))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        y = relax.Var("y", [10, 20], relax.DynTensorType(2, "float32"))
        with bb.function("main", [x, y]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(topi.add, x, y, primfunc_name_hint="fused_add"))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_return_partial_result():
    def te_argmax_idx_val(val):
        from tvm import te

        def f_combine(x, y):
            lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
            rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
            return lhs, rhs

        def f_identity(dtype0: tvm.DataType, dtype1: tvm.DataType):
            return tvm.tir.const(-1, dtype0), tvm.te.min_value(dtype1)

        argmax = te.comm_reducer(f_combine, f_identity, name="argmax")
        m, n = val.shape
        k = te.reduce_axis((0, n), "k")
        max_idx, max_val = te.compute(
            (m,), lambda i: argmax((k.var, val[i, k]), axis=k), name="argmax"
        )
        return max_idx, max_val

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        offset = relax.Var("offset", [10], relax.DynTensorType(1, "int32"))
        with bb.function("fused_argmax_add", [x, offset], attrs={"Primitive": True}):
            with bb.dataflow():
                lv = bb.emit_te(te_argmax_idx_val, x)
                idx = bb.emit(relax.TupleGetItem(lv, 0))
                gv = bb.emit_output(bb.call_te(topi.add, idx, offset))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_argmax_add")
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        offset = relax.Var("x", [10], relax.DynTensorType(1, "int32"))
        with bb.function("main", [x, offset]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(func_gv, [x, offset]))
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_argmax_add(x, offset):
            idx, value = te_argmax_idx_val(x)
            idx = topi.add(idx, offset)
            return idx

        bb = relax.BlockBuilder()
        x = relax.Var("x", [10, 20], relax.DynTensorType(2, "float32"))
        offset = relax.Var("offset", [10], relax.DynTensorType(1, "int32"))
        with bb.function("main", [x, offset]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(fused_argmax_add, x, offset))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


if __name__ == "__main__":
    test_simple()
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
