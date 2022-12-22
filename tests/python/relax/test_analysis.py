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

from typing import List, Set, Union
import pytest

import tvm
from tvm import tir
from tvm import relax as rx
from tvm.relax.analysis import (
    udchain,
    remove_all_unused,
    name_to_binding,
    shape_vars,
    derive_func_ret_shape,
    all_vars,
    free_vars,
    bound_vars,
    all_global_vars,
    called_global_vars,
)
from tvm.script import relax as R


def var_name_set(vars: List[Union[rx.Var, rx.GlobalVar]]) -> Set[str]:
    return set(map(lambda v: v.name_hint, vars))


def test_dispatch_var():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    v0 = rx.Var("v0", R.Tensor([m, n], "float16"))
    v1 = rx.DataflowVar("v1", R.Tensor([n], "float16"))
    t = None

    def fvisit(e):
        nonlocal t
        t = type(e)

    rx.analysis.post_order_visit(v0, fvisit)
    assert t == type(v0)
    rx.analysis.post_order_visit(v1, fvisit)
    assert t == type(v1)


def test_post_order_visit():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    ib = rx.BlockBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow():
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_func_output(gv0)
    expr = ib.get()["func"]

    names = []

    def fvisit(e):
        nonlocal names
        if isinstance(e, tvm.ir.op.Op):
            names.append(e.name)

    rx.analysis.post_order_visit(expr.body, fvisit)
    assert names == ["relax.add", "relax.multiply"]


def test_use_def():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    ib = rx.BlockBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow():
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_func_output(gv0)
    dfb = ib.get()["func"].body.blocks[0]
    udc = udchain(dfb)
    assert set(udc[x]) == {lv0}
    assert set(udc[y]) == {lv0, lv1}
    assert set(udc[lv0]) == {lv1}
    assert set(udc[lv1]) == {gv0}
    assert set(udc[gv0]) == set()


def test_chained_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_tir("my_sigmoid", (x,), (32, 32), dtype="float32")
                unused1 = R.call_tir("my_sigmoid", (unused0,), (32, 32), dtype="float32")
                R.output(lv0)
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_tir("my_sigmoid", (x,), (32, 32), dtype="float32")
                unused1 = R.call_tir("my_sigmoid", (unused0,), (32, 32), dtype="float32")
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(R.Tensor((32, 32), "float32")))
            return z

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(R.Tensor((32, 32), "float32")))
            return z

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_fake_unused_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(R.Tensor((32, 32), "float32")))
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            # This might bring side effect so cannot be removed.
            z = R.call_packed("vm.builtin.copy", lv0, type_args=(R.Tensor((32, 32), "float32")))
            return lv0

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_edge_binding_block_fake_unused_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
            z = R.call_packed("vm.builtin.copy", x, type_args=(R.Tensor((32, 32), "float32")))
            return x

    optimized = remove_all_unused(IdentityUnused["main"])
    tvm.ir.assert_structural_equal(optimized, IdentityUnused["main"])


def test_name_to_binding_var_shadowing():
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            lv1 = lv0
            R.output(lv1)

        with R.dataflow():
            lv0 = lv1  # shadowing
            lv2 = lv0
            R.output(lv2)
        return lv2

    n2binding = name_to_binding(main)

    assert "lv0" in n2binding
    assert "lv1" in n2binding
    assert "lv2" in n2binding

    assert len(n2binding["lv0"]) == 2


def test_shape_var_shape_expr():
    v1 = tir.Var("v1", "int64")
    v2 = tir.Var("v2", "int64")
    v3 = tir.Var("v3", "int64")
    shape_expr = rx.ShapeExpr([v1, v2, tir.Add(v3, v1)])
    vars = shape_vars(shape_expr)

    assert len(vars) == 3
    assert v1 in vars
    assert v2 in vars
    assert v3 in vars

    shape_expr = rx.ShapeExpr([tir.const(1), tir.const(2)])
    vars = shape_vars(shape_expr)
    assert len(vars) == 0


def test_shape_var_nested():
    v1 = rx.Var("v1")
    v2 = rx.Var("v2")
    sv1 = tir.Var("sv1", "int64")
    shape_expr = rx.ShapeExpr([sv1])
    tup = rx.Tuple([v1, v2, shape_expr])
    vars = shape_vars(tup)

    assert len(vars) == 1
    assert sv1 in vars

    x = rx.Var("x", R.Tensor(ndim=-1, dtype="int64"))
    y = rx.Var("y", R.Tensor(ndim=-1, dtype="int64"))

    func = rx.Function([x, y], shape_expr, R.Shape())
    vars = shape_vars(func)

    assert len(vars) == 1
    assert sv1 in vars


def test_derive_func_ret_shape_no_free():
    sv1 = tir.Var("sv1", "int64")
    sv2 = tir.Var("sv2", "int64")
    sv3 = tir.Var("sv3", "int64")
    a1 = rx.Var("a1", R.Tensor([sv1, sv2]))
    a2 = rx.Var("a2", R.Tensor([sv2, sv3]))
    body = a2
    shape_expr = derive_func_ret_shape([a1, a2], body)

    assert isinstance(shape_expr, rx.ShapeExpr)
    assert shape_expr[0] == sv2
    assert shape_expr[1] == sv3


def test_derive_func_ret_shape_free():
    sv1 = tir.Var("sv1", "int64")
    sv2 = tir.Var("sv2", "int64")
    sv3 = tir.Var("sv3", "int64")
    a1 = rx.Var("a1", R.Tensor([sv1, sv2]))
    a2 = rx.Var("a2", R.Tensor([sv2, sv1]))
    # Artifically introducing a free shape variable.
    # This would not be a valid program, but this is being done to test the logic
    body = rx.Var("a3", R.Tensor([sv1, sv3]))
    shape_expr = derive_func_ret_shape([a1, a2], body)
    assert isinstance(shape_expr, rx.RuntimeDepShape)


@tvm.script.ir_module
class VarExample:
    @R.function
    def func(a: R.Tensor) -> R.Tensor:
        # normalized into assigning R.add(a, a) to a var and returning it
        return R.add(a, a)

    @R.function
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        z = R.add(x, y)
        # no binding here
        R.match_shape(x, (5, 5))
        with R.dataflow():
            q = R.add(z, z)
            p = func(q)
            r = R.match_shape(p, (5, 5))
            s = r
            R.output(s)
        return s


def test_all_vars():
    vars = all_vars(VarExample["func"])
    assert len(vars) == 2
    assert vars[0].name_hint == "a"
    # the body of the seq expr in the func body is a var
    assert vars[1] == VarExample["func"].body.body

    var_names = var_name_set(all_vars(VarExample["main"]))
    assert var_names == {"x", "y", "z", "p", "q", "r", "s"}


def test_bound_vars():
    vars = bound_vars(VarExample["func"])
    assert len(vars) == 2
    assert vars[0].name_hint == "a"
    # the body of the seq expr in the func body is a bound var
    assert vars[1] == VarExample["func"].body.body

    # all the vars are bound
    var_names = var_name_set(bound_vars(VarExample["main"]))
    assert var_names == {"x", "y", "z", "p", "q", "r", "s"}

    # if we consider only the body, then the function arguments are not bound
    body_names = var_name_set(bound_vars(VarExample["main"].body))
    assert body_names == {"z", "p", "q", "r", "s"}

    # only binding is in the (normalized) body
    simple_body_vars = bound_vars(VarExample["func"].body)
    assert len(simple_body_vars) == 1
    assert simple_body_vars[0] == VarExample["func"].body.body


def test_free_vars():
    # all the vars are bound
    assert len(free_vars(VarExample["func"])) == 0
    assert len(free_vars(VarExample["main"])) == 0

    # the arguments are free if we look only at the bodies
    func_free = var_name_set(free_vars(VarExample["func"].body))
    main_free = var_name_set(free_vars(VarExample["main"].body))
    assert len(func_free) == 1
    assert len(main_free) == 2
    assert "a" in func_free
    assert main_free == {"x", "y"}

    # function that captures vars
    x = rx.Var("x", R.Tensor(ndim=-1))
    y = rx.Var("y", R.Tensor(ndim=-1))
    z = rx.Var("z", R.Tensor(ndim=-1))
    inner = rx.Function(
        [z],
        rx.op.add(x, rx.op.add(y, z)),
        ret_struct_info=R.Tensor(ndim=-1),
    )
    outer = rx.Function(
        [x, y],
        rx.Call(inner, [y]),
        ret_struct_info=R.Tensor(ndim=-1),
    )
    assert len(free_vars(outer)) == 0
    assert var_name_set(free_vars(inner)) == {"x", "y"}


def test_all_global_vars():
    # there is one call to "func"
    global_vars = all_global_vars(VarExample["main"])
    assert len(global_vars) == 1
    assert global_vars[0].name_hint == "func"

    gv1 = rx.GlobalVar("gv1")
    gv2 = rx.GlobalVar("gv2")
    gv3 = rx.GlobalVar("gv3")
    call = rx.Call(gv1, [gv2, gv3])
    call_var_names = var_name_set(all_global_vars(call))
    assert call_var_names == {"gv1", "gv2", "gv3"}


def test_called_global_vars():
    # there is one call to "func"
    global_vars = called_global_vars(VarExample["main"])
    assert len(global_vars) == 1
    assert global_vars[0].name_hint == "func"

    gv1 = rx.GlobalVar("gv1")
    gv2 = rx.GlobalVar("gv2")
    gv3 = rx.GlobalVar("gv3")
    call = rx.Call(gv1, [gv2, gv3])
    call_vars = called_global_vars(call)
    assert len(call_vars) == 1
    assert call_vars[0].name_hint == "gv1"


if __name__ == "__main__":
    pytest.main([__file__])
